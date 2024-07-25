# =============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import os
import sys
import argparse

# modeltools is present in common as well as dlc_utils
# try importing from common first (currently used by QNN) and if not found import from dlc_utils (used by SNPE)
# TODO: remove modeltools from dlc_utils and update all SNPE tools to use modeltools from common
try:
    from qti.aisw.converters.common import modeltools
except ImportError as ie1:
    from qti.aisw.dlc_utils import modeltools

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.backend_base import BackendTranslationBase
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.translation_utils import get_si_notation
from qti.aisw.converters.qnn_backend.qnn_translations import QnnTranslations
from qti.aisw.converters.qnn_backend.qnn_backend_base import QnnConverterBackendBase
from qti.aisw.converters.qnn_backend.qnn_mappings import *


from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory as CustomFactory



# TODO: updated inheritance to ConverterBackend once alignment of Ops are complete
class DLCBackend(QnnConverterBackendBase):
    class ArgParser(QnnConverterBackendBase.ArgParser):
        def __init__(self, **kwargs):
            super(DLCBackend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')
            self.add_optional_argument('--validation_target', nargs=2,
                                       action=validation_utils.ValidateTargetArgs,
                                       help="Note: This option is deprecated. \n"
                                            "A combination of processor and runtime target against which model "
                                            "will be validated. \n"
                                            "Choices for RUNTIME_TARGET: \n   {cpu, gpu, dsp}. \n"
                                            "Choices for PROCESSOR_TARGET: \n"
                                            "   {snapdragon_801, snapdragon_820, snapdragon_835}.\n"
                                            "If not specified, will validate model against "
                                            "{snapdragon_820, snapdragon_835} across all runtime targets.",
                                       metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'),
                                       default=[], )
            self.add_optional_argument('--strict', dest="enable_strict_validation",
                                       action="store_true",
                                       default=False,
                                       help="Note: This option is deprecated. \n"
                                            "If specified, will validate in strict mode whereby model will not "
                                            "be produced if it violates constraints of the specified validation "
                                            "target. If not specified, will validate model in permissive mode "
                                            "against the specified validation target.")
            self.add_optional_argument("--udo_config_paths", "-udo", nargs='+',
                                       dest="custom_op_config_paths",
                                       action=validation_utils.check_json(),
                                       help="Path to the UDO configs (space separated, if multiple)")

    class ArgParserv2(QnnConverterBackendBase.ArgParserv2):
        def __init__(self, **kwargs):
            super(DLCBackend.ArgParserv2, self).__init__(**kwargs)
            self.add_optional_argument('--model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')
            self.add_optional_argument('--disable_qnn_op_config_validation', action='store_true',
                                       help=argparse.SUPPRESS, default=False)
    def __init__(self, args):
        super(DLCBackend, self).__init__(args)
        # get converter args for saving dlc
        if self.output_model_path is None:
            filename, _ = os.path.splitext(os.path.realpath(self.input_model_path))
            self.output_path = filename + ".dlc"
        else:
            self.output_path = self.output_model_path

        self.model_version = args.model_version
        self.serialize_with_suppl_attr = True

        if hasattr(args, 'validation_target'):
            self.validation_target = args.validation_target
            if args.validation_target:
                log_warning("--validation_target is deprecated.")
        if hasattr(args, 'strict'):
            self.enable_strict_validation = args.enable_strict_validation
            if args.enable_strict_validation:
                log_warning("--strict is deprecated.")

        self.do_qnn_op_config_validation = True
        if hasattr(args, 'strict'):
            self.do_qnn_op_config_validation = not args.disable_qnn_op_config_validation

        # Ensure model version fits in 64 bytes to match dlcv3
        model_version = self.model_version
        if model_version:
            model_version = model_version[:64]
        else:
            model_version = ''

        self.dlc_serializer = modeltools.IrDlcSerializer(self.output_path,
                                                         self.copyright_str,
                                                         model_version,
                                                         self.converter_command)

    # TODO: Cleanup when all ops are aligned to QNN
    """ Start of clean up """
    def add_tensor(self, node_name, tensor_name, tensor_type, tensor: np.ndarray,
                   check_encodings=True, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
                   src_axis_format=None, tensor_axis_format=None, orig_tensor_name=None, is_bias=False):
        data = None
        if tensor_type == qnn_definitions.QNN_TENSOR_TYPE_STATIC:
            data = tensor
        tensor_info = self.create_tensor_info(tensor_name, tensor_type, tensor.shape,
                                              tensor_data_type, src_axis_format, tensor_axis_format, data=data,
                                              encoding=None, is_bias=is_bias)

        is_quantizable = True
        if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 or not check_encodings:
            is_quantizable = False

        if not self.model.add_tensor(node_name, tensor_info, is_quantizable=is_quantizable):
            raise RuntimeError("Adding Tensor {} for Node {} failed.".format(node_name, tensor_name))

    def add_custom_input_tensor(self, node_name, tensor_name, tensor_type, tensor: np.ndarray,
                                tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32, tensor_axis_format=None,
                                quant_params = None, params_count=0):
        """
        Function to add a tensor with the quant_params obtained from Custom IO YAML file.
        :param node_name: the IRGraph name for node.
        :param tensor_name: name to use for the tensor
        :param tensor_type: the QNN tensor type. (i.e: NATIVE, APP_WRITE,...)
        :param tensor: np.ndarray object
        :param tensor_data_type: the data type to use for the tensor
        :param tensor_axis_format: the axis format of the QNN tensor
        :param quant_params: Dictionary containing information regarding the scale and offset
                            of custom input tensor.
        :param params_count: the size of weights for the operation, if applicable
        """

        # TODO: Directly accept FXP8 from the config file rather than combination
        #       of INT8 and QuantParams to infer FXP

        if quant_params:
            if tensor_data_type == ir_graph.QNN_DATATYPE_UINT_8:
                tensor_data_type = ir_graph.QNN_DATATYPE_UFIXED_POINT_8
            elif tensor_data_type == ir_graph.QNN_DATATYPE_INT_8:
                tensor_data_type = ir_graph.QNN_DATATYPE_SFIXED_POINT_8

        tensor_info = self.create_tensor_info(tensor_name, tensor_type, tensor.shape,
                                              tensor_data_type, tensor_axis_format,
                                              data=None, encoding=None)
        tensor_info['quant_params'] = quant_params
        is_quantizable = False
        if quant_params:
            is_quantizable = True
        if not self.model.add_tensor(node_name, tensor_info, is_quantizable=is_quantizable):
            raise RuntimeError("Adding Tensor {} for Node {} failed.".format(node_name, tensor_name))

    def add_node(self, node_name, node_type, input_names, outputs_info, tensor_params={}, scalar_params={},
                 macs=0):
        # resolve package names for each node name
        node_package_name = self.resolve_package_names(node_type)

        if not self.model.add_node(node_name, node_type, node_package_name, tensor_params, scalar_params,
                                   input_names, outputs_info, self.do_qnn_op_config_validation):
            raise RuntimeError("Adding Node {} failed.".format(node_name))

    @staticmethod
    def sanitize_name(name):
        return name

    @staticmethod
    def _sanitize_tensor_name(tensor_name):
        return tensor_name

    """ End of clean up """

    # overrides the set_package_dict method in qnn_backend_base
    # to correctly set the package dict info for snpe 2.0 udo
    def set_package_dict(self, graph):
        if self.package_name:
            package_name_dict = {self.package_name: [node.op.type for node in graph.list_nodes()[1:]]}
        elif CustomFactory.package_resolver:
            package_name_dict = CustomFactory.package_resolver
        else:
            package_name_dict = dict()

        # if there is no package lib provided, then it is assumed that the default qti package will be
        # will used to quantize any custom ops.
        if self.op_package_lib:
            self.quantize_with_default_package = False

        self.package_name_to_qnn_op_types = package_name_dict

    # overrides the resolve_package_names method in qnn_backend_base
    # to correctly resolve the package names for snpe 2.0 udo
    def resolve_package_names(self, node_type):
        default_package_name = qnn_definitions.QNN_OP_PACKAGE_NAME_QTI_AISW
        package_names = [default_package_name]
        for package_name, node_types in self.package_name_to_qnn_op_types.items():
            if node_type.lower() in node_types:
                package_names.append(package_name)
        return package_names[-1]

    def apply_custom_io_dequant(self, graph):
        for entry in graph.user_custom_io:
            buffer_name = str(entry['IOName'])
            log_assert(buffer_name in graph.buffers,"Incorrect IOName provided in custom IO YAML file. Buffer {} not found in graph"
                       .format(buffer_name))
            if 'Datatype' in entry:
                if entry['Datatype'] not in ['int8', 'uint8']:
                    log_assert(self.c_ir_graph is None,"To pass non-quantized inputs/output to quantized model, use the --input_data_type/--output_data_type\
                        option of qnn-net-run. {} datatype provided for Buffer {}".format(entry['Datatype'], buffer_name))
            if "QuantParam" in entry:
                # Default datatype for quantized model is uint8 in case of custom IO.
                custom_datatype = 'uint8'
                if 'Datatype' in entry:
                    custom_datatype = entry['Datatype']
                if custom_datatype == 'int8':
                    log_assert(self.c_ir_graph is None,"Custom IO does not support int8 inputs to quantized model. int8 datatype provided for Buffer {}"
                               .format(buffer_name))
                isInput = False
                # Check if the buffer name provided is input buffer
                for node in graph.get_input_nodes_to_graph():
                    if buffer_name in node.output_names:
                        isInput = True
                #To handle the case when quantized custom inputs are to be provided to a non-quantized model
                if isInput and entry['QuantParam']['Type'] == 'QNN_DEFINITION_DEFINED':
                    consumers = [str(name) for name in graph.buffers[buffer_name].consumers]

                    # Insert a dequant op after the input node. The params for the dequant op are obtianed from graph.quantization_params which
                    # is in-turn filled with the information obtianed from the custom IO YAML file.
                    node = graph.buffers[buffer_name].producer
                    node.op.input_dtype = custom_datatype
                    dequant_op = op_adapter.DequantizeOp(buffer_name+"_dequant", bw=graph.quantization_params[buffer_name]['output_encodings'][0]['bw'],
                                                         scale=graph.quantization_params[buffer_name]['output_encodings'][0]['scale'][0],
                                                         offset=graph.quantization_params[buffer_name]['output_encodings'][0]['offset'][0],
                                                         is_symmetric=graph.quantization_params[buffer_name]['output_encodings'][0]['is_symmetric'])
                    graph.inject(dequant_op, buffer_name, buffer_name+"_custom_dequant", consumer_names=consumers)

    def save(self, graph):
        if graph.dump_qairt_io_config_yaml:
            yaml_dump_dir = os.path.dirname(os.path.abspath(self.output_path))
            yaml_file_name = yaml_dump_dir + "/" + graph.dump_yaml_file_name
            f = open(yaml_file_name, 'w')
            f.write('\n'.join(graph.dump_yaml_file_data))
            log_info("Dumped IO config in file: %s " % graph.dump_yaml_file_name)
            f.close()

        self.dlc_serializer.initialize()
        log_info(code_to_message.get_progress_message("INFO_INITIALIZATION_SUCCESS"))
        # set up the package information for each op type in the graph
        self.set_package_dict(graph)

        # To handle the case when quantized custom inputs are to be provided to the model
        if graph.user_custom_io:
            self.apply_custom_io_dequant(graph)

        # TODO: pass graph as-is
        ir_graph = self.get_ir_graph(graph)
        self.dlc_serializer.serialize(ir_graph)
        log_info(code_to_message.get_progress_message("INFO_CONVERSION_SUCCESS"))
        self.dlc_serializer.finish()
        log_info(code_to_message.get_progress_message("INFO_WRITE_SUCCESS"))
