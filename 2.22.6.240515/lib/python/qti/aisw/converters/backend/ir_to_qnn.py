# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import io
import numpy as np
import json
import os
import re
import sys
import tarfile
from collections import OrderedDict

try:
    from qti.aisw.converters.backend import qnn_modeltools
    from qti.aisw.converters.qnn_backend import qnn_definitions
    from . import ir_graph
    from qti.aisw.converters.common import json_serializer, modeltools
    from qti.aisw.converters.common import encodings_json_serializer
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python/ is in your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common.backend_base import BackendTranslationBase
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning, log_debug3, log_assert
from qti.aisw.converters.common.utils.translation_utils import get_si_notation
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.qnn_backend.qnn_translations import QnnTranslations
from qti.aisw.converters.qnn_backend.qnn_backend_base import QnnConverterBackendBase
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
from qti.aisw.converters.backend.qnn_quantizer import QnnQuantizer

class QnnConverterBackend(QnnConverterBackendBase):
    class ArgParser(QnnConverterBackendBase.ArgParser):
        def __init__(self):
            super(QnnConverterBackend.ArgParser, self).__init__()
            self.add_optional_argument('--overwrite_model_prefix', action='store_true',
                                       help='If option passed, model generator will use the output path name '
                                            'to use as model prefix to name functions in '
                                            '<qnn_model_name>.cpp. (Useful for running multiple models at '
                                            'once) eg: ModelName_composeGraphs. Default is to use generic '
                                            '"QnnModel_".')
            self.add_optional_argument('--exclude_named_tensors', action='store_true',
                                       help='Remove using source framework tensorNames; instead use a '
                                            'counter for naming tensors. Note: This can potentially help to '
                                            'reduce  the final model library that will be generated'
                                            '(Recommended for deploying model). Default is False.')

            self.add_optional_argument('--disable_node_validation', action='store_true',
                                       help=argparse.SUPPRESS, default=False)

            self.add_optional_argument('--disable_qnn_op_config_validation', action='store_true',
                                       help=argparse.SUPPRESS, default=False)
            self.add_optional_argument('--dump_encoding_json', action='store_true', help=argparse.SUPPRESS,
                                       default=False)
            self.add_optional_argument('--include_data_invariant_ops', action='store_true', help=argparse.SUPPRESS,
                                       default=False)

            self.add_optional_argument('--model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')

            self.add_optional_argument("--export_format",
                                       choices=["dlc", "cpp"],
                                       help=argparse.SUPPRESS,
                                       default="cpp")

    def __init__(self, args):
        super(QnnConverterBackend, self).__init__(args)
        self.quantizer = QnnQuantizer(args)
        self.overwrite_model_prefix = args.overwrite_model_prefix
        self.exclude_named_tensors = args.exclude_named_tensors
        self.dump_encoding_json = args.dump_encoding_json
        self.include_data_invariant_ops = args.include_data_invariant_ops
        self.qnn_binary_tar = None
        self.qnn_json_graph = OrderedDict()

        # TODO: Holistic solution involving axis-tracking
        self.do_qnn_op_config_validation = not args.disable_qnn_op_config_validation

        # Disables running backend op validation prior to adding node.
        # Note: Only applies for offline construction when saving model to cpp. The model.cpp
        #       will have validation flag set to value of this parameter when calling
        #       QnnModel::initialize() function
        self.do_node_validation = not args.disable_node_validation

        # used to name tensors counter based when exclude_named_tensors arg is passed
        self._tensor_name_counter = 0
        # used to hold key as actual tensor name, value as counter based name
        # to avoid increment of counter for already generated name. Additionally, this
        # overall tracks name changes from IR to QNN sanitized tensor naming.
        self._generated_tensor_name_map = {}

        self.export_format = args.export_format

        # get converter args for saving qnn .cpp/.bin
        if self.output_model_path is None:
            filename, _ = os.path.splitext(os.path.realpath(self.input_model_path))
            if self.export_format == "cpp":
                self.output_path = filename + ".cpp"
                self.output_bin_path = filename + ".bin"
                self.output_encodings_path = filename + "_net.json"
                self.output_json_path = filename + "_quantized_encoding.json"
            if self.export_format == "dlc":
                 self.output_path = filename + ".dlc"
        else:
            self.output_path = self.output_model_path
            if self.export_format == "cpp":
                self.output_bin_path = os.path.splitext(os.path.realpath(self.output_path))[0] + ".bin"
                self.output_encodings_path = os.path.splitext(os.path.realpath(self.output_path))[0] + "_net.json"
                self.output_json_path = os.path.splitext(os.path.realpath(self.output_path))[0] + "_encoding.json"

        self.model_version = args.model_version
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

    def sanitize_name(self, name):
        """
        Modifies given name to adhere with C++ naming standard as names(node or tensors) are used
        as variable name lookup in generated model.cpp
        """

        # All separators should be _ to follow C++ variable naming standard
        name = re.sub(r'\W+', "_", name)
        # prefix needed as C++ variables cant start with numbers
        return name if name[0].isalpha() else "_" + name

    def _sanitize_tensor_name(self, tensor_name):
        """ Function to support tensor name exclusion in the generated qnn_model """

        new_tensor_name = tensor_name
        if tensor_name in self._generated_tensor_name_map:
            return self._generated_tensor_name_map[tensor_name]
        elif not self.is_online_construction and self.exclude_named_tensors:
            new_tensor_name = str(self._tensor_name_counter)
            self._tensor_name_counter += 1

        new_tensor_name = self.sanitize_name(new_tensor_name)
        self._generated_tensor_name_map.update({tensor_name: new_tensor_name})

        return new_tensor_name

    def add_tensor(self, node_name, tensor_name, tensor_type, tensor: np.ndarray,
                   check_encodings=True, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
                   src_axis_format=None, tensor_axis_format=None,
                   orig_tensor_name=None, is_bias=False):
        """
        This function is called only during Py IRGraph to C IrGraph and in turn calls the C graph addTensor.
        :param node_name: the IRGraph name for node.
        :param tensor_name: name to use for the tensor
        :param tensor_type: the QNN tensor type. (i.e: NATIVE, APP_WRITE,...)
        :param tensor: np.ndarray object
        :param check_encodings: flag to check for quantization encodings for tensor. Quantization is done
                                in op/tensor agnostic manner. Hence, if any tensor specific constraint is needed
                                to keep tensor type as source framework, flag should be set to False, otherwise True
        :param tensor_data_type: the data type to use for the tensor
        :param src_axis_format: the axis format of the source framework tensor
        :param tensor_axis_format: the axis format of the QNN tensor
        :param orig_tensor_name: the IRGraph name for tensor. This can be different from tensor_name param which will
                                 be used for the QNN tensorname.(These two can differ especially given that for QNN
                                 tensorNames are sanitized to comply with C++ naming scheme).
        """

        encoding = None
        if tensor_type == qnn_definitions.QNN_TENSOR_TYPE_STATIC:
            tensor_info = self.create_tensor_info(tensor_name, tensor_type, tensor.shape,
                                                  tensor_data_type, src_axis_format, tensor_axis_format, data=tensor,
                                                  encoding=encoding, is_bias=is_bias)
        else:
            tensor_info = self.create_tensor_info(tensor_name, tensor_type, tensor.shape,
                                                  tensor_data_type, src_axis_format, tensor_axis_format, data=None,
                                                  encoding=encoding)
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
        """
        Depending on graph construction mode(online vs offline), it either calls function to Execute QnnModel addNode
        function or constructs the call string for offline model.cpp
        :param node_name: the IRGraph name for node.
        :param node_type: the QNN node type
        :param input_names: list object of strings for node inputs
        :param outputs_info: list object of tensorInfo dictionaries for node outputs.
        :param tensor_params: dictionary object for Node tensor parameters.
                                key: QNN Op param name, value: numpy.ndarray
        :param scalar_params: dictionary object for Node scalar parameters.
                                key: QNN Op param name, value: numpy scalar type
        :param macs: the macs(multiply and accumulates) value for set for operation, if applicable
        """
        # resolve package names for each node name
        node_package_name = self.resolve_package_names(node_type)

        if not self.model.add_node(node_name, node_type, node_package_name, tensor_params, scalar_params,
                                   input_names, outputs_info, self.do_qnn_op_config_validation):
            raise RuntimeError("Adding Node {} failed.".format(node_name))

    def get_tensor_map(self):
        """ Retrieve the tensor mappings """
        return self._generated_tensor_name_map

    def _reset_tensor_tracking(self):
        self._generated_tensor_name_map.clear()
        self._tensor_name_counter = 0
        self._tensors_info = {}

    def add_tensor_to_qnn_bin(self, tensor_name, tensor):
        # add the actual data to the binary file

        buf = io.BytesIO(tensor.tobytes())
        tensor_tar_info = tarfile.TarInfo(name=tensor_name + ".raw")
        tensor_tar_info.size = len(buf.getbuffer())
        self.qnn_binary_tar.addfile(tarinfo=tensor_tar_info, fileobj=buf)
        buf.close()

    def init_qnn_json_graph(self, model_cpp, model_bin, converter_command, copyright_str):
        tensors = OrderedDict()
        nodes = OrderedDict()
        op_types = set()
        self.qnn_json_graph.update([("model.cpp", model_cpp),
                                    ("model.bin", model_bin),
                                    ("converter_command", converter_command),
                                    ('copyright_str', copyright_str),
                                    ('op_types', op_types),
                                    ('Total parameters', "{} ({} MB assuming single precision float)"
                                     .format(str(self.total_graph_params_count),
                                            int(self.total_graph_params_count * 4 / (1024 ** 2)))),
                                    ('Total MACs per inference', str(get_si_notation(self.total_graph_macs,
                                                                                     self.total_graph_macs))),
                                    ("graph", OrderedDict([("tensors", tensors),
                                                           ("nodes", nodes)]))
                                    ])

    def qnn_json_graph_add_tensor(self, tensor_info, params_count=0):
        tensor = self._get_resolved_tensor_info(tensor_info)
        if params_count:
            tensor[tensor_info['name']].update([("params_count",
                                                 str(get_si_notation(params_count, self.total_graph_params_count)))])
        self.qnn_json_graph['graph']['tensors'].update(tensor)

    def qnn_json_graph_add_node(self, node_name, node_pkg_name, node_type, input_names, outputs_info, tensor_params={},
                                scalar_params={}, macs=0):

        # add the output tensor infos to the tensors section in json
        output_tensor_names = []
        for tensor_info in outputs_info:
            output_tensor_names.append(tensor_info['name'])
            self.qnn_json_graph_add_tensor(tensor_info)

        # resolve the enum values for qnn tensors and scalars
        tensor_params_ = OrderedDict()
        for name, tensor_info in tensor_params.items():
            tensor_params_.update([(name, self._get_resolved_tensor_info(tensor_info))])

        scalar_params_ = OrderedDict()
        for name, scalar_info in scalar_params.items():
            data_type, value = scalar_info
            value = value.item() if isinstance(value, np.generic) else value
            scalar_params_.update([(name, {int(data_type): value})])

        node_info = OrderedDict([
            ("package", node_pkg_name),
            ("type", str(node_type)),
            ("tensor_params", tensor_params_),
            ("scalar_params", scalar_params_),
            ("input_names", input_names),
            ("output_names", output_tensor_names)
        ])
        if macs:
            node_info.update([("macs_per_inference", str(get_si_notation(macs, self.total_graph_macs)))])

        self.qnn_json_graph['graph']['nodes'].update({node_name: node_info})
        self.qnn_json_graph['op_types'].add(str(node_type))

    def dump_qnn_graph_json(self, filepath):
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, np.int32):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    # Let the base class default method raise the TypeError
                    return json.JSONEncoder.default(self, obj)

        with open(filepath, 'w') as f:
            json.dump(self.qnn_json_graph, f, indent=2, cls=Encoder)
            log_info("Model Network JSON saved at: %s " % filepath)

    def construct_model(self, graph, modelgen_backend, modelgen_interface, context, graph_configs_info,
                        num_graph_configs_info):
        log_info("Constructing QNN Model...")
        self.is_online_construction = True
        self._reset_tensor_tracking()

        # set up the package information for each op type in the graph
        self.set_package_dict(graph)
        if not self.model.init_online_model_generator(modelgen_backend, modelgen_interface, context, False,
                                                      graph_configs_info, num_graph_configs_info):
            raise ValueError("Online model init failed when lowering to qnn.")

        try:
            QnnTranslations.apply_method_to_all_ops(BackendTranslationBase.ADD_OP_TO_BACKEND, graph, self)
        except BaseException as e:
            print('Error constructing online model!')
            raise e

        log_info("Construction complete!")
        self.is_online_construction = False
        return self.model.get_model()

    def resolve_package_names(self, node_type):
        default_package_name = "qti.aisw"
        package_names = [default_package_name]
        for package_name, node_types in self.package_name_to_qnn_op_types.items():
            if node_type.lower() in node_types:
                # Custom op override ops are added to the default op collection during
                # conversion. If quantize_with_default_package is set, the default qti.aisw
                # package will be set so it can be used for quantization. If online
                # construction is not set (i.e no quantization) then any user provided package names
                # are used.
                if self.is_online_construction and self.quantize_with_default_package and \
                        node_type.lower() in QnnCustomOpFactory.default_op_collection:
                    continue
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
                if isInput and entry['QuantParam']['Type'] == 'QNN_DEFINITION_DEFINED' and not self.quantizer.opts.input_list:
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

    def set_package_dict(self, graph):
        if self.package_name:
            package_name_dict = {self.package_name: [node.op.type for node in graph.list_nodes()[1:]]}
        elif QnnCustomOpFactory.package_resolver:
            package_name_dict = QnnCustomOpFactory.package_resolver
        else:
            package_name_dict = dict()

        # if there is no package lib provided, then it is assumed that the default qti package will be
        # will used to quantize any custom ops.
        if self.op_package_lib:
            self.quantize_with_default_package = False

        self.package_name_to_qnn_op_types = package_name_dict

    def save_to_dlc(self, ir_graph):
        log_info("Saving DLC Model...")
        self.dlc_serializer.initialize()
        self.dlc_serializer.serialize(ir_graph)
        self.dlc_serializer.finish()

    def save_to_cpp(self, ir_graph):
        log_info("Saving QNN Model...")
        try:
            # initialize binary file
            self.qnn_binary_tar = tarfile.TarFile(self.output_bin_path, 'w')
            self.model = qnn_modeltools.QnnModel()
            model_prefix = ""
            if self.overwrite_model_prefix:
                # get name prefix to use for functions in model.cpp. e.g ModelPrefix_composeGraphs()
                model_prefix, _ = os.path.splitext(os.path.basename(self.output_path))
                # do PascalCase to follow c++ naming standard
                model_prefix = self.sanitize_name(str(model_prefix).title()).replace("_", "")

            if not self.model.init_model_src_serializer(self.output_path, model_prefix,
                                                           self.copyright_str, self.converter_command,
                                                           self.do_node_validation, self):
                raise ValueError("Model init failed when lowering to qnn.")

            self.model.serialize(ir_graph, self)
            if not self.model.save():
                raise ValueError("Model save failed when lowering to qnn.")
            log_info("Model CPP saved at: %s " % self.output_path)
            qnn_raw_files = self.qnn_binary_tar.getmembers()
            self.qnn_binary_tar.close()
            if not len(qnn_raw_files):
                log_warning("No raw files found for Model. Saving Model BIN skipped.")
                os.path.exists(self.output_bin_path) and os.remove(self.output_bin_path)
            else:
                log_info("Model BIN saved at: %s " % self.output_bin_path)

            # Serialize QNNIR to Net JSON
            self.json_serializer = json_serializer.IrJsonSerializer()
            self.json_serializer.init_json_serializer(os.path.realpath(self.output_path), self.output_bin_path,
                                                      self.converter_command, self.copyright_str)
            self.json_serializer.serialize(ir_graph)
            ir_json = self.json_serializer.get_graph_json()
            with open(self.output_encodings_path, "w") as json_file:
                json_file.write(ir_json)

            # Serialize QNNIR to ENCODING JSON
            if (self.quantizer.opts.input_list or self.quantizer.use_fallback_to_float) and self.dump_encoding_json:
                self.encodings_json_serializer = encodings_json_serializer.IrEncodingsJsonSerializer(self.output_json_path, self.include_data_invariant_ops)
                self.encodings_json_serializer.serialize(ir_graph)
                ir_json2 = self.encodings_json_serializer.get_graph_json()
                with open(self.output_json_path, "w") as json_file:
                    json_file.write(ir_json2)
                log_info("encodings JSON saved at: %s " % self.output_json_path)
        except BaseException as e:
            # clean file if errors
            os.path.exists(self.output_path) and os.remove(self.output_path)
            os.path.exists(self.output_bin_path) and os.remove(self.output_bin_path)
            os.path.exists(self.output_encodings_path) and os.remove(self.output_encodings_path)
            os.path.exists(self.output_json_path) and os.remove(self.output_json_path)
            raise e

    def save(self, graph):
        # set up the package information for each op type in the graph
        self.set_package_dict(graph)

        # To handle the case when quantized custom inputs are to be provided to a non-quantized model
        if graph.user_custom_io and not self.quantizer.opts.input_list and not self.quantizer.use_fallback_to_float:
            self.apply_custom_io_dequant(graph)

        if self.quantizer.opts.input_list or self.quantizer.use_fallback_to_float:
            self.is_online_construction = True
        ir_graph = self.get_ir_graph(graph)
        self.is_online_construction = False

        if graph.preserve_io_datatype_passed:
            self.quantizer.opts.use_native_input_dtype = True

        self.quantizer.quantize(ir_graph, self)
        if self.quantizer.ir_graph_reader:
            ir_graph = self.quantizer.ir_graph_reader.get_ir_graph()

        if graph.custom_datatype_tensors:
            ir_graph.modify_io_datatype(graph.custom_datatype_tensors)

        if graph.preserve_io_datatype_passed:
            ir_graph.modify_io_datatype(graph.preserve_datatype_tensors)

        if graph.dump_qairt_io_config_yaml:
            yaml_dump_dir = os.path.dirname(os.path.abspath(self.output_path))
            yaml_file_name = yaml_dump_dir + "/" + graph.dump_yaml_file_name
            f = open(yaml_file_name, 'w')
            f.write('\n'.join(graph.dump_yaml_file_data))
            log_info("Dumped IO config in file: %s " % graph.dump_yaml_file_name)
            f.close()

        if self.export_format == "cpp":
            self.save_to_cpp(ir_graph)
        elif self.export_format == "dlc":
            self.save_to_dlc(ir_graph)
        log_info("Conversion complete!")
