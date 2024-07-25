# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import numpy as np
import sys

from qti.aisw.converters.common.backend_base import BackendTranslationBase
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.translation_utils import get_si_notation
from qti.aisw.converters.qnn_backend.qnn_translations import QnnTranslations

try:
    from . import qnn_definitions
    from . import ir_graph
    from . import qnn_ir
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that <SDK_ROOT>/lib/python is in your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.converter_utils import log_assert
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrder
from qti.aisw.converters.common.backend_base import ConverterBackend
from qti.aisw.converters.qnn_backend.qnn_mappings import *
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory


class QnnConverterBackendBase(ConverterBackend):
    class ArgParser(ConverterBackend.ArgParser):
        def __init__(self):
            super(QnnConverterBackendBase.ArgParser, self).__init__()
            #TODO: Remove the deprecated argument "--float_bw"/"--float_bias_bw" in future release
            self.add_optional_argument('--float_bitwidth', type=int, default=32,
                                       help='Use the --float_bitwidth option to convert the graph to the specified float'
                                       ' bitwidth, either 32 (default) or 16.')
            self.add_optional_argument('--float_bw', type=int, default=32,
                                       help='Note: --float_bw is deprecated, use --float_bitwidth.')
            self.add_optional_argument('--float_bias_bitwidth', type=int, default=0,
                                       help='Use the --float_bias_bitwidth option to select the bitwidth to use for '
                                            'float bias tensor')
            self.add_optional_argument('--float_bias_bw', type=int, default=0,
                                       help='Note: --float_bias_bw is deprecated, use --float_bias_bitwidth.')

    class ArgParserv2(ConverterBackend.ArgParserv2):
        def __init__(self):
            super(QnnConverterBackendBase.ArgParserv2, self).__init__()
            self.add_optional_argument('--float_bitwidth', type=int, default=32,
                                       help='Use the --float_bitwidth option to convert the graph to the specified float'
                                            ' bitwidth, either 32 (default) or 16.')
            self.add_optional_argument('--float_bias_bitwidth', type=int, default=0,
                                       help='Use the --float_bias_bitwidth option to select the bitwidth to use for '
                                            'float bias tensor')
    def __init__(self, args):
        super(QnnConverterBackendBase, self).__init__(args)
        self.quantization_overrides = args.quantization_overrides
        self.input_network = args.input_network

        if "--float_bw" in sys.argv:
            log_warning("--float_bw option is deprecated, use --float_bitwidth.")
            if "--float_bitwidth" in sys.argv:
                raise Exception("Invalid combination: --float_bw and --float_bitwidth "
                                "cannot be provided at the same time.")
        if "--float_bias_bw" in sys.argv:
            log_warning("--float_bias_bw option is deprecated, use --float_bias_bitwidth.")
            if "--float_bias_bitwidth" in sys.argv:
                raise Exception("Invalid combination: --float_bias_bw and --float_bias_bitwidth "
                                "cannot be provided at the same time.")

        self.float_bitwidth = args.float_bitwidth
        if "--float_bitwidth" not in sys.argv and "--float_bw" in sys.argv:
            self.float_bitwidth = args.float_bw

        self.float_bias_bitwidth = args.float_bias_bitwidth
        if "--float_bias_bitwidth" not in sys.argv and "--float_bias_bw" in sys.argv:
            self.float_bias_bitwidth = args.float_bias_bw

        self.total_graph_macs = 0
        self.total_graph_params_count = 0
        self.serialize_with_suppl_attr = False

        # stores a mapping of all known packages to all its associated op types.
        self.package_name_to_qnn_op_types = {}
        self.quantize_with_default_package = True

        self.is_online_construction = False
        # When leveraging quantization if this C ir graph is set, pull tensor data from
        # here rather than the python graph
        self.c_ir_graph = None
        self.c_utils = ir_graph.IrUtils()

        # holds all created tensor info across different translations
        self._tensors_info = {}

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

    def resolve_package_names(self, node_type):
        default_package_name = qnn_definitions.QNN_OP_PACKAGE_NAME_QTI_AISW
        package_names = [default_package_name]

        return package_names[-1]

    def check_qnn_type_is_custom(self, node_type):
        if self.resolve_package_names(node_type) == qnn_definitions.QNN_OP_PACKAGE_NAME_QTI_AISW:
            return False
        return True

    def process_float_datatype(self, is_bias=False):
        if self.is_online_construction or self.float_bitwidth == 32 or (self.float_bias_bitwidth == 32 and is_bias):
            return ir_graph.QNN_DATATYPE_FLOAT_32
        elif self.float_bitwidth == 16:
            return ir_graph.QNN_DATATYPE_FLOAT_16
        else:
            raise ValueError('Invalid float bitwidth = ' + str(self.float_bitwidth))

    @staticmethod
    def is_float(data_type):
        if ((data_type == ir_graph.QNN_DATATYPE_FLOAT_32) or
                (data_type == ir_graph.QNN_DATATYPE_FLOAT_16)):
            return True
        else:
            return False

    def create_unique_qnn_tensor_name(self, node_name, tensor_name):
        """
        Useful for naming static tensors whose names are not unique across nodes.
        Eg: weights, strides Since node_names are required to be unique in IR, prefixing the
        tensorName will thus add uniqueness.`
        """
        node_name = self.sanitize_name(node_name)
        tensor_name = self.sanitize_name(tensor_name)
        unique_tensor_name = node_name + "_" + tensor_name
        return unique_tensor_name

    @staticmethod
    def get_qnn_quant_params(encoding):
        """
        Queries encoding to construct a QNN style dictionary info for quantization parameters
        :param encoding: IrQuantizationData object for a tensor
        :return: dictionary for quantization parameters and (the raw IrQuantizationInfo or
                 IrAxisQuantization object depending on quantization type)
        :raises ValueError if the encoding object passed has unsupported quantization type for constructing
                           the dictionary info object
        """
        if encoding is None:
            quant_params = {
                "definition": qnn_definitions.QNN_DEFINITION_UNDEFINED,
                "encoding": ir_graph.QNN_QUANTIZATION_ENCODING_UNDEFINED,
                "scale_offset": {"scale": 0.0, "offset": 0},
            }
            return quant_params, None

        quant_params = {"definition": qnn_definitions.QNN_DEFINITION_DEFINED}
        if encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
            tensor_encoding = encoding.encInfo
            quant_params.update({"encoding": ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                 "scale_offset": {"scale": tensor_encoding.scale, "offset": tensor_encoding.offset},
                                 })
        elif encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
            tensor_encoding = encoding.axisEncInfo
            scale_offsets = []
            for q in tensor_encoding.encInfos:
                scale_offsets.append({"scale": q.scale, "offset": q.offset})
            quant_params.update({"encoding": ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
                                 "axis_scale_offset": {"axis": tensor_encoding.axis,
                                                       "num_scale_offsets": len(tensor_encoding.encInfos),
                                                       "scale_offsets": scale_offsets}})
        elif encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
            tensor_encoding = encoding.encInfo
            quant_params.update({"encoding": ir_graph.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET,
                                 "bw_scale_offset": {"bitwidth": tensor_encoding.bw,
                                                     "scale": tensor_encoding.scale,
                                                     "offset": tensor_encoding.offset},
                                 })
        elif encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
            tensor_encoding = encoding.axisEncInfo
            scales = []
            offsets = []
            # Offsets don't need to be included if they are all 0
            get_offsets = any(e.offset != 0 for e in tensor_encoding.encInfos)
            for q in tensor_encoding.encInfos:
                scales.append(q.scale)
                if get_offsets:
                    offsets.append(q.offset)
            quant_params.update({"encoding": ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET,
                                 "bw_axis_scale_offset": {"bitwidth": tensor_encoding.encInfos[0].bw,
                                                          "axis": tensor_encoding.axis,
                                                          "num_elements": len(tensor_encoding.encInfos),
                                                          "scales": scales,
                                                          "offsets": offsets}})
        else:
            raise ValueError("Unsupported quantization encoding type: {}".format(encoding.type))

        return quant_params, tensor_encoding

    def create_tensor_info(self, tensor_name, tensor_type, dims, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
                           src_axis_format=None, tensor_axis_format=None, data=None, encoding=None, quantizable=True,
                           is_bias=False, sparse_params=None):
        """
        Constructs a python dictionary variant of the Qnn TensorInfo struct object for a given tensor
        :param tensor_name: name to use for the tensor
        :param tensor_type: the IrTensorType. (i.e: NATIVE, APP_WRITE,...)
        :param dims: a list object for the shape of the tensor
        :param tensor_data_type: the data type to use for the tensor
        :param src_axis_format: the axis format of the source framework tensor
        :param tensor_axis_format: the axis format of the QNN tensor
        :param data: a numpy ndarray that stores static tensor.
        :param encoding: IrQuantizationData cpp class object.
        :param quantizable: flag to indicate if tensor is quantizable.
        :param sparse_params: sparse params for the tensor
        :return:
        """

        quant_params, _ = self.get_qnn_quant_params(encoding)
        if quantizable and self.is_float(tensor_data_type):
            tensor_data_type = self.process_float_datatype(is_bias)
            if tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_16 and data is not None:
                data = self.c_utils.buffer_fp32_to_fp16(data.flatten())

        tensor_info = {
            "id": 0,  # don't set by client
            "name": tensor_name,
            "type": tensor_type,
            "dataFormat": qnn_definitions.QNN_TENSOR_DATA_FORMAT_DENSE,
            "data_type": tensor_data_type,
            "quant_params": quant_params,
            "dims": dims,
            "quantizable": quantizable
        }

        if data is not None:
            tensor_info["data"] = data

        if src_axis_format is not None:
            tensor_info["src_axis_format"] = src_axis_format
        else:
            tensor_info["src_axis_format"] = AxisTracker.AxisFormat.NOT_YET_DEFINED

        if tensor_axis_format is not None:
            tensor_info["axis_format"] = tensor_axis_format
        else:
            tensor_info["axis_format"] = AxisOrder().get_axis_format(len(dims))

        if sparse_params is not None:
            if sparse_params.layout != ir_graph.QNN_SPARSE_LAYOUT_UNDEFINED:
                tensor_info["sparse_params"] = sparse_params
                tensor_info["data_format"] = qnn_definitions.QNN_TENSOR_DATA_FORMAT_SPARSE

        self._tensors_info.update({tensor_name: tensor_info})
        return tensor_info

    def retrieve_tensor_info(self, tensor_name):
        """
        Queries the dictionary of tensor info objects that tracks all tensor additions for the network.
        :param tensor_name: the tensor name to lookup
        :return: the tensor_info dictionary object associated with the tensor name
        :raises: ValueError if tensor name not found
        """
        if tensor_name not in self._tensors_info:
            raise ValueError("Requested tensor name ({}) not found.".format(tensor_name))
        return self._tensors_info[tensor_name]

    def update_tensors_info(self, tensor_info, new_tensor_name):
        """
        Updates dictionary entry for a tensor_info object to a new tensor name. The passed in tensor_info
        object's name field also gets updated.
        Note: the new tensor name will be sanitized per converter naming requirements
        :param tensor_info: the tensor_info object to lookup for updating
        :param new_tensor_name: the new tensor name to use for updating
        :raises: ValueError if the tensor_info is not found
        """
        old_tensor_name = tensor_info['name']
        if old_tensor_name not in self._tensors_info:
            raise ValueError("Requested update for tensor name ({}) not found.".format(old_tensor_name))

        del self._tensors_info[old_tensor_name]
        tensor_info['name'] = new_tensor_name
        self._tensors_info[new_tensor_name] = tensor_info

    @staticmethod
    def update_quant_param_info(node, graph, backend, output_tensor_info, recompute_quant_params=True):
        # TODO: only work for QNN workflow
        if backend.c_ir_graph is None:
            return
        producer_encoding = backend.get_producer_encoding(node, graph)
        quant_params, producer_tensor_encoding = backend.get_qnn_quant_params(producer_encoding)
        if producer_tensor_encoding is None:
            return
        input_tensor_bw = producer_tensor_encoding.bw
        old_num_steps = pow(2, input_tensor_bw) - 1
        new_scale = old_scale = producer_tensor_encoding.scale
        new_offset = old_offset = producer_tensor_encoding.offset
        consumer_node = graph.get_op_output_nodes(node)

        if consumer_node:
            # currently only those consumer nodes that got just one quantized output tensor are supported
            # we loop over all output tensors and break at the first quantized output
            for output_name in consumer_node[0].output_names:
                consumer_output_tensor = backend.c_ir_graph.get_output_tensor(output_name)
                if consumer_output_tensor is not None and consumer_output_tensor.is_quantized():
                    output_tensor_info['data_type'] = consumer_output_tensor.data_type()
                    consumer_output_encoding = consumer_output_tensor.get_encoding()
                    consumer_output_tensor_qinfo = consumer_output_encoding.encInfo
                    output_tensor_bw = consumer_output_tensor_qinfo.bw
                    if input_tensor_bw != output_tensor_bw:
                        if recompute_quant_params:
                            # recompute scale offset params using theoretical formula
                            new_num_steps = pow(2, output_tensor_bw) - 1
                            new_scale = (old_scale * old_num_steps) / new_num_steps
                            new_offset = round((old_offset * old_scale) / new_scale)
                        else:
                            # set scale offset params using a performance optimized scale factor
                            if input_tensor_bw == 8 and output_tensor_bw == 16:
                                new_scale = old_scale / 256.0
                                new_offset = round(old_offset * 256.0)
                            elif input_tensor_bw == 16 and output_tensor_bw == 8:
                                new_scale = old_scale * 256.0
                                new_offset = round(old_offset / 256.0)
                            else:
                                raise ValueError("Activation bitwidth conversion from {} to {} is not supported. "
                                                 "Supported conversions are 8->16 and 16->8.".format(input_tensor_bw,
                                                                                                     output_tensor_bw))
                        break

            # update scale offset as per consumer node's bw
            quant_params['scale_offset']['scale'] = new_scale
            quant_params['scale_offset']['offset'] = new_offset
            output_tensor_info['quant_params'] = quant_params

    def get_producer_encoding(self, node, graph):
        # TODO: only work for QNN workflow
        if hasattr(self, "c_ir_graph") and self.c_ir_graph is not None:
            input_nodes = graph.get_op_input_nodes(node)
            input_name = node.input_names[0]

            # Quantize node is preceded by Dequantize node
            if input_nodes[0].op.type == op_adapter.DequantizeOp.TRANSLATION_KEY:
                input_name = input_nodes[0].input_names[0]

            input_tensor = self.c_ir_graph.get_output_tensor(input_name)
            if input_tensor.is_quantized():
                return input_tensor.get_encoding()

        return None

    def get_output_info(self, graph, tensor_name, tensor_type=qnn_definitions.QNN_TENSOR_TYPE_NATIVE, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
                        check_encodings=True, orig_tensor_name=None):
        """
        Constructs a python dictionary variant of the Qnn TensorInfo struct object for a given output tensor
        :param tensor_name: name to use for the tensor
        :param tensor_shape: a list object for the shape of the tensor
        :param tensor_type: the IrTensorType. (i.e: NATIVE, APP_WRITE,...)
        :param tensor_data_type: the data type to use for the tensor
        :param src_axis_format: the axis format of the source framework tensor
        :param tensor_axis_format: the axis format of the QNN tensor
        :param check_encodings: flag to check for quantization encodings for each tensor output of node. Quantization
                                is done in op_agnostic manner. Hence, if any op specific constraint is needed
                                to keep tensor type as source framework, flag should be set to False, otherwise True
        :return: a dictionary object with tensorinfo
        """
        encoding = None

        if not orig_tensor_name:
            orig_tensor_name = tensor_name

        output_buf = graph.get_buffer(tensor_name)
        tensor_shape = output_buf.shape
        # retrieve source and QNN tensor axis format from output buffer
        src_axis_format = output_buf.get_src_axis_format()
        tensor_axis_format = output_buf.get_axis_format()
        sparse_params = output_buf.get_sparse_params()

        # tensor name can follow (colon):num which is stripped to check for command-line name
        tensor_name_ = tensor_name.split(":")[0]

        # TODO: Remove once other frameworks than onnx follows new APP_READ tensor alignment
        # Currently QNN other than onnx framework relies on output buffer with no consumers
        # Enabling the new APP_READ tensor assignment for qnn-onnx-converter
        # AISW - 69823 for tracking
        if self.input_network.split(".")[-1]!='onnx':
            if len(output_buf.consumers) == 0:
                tensor_type = qnn_definitions.QNN_TENSOR_TYPE_APP_READ

        # determine if tensor is native(i.e. intra graph) or output based on whether:
        #   - Default output is present in the network (default), or
        #   - User has provided output_name as output for the network.
        if tensor_name_ in graph.output_names or tensor_name in graph.output_names:
            tensor_type = qnn_definitions.QNN_TENSOR_TYPE_APP_READ

        if hasattr(self, "c_ir_graph") and self.c_ir_graph is not None and check_encodings:
            t = self.c_ir_graph.get_output_tensor(orig_tensor_name)
            if t is not None and t.is_quantized():
                encoding = t.get_encoding()
                tensor_data_type = t.data_type()

        output_tensor_info = self.create_tensor_info(tensor_name,
                                                     tensor_type,
                                                     tensor_shape,
                                                     tensor_data_type,
                                                     src_axis_format,
                                                     tensor_axis_format,
                                                     encoding=encoding,
                                                     quantizable=check_encodings,
                                                     sparse_params=sparse_params)

        return output_tensor_info

    def get_outputs_info(self, node, graph, tensor_data_type=None, check_encodings=True, original_output=None):
        """
        Constructs a python dictionary variant of the Qnn TensorInfo struct object for a given node's output
        :param node: the node object to construct output tensorInfo
        :param graph: the IROpgraph object
        :param tensor_data_type: the data type to use for each of the output tensors
        :param check_encodings: flag to check for quantization encodings for each tensor output of node. (see
                                get_output_info for further detail on param)
        :return: a list of the construct output info objects.
        """
        outputs_info = []
        if tensor_data_type is None:
            # by default match the input and outputs
            input_data_types = [self.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names]
            tensor_data_type = input_data_types[0]
        if tensor_data_type != ir_graph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32 and tensor_data_type not in qnn_quantized_types:
            check_encodings = False

        for output_name in node.output_names:
            output_tensor_info = self.get_output_info(graph,
                                                      tensor_name=output_name,
                                                      tensor_data_type=tensor_data_type,
                                                      check_encodings=check_encodings,
                                                      orig_tensor_name=original_output)
            outputs_info.append(output_tensor_info)
        return outputs_info

    @staticmethod
    def _get_resolved_tensor_info(tensor_info):
        """ Resolves any enum or numpy types for json encoding
        @return: tensor with name as key and rest of config info as value"""

        # resolve quant param enums
        quant_params = OrderedDict([
            ("definition", int(tensor_info['quant_params']['definition'])),
            ("encoding", int(tensor_info['quant_params']['encoding'])),
        ])
        if tensor_info['quant_params']['encoding'] == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
            quant_params.update([
                ("axis_scale_offset", tensor_info['quant_params']['axis_scale_offset'])
            ])
        elif tensor_info['quant_params']['encoding'] == ir_graph.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
            quant_params.update([
                ("bw_scale_offset", tensor_info['quant_params']['bw_scale_offset'])
            ])
        elif tensor_info['quant_params']['encoding'] == ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
            quant_params.update([
                            ("bw_axis_scale_offset", tensor_info['quant_params']['bw_axis_scale_offset'])
            ])
        else:
            quant_params.update([
                ("scale_offset", tensor_info['quant_params']['scale_offset'])
            ])

        tensor_info_ = OrderedDict([
            ("id", tensor_info['id']),
            ("type", int(tensor_info['type'])),
            ("dataFormat", int(tensor_info['dataFormat'])),
            ("data_type", int(tensor_info['data_type'])),
            ('src_axis_format', tensor_info['src_axis_format']),
            ('axis_format', tensor_info['axis_format']),
            ("quant_params", quant_params),
            ("dims", list(map(int, tensor_info['dims']))),
        ])

        if 'data' in tensor_info:
            data = tensor_info['data']
            if type(data) is np.ndarray:
                data = data.tolist()
            tensor_info_.update({'data': data})

        tensor = {tensor_info['name']: tensor_info_}

        return tensor

    @staticmethod
    def _get_c_input_types_dict(input_types):
        input_types_dict = {}
        for name_, type_ in input_types.items():
            # construct with both name provided and indexed name to account for fws(like tf) that change tensor names
            # to include index. (using zero since this is for input tensors).
            indexed_name_ = '{}:{}'.format(name_, 0) if ':' not in name_ else name_
            input_types_dict.update({name_: ir_graph.InputTypeClass.get_str_as_input_type(type_),
                                     indexed_name_: ir_graph.InputTypeClass.get_str_as_input_type(type_)})
        return input_types_dict

    def get_ir_graph(self, graph):
        """
        Temp acceleration solution to resolve all ir graph ops using ir to qnn translations which we then reconstruct
        a qnn ir graph to be serialized for dlc.
        """
        log_debug3("Converting QuIR -> QNN Model -> QNNIR Graph...")

        graph_name, _ = os.path.splitext(os.path.basename(self.output_path))
        self.model = qnn_ir.QnnModel(graph_name,
                                                      input_types_map=self._get_c_input_types_dict(
                                                          graph.inputs_type_dict))
        self.total_graph_macs = graph.total_macs
        self.total_graph_params_count = graph.total_params_count

        log_debug3("Total parameters: {} ({} MB assuming single precision float)"
                   .format(str(self.total_graph_params_count),
                           int(self.total_graph_params_count * 4 / (1024 ** 2))))
        log_debug3("Total Macs per inference: {}".format(str(get_si_notation(self.total_graph_macs,
                                                                             self.total_graph_macs))))

        try:

            if graph.quantization_params:
                self.model.set_tensor_overrides(graph.quantization_params)

            # lower to qnn api calls
            QnnTranslations.apply_method_to_all_ops(BackendTranslationBase.ADD_OP_TO_BACKEND, graph, self)

            ir_graph = self.model.get_ir_graph()

        except BaseException as e:
            raise e

        log_debug3("QNNIR Graph construction successful")

        return ir_graph

    def mixed_precision_processing(self, graph):
        # Todo: only for QNN workflow
        # loop through the graph to identify nodes using different quantized tensor data types for input and output
        # insert Convert node between the identified node and each of it's input nodes
        per_channel_ops = [op_adapter.Conv2dOp.TRANSLATION_KEY,
                           op_adapter.Conv3dOp.TRANSLATION_KEY,
                           op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY,
                           op_adapter.TransposeConv2dOp.TRANSLATION_KEY,
                           op_adapter.TransposeConv3dOp.TRANSLATION_KEY]
        for node in graph.list_nodes():
            if not node.input_names:
                continue

            output_tensor_data_types = []

            for output_name in node.output_names:
                output_tensor = self.c_ir_graph.get_output_tensor(output_name)
                if output_tensor is not None and not output_tensor.is_static():
                    output_tensor_data_type = output_tensor.data_type()
                    output_tensor_data_types.append(output_tensor_data_type)

            if len(output_tensor_data_types) > 1:
                all_equal = all(type_str == output_tensor_data_types[0] for type_str in output_tensor_data_types)
                log_assert(all_equal,
                           "Mixed precision feature only supported for Ops with all outputs of same type."
                           "But node {} has multiple outputs with different types.".format(node.op.name))

            if not output_tensor_data_types:
                continue

            for idx, input_name in enumerate(node.input_names):
                # compare only in[0] activation bitwidth with out[0] for per-channel ops, skip weights and bias
                if node.op.type in per_channel_ops and idx != 0:
                    continue
                input_name = node.input_names[idx]
                input_tensor = self.c_ir_graph.get_output_tensor(input_name)
                static_input = input_tensor.is_static()
                if input_tensor is not None and not input_tensor.is_static():
                    input_tensor_data_type = input_tensor.data_type()
                    if input_tensor_data_type != output_tensor_data_types[0]:
                        convert_name = input_name + "_converted_" + str(output_tensor_data_types[0]).split('.')[1]
                        convert_op = op_adapter.ConvertOp(convert_name)

                        # if convert node already exists from the same input node
                        # to convert to current output tensor data type then reuse it
                        if graph.has_buffer(convert_name):
                            convert_buffer = graph.buffers[convert_name]
                            old_consumers = list(convert_buffer.consumers)
                            consumer = graph.nodes_by_name[node.op.name]
                            old_consumers.append(consumer)
                            node.input_names[idx] = convert_name
                            input_buffer = graph.buffers[input_name]
                            input_buffer.consumers.remove(consumer)
                        else:
                            graph.inject(convert_op, input_name, convert_name, consumer_names=[node.op.name])

    """ Abstract functions """
    def sanitize_name(self, name):
        """
        Modifies given name to adhere with C++ naming standard as names(node or tensors) are used
        as variable name lookup in generated model.cpp
        """
        raise NotImplementedError("sanitize_name() needs to be overridden and implemented")

    def _sanitize_tensor_name(self, tensor_name):
        """ Function to support tensor name exclusion in the generated qnn_model """

        raise NotImplementedError("_sanitize_tensor_name() needs to be overridden and implemented")

    def add_tensor(self, orig_node_name, tensor_name, tensor_type, tensor: np.ndarray,
                   check_encodings=True, is_static=False, store_in_bin=True,
                   tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32, src_axis_format=None, tensor_axis_format=None,
                   orig_tensor_name=None, params_count=0):
        """
        Depending on graph construction mode(online vs offline), it either calls function to Execute QnnModel addTensor
        function or constructs the call string for offline model.cpp
        :param orig_node_name: the IRGraph name for node.
        :param tensor_name: name to use for the tensor
        :param tensor_type: the QNN tensor type. (i.e: NATIVE, APP_WRITE,...)
        :param tensor: np.ndarray object
        :param check_encodings: flag to check for quantization encodings for tensor. Quantization is done
                                in op/tensor agnostic manner. Hence, if any tensor specific constraint is needed
                                to keep tensor type as source framework, flag should be set to False, otherwise True
        :param is_static: flag to indicate if tensor added has static data
        :param store_in_bin: flag to indicate if tensor data should be stored in tarfile or within the generate .cpp
        :param tensor_data_type: the data type to use for the tensor
        :param src_axis_format: the axis format of the source framework tensor
        :param tensor_axis_format: the axis format of the QNN tensor
        :param orig_tensor_name: the IRGraph name for tensor. This can be different from tensor_name param which will
                                 be used for the QNN tensorname.(These two can differ especially given that for QNN
                                 tensorNames are sanitized to comply with C++ naming scheme).
        :param params_count: the size of weights for the operation, if applicable
        """

        raise NotImplementedError("add_tensor() needs to be overridden and implemented")

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
        raise NotImplementedError("add_node() needs to be overridden and implemented")

    def save(self, graph):
        raise NotImplementedError("save() needs to be overridden and implemented")
