# ==============================================================================
#
#  Copyright (c) 2018-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import math
import sys
import os
import tempfile
from packaging.version import parse as parse_version
from collections import defaultdict, OrderedDict
from collections.abc import Iterable
from functools import reduce
from operator import mul
from typing import IO, Any, Dict, List, Optional, Set, Text, Tuple, Union
from qti.aisw.converters.common.utils import code_to_message

import numpy as np

try:
    import onnx
    from onnx import (AttributeProto, GraphProto, ModelProto, NodeProto,
                      TensorProto, helper, mapping, onnx_pb)
    from onnx.external_data_helper import (
        load_external_data_for_model,
        set_external_data,
        uses_external_data,
    )
    from onnx.numpy_helper import to_array as extract_onnx_tensor
except:
    onnx = None  # converter will throw before we try anything in here

from qti.aisw.converters.common.converter_ir.op_adapter import \
    IRPaddingStrategies
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import (log_assert,
                                                              log_debug,
                                                              log_debug1,
                                                              log_error,
                                                              log_warning,
                                                              log_info,
                                                              converter_type)
from qti.aisw.converters.common.utils.framework_utils import (TensorInfo,
                                                              determine_layout,
                                                              make_dirs)
from qti.aisw.converters.common.utils.translation_utils import broadcastable


class OnnxAttrProtoUtil(object):
    strCode_to_enum = {'i': onnx.AttributeProto.INT,
                       'f': onnx.AttributeProto.FLOAT,
                       's': onnx.AttributeProto.STRING,
                       't': onnx.AttributeProto.TENSOR,
                       'g': onnx.AttributeProto.GRAPH,
                       'li': onnx.AttributeProto.INTS,
                       'lf': onnx.AttributeProto.FLOATS,
                       'ls': onnx.AttributeProto.STRINGS,
                       'lt': onnx.AttributeProto.TENSORS,
                       'lg': onnx.AttributeProto.GRAPHS}

    enum_to_strCode = {onnx.AttributeProto.INT: 'i',
                       onnx.AttributeProto.FLOAT: 'f',
                       onnx.AttributeProto.STRING: 's',
                       onnx.AttributeProto.TENSOR: 't',
                       onnx.AttributeProto.GRAPH: 'g',
                       onnx.AttributeProto.INTS: 'li',
                       onnx.AttributeProto.FLOATS: 'lf',
                       onnx.AttributeProto.STRINGS: 'ls',
                       onnx.AttributeProto.TENSORS: 'lt',
                       onnx.AttributeProto.GRAPHS: 'lg'}


TensorsType = Dict[str, np.ndarray]
TensorShapeType = List[int]
TensorSpecType = Dict[str, TensorShapeType]

numpy_to_onnx_type = {
    np.float32: onnx.TensorProto.FLOAT,
    np.float64: onnx.TensorProto.DOUBLE,
    np.uint8: onnx.TensorProto.UINT8,
    np.int8: onnx.TensorProto.INT8,
    np.int16: onnx.TensorProto.INT16,
    np.int32: onnx.TensorProto.INT32,
    np.int64: onnx.TensorProto.INT64,
    np.bool_: onnx.TensorProto.BOOL,
}

ONNX_EXTERNAL_DATA_THRESHOLD = 1024


def elem_type_to_name(elem_type: int) -> str:
    """
    Function to convert the elem_type to its equivalent name based on onnx proto

    :param type(int): elem_type int
    :return type(str): elem_type name
    """
    type_map = mapping.TENSOR_TYPE_TO_NP_TYPE
    if elem_type in type_map.keys():
        return type_map[elem_type].name
    else:
        return "undefined"


onnx_to_np_dtype = {
    # int types
    TensorProto.INT8: np.dtype('int8'),
    TensorProto.INT16: np.dtype('int16'),
    TensorProto.INT32: np.dtype('int32'),
    # downcast as yet not supported in conversion
    TensorProto.INT64: np.dtype('int32'),
    TensorProto.UINT8: np.dtype('uint8'),
    TensorProto.UINT16: np.dtype('uint16'),
    TensorProto.UINT32: np.dtype('uint32'),
    # downcast as yet not supported in conversion
    TensorProto.UINT64: np.dtype('uint32'),

    # float types
    TensorProto.FLOAT16: np.dtype('float16'),
    TensorProto.FLOAT: np.dtype('float32'),
    TensorProto.DOUBLE: np.dtype('float64'),
    TensorProto.BOOL: np.dtype('bool_')
}

KNOWN_ATTRIBUTE_DEFAULTS = dict(dilations=[1, 1],
                                strides=[1, 1],
                                pads=[0, 0, 0, 0],
                                output_shape=[],
                                axes=[],
                                consumed_inputs=[],
                                kernel_shape=[])


def is_broadcast(onnx_op, graph=None):
    attrs = extract_attributes(
        onnx_op, [('axis', 'i', 0), ('broadcast', 'i', 0)])

    if graph is not None:
        # newer version of onnx(e.g version 7 of Mul or Add) do not have axis and broadcast attributes
        # hence another way to check would be to make sure all inputs to op are the same shape
        input_names = list(map(str, onnx_op.input))
        input_buffers_shape = []
        for name in input_names:
            if graph.has_buffer(name):
                input_buffers_shape.append(list(graph.get_buffer(name).shape))
            else:
                input_buffers_shape.append(
                    list(graph.weights.fetch(name).shape))
        if any(shape != input_buffers_shape[0] for shape in input_buffers_shape):
            return True

    return attrs['axis'] != 0 or attrs['broadcast'] == 1


def assert_no_broadcast(onnx_op):
    log_assert(not is_broadcast(onnx_op),
               code_to_message.get_error_message("ERROR_BROADCAST_NOT_SUPPORTED")(onnx_op.name))


class NamedDict(dict):
    def __getattr__(self, key):
        return self[key]


def extract_initializer_tensor(initializer):
    return extract_onnx_tensor(initializer)


def extract_attributes(onnx_op, attr_infos=None, schema=None, validate=False, default_attrs=NamedDict()):
    """Ensure the existence and extract well typed attributes from an onnx
    NodeProto.
    :param attr_infos: a list of attributes to extract in the form [(attr_name, attr_type, attr_value)]
    :param schema:   an op_schema object for the onnx_op
    :param validate:  an optional validator function that is registered with the schema
                     of the form:  validator(src_op, attr_name, attr_value)
    :param default_attrs: a object have a named property for default attributes info value

    Each entry in attr_info should be either a 2- or 3-tuple.
    * The first element should be the string name of an attribute.
    * The second element should by a type code for the attribute corresponding to:
      - i for int attributes
      - f for float attributes
      - s for string attributes
      - t for tensor attributes (returned as a numpy array)
      - g for graph attributes
      - lx, where x is one of the preceding attribute type identifiers, for list valued attributes
    * The third element, if present, specifies a default value should the attribute not be present.
      If no default is specified, this function will thrown an error.

    The return object will have a named property for each attribute info."""
    onnx_attrs = {}
    if not attr_infos and schema:
        attr_infos = schema.attributes()

    for attr in onnx_op.attribute:
        onnx_attrs[attr.name] = attr
        if schema and not validate:
            if not schema.check_supported_attributes(str(attr.name)):
                log_warning(code_to_message.get_warning_message("WARNING_UNSUPPORTED_ATTRIBUTE")
                            (attr.name, onnx_op.op_type, onnx_op.input[0]))
    ret = NamedDict()
    for attr_info in attr_infos:
        name = attr_info[0]
        if not name in onnx_attrs:
            if len(default_attrs) == 0:
                if len(attr_info) == 3:
                    ret[name] = attr_info[2]
                    continue
                else:
                    try:
                        ret[name] = KNOWN_ATTRIBUTE_DEFAULTS[name]
                        continue
                    except KeyError:
                        raise ValueError(code_to_message.get_error_message(
                            "ERROR_ATTRIBUTE_MISSING")(onnx_op.name, name))
            elif len(default_attrs) > 0 and name in default_attrs.keys():
                ret[name] = default_attrs[name]
                continue

        attr = onnx_attrs[name]
        code = attr_info[1]
        requested_type = OnnxAttrProtoUtil.strCode_to_enum[code]
        if attr.type != requested_type:
            msg = code_to_message.get_error_message("ERROR_ATTRIBUTE_WRONG_TYPE")(onnx_op.name,
                                                                                  name,
                                                                                  onnx.AttributeProto.AttributeType.Name(
                                                                                      requested_type),
                                                                                  onnx.AttributeProto.AttributeType.Name(attr.type))
            raise TypeError(msg)
        value = extract_onnx_type(code, attr)
        if validate and schema:
            schema.validate_data_constraints(onnx_op)
            schema.get_validate_method(
                "validate_attribute_values")(onnx_op, name, value)
        ret[name] = value

    return ret


def extract_onnx_type(code, attr):
    ret = ''
    if code == 'i':
        ret = int(attr.i)
    elif code == 'f':
        ret = float(attr.f)
    elif code == 's':
        ret = str((attr.s).decode('utf-8'))
    elif code == 'g':
        ret = attr.g
    elif code == 't':
        ret = extract_onnx_tensor(attr.t)
    elif code == 'li':
        ret = list(map(int, attr.ints))
    elif code == 'lf':
        ret = list(map(float, attr.floats))
    elif code == 'ls':
        ret = list(map(str, [data.decode('utf-8') for data in attr.strings]))
    elif code == 'lg':
        ret = list(attr.graphs)
    elif code == 'lt':
        ret = list(map(extract_onnx_tensor, attr.tensors))
    return ret


def extract_padding_mode(auto_pad, node_name, ceil_mode=0, right_handed=False):
    if right_handed == True:
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED
    elif auto_pad == 'VALID':
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID
    elif auto_pad == 'SAME_UPPER':
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END
    elif auto_pad == 'SAME_LOWER':
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN
    elif ceil_mode != 0:
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT
    elif auto_pad == 'NOTSET':
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
    else:
        raise ValueError(code_to_message.get_error_message(
            "ERROR_PADDING_TYPE_UNSUPPORTED")(node_name, auto_pad))


def broadcast_to(data, new_shape):
    """
    Broadcasts data into a new shape if possible
    :param new_shape: shape to be broadcasted into
    :param data: data to be broadcasted
    :return: broadcasted data if possible or original data if not
    """
    if data.shape != new_shape and broadcastable(data.shape, new_shape):
        return np.broadcast_to(data, new_shape).astype(np.float32)
    return data


def product(nums):
    if len(nums) == 0:
        return 1
    else:
        return reduce(mul, nums)


def get_quant_info(zp):
    log_assert(isinstance(zp, np.ndarray),
               "Zero point is not a numpy array")
    if zp.dtype == np.uint8:
        return False, 8, -int(zp[0])
    elif zp.dtype == np.int8:
        return True, 8, -2**(8-1)
    elif zp.dtype == np.uint16:
        return False, 16, -int(zp[0])
    elif zp.dtype == np.int16:
        return True, 16, -2**(16-1)
    elif zp.dtype == np.uint32:
        return False, 32, -int(zp[0])
    elif zp.dtype == np.int32:
        return True, 32, -2**(32-1)
    else:
        raise ValueError("Unsupported zero point type: ", zp.dtype)


def get_encoding(name, scale, zp):
    is_symmetric, bw, new_zp = get_quant_info(zp)
    return {"name": name,
            "bw": bw,
            "min": ((np.float32(np.iinfo(zp.dtype).min) - zp) * scale)[0],
            "max": ((np.float32(np.iinfo(zp.dtype).max) - zp) * scale)[0],
            "scale": scale[0],
            "offset": new_zp,
            "is_symmetric": is_symmetric}


def downcast_dtype_64bit_to_32bit(tensor_name, tensor_dtype):
    numpy_dtype_downcast = {
        np.dtype('int64'): np.int32,
        np.dtype('uint64'): np.uint32,
        np.dtype('float64'): np.float32,
    }
    if tensor_dtype in numpy_dtype_downcast:
        prev_dtype = tensor_dtype
        tensor_dtype = numpy_dtype_downcast[tensor_dtype]
        log_debug1(code_to_message.get_debugging_message("DEBUG_DOWNCAST_TENSOR")
                   (prev_dtype, np.dtype(tensor_dtype), tensor_name))

    return tensor_dtype


class WeightData(object):
    def __init__(self, weights, was_scalar=False):
        """
        :param weights: weights from the network
        :param was_scalar: boolean to determine if original weight was initialized as scalar.
                           Since QuIR expects that all inputs are tensors, this information will be
                           helpful for any op specific usecases such as determining output_shape
        """
        self.weights = weights
        # Track if the weights have been retrieved for use in another layer
        # Weights can be provided in one of two ways: initializers or constant ops
        # Constant ops being used as weight providers are setup with the weights from
        # the start and thus don't need to retrieve weights from the weight provider
        # again. SNPE layers like Conv/Matmul/GEMM/etc store weights internally and
        # will attempt to retrieve the weights. The consumed field will track which
        # Constant ops are being used as weight providers so they can be pruned from
        # the network at the end
        self.consumed = False
        self.was_scalar = was_scalar


# ------------------------------------------------------------------------------
#   WeightProvider
# ------------------------------------------------------------------------------
class WeightProvider(object):
    def __init__(self, model, remove_model_weights=True):
        self.weight_map = {}
        count = 0
        total_weights = len(model.graph.initializer)
        while count < total_weights:
            if remove_model_weights:
                # index of the weight to be inserted will be 0 since previous weights are already removed
                tensor = model.graph.initializer[0]
                # remove weights from the weight initializer
                model.graph.initializer.remove(tensor)
            else:
                tensor = model.graph.initializer[count]

            onnx_tensor = extract_onnx_tensor(tensor)
            was_scalar = False

            if not tensor.dims:
                # tensor of dim 1 is empty array in onnx resulting in broken numpy array,
                # since unable to modify tensor.dims, reshaping to have numpy array proper config
                onnx_tensor = onnx_tensor.reshape(1)
                was_scalar = True
            self.weight_map[str(tensor.name)] = WeightData(onnx_tensor, was_scalar)
            count += 1

    def was_scalar(self, key):
        return self.weight_map[key].was_scalar

    def consumed(self, key):
        if not key in self.weight_map:
            return False
        return self.weight_map[key].consumed

    def fetch(self, *keys, **kwargs):
        ret = []
        # Prunable indicates whether the weights have been consumed in such a way as to
        # allow pruning of the node (eg Const ops that contain weights are consumed by
        # Conv/FC/etc and thus can be pruned from the network. Const ops that are inputs
        # to a node cannot
        consumed = kwargs.get('prunable', True)
        for key in keys:
            key = str(key)
            log_debug(code_to_message.get_debugging_message(
                "DEBUG_RETRIEVE_WEIGHTS")(key))
            if key not in self.weight_map:
                raise KeyError(code_to_message.get_error_message(
                    "ERROR_WEIGHTS_MISSING_KEY")(key))
            self.weight_map[key].consumed = consumed
            if kwargs.get('dtype') is None:
                tensor_dtype = downcast_dtype_64bit_to_32bit(
                    key, self.weight_map[key].weights.dtype)
            else:
                # Assumes downcasting of provided dtype, if required, will be handled by fetch caller
                tensor_dtype = kwargs.get('dtype')

            # This function returns the original weights present in the weight map.
            # If the requested dtype is different from the actual tensor dtype, then it returns a
            # copy of the weight.
            # The original weight should not be modified by the translations.
            if tensor_dtype != self.weight_map[key].weights.dtype:
                ret.append(np.require(
                    self.weight_map[key].weights.copy(), dtype=tensor_dtype))
            else:
                ret.append(self.weight_map[key].weights)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def has(self, key):
        return key in self.weight_map

    def type(self, key):
        if key not in self.weight_map:
            raise KeyError(code_to_message.get_error_message(
                "ERROR_WEIGHTS_MISSING_KEY")(key))
        return self.weight_map[key].weights.dtype

    def has_all(self, keys):
        return all(self.has(key) for key in keys)

    def insert(self, key, weights, was_scalar=False):
        log_debug("Inserting weights for {}, was_scalar:{}", key, was_scalar)
        self.weight_map[key] = WeightData(weights, was_scalar)


def get_type_dims_info(source, node_name):
    """
    :param source: structure to query for node_name's info
    :param node_name: the name of the node to query info for
    :return: (bool, elem_type, dims)
    """
    for info in source:
        if info.name == node_name:
            dims = [int(dim.dim_value)
                    for dim in info.type.tensor_type.shape.dim]
            return True, info.type.tensor_type.elem_type, dims
    return False, None, None


def get_all_dims_info(source):
    """
    :param source: structure to query for input/output shape info
    :return: [(str_name1, elem_type1, dims1), (str_name2, elem_type2, dims2)]
    """
    dims_info = []
    for info in source:
        dims = [str(dim.dim_param) if dim.dim_param else int(dim.dim_value)
                for dim in info.type.tensor_type.shape.dim]
        dims_info.append((info.name, info.type.tensor_type.elem_type, dims))
    return dims_info

# ------------------------------------------------------------------------------
#   ONNX Common Utils
# ------------------------------------------------------------------------------


def get_unique_ops(model) -> Dict[str, int]:
    """
    Get the unique ops present in the model and also, get the total ops. count.

    :param model: path to the loaded onnx model proto
    :return Dict[str, int]: Mapping of each op with its count.
    """
    count = defaultdict(int)
    for node in get_nodes(model):
        count[node.op_type] += 1
    return count


def get_inputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph inputs tensors except initializers.

    :param model: path to the loaded onnx model proto
    :return List: List of graph inputs
    """
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


def get_all_inputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph inputs tensors including the inputs which are also initializers.

    :param model: path to the loaded onnx model proto
    :return List: List of graph inputs
    """
    return [ipt for ipt in model.graph.input]


def get_outputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph outputs tensors.

    :param model: path to the loaded onnx model proto
    :return List: List of graph iutputs
    """
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.output if ipt.name not in initializer_names]


def get_graphs(model: ModelProto) -> List[GraphProto]:
    """
    Function to return all the graph present in model.

    :param ModelProto model: Onnx Model graph proto.
    :return List[GraphProto]: Onnx graphs
    """
    all_graphs = []
    graph_queue = [model.graph]
    while graph_queue:
        graph = graph_queue.pop(0)
        all_graphs.append(graph)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == AttributeProto.AttributeType.GRAPH:
                    assert isinstance(attr.g, onnx_pb.GraphProto)
                    graph_queue.append(attr.g)
                if attr.type == AttributeProto.AttributeType.GRAPHS:
                    for g in attr.graphs:
                        assert isinstance(g, onnx_pb.GraphProto)
                        graph_queue.append(g)
    return all_graphs


def get_nodes(model: ModelProto) -> List[NodeProto]:
    """
    Function to return the nodes information.

    :param ModelProto model: Onnx Model graph proto.
    :return List[NodeProto]: Underlying onnx nodes.
    """
    nodes_all = []
    for graph in get_graphs(model):
        for node in graph.node:
            nodes_all.append(node)
    return nodes_all


def get_graph_by_node(
    model: ModelProto, node: NodeProto
) -> Union[None, GraphProto]:
    """
    Get graph which contains given node.

    :param ModelProto model: Onnx Model graph proto.
    :param NodeProto node: onnx node
    :return Union[None, GraphProto]: Underlying onnx graph if found else None.
    """
    for graph in get_graphs(model):
        if node in graph.node:
            return graph
    return None


def get_graph_by_initializer(
    model: ModelProto, init: TensorProto
) -> Union[None, GraphProto]:
    """
    Get graph which contains given initializer.

    :param ModelProto model: Onnx Model graph proto.
    :param TensorProto node: Initializer instance.
    :return Union[None, GraphProto]: Underlying onnx graph if found else None.
    """
    for graph in get_graphs(model):
        if init in graph.initializer:
            return graph
    return None


def get_nodes_by_op_type(model: ModelProto, op_type: str) -> List[NodeProto]:
    """
    Get all the nodes from underlying onnx model based on op type.

    :param ModelProto model: Onnx Model graph proto.
    :param str op_type: optype
    :return List[NodeProto]: Underlying onnx nodes.
    """
    nodes = []
    for node in get_nodes(model):
        if node.op_type == op_type:
            nodes.append(node)
    return nodes


def get_graph_by_name(model: ModelProto, graph_name: str) -> Union[None, GraphProto]:
    """
    Get the graph by their name.

    :param ModelProto model: Onnx Model Proto.
    :param str graph_name: name of the graph.
    :return Union[None, GraphProto]: onnx graph proto instance if found else None.
    """
    for graph in get_graphs(model):
        if graph_name == graph.name:
            return graph
    return None


def remove_unused_inputs_outputs(model: ModelProto) -> ModelProto:
    """
    Remove the unused/dangling input and output nodes from graph's inputs
    and outputs.

    :param model: Loaded Onnx Model Proto instance
    :return model: Cleaned model with dangling inputs and outputs removed.
    """
    main_graph = model.graph

    node_by_input_name = get_node_by_input_name(model)
    node_by_output_name = get_node_by_output_name(model)

    def is_dangling_graph_output(graph_output_name: str) -> bool:
        """
        Function to check if the graph output is dangling output (e.g. not an
        output of any nodes.)

        :param str graph_output_name: Name of the graph output tensor to check.
        :return bool: True if the given output is a dangling output, else False.
        """
        if graph_output_name not in node_by_output_name.keys():
            return True
        else:
            return False

    def is_dangling_graph_input(graph_input_name: str) -> bool:
        """
        Function to check if the graph input is dangling input (e.g. not an
        input of any nodes.)

        :param str graph_input_name: Name of the graph input tensor to check.
        :return bool: True if the given input is a dangling input, else False.
        """
        if graph_input_name not in node_by_input_name.keys():
            return True
        else:
            return False

    # Identify and remove any dangling graph inputs and graph outputs
    # Note: We shall not remove dangling input/output of subgraph as they may
    #       have connection to parent graph.
    remove_graph_outputs = [
        g_op for g_op in main_graph.output if is_dangling_graph_output(g_op.name)
    ]
    remove_graph_inputs = [
        g_op for g_op in main_graph.input if is_dangling_graph_input(g_op.name)
    ]

    for r_g_op in remove_graph_outputs:
        main_graph.output.remove(r_g_op)

    for r_g_ip in remove_graph_inputs:
        main_graph.input.remove(r_g_ip)

    return model


def get_shape_from_value_info_proto(
    val_info: onnx.ValueInfoProto,
    use_onnx_def_symbols: bool = False,
) -> List[Union[str, int]]:
    """
    Function to get the shape from value info proto.

    :param val_info: onnx.ValueInfoProto
    :param use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return List[Union[str, int]]: Tensor shape of given value info proto.
    """
    tensor_shape = []
    tensor_type = val_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        log_warning(f"Shape not found for tensor: {val_info.name}")
        return tensor_shape

    # iterate through dimensions of the shape:
    for d in tensor_type.shape.dim:
        # the dimension may have a definite (integer) value or a symbolic identifier or neither:
        if d.HasField("dim_value"):
            tensor_shape.append(d.dim_value)
        elif d.HasField("dim_param"):
            # unknown dimension with symbolic name
            if use_onnx_def_symbols:
                tensor_shape.append(d.dim_param)
            else:
                tensor_shape.append(-1)
        else:
            tensor_shape.append("?")
    return tensor_shape


def get_tensor_info_from_value_info_proto(
    val_info: onnx.ValueInfoProto, use_onnx_def_symbols: bool = False
) -> TensorInfo:
    """
    Converts tensor's value info into TensorInfo object which posses information
    about name, shape, dtype and layout.

    :param onnx.ValueInfoProto val_info: Tensor information in terms of ValueInfoProto.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return TensorInfo: Information about tensor in terms of TensorInfo which
        contains its name, shape, dtype and layout.
    """
    tensor_name = val_info.name
    tensor_shape = get_shape_from_value_info_proto(val_info, use_onnx_def_symbols)
    tensor_dtype = elem_type_to_name(val_info.type.tensor_type.elem_type)
    tensor_layout = determine_layout(tensor_shape)
    return TensorInfo(tensor_name, tensor_shape, tensor_dtype, tensor_layout)


def get_input_info(model: ModelProto, use_onnx_def_symbols: bool=False) -> Dict[str, TensorInfo]:
    """
    Get the graph input info as TensorInfo object.

    :param ModelProto model: Onnx ModelProto instance.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return Dict[str, TensorInfo]: Mapping of graph's input name to its TensorInfo.
    """
    model_inputs = get_inputs(model)
    input_specs = OrderedDict()
    for model_input in model_inputs:
        input_specs[model_input.name] = get_tensor_info_from_value_info_proto(model_input, use_onnx_def_symbols)
    return input_specs


def get_output_info(model: ModelProto, use_onnx_def_symbols: bool=False) -> Dict[str, TensorInfo]:
    """
    Get the graph output info as TensorInfo object.

    :param ModelProto model: Onnx ModelProto instance.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return Dict[str, TensorInfo]: Mapping of graph's output name to its TensorInfo.
    """
    model_outputs = get_outputs(model)
    output_specs = OrderedDict()
    for model_output in model_outputs:
        output_specs[model_output.name] = get_tensor_info_from_value_info_proto(model_output, use_onnx_def_symbols)
    return output_specs


def get_millify_number(num: int) -> int:
    """
    Convert the large numbers to human readble format.

    :param int num: number to convert
    :return int: converted number in readble format
    """
    millnames = ["", " Thousand", " Million", " Billion", " Trillion"]
    n = float(num)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def get_model_params(model: ModelProto) -> int:
    """
    Return the total number of model parameters.

    :param ModelProto model: Onnx model proto instance.
    :return int: Total parameters in model.
    """
    onnx_constants_mapping = get_initializer_mappings(model)
    params = 0
    for _, const_tensor in onnx_constants_mapping.items():
        try:
            weight_shape = [d for d in const_tensor.dims]
            params += np.prod(weight_shape)
        except Exception as e:
            log_warning(f"Invalid weight params: {e}")
            continue
    return params


def get_model_size(model: ModelProto) -> float:
    """
    Provides the size of the model in GB.

    :param model (ModelProto): Onnx model proto instance.
    :return size_gb (float): size of the model in GB
    """
    NUM_GB_BYTES = 1024**3
    size_bytes = model.ByteSize()
    size_gb = size_bytes / NUM_GB_BYTES
    return size_gb


def get_opset_version(model: ModelProto) -> int:
    """
    Return the model opset version for default domain.

    :param ModelProto model: Onnx model proto instance.
    :raises RuntimeError: If no default domains found in model.
    :return int: opset version of onnx domain
    """
    for opset in model.opset_import:
        if opset.domain in ["", "ai.onnx"]:
            return opset.version
    raise RuntimeError("Onnx model has no opset for default domain")


def remove_zero_dim_input_from_node(
    model: ModelProto, op_type: str
) -> ModelProto:
    """
    This function is removing initializer's zero Sized tensor from the Input node

    :param ModelProto model: Input ONNX Model.
    :param str op_type: ONNX operation type.
    :return ModelProto: Modified ONNX Model.
    """
    # Aggregating zero dims tensors from initializer.
    filtered_nodes = get_nodes_by_op_type(model, op_type)
    zero_dim_init_dict = {
        init_name: init for init_name, init in get_initializer_mappings(model).items() if init.dims == [0]
    }

    if not zero_dim_init_dict:
        return model

    node_idxs_to_check = set()
    initializers_to_remove = set()
    # Removing node inputs which are having zero dimensions
    # and part of initializer also.
    for node_idx, node in enumerate(filtered_nodes):
        for input in node.input:
            if input not in zero_dim_init_dict:
                continue
            init = zero_dim_init_dict[input]
            initializers_to_remove.add(init.name)

            graph = get_graph_by_node(model, node)
            node_graph_idx = list(graph.node).index(node)
            graph.node[node_graph_idx].input.remove(input)

            node_idxs_to_check.add(node_idx)

    # Removing Initializer which are having zero dimensions.
    for name in initializers_to_remove:
        init = zero_dim_init_dict[name]
        graph = get_graph_by_initializer(model, init)
        graph.initializer.remove(init)

    # Removing nodes which are having zero inputs
    for idx in node_idxs_to_check:
        node = filtered_nodes[idx]
        if len(node.input) == 0:
            graph = get_graph_by_node(model, node)
            graph.node.remove(node)
    return model


def remove_zero_dim_init_input_from_concat(model: ModelProto) -> ModelProto:
    """
    This function is removing initializer's zero Sized tensor from the concat node

    :param ModelProto model: Input ONNX Model.
    :return ModelProto: Modified ONNX Model.
    """
    return remove_zero_dim_input_from_node(model, "Concat")


def cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Cleans up the model by removing unused nodes and dangling inputs/outputs
    :param model: Onnx ModelProto
    :return onnx.ModelProto: Cleaned Onnx ModelProto
    """
    # Removing empty input tensors from the Concat Node
    model = remove_zero_dim_init_input_from_concat(model)

    # Traverse back from output node till the input node
    # Remove unused nodes
    # Then remove unused input and outputs
    graph = model.graph

    # Creating two separate dict for traversing the graph easily
    node_by_output_name = {}
    node_by_input_name = {}
    for n in graph.node:
        # Only single node can be referenced by any output node name
        for n_op in n.output:
            node_by_output_name[n_op] = n
        # More than one node can be referenced by any input node name
        # That's why using list of nodes
        for n_ip in n.input:
            if n_ip not in node_by_input_name.keys():
                node_by_input_name[n_ip] = [n]
            else:
                node_by_input_name[n_ip].append(n)

    # Traverse the graph from the graph output nodes.
    visited_nodes = set()
    stack = []

    for g_op in graph.output:
        if g_op.name in node_by_output_name.keys():
            node = node_by_output_name[g_op.name]
            stack.append(node)

    while len(stack) != 0:
        node = stack.pop()
        if node.name in visited_nodes:
            continue
        visited_nodes.add(node.name)
        for ip_name in node.input:
            if ip_name in node_by_output_name.keys():
                parent_node = node_by_output_name[ip_name]
                if parent_node.name not in visited_nodes:
                    stack.append(parent_node)

    # Till now visited_nodes is populated with nodes connected with graph outputs
    remove_nodes = [n for n in graph.node if n.name not in visited_nodes]

    # Remove nodes from the graph as well as from two dictionaries.
    for r_n in remove_nodes:
        graph.node.remove(r_n)

        # Delete its entries from node_by_input_name, node_by_output_name
        for n_op in r_n.output:
            node_by_output_name.pop(n_op)

        for n_ip in r_n.input:
            if len(node_by_input_name[n_ip]) == 1:
                node_by_input_name.pop(n_ip)
            else:
                node_by_input_name[n_ip].remove(r_n)
                # Remove node "r_n" from list if the input is used in multiple nodes
                # We only want to remove that "r_n" node and want to keep other nodes which uses
                # the same input.

    # Remove unused initializers from the graph.
    unused_initializers = [
        init for init in graph.initializer if init.name not in node_by_input_name
    ]

    for init in unused_initializers:
        # Remove the initializers which are not connected to any node.
        graph.initializer.remove(init)

    # Remove unused value_info from the graph.
    unused_value_infos = [
        val_info
        for val_info in graph.value_info
        if val_info.name not in node_by_input_name
    ]

    for val_info in unused_value_infos:
        # Remove the value_info which are not connected to any node.
        graph.value_info.remove(val_info)

    # Clean any dangling input or output from model's main graph.
    model = remove_unused_inputs_outputs(model)
    return model


def create_node_name(
    graph: GraphProto,
    op_type: str,
    _node_name_suffix: Dict[str, int],
    name_prefix: str = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Create a unique node name that starts with a prefix (default is operator type).
    The name will not be duplicated with any name that generated or existed in current graphs.

    :param graph (GraphProto): Onnx GraphProto instance of model
    :param op_type (str): Operator type for which the name is to be generated.
    :param _node_name_suffix (Dict[str, int]): Dict mapping of node_name to its suffix.
    :param name_prefix (str, optional): Prefix of node name. Defaults to None.
    :returns Tuple[str, Dict[str, int]]: Node name for given node op type and updated
                                         Dict mapping of node_name to its suffix.
    """
    # TODO: This functionality shall be redesigned where node can be created
    #        first and then added to graph and post that we shall call an API
    #        assign_name on graph to address issues related to empty name.
    if name_prefix:
        prefix = name_prefix if name_prefix.endswith("_") else (name_prefix + "_")
    else:
        prefix = op_type + "_"
    suffix: int = 0
    if prefix in _node_name_suffix:
        suffix = _node_name_suffix[prefix] + 1
    else:
        # Check existed node name only once for a prefix as we assume
        # create_node_name is called for every new node in fusion.
        for node in graph.node:
            if node.name and node.name.startswith(prefix):
                try:
                    index = int(node.name[len(prefix) :])
                    suffix = max(index + 1, suffix)
                except ValueError:
                    continue
    # Record the generated suffix so that we can avoid generating duplicated name.
    _node_name_suffix[prefix] = suffix
    return prefix + str(suffix), _node_name_suffix


def create_tensor_name(
    proposed_tensor_name: str, tensor_name_set: Set[str]
) -> Tuple[str, Set[str]]:
    """
    Function to create a new tensor name which doesnt conflict with existing
    tensor names.

    :param  proposed_tensor_name (str): Proposed name of the new tensor.
    :param tensor_name_set (Set[str]): Set of output tensor names of the model.
    :returns Tuple[str, Set[str]]: Tuple of updated name of the new tensor and
        updated set of the output tensor names of the model.
    """
    new_name = proposed_tensor_name
    counter = 1
    while new_name in tensor_name_set:
        new_name = f"{proposed_tensor_name.split('_')[0]}_{counter}"
        counter += 1
    tensor_name_set.add(new_name)
    return new_name, tensor_name_set


def assign_names_to_empty_nodes(model: ModelProto) -> None:
    """
    Function to add new node names for nodes whose name property is "".

    :param ModelProto model: Onnx model reference.
    """
    _node_name_suffix_mapping = {}
    for node in get_nodes(model):
        if node.name == "":
            new_node_name, _node_name_suffix_mapping = create_node_name(
                model.graph, node.op_type, _node_name_suffix_mapping
            )
            node.name = new_node_name
    return model


def check_duplicates_in_attribute(graph: GraphProto, attribute_name: str) -> bool:
    """
    Function to check for duplicates in the given attribute_name.

    :param GraphProto model: Graph reference from model.
    :param str attribute_name: Attribute name of the model. e.g. node, input or output.
    :return bool: boolean status indicating graph is valid or not.
    """
    node_check_status = True
    property_name_set = set()
    output_tensor_name_set = set()

    if not hasattr(graph, attribute_name):
        return False

    list_of_attributes = getattr(graph, attribute_name)

    if not isinstance(list_of_attributes, Iterable):
        return False

    for property in list_of_attributes:
        if property.name != "" and property.name not in property_name_set:
            property_name_set.add(property.name)
        else:
            node_check_status = node_check_status and False
            if property.name == "":
                log_debug1(
                    "Graph checker: No {} name found for.".format(attribute_name)
                )
            else:
                log_debug1(
                    f"Graph checker: {attribute_name} '{property.name}' is duplicate."
                )

        # Check for intermediate tensors. Each node's output tensor shall have
        # unique name. Two different nodes shall not have same output tensor name.
        if hasattr(property, "output"):
            for node_output in property.output:
                if node_output not in output_tensor_name_set:
                    output_tensor_name_set.add(node_output)
                else:
                    node_check_status = False
                    log_debug1(
                        f"Graph checker: Tensor '{node_output}' is the output of two different nodes."
                    )
                    return node_check_status

    return node_check_status


def graph_checker(model: ModelProto) -> bool:
    """
    Function to check the validity of graph. e.g Node names are present or not,
    Is there any duplicate node names, node's input tensor names or node's
    output tensor names etc. present or not.
    :param ModelProto model: Onnx model
    :return bool: boolean status indicating graph is valid or not.
    """
    node_check_status = True
    for graph in get_graphs(model):
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "node"
        )
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "input"
        )
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "output"
        )
    return node_check_status


def get_node_mappings(model: onnx.ModelProto) -> Dict:
    """
    Node name to nodes mapping.

    :param onnx.ModelProto model:  onnx model.
    :return Dict: Node name to nodes mapping
    """

    return {n.name: n for n in model.graph.node}


def get_initializer_mappings(model: onnx.ModelProto) -> Dict:
    """
    Initializer name to initializer mapping

    :param onnx.ModelProto model: model (onnx.ModelProto): onnx model.
    :return Dict: initializer name to initializer mapping
    """
    return {n.name: n for n in model.graph.initializer}


def get_value_info_proto_mapping(model: onnx.ModelProto) -> Dict:
    """
    Value name to value mapping.
    :param model (onnx.ModelProto): onnx model.
    :returns:Dict: value name to value mapping
    """
    return {v.name: v for v in model.graph.value_info}


def get_node_by_input_name(model: onnx.ModelProto) -> Dict:
    """
    Input name to node mappings.
    :param  loader (onnx.ModelProto): onnx.ModelProto model instance.
    :returns: Dict: Input name to node mappings.
    """
    get_node_by_input = {}
    for n in model.graph.node:
        for n_ip in n.input:
            if n_ip not in get_node_by_input:
                get_node_by_input[n_ip] = [n]
            else:
                get_node_by_input[n_ip].append(n)
    return get_node_by_input


def get_node_by_output_name(model: onnx.ModelProto) -> Dict:
    """
    Output name to node mappings.
    :param model (onnx.ModelProto): OnnxModel model instance.
    :return : Dict: Output name to node mappings.
    """
    get_node_by_output = {}
    for n in model.graph.node:
        for n_ip in n.output:
            if n_ip not in get_node_by_output:
                get_node_by_output[n_ip] = n
    return get_node_by_output


def get_parent_at_any_level(
    output_name: str, get_node_by_output_name: Dict[str, NodeProto], level: int = 1,
) -> List[NodeProto]:
    """
    Function to get the parent of the specified node at given level.
    level 1 - immediate parent
    level 2 - parent of parent and so on.

    :param str output_name: Output name of the node whose parent needs to be
        identified.
    :param Dict[str, NodeProto] get_node_by_output_name: Dict to get the
        node by its output name.
    :param int level: Parent level to be identified.
    :raises Exception: _description_
    :return List[NodeProto]: List of parent nodes.
    """
    output_node = get_node_by_output_name[output_name]

    def get_parent(node, get_node_by_output_name):
        im_parent_nodes = []
        for inp in node.input:
            if inp in get_node_by_output_name:
                it_node = get_node_by_output_name[inp]
                im_parent_nodes.append(it_node)
        return im_parent_nodes

    parent_nodes = get_parent(output_node, get_node_by_output_name)

    if level == 1:
        return parent_nodes
    elif level > 1:
        for i in range(level - 1):
            iterating_nodes = parent_nodes
            final_nodes = []
            for nodes in iterating_nodes:
                candidate_nodes = get_parent(nodes, get_node_by_output_name)
                for node in candidate_nodes:
                    final_nodes.append(node)
            parent_nodes = final_nodes
        return parent_nodes
    else:
        log_error(f"Can't find the parent at given level: {level}")
        return None


def get_initializer_value(init: onnx.ValueInfoProto) -> np.ndarray:
    """
    Function to get value of initializer of a particular initializer.

    :param onnx.ValueInfoProto init: Initializer instance.
    :return np.ndarray: Numpy array constructed from initializer's data.
    """
    return onnx.numpy_helper.to_array(init)


def get_initializer_by_name(model: ModelProto, name: str) -> Optional[TensorProto]:
    """
    Function to get the initializer of particular name

    :param ModelProto model: Onnx model proto instance.
    :param str name: Name of the initializer to be found.
    :return Optional[TensorProto]: Initializer tensor if found else None.
    """
    for graph in get_graphs(model):
        for tensor in graph.initializer:
            if tensor.name == name:
                return tensor
    return None


def get_dim_from_type_proto(dim) -> Union[str, int]:
    """
    Function to get the dim value or dim param from type proto

    :param onnx.onnx_ml_pb2.TensorShapeProto.Dimension dim: Dimension info from tensor.
    :return Union[str, int]: Dim value or Dim param from onnx model.
    """
    return (
        getattr(dim, dim.WhichOneof("value"))
        if type(dim.WhichOneof("value")) == str
        else None
    )


def get_shape_from_type_proto(type_proto: onnx.TypeProto) -> List:
    """
    Function to get the shape info from type proto

    :param onnx.TypeProto type_proto: Onnx Type proto of tensor.
    :return List: Shape of the tensor type proto as list.
    """
    return [get_dim_from_type_proto(d) for d in type_proto.tensor_type.shape.dim]


def get_tensor_dtype(onnx_tensor: Union[onnx.ValueInfoProto, TensorProto]) -> np.dtype:
    """
    Get the numpy dtype of the given value info or tensor proto object.

    :param Union[onnx.ValueInfoProto, TensorProto] onnx_tensor: Value info or tensor proto object.
    :return np.dtype: Numpy dtype of the given tensor.
    """
    if isinstance(onnx_tensor, TensorProto):
        onnx_type = onnx_tensor.data_type
    else:
        onnx_type = onnx_tensor.type.tensor_type.elem_type
    if onnx_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type]
    return None


def extract_large_model_helper(
    loader: FrameworkModelLoader, input_tensor_names: List[str], output_tensor_names: List[str]
) -> FrameworkModelLoader:
    """
    Function to extract subgraphs from large models (> 2 GB)

    :param FrameworkModelLoader loader: Onne model loader instance.
    :param List[str] input_tensor_names: List of input tensors which should
        be made subgraph's input.
    :param List[str] output_tensor_names: List of output tensors which shouldbe made subgraph's outputs.
    :return FrameworkModelLoader: Extracted subgraph in the form of FrameworkModelLoader instance
    """
    # Create a deepcopy of original model and remove unnecessary nodes.
    result_loader = loader.clone_loader()
    result_loader.utils.native_shape_inference()
    result_loader.utils.add_inputs(input_tensor_names)
    result_loader.utils.add_outputs(output_tensor_names)
    for model_output in result_loader.get_output_names():
        # To extract subgraph remove the outputs which are not
        # required.
        if model_output not in output_tensor_names:
            result_loader.utils.remove_outputs([model_output])
    result_loader.utils.clean_model().topological_sort()
    return result_loader


def get_extracted_model(
    loader: FrameworkModelLoader, input_tensor_names: List[str], output_tensor_names: List[str]
) -> FrameworkModelLoader:
    """
    Function to extract subgraphs from given onnx model loader.

    :param FrameworkModelLoader loader: Onnx model loader instance.
    :param List[str] input_tensor_names: List of input tensors which should
        be made subgraph's input.
    :param List[str] output_tensor_names: List of output tensors which should
        be made subgraph's outputs
    :return FrameworkModelLoader: Extracted subgraph in the form of FrameworkModelLoader instance
    """
    onnx_version = onnx.version.version
    if loader.size > 2 or  parse_version(onnx_version) <= parse_version("1.6.0"):
        log_debug1("Model size is > 2GB or onnx version <= 1.6.0. Calling own graph extractor APIs.")
        return extract_large_model_helper(loader, input_tensor_names, output_tensor_names)
    else:
        # Onnx model Extractor API is only working if the given model is of
        # size < 2GB. It is internally applying shape inference as a first
        # step and that is where it is breaking.
        from onnx.utils import Extractor
        loader_tmp = loader.clone_loader()
        loader_tmp.utils.native_shape_inference()
        e = Extractor(loader_tmp.model)
        extracted_model = e.extract_model(
            input_tensor_names, output_tensor_names)
        loader_tmp.update_model_shallow(extracted_model)
        return loader_tmp


def get_value_info_by_name(
    model: ModelProto, name: str
) -> Optional[onnx.ValueInfoProto]:
    """
    Function to get the value info by name.

    :param ModelProto model: Onnx model proto instance.
    :param str name: Name of the value info to be found.
    :return Optional[onnx.ValueInfoProto]: Value info proto if found else None.
    """
    for v in model.graph.value_info:
        if v.name == name:
            return v
    for v in model.graph.input:
        if v.name == name:
            return v
    for v in model.graph.output:
        if v.name == name:
            return v
    return None


def get_attribute_value(attr: AttributeProto) -> List[int]:
    """
    Function to get the value of attribute from attribute proto

    :param AttributeProto attr: AttributeProto instance.
    :return List[int]: Values extracted from AttributeProto as np array.
    """
    value = None
    if not isinstance(attr, onnx.AttributeProto):
        return None

    if attr.type == onnx.AttributeProto.TENSOR:
        dtype = np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[attr.t.data_type])
        data = attr.t.raw_data
        value = np.frombuffer(data, dtype=dtype, count=(
            len(data) // dtype.itemsize))
    elif attr.type == onnx.AttributeProto.STRING:
        value = attr.s
        value = value.decode() if isinstance(value, bytes) else value
    else:
        value = helper.get_attribute_value(attr)

    return value


def get_shape(model: ModelProto, name: str, use_onnx_def_symbols: bool = False) -> List[int]:
    """
    Return the shape of tensor.
    Note: Please perform shape inference before calling this API to introduce
    shapes in the model.

    :param ModelProto model: Onnx model proto instance.
    :param str name: Name of the tensor.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :raises RuntimeError: If shape of the tensor can't be found.
    :return List[int]: Shape of the given tensor.
    """
    v = get_value_info_by_name(model, name)
    if v is not None:
        return get_shape_from_value_info_proto(v, use_onnx_def_symbols)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_all_tensors(model: onnx.ModelProto) -> List[onnx.TensorProto]:
    """
    Get the list of all the tensors e.g. Initializer and constant attribute tensors
    from the onnx model.

    :param onnx.ModelProto model: Onnx model proto instance.
    :return List[onnx.TensorProto]: List of all the tensors in the model.
    """
    tensors = []
    for graph in get_graphs(model):
        tensors.extend(graph.initializer)

        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                tensors.extend(attribute.tensors)
    return tensors


def convert_model_to_external_data(model: onnx.ModelProto, file_name: str) -> None:
    """
    Convert the tensors in the given model by updating their data location and
    data offset parameters. Note: This API will not convert data to external data
    but it will populate external data fields in each tensors. Actual conversion
    to external data will happen via onnx.save API.

    :param onnx.ModelProto model: Onnx model proto instance.
    :param str file_name: Path of the onnx external data file.
    """
    for tensor in get_all_tensors(model):
        if (
            tensor.HasField("raw_data")
            and sys.getsizeof(tensor.raw_data) >= ONNX_EXTERNAL_DATA_THRESHOLD
        ):
            set_external_data(tensor, file_name)


def remove_external_data_from_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove external data fields from Model Proto object.

    :param onnx.ModelProto model: Onnx model reference.
    :return onnx.ModelProto: Updated onnx model.
    """
    for tensor in get_all_tensors(model):
        if uses_external_data(tensor):
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]

    return model


def save_model(model: ModelProto, path: Text):
    """
    Save the onnx model to disk.

    :param ModelProto model: Onnx model proto instance.
    :param Text path: Path at which the model is to be saved.
    """
    make_dirs(path)
    model_name = "qnn_onnx_converter_model.onnx"
    if os.path.isfile(path):
        model_name = os.path.splitext(os.path.basename(path))[0]

    weight_file_name = model_name.split(".")[0] + ".data"

    # onnx.save doesn't support saving the model > 2GB ,
    if get_model_size(model) > 2:  # GB
        model_dir = os.path.abspath(os.path.dirname(path))
        convert_model_to_external_data(model, file_name=weight_file_name)
        onnx.save(model, path)
        load_external_data_for_model(model, model_dir)
        # For onnx-1.6, the load_external_data_for_model API loads the external
        # data but it is not removing external data related artifacts from model
        # proto which is wrong.
        if parse_version(onnx.version.version) <= parse_version("1.6.0"):
            model = remove_external_data_from_model(model)
    else:
        onnx.save(model, path)


def create_ort_session(
    model: ModelProto,
    optimize: bool=False,
    custom_lib: Optional[str] = None,
    graph_optimization_level = None,
    **kwargs,
) -> Tuple:
    """
    Function to create Onnx Runtime Session and Verify the Correctness
    If loading is successful, that will verify the model correctness on ort.

    :param ModelProto model: Onnx model proto instance.
    :param bool optimize: Optimize the model using onnxruntime and return the
        optimized model, defaults to False
    :param Optional[str] custom_lib: Path of CustomOp library for onnxruntime
        session, defaults to None
    :param Optional[onnxruntime.GraphOptimizationLevel]
        graph_optimization_level: Graph Optimization to be applied as per the
        level., defaults to None
    :return Tuple[bool, Union[None, onnxruntime.InferenceSession],
        Union[None, ModelProto]]: Tuple of boolean status, Onnx Runtime Session
        Object and Optimized ModelProto.
    """
    # Models having custom op can't run on onnxruntime session.
    try:
        import onnxruntime
        optimized_model = None
        sess_options = onnxruntime.SessionOptions()
        # Disable the Warnings from Onnx Runtime During the Execution
        sess_options.log_severity_level = 3
        if graph_optimization_level:
            sess_options.graph_optimization_level = graph_optimization_level
        # Load the Custom lib is any
        if custom_lib:
            sess_options.register_custom_ops_library(custom_lib)
        execution_providers = ["CPUExecutionProvider"]
        status = True
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_model_path = os.path.join(tmpdirname, "model.onnx")

            if optimize:
                sess_options.optimized_model_filepath = os.path.join(
                    tmpdirname, "model_optimized.onnx")

            save_model(model, temp_model_path)

            session = onnxruntime.InferenceSession(
                temp_model_path, sess_options, providers=execution_providers, **kwargs
            )
            if optimize:
                optimized_model = onnx.load(
                    sess_options.optimized_model_filepath)

    except ImportError as e:
        session = None
        status = False
        optimized_model = None
        log_warning("Onnxruntime not found in current environment. Hence " \
            "Onnxruntime session is not created.")

    except Exception as e:
        session = None
        status = False
        optimized_model = None
        log_warning(f"Creation of ORT session is failed! with Exception : {str(e)}")

    return status, session, optimized_model


def run_ort_inference(model: ModelProto, input_data: Dict[Text, np.ndarray], output_node_names: List[Text], custom_lib: Optional[Text] = None) -> Dict[Text, np.ndarray]:
    """
    Run the inference of given onnx model via Onnx Runtime

    :param ModelProto model: Onnx model proto instance.
    :param Dict[Text, np.ndarray] input_data: Dict containing input tensor name
        to corresponding tensor data.
    :param List[Text] output_node_names: List of output names.
    :param Optional[Text] custom_lib: Path of custom lib, defaults to None
    :return Dict[Text, np.ndarray]: Dict containing output tensor name as key and
        its computed numpy array output as value.
    """
    status, session, _ = create_ort_session(
        model, optimize=False, custom_lib=custom_lib
    )
    if not status:
        log_error("Failed to create Onnxruntime Session.")
        return None

    model_output_names = [x.name for x in session.get_outputs()]

    if not output_node_names:
        output_node_names = model_output_names
    else:
        for output_name in output_node_names:
            if output_name not in model_output_names:
                log_error(
                    f"Given output: {output_name} is not found in "
                    "model's outputs. Please create that tensor as model's "
                    "output and then supply for inference."
                )
                return None

    input_node_names = [x.name for x in session.get_inputs()]
    filtered_input_data = {}
    for input_name in input_node_names:
        if input_name in input_data:
            filtered_input_data[input_name] = input_data[input_name]
        else:
            log_error(
                f"Inference input data for input: {input_name} is "
                "not found. Please provide the same for inference."
            )
            return None

    try:
        import onnxruntime
    except ImportError as e:
        log_warning("Onnxruntime not found in current environment. Hence " \
            "Onnxruntime inference is not performed.")
        return None

    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 3
    outputs = OrderedDict(
        zip(
            output_node_names,
            session.run(
                output_node_names, filtered_input_data, run_options=run_options
            ),
        )
    )
    return outputs


def order_repeated_field(
    repeated_proto: AttributeProto, key_name: str, order: List[str]
) -> None:
    """
    Function to sort the fields in NodeProto.

    :param AttributeProto repeated_proto: NodeProto of a node
    :param str key_name: key_name for each attribute
    :param List[str] order: List of arguments for a node
    """
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_tensor(name: str, data: Union[np.ndarray, List]) -> TensorProto:
    """
    Function to generate TensorProto object based on given datatype, dims and values.

    :param str name: Name of the TensorProto.
    :param Union[np.ndarray, List] data: Actual data to be used for the TensorProto.
    :return TensorProto: return tensor proto.
    """
    if isinstance(data, List):
        data = np.array(data, dtype=np.float32)

    tensor = helper.make_tensor(
        name=name,
        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype],
        dims=data.shape,
        vals=data.flatten().tolist(),
    )
    return tensor


def make_node(
    op_type: str,
    inputs: List[str],
    outputs: List[str],
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    domain: Optional[str] = None,
    **kwargs: Dict,
) -> NodeProto:
    """
    Function to generate node based on given params and doc_string

    :param str op_type: Node operator type
    :param List[str] inputs: List of input node names
    :param List[str] outputs: List of output node names
    :param Optional[str] name: Name of the node. Defaults to None.
    :param Optional[str] doc_string: Doc string used to describe the graph.
        Defaults to None.
    :param Optional[str] domain: Domain name for the node. Defaults to None.
    :return NodeProto: NodeProto of the generated Node
    """
    node = helper.make_node(
        op_type, inputs, outputs, name, doc_string, domain, **kwargs
    )
    if doc_string == "":
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def make_graph(*args: List, doc_string: str = "", **kwargs: Dict) -> GraphProto:
    """
    Function to generate graph based on given args and doc_string.

    :param str doc_string: Doc string used to describe the graph, defaults to ""
    :return GraphProto: GraphProto instance based on given inputs, outputs,
        value_infos and nodes required for generating graph.
    """
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == "":
        graph.doc_string = ""
    return graph


def opset_to_ir_version(opset_version: int) -> int:
    """
    Function to get the IR version corresponding to opset_version.

    :param int opset_version:: Opset version of default domain which is "" or "ai.onnx".
    :return int: IR Version corresponding to Opset Version.
    """
    if ("ai.onnx", opset_version) not in helper.OP_SET_ID_VERSION_MAP:
        log_warning(
            "Fail to get the IR version for given opset_version: {opset_version}. "
            "Using IR Version 6 by default."
        )
        return 6

    return helper.OP_SET_ID_VERSION_MAP[("ai.onnx", opset_version)]


def remove_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove shape information from the onnx model.

    :param onnx.ModelProto model: Onnx model reference.
    :return onnx.ModelProto: Updated onnx model.
    """
    for val_info in model.graph.value_info:
        val_info.type.tensor_type.ClearField("shape")
    return model
