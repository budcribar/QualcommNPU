# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import re
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.translation_utils import are_shapes_broadcastable
from qti.aisw.converters.common.converter_ir.axis_tracker import RelayAxisOrder
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_adapter import (
    CastOp,
    ConcatOp,
    ConstantOp,
    DequantizeOp,
    ExpandOp,
    GatherOp,
    GatherElementsOp,
    GatherNDOp,
    IdentityOp,
    ElementwiseNeuronOp,
    OneHotOp,
    PackOp,
    QuantizeOp,
    ReshapeOp,
    ResizeOp,
    ScatterNDOp,
    SplitOp,
    StridedSliceOp,
    TileOp,
    TransposeOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase, RelayQuantization

from qti.aisw.converters.relay.translations import RelayTranslations
from qti.aisw.converters.relay.utils import get_prim_type

import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type


# ------------------------------------------------------------------------------
#   Broadcast_to
# ------------------------------------------------------------------------------
class RelayBroadcastToTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBroadcastToTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        broadcast_to_attrs = relay_expr.attrs
        attr_dict = {}

        if isinstance(broadcast_to_attrs.shape, tvm.ir.container.Array):
            attr_dict["shape"] = [int(dim) for dim in broadcast_to_attrs.shape]
        else:
            raise ValueError("Relay Broadcast_to {} only support static shape tensor".format(relay_expr))

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        output_shape = attr_dict['shape']

        if len(input_shape) > len(output_shape):
            raise ValueError("Broadcast_to Op {} output rank ({}) shall not be smaller than input rank ({})"
                             .format(input_names[0] ,len(output_shape), len(input_shape)))

        if not are_shapes_broadcastable(shapes_list=[input_shape, output_shape], axis_order=RelayAxisOrder, align_channels=False):
            raise ValueError("Broadcast_to Op input shape: {} can't be broadcast to {}".format(input_shape, output_shape))

        op_name = converter_context.get_op_name(relay_expr, ExpandOp.TRANSLATION_KEY, ExpandOp.LEGACY_TRANSLATION_KEY)
        ir_op = ExpandOp(name=op_name, shape=output_shape)

        return ir_op


RelayTranslations.register_translation(RelayBroadcastToTranslation(),
                                       converter_type('broadcast_to', 'relay'))


# ------------------------------------------------------------------------------
#   Cast
# ------------------------------------------------------------------------------
class RelayCastTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCastTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        cast_attrs = relay_expr.attrs
        attr_dict = {}
        attr_dict["to_dtype"] = cast_attrs.dtype
        attr_dict["from_dtype"] = relay_expr.args[0].checked_type.dtype

        log_debug3("\tto_dtype {}", attr_dict["to_dtype"])
        log_debug3("\tfrom_dtype {}", attr_dict["from_dtype"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, CastOp.TRANSLATION_KEY, CastOp.LEGACY_TRANSLATION_KEY)
        to_dtype = attr_dict["to_dtype"]
        from_dtype = attr_dict["from_dtype"]

        ir_op = CastOp(op_name, to_type=to_dtype, from_type=from_dtype)
        return ir_op


RelayTranslations.register_translation(RelayCastTranslation(),
                                       converter_type('cast', 'relay'))


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class RelayClipTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayClipTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["clip_min"] = relay_expr.attrs.a_min
        attr_dict["clip_max"] = relay_expr.attrs.a_max

        log_debug3("\tclip min {}", attr_dict["clip_min"])
        log_debug3("\tclip max {}", attr_dict["clip_max"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ElementwiseNeuronOp.TRANSLATION_KEY, ElementwiseNeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = ElementwiseNeuronOp(op_name,
                         operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                         min_value=attr_dict["clip_min"],
                         max_value=attr_dict["clip_max"])
        return ir_op


RelayTranslations.register_translation(RelayClipTranslation(),
                                       converter_type('clip', 'relay'))


# ------------------------------------------------------------------------------
#   Copy
# ------------------------------------------------------------------------------
class RelayCopyTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCopyTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, IdentityOp.TRANSLATION_KEY, IdentityOp.LEGACY_TRANSLATION_KEY)

        return IdentityOp(op_name)


RelayTranslations.register_translation(RelayCopyTranslation(),
                                       converter_type('copy', 'relay'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class RelayConcatTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConcatTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ConcatOp.TRANSLATION_KEY, ConcatOp.LEGACY_TRANSLATION_KEY)

        if len(input_names) == 1:
            return IdentityOp(op_name)

        ir_op = ConcatOp(op_name, axis=attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelayConcatTranslation(),
                                       converter_type('concatenate', 'relay'))


# ------------------------------------------------------------------------------
#   Dequantize
# ------------------------------------------------------------------------------
class RelayDequantizeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDequantizeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        quant_attrs = relay_expr.attrs

        attr_dict = {}
        attr_dict['axis'] = quant_attrs.axis
        attr_dict['dtype'] = relay_expr.args[0].checked_type.dtype
        m = re.search(r'\d+$', relay_expr.args[0].checked_type.dtype)
        attr_dict['bw'] = int(m.group()) if m else RelayQuantization.DefaultBw
        if attr_dict['bw'] not in [8, 16]:
            log_error("Relay Dequantize {} dtype only support [uint8, int8, uint16, int16]".format(relay_expr))

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, DequantizeOp.TRANSLATION_KEY)
        q_params = RelayQuantization.get_quantization_params(op_name, input_names, relay_params, attr_dict)
        q_params['bw'] = attr_dict['bw']
        log_debug3("\tQuantization attributes {}", attr_dict)
        log_debug3("\tQuantization params {}", q_params)

        # Strip the additional quantization inputs
        for name in input_names[1:]:
            input_names.remove(name)

        quir_graph.add_quantization_params(op_name,
                                           output_encodings=q_params)

        return DequantizeOp(op_name, axis=q_params['axis'], bw=q_params['bw'], scale=q_params['scale'], offset=q_params['offset'])


RelayTranslations.register_translation(RelayDequantizeTranslation(),
                                       converter_type('qnn.dequantize', 'relay'))


# ------------------------------------------------------------------------------
#   ExpandDims
# ------------------------------------------------------------------------------
class RelayExpandDimsTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayExpandDimsTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        expand_dims_attrs = relay_expr.attrs
        attr_dict['axis'] = expand_dims_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)
        axis = attr_dict['axis']
        log_debug3("\taxis {}", axis)

        mod = tvm.IRModule.from_expr(relay_expr)
        mod = relay.transform.InferType()(mod)
        output_shape = mod["main"].ret_type.shape
        if isinstance(output_shape, tvm.ir.container.Array):
            log_debug3("\toutput shape {}", output_shape)
            output_shape = [int(x) for x in output_shape]

        ir_op = ReshapeOp(op_name, shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelayExpandDimsTranslation(),
                                       converter_type('expand_dims', 'relay'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class RelayFlattenTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayFlattenTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        output_shape = list()
        output_shape.append(input_shape[0]) # batch
        output_shape.append(int(np.prod(input_shape[1:])))

        log_debug3("\tOp input shape {}", input_shape)
        log_debug3("\tOp new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelayFlattenTranslation(),
                                       converter_type('batch_flatten', 'relay'))


# ------------------------------------------------------------------------------
#   Full
# ------------------------------------------------------------------------------
class RelayFullTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayFullTranslation, self).__init__()
        self.value = None

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        relay_attrs = relay_expr.attrs
        attr_dict['shape'] = [int(m) for m in relay_attrs.shape]
        attr_dict['value'] = getattr(relay_attrs, 'fill_value', float(self.value))
        attr_dict['dtype'] = relay_attrs.dtype if relay_attrs.dtype else np.float32

        log_debug3("\t shape {}", attr_dict["shape"])
        log_debug3("\t value {}", attr_dict["value"])
        log_debug3("\t dtype {}", attr_dict["dtype"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ConstantOp.TRANSLATION_KEY, ConstantOp.LEGACY_TRANSLATION_KEY)

        shape = attr_dict['shape']
        value = attr_dict['value']
        dtype = attr_dict['dtype']
        tensor = np.full(shape, value, dtype=dtype)

        ir_op = ConstantOp(op_name, tensor=tensor)
        return ir_op


RelayTranslations.register_translation(RelayFullTranslation(),
                                       converter_type('full', 'relay'))


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class RelayGatherTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGatherTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        gather_attrs = relay_expr.attrs
        attr_dict['axis'] = gather_attrs.axis

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GatherElementsOp.TRANSLATION_KEY, GatherElementsOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict['axis']

        if input_names[0] in relay_params:
            data = relay_params[input_names[0]]
            if isinstance(data, tvm.runtime.ndarray.NDArray) or isinstance(data, tvm.runtime.NDArray):
                data = data.asnumpy()
            data_output = input_names[0]
            quir_graph.add(ConstantOp(data_output, data), [], [data_output])

        if input_names[1] in relay_params:
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            indices_output = input_names[1]
            quir_graph.add(ConstantOp(indices_output, indices), [], [indices_output])

        ir_op = GatherElementsOp(op_name, axis=axis)
        return ir_op


RelayTranslations.register_translation(RelayGatherTranslation(),
                                       converter_type('gather', 'relay'))


# ------------------------------------------------------------------------------
#   GatherND
# ------------------------------------------------------------------------------
class RelayGatherNDTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGatherNDTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        attrs = relay_expr.attrs
        attr_dict['batch_dims'] = attrs.batch_dims

        log_debug3("\batch_dims {}", attr_dict["batch_dims"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GatherNDOp.TRANSLATION_KEY, GatherNDOp.LEGACY_TRANSLATION_KEY)

        batch_dims = attr_dict['batch_dims']

        if input_names[0] in relay_params:
            data = relay_params[input_names[0]]
            if isinstance(data, tvm.runtime.ndarray.NDArray) or isinstance(data, tvm.runtime.NDArray):
                data = data.asnumpy()
            log_debug3("\tdata shape {}", data.shape)
            quir_graph.add(ConstantOp(input_names[0], data), [], [input_names[0]])

        # relay frontend will add additional transpose into relay mod since indices of relay.gather_nd is column based,
        # we should add another transpose to negate the transpose added by frontend since QNNIR is row based.
        if input_names[1] in relay_params:
            # if indices is constant, transpose it directly
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            indices_rank = len(indices.shape)
            perm = list(range(1, indices_rank)) + [0]
            indices = np.ascontiguousarray(np.transpose(indices, perm))
            log_debug3("\tindices shape {}", indices.shape)
            quir_graph.add(ConstantOp(input_names[1], indices), [], [input_names[1]])
        else:
            indices_rank = len(quir_graph.get_buffer(input_names[1]).shape)
            perm = list(range(1, indices_rank)) + [0]
            transpose_op_name = input_names[1] + '_permute'
            transpose_op = TransposeOp(name=transpose_op_name, perm=perm)
            transpose_node = quir_graph.add(transpose_op, input_names=[input_names[1]], output_names=[transpose_op_name])

            input_names[1] = transpose_node.output_names[0]

        ir_op = GatherNDOp(op_name, batch_dims=batch_dims)
        return ir_op


RelayTranslations.register_translation(RelayGatherNDTranslation(),
                                       converter_type('gather_nd', 'relay'))


# ------------------------------------------------------------------------------
#   LayoutTransform
# ------------------------------------------------------------------------------
class RelayLayoutTransformTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLayoutTransformTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        attr_dict["src_layout"] = relay_expr.attrs.src_layout
        attr_dict["dst_layout"] = relay_expr.attrs.dst_layout

        log_debug3("\t src_layout {}", attr_dict["src_layout"])
        log_debug3("\t dst_layout {}", attr_dict["dst_layout"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TransposeOp.TRANSLATION_KEY,
                                                TransposeOp.LEGACY_TRANSLATION_KEY)

        src_layout = attr_dict["src_layout"]
        dst_layout = attr_dict["dst_layout"]

        permute_order = [src_layout.index(axis_name) for axis_name in dst_layout]

        log_debug3("\t permute_order {}", permute_order)

        # transform src_layout to IR axis_format after permute_order is extracted
        # TODO: Add more valid transformation for other TVM layouts
        if src_layout in ["NCW", "NWC", "WIO", "WOI", "OIW", "IOW"]:
            src_layout = src_layout.replace("W", "F")

        input_buffer = quir_graph.buffers[input_names[0]]
        if src_layout in AxisTracker.AxisFormat.ir_to_c_axis_format:
            input_buffer.axis_format = src_layout
        # The reason why we don't raise error in else case is that
        # sometimes TVM ConvertLayout pass will generate strange layout like "CNHW",
        # which should be labelled as NONTRIVIAL in populate_data_axis_format().
        # For this case, we let TransposeOp's populate_data_axis_format()
        # to handle the input NONTRIVIAL and propagate the NONTRIVIAL layout.

        return TransposeOp(op_name, permute_order)


RelayTranslations.register_translation(RelayLayoutTransformTranslation(),
                                       converter_type('layout_transform', 'relay'))


# ------------------------------------------------------------------------------
#   OneHot
# ------------------------------------------------------------------------------
class RelayOneHotTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayOneHotTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        # Based on tvm.relay.one_hot documentation, depth can be extracted from attribute or relay.expr
        if hasattr(relay_expr.attrs, "depth"):
            attr_dict["depth"] = relay_expr.attrs.depth
            log_debug3("\tdepth {}", attr_dict["depth"])

        attr_dict["axis"] = relay_expr.attrs.axis
        log_debug3("\taxis {}", attr_dict["axis"])
        attr_dict["dtype"] = relay_expr.attrs.dtype
        log_debug3("\tdtype {}", attr_dict["dtype"])

        return attr_dict

    def parse_on_off_value(self, value, dtype):
        if isinstance(value, tvm.runtime.ndarray.NDArray) or isinstance(value, tvm.runtime.NDArray):
            value = value.asnumpy()
        # To extract the numpy value from a 0-d numpy array
        value = np.atleast_1d(value)[0]
        value = value.astype(dtype)

        return value

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, OneHotOp.TRANSLATION_KEY, OneHotOp.LEGACY_TRANSLATION_KEY)

        # Extract constant indices and add it to the graph
        if input_names[0] in relay_params and not quir_graph.has_buffer(input_names[0]):
            indices = relay_params[input_names[0]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            log_debug3("\tindices {}", indices)
            input_const_op = ConstantOp(input_names[0], indices)
            quir_graph.add(input_const_op, [], [input_names[0]])

        # on_value and off_value are as type relay.expr in tvm while they are defined as attribute
        # in QnnIR, so we could only extract them from relay_params as relay.const
        on_value_name = input_names[1]
        if on_value_name not in relay_params:
            raise ValueError("OneHotOp {} only support on_value as an attribute, got dynamic input {}".format(op_name, on_value_name))
        on_value = self.parse_on_off_value(relay_params[on_value_name], attr_dict["dtype"])
        log_debug3("\tOneHotOp {} has on_value {}", op_name, on_value)

        off_value_name = input_names[2]
        if off_value_name not in relay_params:
            raise ValueError("OneHotOp {} only support off_value as an attribute, got dynamic input {}".format(op_name, off_value_name))
        off_value = self.parse_on_off_value(relay_params[off_value_name], attr_dict["dtype"])
        log_debug3("\tOneHotOp {} has off_value {}", op_name, off_value)

        # depth is as an type int or relay.expr in tvm, but it is defined as an attribute in QnnIR.
        # If depth exists in input_names, we could only take it as relay.const and extract it from relay_params
        if len(input_names) > 3:
            if input_names[3] not in relay_params:
                raise ValueError("OneHotOp {} only support depth as an attribute, got dynamic input {}".format(op_name, input_names[3]))
            depth = relay_params[input_names[3]]
            if isinstance(depth, tvm.runtime.ndarray.NDArray) or isinstance(depth, tvm.runtime.NDArray):
                depth = depth.asnumpy()
            attr_dict["depth"] = int(depth)
            log_debug3("\tdepth {}", attr_dict["depth"])

        if attr_dict["depth"] < 0:
            raise ValueError("Invalid attribute for OneHotOp, expected non-negative depth, got {}".format(attr_dict["depth"]))

        # Remove additional OneHotOp inputs from the input names list
        for name in input_names[1:]:
            input_names.remove(name)

        ir_op = OneHotOp(op_name, on_value=on_value, off_value=off_value, depth=attr_dict["depth"], axis=attr_dict["axis"])

        return ir_op


RelayTranslations.register_translation(RelayOneHotTranslation(),
                                       converter_type('one_hot', 'relay'))


# ------------------------------------------------------------------------------
#   Ones
# ------------------------------------------------------------------------------
class RelayOnesTranslation(RelayFullTranslation):
    def __init__(self):
        super(RelayOnesTranslation, self).__init__()
        self.value = 1


RelayTranslations.register_translation(RelayOnesTranslation(),
                                       converter_type('ones', 'relay'))


# ------------------------------------------------------------------------------
#   Quantize
# ------------------------------------------------------------------------------
class RelayQuantizeTranslation(RelayTranslationBase):

    def __init__(self):
        super(RelayQuantizeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        quant_attrs = relay_expr.attrs

        attr_dict = {}
        attr_dict['dtype'] = quant_attrs.out_dtype
        attr_dict['axis'] = quant_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, QuantizeOp.TRANSLATION_KEY)
        q_params = RelayQuantization.get_quantization_params(op_name, input_names, relay_params, attr_dict)

        log_debug3("\tQuantization attributes {}", attr_dict)
        log_debug3("\tQuantization params {}", q_params)

        # Strip the additional quantization inputs
        #input_names = input_names[0:1]
        for name in input_names[1:]:
            input_names.remove(name)

        quir_graph.add_quantization_params(op_name,
                                           output_encodings=q_params)

        return QuantizeOp(op_name, axis=q_params['axis'], bw=q_params['bw'], scale=q_params['scale'], offset=q_params['offset'])


RelayTranslations.register_translation(RelayQuantizeTranslation(),
                                       converter_type('qnn.quantize', 'relay'))


# ------------------------------------------------------------------------------
#   Repeat
# ------------------------------------------------------------------------------
class RelayRepeatTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayRepeatTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["repeats"] = int(relay_expr.attrs.repeats)
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TileOp.TRANSLATION_KEY, TileOp.LEGACY_TRANSLATION_KEY)
        axis = attr_dict["axis"]
        repeats = attr_dict["repeats"]

        input_rank = quir_graph.get_buffer(input_names[0]).rank()

        if axis < 0:
            axis += input_rank

        log_assert(axis < input_rank and axis >= 0, "Axis value shall be less than the number of data dimension")

        multiples = [1]*input_rank
        multiples[axis] = repeats

        ir_op = TileOp(op_name, multiples=multiples)
        return ir_op


RelayTranslations.register_translation(RelayRepeatTranslation(),
                                       converter_type('repeat', 'relay'))


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class RelayReshapeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReshapeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["new_shape"] = [int(val) for val in relay_expr.attrs.newshape]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)

        new_shape = attr_dict["new_shape"]
        log_debug3("\tReshape Op attribute new shape {}", new_shape)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        log_debug3("\tReshape Op input shape {}", input_shape)

        ir_op = ReshapeOp(op_name, shape=new_shape)
        return ir_op


RelayTranslations.register_translation(RelayReshapeTranslation(),
                                       converter_type('Reshape', 'relay'))


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class RelayResizeTranslationBase(RelayTranslationBase):

    class RelayTransformModes:
        ALIGN_CORNERS = "align_corners"
        ASYMMETRIC = "asymmetric"
        HALF_PIXEL = "half_pixel"
        PYTORCH_HALF_PIXEL = "pytorch_half_pixel"
        TF_HALF_PIXEL = "tf_half_pixel_for_nn"
        TF_CROP_AND_RESIZE = "tf_crop_and_resize"

    class RelayScaleModes:
        CUBIC = "cubic"
        LINEAR = "linear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    class RelayRoundingModes:
        CEIL = "ceil"
        FLOOR = "floor"
        ROUND = "round"
        # option not valid in relay doc, but may appear from onnx frontend
        ROUND_PREFER_FLOOR = "round_prefer_floor"
        ROUND_PREFER_CEIL = "round_prefer_ceil"

    RelayToQnn_NearestMode = {
        RelayRoundingModes.CEIL: ir_graph.QNN_OP_RESIZE_NEAREST_MODE_CEIL,
        RelayRoundingModes.FLOOR: ir_graph.QNN_OP_RESIZE_NEAREST_MODE_FLOOR,
        RelayRoundingModes.ROUND: ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR,
        RelayRoundingModes.ROUND_PREFER_CEIL: ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL,
        RelayRoundingModes.ROUND_PREFER_FLOOR: ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR,
    }

    RelayToQnn_InterpolationMode = {
        RelayScaleModes.CUBIC: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC,
        RelayScaleModes.LINEAR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        RelayScaleModes.NEAREST_NEIGHBOR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
    }


    RelayToQnn_TransformMode = {
        RelayTransformModes.ALIGN_CORNERS: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS,
        RelayTransformModes.ASYMMETRIC: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC,
        RelayTransformModes.HALF_PIXEL: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
        RelayTransformModes.PYTORCH_HALF_PIXEL: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_PYTORCH_HALF_PIXEL,
        RelayTransformModes.TF_HALF_PIXEL: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
    }


    def __init__(self):
        super(RelayResizeTranslationBase, self).__init__()

    def extract_attributes(self, relay_expr: relay.expr.Call, relay_params: dict, **kwargs):
        attr_dict = {}
        resize_attrs = relay_expr.attrs

        attr_dict['size'] = [int(num) for num in getattr(resize_attrs, 'size')]
        attr_dict['layout'] = getattr(resize_attrs, 'layout')

        log_debug3("\tsize {}", attr_dict['size'])
        log_debug3("\tlayout {}", attr_dict['layout'])

        output_dtype = getattr(resize_attrs, "output_dtype", None)
        if output_dtype is not None:
            raise ValueError("Unsupported conversion to output dtype {} for resize expr".format(output_dtype))

        scale_mode = getattr(resize_attrs, "method", self.RelayScaleModes.LINEAR)

        attr_dict["interpolation_mode"] = self.RelayToQnn_InterpolationMode[scale_mode]
        log_debug3("\tinterpolation_mode mode {}", attr_dict['interpolation_mode'])

        transform_mode = getattr(resize_attrs, "coordinate_transformation_mode", self.RelayTransformModes.HALF_PIXEL)

        if transform_mode == self.RelayTransformModes.TF_CROP_AND_RESIZE:
            raise ValueError(f"Unsupported coordinate transformation_mode: {self.RelayTransformModes.TF_CROP_AND_RESIZE}")
        attr_dict["transformation_mode"] = self.RelayToQnn_TransformMode[transform_mode]
        log_debug3("\ttransformation_mode mode {}", attr_dict['transformation_mode'])

        # round_mode is only effective if scale_mode == self.ScaleModes.NEAREST_NEIGHBOR
        round_mode = getattr(resize_attrs, "rounding_method", self.RelayRoundingModes.ROUND)
        # For some models the rounding_method attr is an empty string
        round_mode = round_mode if round_mode else self.RelayRoundingModes.ROUND
        attr_dict["nearest_mode"] = self.RelayToQnn_NearestMode[round_mode]
        log_debug3("\tnearest_mode mode {}", attr_dict["nearest_mode"])

        # cubic_coeff is only effective if scale_mode == self.ScaleModes.CUBIC
        cubic_coeff = getattr(resize_attrs, "cubic_alpha", -0.5)
        attr_dict["cubic_coeff"] = cubic_coeff
        log_debug3("\tcubic_coeff mode {}", cubic_coeff)

        cubic_exclude = getattr(resize_attrs, "cubic_exclude", 0)
        exclude_outside = bool(cubic_exclude)
        attr_dict["exclude_outside"] = exclude_outside
        log_debug3("\texclude_outside mode {}", exclude_outside)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ResizeOp.TRANSLATION_KEY, ResizeOp.LEGACY_TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        rank = len(input_shape)

        if attr_dict['layout'] not in ["NWC", "NHWC", "NDHWC"]:
            raise ValueError(f"invalid layout: {attr_dict['layout']}")

        input_spatial = input_shape[1:-1]
        output_spatial = attr_dict['size']
        scale_width = output_spatial[-1] / input_spatial[-1]
        scale_height = output_spatial[-2] / input_spatial[-2] if rank in [4,5] else None
        scale_depth = output_spatial[-3] / input_spatial[-3] if rank in [5] else None

        ir_op = ResizeOp(op_name,
                         transformation_mode=attr_dict["transformation_mode"],
                         interpolation_mode=attr_dict["interpolation_mode"],
                         cubic_coeff=attr_dict["cubic_coeff"],
                         exclude_outside=attr_dict["exclude_outside"],
                         nearest_mode=attr_dict["nearest_mode"],
                         scale_height=scale_height,
                         scale_width=scale_width,
                         scale_depth=scale_depth)

        return ir_op


# ------------------------------------------------------------------------------
#   Resize1D
# ------------------------------------------------------------------------------
class RelayResize1DTranslation(RelayResizeTranslationBase):
    def __init__(self):
        super(RelayResize1DTranslation, self).__init__()


RelayTranslations.register_translation(RelayResize1DTranslation(),
                                       converter_type('resize1d', 'relay'))


# ------------------------------------------------------------------------------
#   Resize2D
# ------------------------------------------------------------------------------
class RelayResize2DTranslation(RelayResizeTranslationBase):
    def __init__(self):
        super(RelayResize2DTranslation, self).__init__()


RelayTranslations.register_translation(RelayResize2DTranslation(),
                                       converter_type('resize2d', 'relay'))


# ------------------------------------------------------------------------------
#   Resize3D
# ------------------------------------------------------------------------------
class RelayResize3DTranslation(RelayResizeTranslationBase):
    def __init__(self):
        super(RelayResize3DTranslation, self).__init__()


RelayTranslations.register_translation(RelayResize3DTranslation(),
                                       converter_type('resize3d','relay'))


# ------------------------------------------------------------------------------
#   Reverse
# ------------------------------------------------------------------------------
class RelayReverseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReverseTranslation, self).__init__()

    def extract_attributes(self, relay_expr: relay.expr.Call, relay_params: dict, **kwargs):

        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        input_shape = quir_graph.get_buffer(input_names[0]).shape
        reverse_axis = attr_dict["axis"]

        begin = [0]*len(input_shape)
        end = list(input_shape)
        strides = [1]*len(input_shape)

        # We use StridedSlice with stride = -1, end = -1 to mimic the effect of reverse.
        # For the axis which is reversed, (begin, end, stride) = (dim[axis]-1, -1, -1).
        # Otherwise, (begin, end, stride) = (0, dim[axis], 1)
        begin[reverse_axis] = input_shape[reverse_axis] - 1
        end[reverse_axis] = -1
        strides[reverse_axis] = -1

        op_name = converter_context.get_op_name(relay_expr, StridedSliceOp.TRANSLATION_KEY,
                                                StridedSliceOp.LEGACY_TRANSLATION_KEY)
        ranges = list(map(list, zip(begin, end, strides)))
        ir_op = StridedSliceOp(op_name, ranges=ranges)
        return ir_op


RelayTranslations.register_translation(RelayReverseTranslation(),
                                       converter_type('reverse', 'relay'))


# ------------------------------------------------------------------------------
#   ScatterND
# ------------------------------------------------------------------------------
class RelayScatterNDTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayScatterNDTranslation, self).__init__()
        self.reduction_types = {"none": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_NONE,
                                "add": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_ADD,
                                "mul": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_MUL}

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        scatternd_attrs = relay_expr.attrs
        attr_dict['mode'] = "none"
        if scatternd_attrs.mode == 'add':
            attr_dict['mode'] = "add"
        elif scatternd_attrs.mode != 'update':
            raise TypeError("Unsupported mode for scatter_nd: {}".format(scatternd_attrs.mode))
        log_debug3("\taccumulation mode {}", attr_dict['mode'])
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ScatterNDOp.TRANSLATION_KEY)

        # indices of relay.scatter_nd is column-based while indices of QNNIR ScatterND is row-based, so transpose the indices
        if not quir_graph.has_buffer(input_names[1]) and input_names[1] in relay_params:
            # if indices is constant, transpose it directly
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            indices_rank = len(indices.shape)
            perm = list(range(1, indices_rank)) + [0]
            indices = np.ascontiguousarray(np.transpose(indices, perm))
            log_debug3("\tindices shape {}", indices.shape)
            indices_output = input_names[1]
            # we don't need to populate quantization params for input[1] since it is indices and its dtype is int32
            quir_graph.add(ConstantOp(indices_output, indices), [], [indices_output])
        else:
            indices_rank = len(quir_graph.get_buffer(input_names[1]).shape)
            perm = list(range(1, indices_rank)) + [0]
            transpose_op_name = input_names[1] + '_permute'
            transpose_op = TransposeOp(name=transpose_op_name, perm=perm)
            transpose_node = quir_graph.add(transpose_op, input_names=[input_names[1]], output_names=[transpose_op_name])
            input_names[1] = transpose_node.output_names[0]

        if not quir_graph.has_buffer(input_names[2]) and input_names[2] in relay_params:
            updates = relay_params[input_names[2]]
            if isinstance(updates, tvm.runtime.ndarray.NDArray) or isinstance(updates, tvm.runtime.NDArray):
                updates = updates.asnumpy()
            log_debug3("\tupdates shape {}", updates.shape)
            updates_output = input_names[2]
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [updates_output], is_param=True)
            quir_graph.add(ConstantOp(updates_output, updates), [], [updates_output])

        ir_op = ScatterNDOp(op_name, reduction=self.reduction_types[attr_dict['mode']])
        return ir_op

RelayTranslations.register_translation(RelayScatterNDTranslation(),
                                       converter_type('scatter_nd', 'relay'))


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class RelaySplitTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySplitTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)
        attr_dict["slice_points"] = relay_expr.attrs.indices_or_sections

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SplitOp.TRANSLATION_KEY, SplitOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict["axis"]
        slice_points = attr_dict["slice_points"]

        output_shapes = []
        slices = []
        num_outputs = 0

        input_shapes = converter_context.get_input_shapes(relay_expr)
        slice_input_shape = input_shapes[0][:]
        if isinstance(slice_points, tvm.ir.container.Array):
            log_debug3("\tslice points {}", slice_points)
            num_outputs = len(slice_points) + 1
            slices = [int(val) for val in slice_points]

            log_debug3("\tmax dim {}", slice_input_shape[axis])
            slice_sizes = [0] + slices + [slice_input_shape[axis]]
            log_debug3("\tslice sizes {}", slice_sizes)

            for i in range(num_outputs):
                output_shapes.append(slice_input_shape[:])
                output_shapes[i][axis] = slice_sizes[i + 1] - slice_sizes[i]
        elif isinstance(slice_points, tvm.tir.expr.IntImm):
            log_debug3("\tslice points {}", int(slice_points))
            num_outputs = int(slice_points)

            # IR can handle [] and split the output evenly using the num of outputs
            slices = []

            for i in range(num_outputs):
                output_shapes.append(input_shapes[0][:])
                output_shapes[i][axis] = int(int(output_shapes[i][axis]) / num_outputs)
        else:
            raise TypeError("Unsupported type {} for slice_points in SplitOp".format(type(slice_points)))

        log_debug3("\tnum_outputs {}", num_outputs)
        log_debug3("\tslices {}", slices)
        log_debug3("\toutput shapes {}", output_shapes)

        ir_op = SplitOp(op_name, axis=axis, split_index=slices, output_shape=output_shapes)
        ir_op.num_outputs = num_outputs
        return ir_op


RelayTranslations.register_translation(RelaySplitTranslation(),
                                       converter_type('Split', 'relay'))


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class RelaySqueezeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySqueezeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        squeeze_attrs = relay_expr.attrs
        attr_dict["axis"] = squeeze_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        log_debug3("\tSqueeze Op input shape {}", input_shape)

        axis = attr_dict['axis']

        if axis is None:
            output_shape = [dim for dim in input_shape if dim != 1]
        else:
            axis = [ax % len(input_shape) for ax in axis]
            log_debug3("\taxis {}", axis)

            output_shape = []
            for index, shape in enumerate(input_shape):
                if index in axis:
                    if shape != 1:
                        raise ValueError("Input shape {} at axis {} should be 1", input_shape, index)
                    continue
                output_shape.append(shape)
        log_debug3("\tSqueeze Op new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelaySqueezeTranslation(),
                                       converter_type('squeeze', 'relay'))


# ------------------------------------------------------------------------------
#   Stack
# ------------------------------------------------------------------------------
class RelayStackTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayStackTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        stack_attrs = relay_expr.attrs
        attr_dict["axis"] = stack_attrs.axis
        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PackOp.TRANSLATION_KEY, PackOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict["axis"]
        if isinstance(axis, tvm.tir.expr.IntImm):
            axis = int(axis)

        ir_op = PackOp(op_name, axis=axis)

        return ir_op


RelayTranslations.register_translation(RelayStackTranslation(),
                                       converter_type('stack', 'relay'))


# ------------------------------------------------------------------------------
#   StridedSlice
# ------------------------------------------------------------------------------
class RelayStridedSliceTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayStridedSliceTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        strided_slice_attrs = relay_expr.attrs
        attr_dict['begin'] = strided_slice_attrs.begin
        attr_dict['end'] = strided_slice_attrs.end
        attr_dict['strides'] = strided_slice_attrs.strides
        attr_dict['slice_mode'] = strided_slice_attrs.slice_mode
        attr_dict['axes'] = strided_slice_attrs.axes

        for k, v in attr_dict.items():
            attr_dict[k] = get_prim_type(v)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        rank = len(input_shape)
        if attr_dict['axes'] is None:
            attr_dict['axes'] = list(range(rank))

        if len(attr_dict['strides']) == 1 and len(attr_dict['axes']) != len(attr_dict['strides']):
            attr_dict['strides'] = list(attr_dict['strides']) * len(attr_dict['axes'])

        for attr in ['begin', 'end', 'strides']:
            if isinstance(attr_dict[attr], int):
                attr_dict[attr] = [attr_dict[attr]] * rank
            elif len(attr_dict[attr]) == 1:
                attr_dict[attr] = list(attr_dict[attr]) * rank

        op_name = converter_context.get_op_name(relay_expr, StridedSliceOp.TRANSLATION_KEY,
                                                StridedSliceOp.LEGACY_TRANSLATION_KEY)

        slice_mode = attr_dict['slice_mode']
        if slice_mode == 'size':
            raise ValueError("Unsupported slice mode {} in StridedSliceOp".format(slice_mode))

        begin = [0] * rank
        end = input_shape.copy()
        strides = [1] * rank

        axes = attr_dict['axes']
        for i, axis in enumerate(axes):
            strides[axis] = attr_dict['strides'][i]
            begin[axis] = attr_dict['begin'][i]
            end[axis] = attr_dict['end'][i]

        ranges = list(map(list, zip(begin, end, strides)))
        ir_op = StridedSliceOp(op_name, ranges=ranges)

        log_debug3("\tbegin {}", ir_op.ranges[:,0])
        log_debug3("\tend {}", ir_op.ranges[:,1])
        log_debug3("\tstrides {}", ir_op.ranges[:,2])

        return ir_op


RelayTranslations.register_translation(RelayStridedSliceTranslation(),
                                       converter_type('strided_slice', 'relay'))


# ------------------------------------------------------------------------------
#   Take
# ------------------------------------------------------------------------------
class RelayTakeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTakeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        take_attrs = relay_expr.attrs
        attr_dict['axis'] = take_attrs.axis

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def add_op(self,
               relay_expr: relay.expr.Call,
               quir_graph: IROpGraph,
               **kwargs):
        converter_context = kwargs.get("converter_context")
        relay_params = kwargs.get("relay_params")

        attr_dict = self.extract_attributes(relay_expr, relay_params)

        input_names = self.extract_input_names(relay_expr,
                                               converter_context=converter_context)

        ir_ops = self.translate_op(relay_expr,
                                   relay_params,
                                   converter_context,
                                   quir_graph,
                                   attr_dict,
                                   input_names)

        # TODO: deprecate it after 0d tensor is fully supported
        # When handling 0d indices case
        # Post reshape will be inserted to make sure the shape consistency
        if len(ir_ops) == 2:
            # add gather op
            gather_op = ir_ops[0]
            num_outputs = gather_op.num_outputs
            output_names = self.extract_output_names(relay_expr,
                                                     converter_context=converter_context,
                                                     num_outputs=num_outputs)
            gather_output_names = [output_names[0] + '_pre_reshape']
            log_debug1("Op {} Type {} inputs {}", gather_op.name, gather_op.type, input_names)
            log_debug1("Op {} Type {} outputs {}", gather_op.name, gather_op.type, gather_output_names)
            self.populate_quantization_params(relay_expr, converter_context, quir_graph, gather_output_names, is_param=False)
            gather_node = converter_context.add_op_to_graph(relay_expr, gather_op, input_names, gather_output_names)
            quir_graph.add_src_op_info(gather_node.op.name, input_names, gather_output_names)

            # add post reshape op
            reshape_op = ir_ops[1]
            num_outputs = reshape_op.num_outputs
            # gather's output is reshape's input
            reshape_input_names = gather_output_names
            reshape_output_names = output_names[:num_outputs]
            log_debug1("Op {} Type {} inputs {}", reshape_op.name, reshape_op.type, reshape_input_names)
            log_debug1("Op {} Type {} outputs {}", reshape_op.name, reshape_op.type, reshape_output_names)
            reshape_node = converter_context.add_op_to_graph(relay_expr, reshape_op, reshape_input_names, reshape_output_names)
            quir_graph.add_src_op_info(reshape_node.op.name, reshape_input_names, reshape_output_names)
            return reshape_node

        ir_op = ir_ops[0]
        num_outputs = ir_op.num_outputs
        output_names = self.extract_output_names(relay_expr,
                                                 converter_context=converter_context,
                                                 num_outputs=num_outputs)

        log_debug1("Op {} Type {} inputs {}", ir_op.name, ir_op.type, input_names)
        log_debug1("Op {} Type {} outputs {}", ir_op.name, ir_op.type, output_names[:num_outputs])

        self.populate_quantization_params(relay_expr, converter_context, quir_graph, output_names[:num_outputs], is_param=False)
        ir_node = converter_context.add_op_to_graph(relay_expr, ir_op, input_names, output_names[:num_outputs])
        quir_graph.add_src_op_info(ir_node.op.name, input_names, output_names[:num_outputs])
        return ir_node

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GatherOp.TRANSLATION_KEY, GatherOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict['axis']

        if axis is None:
            axis = 0

            reshape_op_name = input_names[0] + '_flatten'
            reshape_op = ReshapeOp(name=reshape_op_name, shape=[-1])
            reshape_node = quir_graph.add(reshape_op, input_names=[input_names[0]], output_names=[reshape_op_name])
            input_names[0] = reshape_node.output_names[0]

        if not quir_graph.has_buffer(input_names[0]) and input_names[0] in relay_params:
            data = relay_params[input_names[0]]
            if isinstance(data, tvm.runtime.ndarray.NDArray) or isinstance(data, tvm.runtime.NDArray):
                data = data.asnumpy()
            data_output = input_names[0]
            self.populate_quantization_params(relay_expr.args[0], converter_context, quir_graph, [data_output], is_param=True)
            quir_graph.add(ConstantOp(data_output, data), [], [data_output])

        is_0d_indices = False
        if not quir_graph.has_buffer(input_names[1]) and input_names[1] in relay_params:
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()

            # TODO: deprecate it after 0d tensor is fully supported
            # Handling 0d indices tensor case
            if indices.ndim == 0:
                indices = np.atleast_1d(indices)
                is_0d_indices = True
            indices_output = input_names[1]
            # we don't need to populate quantization params for input[1] since it is indices and its dtype is int32
            quir_graph.add(ConstantOp(indices_output, indices), [], [indices_output])
        elif quir_graph.has_buffer(input_names[1]) and input_names[1] in relay_params:
            # handle shared tensor for indices
            src_rank = relay_params[input_names[1]].asnumpy().ndim
            cur_rank = len(quir_graph.get_buffer(input_names[1]).shape)
            if src_rank == 0 and cur_rank == 1:
                # other node has update the indices from 0d to 1d
                # need to insert post reshape to ensure shape consistency
                is_0d_indices = True
            elif src_rank == 0 and cur_rank != 1:
                # other node do the invalid process to the indices scalar
                # raise error for this case
                raise ValueError("Other node transform scalar tensor from 0d to {}d, which is invalid.".format(cur_rank))

        ir_ops = []
        input_rank = len(quir_graph.get_buffer(input_names[0]).shape)
        axis = input_rank + int(axis) if axis < 0 else int(axis)
        ir_ops.append(GatherOp(op_name, axis=axis))

        # TODO: deprecate it after 0d tensor is fully supported
        # Handling 0d indices tensor case
        if is_0d_indices:
            input_shape = converter_context.get_input_shapes(relay_expr)[0]
            # infer output shape for source take op
            output_shape = input_shape[:axis] + input_shape[axis+1:]
            # we have transformed 0d into 1d indices, so it's needed to insert post_reshape to squeeze at axis
            post_reshape_op_name = 'Reshape_post_' + op_name
            ir_ops.append(ReshapeOp(post_reshape_op_name, shape=output_shape))

        return ir_ops


RelayTranslations.register_translation(RelayTakeTranslation(),
                                       converter_type('take', 'relay'))


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class RelayTileTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTileTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        tile_attrs = relay_expr.attrs
        attr_dict['multiples'] = [int(m) for m in tile_attrs.reps]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TileOp.TRANSLATION_KEY, TileOp.LEGACY_TRANSLATION_KEY)

        multiples =  attr_dict['multiples']

        log_assert(all([m > 0 for m in multiples]), "Multiples {} shall be all postive value", multiples)

        ir_op = TileOp(op_name, multiples=multiples)
        return ir_op


RelayTranslations.register_translation(RelayTileTranslation(),
                                       converter_type('tile', 'relay'))


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class RelayTransposeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTransposeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        transpose_attr = relay_expr.attrs
        axes = transpose_attr.axes if hasattr(transpose_attr, 'axes') else None
        attr_dict['axes'] = axes

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TransposeOp.TRANSLATION_KEY,
                                                TransposeOp.LEGACY_TRANSLATION_KEY)

        if attr_dict['axes'] is None:
            # reverse order if not specified
            input_shape = converter_context.get_input_shapes(relay_expr)[0]
            input_dimensions = len(input_shape)
            axes = [i for i in reversed(range(input_dimensions))]
        else:
            axes = [int(i) for i in attr_dict['axes']]

        log_debug3("\taxes {}", axes)

        return TransposeOp(op_name, axes)


RelayTranslations.register_translation(RelayTransposeTranslation(),
                                       converter_type('transpose', 'relay'))


# ------------------------------------------------------------------------------
#   Zeros
# ------------------------------------------------------------------------------
class RelayZerosTranslation(RelayFullTranslation):
    def __init__(self):
        super(RelayZerosTranslation, self).__init__()
        self.value = 0


RelayTranslations.register_translation(RelayZerosTranslation(),
                                       converter_type('zeros', 'relay'))
