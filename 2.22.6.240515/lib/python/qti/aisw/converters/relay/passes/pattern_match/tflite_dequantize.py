# ==============================================================================
#
#  Copyright (c) 2022-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import tvm
from tvm.relay.dataflow_pattern import *
from tvm.relay.frontend.common import set_span
from qti.aisw.converters.relay.utils import get_key_from_expr

DataTypeBitwidths = {
    'float16': 16,
    'float32': 32,
    'int8': 8,
    'int16': 16,
    'int32': 32,
    'uint8': 8,
    'uint16': 16,
    'uint32': 32,
    None: 8
}

def force_set_span(expr, span):
    if isinstance(expr, tvm.relay.Call):
        return tvm.relay.expr.CallWithFields(
            expr, expr.op, expr.args, expr.attrs, expr.type_args, None, span
        )
    elif isinstance(expr, tvm.relay.Var):
        return tvm.relay.expr.VarWithFields(expr, expr.vid, expr.type_annotation, None, span)
    elif isinstance(expr, tvm.relay.TupleGetItem):
        return tvm.relay.expr.TupleGetItemWithFields(
            expr, expr.tuple_value, expr.index, None, span
        )
    elif isinstance(expr, tvm.relay.Constant):
        return tvm.relay.expr.ConstantWithFields(expr, expr.data, None, span)
    elif isinstance(expr, tvm.relay.Tuple):
        return tvm.relay.expr.TupleWithFields(expr, expr.fields, None, span)
    elif isinstance(expr, tvm.relay.TupleWrapper):
        return tvm.relay.expr.TupleWrapper(force_set_span(expr.tuple_value, span), expr.size)

def dequantize(data, scale, zp):
    return (data.astype(scale.dtype)-zp.astype(scale.dtype))*scale


class DequantizeQnnPattern(tvm.relay.ExprMutator):

    def __init__(self, dtype_dict, span_to_encodings):
        super().__init__()
        self.dtype_dict = dtype_dict
        self.pattern = self.get_pattern()
        self.span_to_encodings = span_to_encodings

    def get_pattern(self):
        """all dequantize pattern need to implement this function"""
        raise NotImplementedError()

    def dequantize_qnn_expr(self, expr, args):
        """all dequantize pattern need to implement this function"""
        raise NotImplementedError()

    def visit_call(self, call):
        # if match quantized pattern, dequantize them
        # else recursive visit the args and create a new call since their args may change in dequantize pass
        if self.pattern.match(call):
            args = [self.visit(arg) for arg in call.args]
            # after vistiing, the args may be updated, so we need to pass new args in this way
            new_call = self.dequantize_qnn_expr(call, args)
        else:
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            new_call = tvm.relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        return new_call

    def visit_function(self, fn):
        # enter point
        new_body = self.visit(fn.body)

        # we need to get free_vars after visit body since new relay.Var may be added
        new_params = tvm.relay.analysis.free_vars(new_body)

        return tvm.relay.Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)

    def visit_var(self, var):
        if self.pattern.match(var):
            new_var = self.dequantize_qnn_expr(var, None)
        else:
            new_var = var
        return new_var

    def populate_quantization_info(self, new_expr, offsets, scales, dtype=None):
        """
        this function should be overrided for those op having multiple output, e.g. relay.split
        since they need to populate each encoding for each output
        """

        # need to broadcast offset for per channel op
        offsets = np.broadcast_to(offsets, scales.shape)

        # get bitwidth from dtype
        bw = DataTypeBitwidths[dtype]

        # tflite: scale*(q-offset)
        # QNN:    scale*(q+offset)
        # offset need to be negated here
        if scales.size > 1:
            # per channel quantization
            q_info = []
            for offset, scale in zip(offsets, scales):
                q_info.append({
                    "bitwidth": bw,
                    "offset": -offset,
                    "scale": scale,
                    "is_symmetric": "True", # symmetric is True for per channel quantization
                })
        else:
            # case1: tensor is symmetric with signed int, offset must be zero
            # case2: tensor is not symmetric with unsigned int dtype
            # case3: tensor is not symmetric with signed int dtype
            is_symmetric = dtype and dtype.startswith('int') and np.allclose(offsets, 0)
            if dtype and dtype.startswith('int') and not is_symmetric:
                # activation is quantized to unsigned integer in SNPE/QNN,
                # so we need to shift offsets for signed integer
                offsets = offsets + 2**(bw-1)
            q_info = {
                "bitwidth": bw,
                "offset": -offsets,
                "scale": scales,
                "is_symmetric": str(is_symmetric),
            }
        # the length of encodings should align length of output_names here
        # so for op with multiple output, they should override this function to create q_info
        # for each output
        # e.g,
        # span_to_encodings.encodings = [[encodings1], [encodings2], ...]
        # span_to_encodings.output_names = [output_name1, output_name2, ...]
        self.span_to_encodings[new_expr.span] = [q_info]

    def dequantize_constant_expr(self, constant_expr, constant_scale, constant_zero_point):
        if isinstance(constant_expr, tvm.relay.Constant):
            constant_array = constant_expr.data.asnumpy()
            dequantized_constant_array = dequantize(constant_array, constant_scale, constant_zero_point)
            new_constant_expr = tvm.relay.const(dequantized_constant_array)
            new_constant_expr = set_span(new_constant_expr, constant_expr.span)
            self.populate_quantization_info(new_constant_expr, constant_zero_point, constant_scale,
                                            dtype=constant_expr.data.dtype)
        else:
            new_constant_expr = constant_expr
        return new_constant_expr

    def dequantize_output_expr(self, output_expr, output_span, output_scale, output_zero_point, output_dtype):
        # after converting to tflite model, some activations are squashed(e.g., relu), the following clip try to recover origin range for floating model
        fmin = (float(tvm.tir.op.min_value(output_dtype).value) - output_zero_point) * output_scale
        fmax = (float(tvm.tir.op.max_value(output_dtype).value) - output_zero_point) * output_scale
        new_output_expr = tvm.relay.clip(output_expr, fmin, fmax)

        # populate quantization info for new clip
        new_output_expr = set_span(new_output_expr, output_span)
        self.populate_quantization_info(new_output_expr, output_zero_point, output_scale, output_dtype)
        return new_output_expr

class DequantizeQnnElementwiseBinaryPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L1405
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L1426
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L1433

    this pass will transform
    IRModule(
        %1 = qnn.add | qnn.mul | qnn.subtract
        %2 = clip (optional)
    )
    to
    IRModule(
        %1 = add | mul | subtract
        %2 = clip
    )
    """
    def get_pattern(self):
        self._eltwise = is_op('qnn.add')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()) | \
                        is_op('qnn.mul')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()) | \
                        is_op('qnn.subtract')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        self._clip = is_op('clip')(self._eltwise)
        return self._eltwise

    def visit_call(self, call):
        if self._clip.match(call):
            args = [self.visit(arg) for arg in call.args]
            return args[0]
        else:
            return super().visit_call(call)

    def dequantize_qnn_expr(self, eltwise_expr, args):
        # Input expressions
        lhs_expr = args[0]
        rhs_expr = args[1]
        output_scale = args[6].data.asnumpy()
        output_zero_point = args[7].data.asnumpy()

        # Extract quantized attributes
        lhs_scale = args[2].data.asnumpy()
        lhs_zero_point = args[3].data.asnumpy()
        rhs_scale = args[4].data.asnumpy()
        rhs_zero_point = args[5].data.asnumpy()

        # Dequantize input expressions if they are Constant
        new_lhs_expr = self.dequantize_constant_expr(lhs_expr, lhs_scale, lhs_zero_point)
        new_rhs_expr = self.dequantize_constant_expr(rhs_expr, rhs_scale, rhs_zero_point)

        if eltwise_expr.op.name == 'qnn.add':
            new_eltwise_expr = tvm.relay.add(new_lhs_expr, new_rhs_expr)
        elif eltwise_expr.op.name == 'qnn.mul':
            new_eltwise_expr = tvm.relay.multiply(new_lhs_expr, new_rhs_expr)
        elif eltwise_expr.op.name == 'qnn.subtract':
            new_eltwise_expr = tvm.relay.subtract(new_lhs_expr, new_rhs_expr)
        else:
            raise ValueError("{} op is not supported in dequantize pass".format(eltwise_expr.op.name))

        # Search data type from source nodes
        src_expr = lhs_expr
        while not (hasattr(src_expr, 'op') and src_expr.op.name in ['qnn.requantize', 'qnn.quantize']) \
              and not isinstance(src_expr, tvm.relay.Constant) and not isinstance(src_expr, tvm.relay.Var):
            src_expr = src_expr.args[0]

        if hasattr(src_expr, 'op') and src_expr.op.name in ['qnn.requantize', 'qnn.quantize']:
            out_dtype = src_expr.attrs['out_dtype']
        elif isinstance(src_expr, tvm.relay.Constant):
            out_dtype = src_expr.data.dtype
        elif isinstance(src_expr, tvm.relay.Var):
            out_dtype = src_expr.type_annotation.dtype
        else:
            raise ValueError("Data type is not avaliable in source expression")

        new_output_expr = self.dequantize_output_expr(new_eltwise_expr, eltwise_expr.span, output_scale, output_zero_point, out_dtype)

        return new_output_expr


class DequantizeQnnAvgPool2dPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L2585

    this pass will transform
    IRModule(
        %1 = cast
        %2 = nn.avg_pool2d
        %3 = cast
    )
    to
    IRModule(
        %1 = nn.avg_pool2d
    )
    """
    def get_pattern(self):
        _cast = is_op('cast')(wildcard()).has_attr({'dtype': 'int32'})
        _avg_pool2d = is_op('nn.avg_pool2d')(_cast)
        _cast = is_op('cast')(_avg_pool2d)
        return _cast

    def dequantize_qnn_expr(self, cast_expr, args):
        avg_pool2d_expr = args[0]
        new_avg_pool2d_expr = tvm.relay.nn.avg_pool2d(avg_pool2d_expr.args[0].args[0], **avg_pool2d_expr.attrs)
        new_avg_pool2d_expr = set_span(new_avg_pool2d_expr, cast_expr.span)
        return new_avg_pool2d_expr


class DequantizeQnnConcatPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L1158

    this pass will transform
    IRModule(
        %1 = qnn.concatenate
    )
    to
    IRModule(
        %1 = concatenate
    )
    """
    def get_pattern(self):
        _concatenate = is_op('qnn.concatenate')(wildcard(), wildcard(), wildcard(), wildcard(), wildcard())
        return _concatenate

    def dequantize_qnn_expr(self, concatenate_expr, args):
        # rewrite qnn.concatenate to concatenate
        # args[0] is relay expr to concatenate
        # refer https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/qnn/op/qnn.py#L326
        input_exprs = args[0]
        input_scales = args[1]
        input_zero_points  = args[2]
        tuple_value = [self.dequantize_constant_expr(expr, scale.data.asnumpy(), zero_point.data.asnumpy()) \
                       for (expr, scale, zero_point) in zip(input_exprs, input_scales, input_zero_points)]
        tuple_value = tvm.relay.Tuple(tuple_value)
        new_concatenate_expr = tvm.relay.concatenate(tuple_value, **concatenate_expr.attrs)
        new_concatenate_expr = set_span(new_concatenate_expr, concatenate_expr.span)
        return new_concatenate_expr


class DequantizeQnnConvPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L2110

    this pass will transform
    IRModule(
        %1 = qnn.conv2d
        %2 = nn.bias_add
        %3 = qnn.requantize
        %4 = clip (optional)
    )
    to
    IRModule(
        %1 = nn.conv2d
        %2 = nn.bias_add
        %3 = clip
    )
    """
    def get_pattern(self):
        _conv2d = is_op('qnn.conv2d')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        _bias_add = is_op('nn.bias_add')(_conv2d, is_constant())
        self._requantize = is_op('qnn.requantize')(_bias_add, is_constant(), is_constant(), is_constant(), is_constant())
        self._clip = is_op('clip')(self._requantize)
        return self._requantize

    def visit_call(self, call):
        if self._clip.match(call):
            args = [self.visit(arg) for arg in call.args]
            return args[0]
        else:
            return super().visit_call(call)

    def dequantize_qnn_expr(self, requantize_expr, args):

        # requantize
        bias_add_expr = args[0]
        bias_scale = args[1].data.asnumpy()
        bias_zero_point = args[2].data.asnumpy()
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        # bias add
        conv2d_expr = bias_add_expr.args[0]
        bias_expr = bias_add_expr.args[1]

        # conv
        data_expr = conv2d_expr.args[0]
        kernel_expr = conv2d_expr.args[1]
        kernel_zero_point = conv2d_expr.args[3].data.asnumpy()
        kernel_scale = conv2d_expr.args[5].data.asnumpy()
        input_zero_point = conv2d_expr.args[2].data.asnumpy()
        input_scale = conv2d_expr.args[4].data.asnumpy()

        # Expand dimension for depthwise convolution so they can broadcast later
        if conv2d_expr.attrs['kernel_layout'] == 'HWOI':
            kernel_scale = np.expand_dims(kernel_scale, axis=-1)

        # dequantize and populate quantization info for data, kernel and bias
        new_data_expr = self.dequantize_constant_expr(data_expr, input_scale, input_zero_point)
        new_kernel_expr = self.dequantize_constant_expr(kernel_expr, kernel_scale, kernel_zero_point)
        new_bias_expr = self.dequantize_constant_expr(bias_expr, bias_scale, bias_zero_point)

        # create dequantized relay expr
        conv2d_attrs = dict(conv2d_expr.attrs)
        conv2d_attrs['out_dtype'] = ''
        new_conv2d_expr = tvm.relay.nn.conv2d(new_data_expr, new_kernel_expr, **conv2d_attrs)
        new_bias_add_expr = tvm.relay.nn.bias_add(new_conv2d_expr, new_bias_expr, **bias_add_expr.attrs)

        new_output_expr = self.dequantize_output_expr(new_bias_add_expr, requantize_expr.span, output_scale, output_zero_point, requantize_expr.attrs['out_dtype'])

        return new_output_expr


class DequantizeQnnPadPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L2661
    """
    def get_pattern(self):
        self._add = is_op('qnn.add')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())

        # use pattern from conv pattern
        conv_pattern = DequantizeQnnConvPattern({}, {})
        self._conv_pattern = conv_pattern.get_pattern()
        self._conv_requantize = conv_pattern._requantize
        self._conv_clip = conv_pattern._clip

        _in_expr = self._add | self._conv_pattern | self._conv_clip
        _pad = is_op('nn.pad')(_in_expr, wildcard())
        return _pad

    def dequantize_qnn_expr(self, pad_expr, args):
        in_expr = args[0]
        pad_value = args[1].data.asnumpy()

        if self._add.match(in_expr):
            output_offset = in_expr.args[7].data.asnumpy()
        elif self._conv_requantize.match(in_expr):
            output_offset = in_expr.args[4].data.asnumpy()
        elif self._conv_clip.match(in_expr):
            output_offset = in_expr.args[0].args[4].data.asnumpy()
        else:
            # if there is no pattern matched, should not enter here
            raise ValueError()

        pad_value_expr = tvm.relay.const(pad_value-output_offset)
        return tvm.relay.nn.pad(in_expr, pad_value=pad_value_expr, **pad_expr.attrs)


class DequantizeQnnReluPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L948

    this pass will transform
    IRModule(
        %1 = clip
        %2 = qnn.requantize
    )
    to
    IRModule(
        %1 = clip
    )
    """
    def get_pattern(self):
        _clip = is_op('clip')(wildcard())
        _requantize = is_op('qnn.requantize')(_clip, is_constant(), is_constant(), is_constant(), is_constant())
        return _requantize

    def dequantize_qnn_expr(self, requantize_expr, args):
        clip = args[0]
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        new_output_expr = self.dequantize_output_expr(clip.args[0], requantize_expr.span, output_scale, output_zero_point, requantize_expr.attrs['out_dtype'])

        return new_output_expr


class DequantizeQnnConv2dTransposePattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L3227

    this pass will transform
    IRModule(
        %1 = qnn.conv2d_transpose
        %2 = nn.bias_add
        %3 = qnn.requantize
    )
    to
    IRModule(
        %1 = nn.conv2d_transpose
        %2 = nn.bias_add
        %3 = clip
    )
    """
    def get_pattern(self):
        _conv2d_transpose = is_op('qnn.conv2d_transpose')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        _bias_add = is_op('nn.bias_add')(_conv2d_transpose, is_constant())
        _requantize = is_op('qnn.requantize')(_bias_add, is_constant(), is_constant(), is_constant(), is_constant())

        return _requantize

    def dequantize_qnn_expr(self, requantize_expr, args):
        # requantize
        bias_add_expr = args[0]
        bias_scale = args[1].data.asnumpy()
        bias_zero_point = args[2].data.asnumpy()
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        # bias add
        conv2d_transpose_expr = bias_add_expr.args[0]
        bias_expr = bias_add_expr.args[1]

        # conv
        data_expr = conv2d_transpose_expr.args[0]
        kernel_expr = conv2d_transpose_expr.args[1]
        input_zero_point = conv2d_transpose_expr.args[2].data.asnumpy()
        kernel_zero_point = conv2d_transpose_expr.args[3].data.asnumpy()
        input_scale = conv2d_transpose_expr.args[4].data.asnumpy()
        kernel_scale = conv2d_transpose_expr.args[5].data.asnumpy()

        if conv2d_transpose_expr.attrs['kernel_layout'] == 'OIHW':
            kernel_scale = np.expand_dims(kernel_scale, axis=1)
            kernel_scale = np.expand_dims(kernel_scale, axis=2)

        # dequantize and populate quantization info for data, kernel and bias
        new_data_expr = self.dequantize_constant_expr(data_expr, input_scale, input_zero_point)
        new_kernel_expr = self.dequantize_constant_expr(kernel_expr, kernel_scale, kernel_zero_point)
        new_bias_expr = self.dequantize_constant_expr(bias_expr, bias_scale, bias_zero_point)

        # create dequantized relay expr
        conv2d_transpose_attrs = dict(conv2d_transpose_expr.attrs)
        conv2d_transpose_attrs['out_dtype'] = 'float32'
        new_conv2d_transpose_expr = tvm.relay.op.nn.conv2d_transpose(new_data_expr, new_kernel_expr, **conv2d_transpose_attrs)
        new_bias_add_expr = tvm.relay.nn.bias_add(new_conv2d_transpose_expr, new_bias_expr, **bias_add_expr.attrs)

        new_output_expr = self.dequantize_output_expr(new_bias_add_expr, requantize_expr.span, output_scale, output_zero_point, requantize_expr.attrs['out_dtype'])

        return new_output_expr


class DequantizeQnnDensePattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L1929

    this pass will transform
    IRModule(
        %1 = qnn.dense
        %2 = nn.bias_add (optional)
        %3 = qnn.requantize
        %4 = clip (optional)
    )
    to
    IRModule(
        %1 = nn.dense
        %2 = nn.bias_add
        %3 = clip
    )
    """
    def get_pattern(self):
        _dense = is_op('qnn.dense')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        _bias_add = _dense.optional(lambda x: is_op('nn.bias_add')(x, is_constant()))
        _requantize = is_op('qnn.requantize')(_bias_add, is_constant(), is_constant(), is_constant(), is_constant())
        self._clip = is_op('clip')(_requantize)
        return _requantize

    def visit_call(self, call):
        if self._clip.match(call):
            args = [self.visit(arg) for arg in call.args]
            return args[0]
        else:
            return super().visit_call(call)

    def dequantize_qnn_expr(self, requantize_expr, args):

        if args[0].op.name == 'nn.bias_add':
            # requantize
            bias_add_expr = args[0]
            bias_scale = args[1].data.asnumpy()
            bias_zero_point = args[2].data.asnumpy()

            # bias add
            dense_expr = bias_add_expr.args[0]
            bias_expr = bias_add_expr.args[1]
        else:
            # requantize
            dense_expr = args[0]

        # requantize
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        # dense
        data_expr = dense_expr.args[0]
        weight_expr = dense_expr.args[1]
        weight_zero_point = dense_expr.args[3].data.asnumpy()
        weight_scale = dense_expr.args[5].data.asnumpy()
        input_zero_point = dense_expr.args[2].data.asnumpy()
        input_scale = dense_expr.args[4].data.asnumpy()

        # dequantize and populate quantization info for data and kernel
        new_data_expr = self.dequantize_constant_expr(data_expr, input_scale, input_zero_point)
        new_weight_expr = self.dequantize_constant_expr(weight_expr, weight_scale, weight_zero_point)

        # create dequantized relay expr
        dense_attrs = dict(dense_expr.attrs)
        dense_attrs['out_dtype'] = ''
        new_dense_expr = tvm.relay.nn.dense(new_data_expr, new_weight_expr, **dense_attrs)
        if args[0].op.name == 'nn.bias_add':
            # dequantize, populate quantization info, and create dequantized relay expr for bias
            new_bias_expr = self.dequantize_constant_expr(bias_expr, bias_scale, bias_zero_point)
            new_dense_expr = tvm.relay.nn.bias_add(new_dense_expr, new_bias_expr, **bias_add_expr.attrs)

        new_output_expr = self.dequantize_output_expr(new_dense_expr, requantize_expr.span, output_scale, output_zero_point, requantize_expr.attrs['out_dtype'])

        return new_output_expr


class DequantizeQnnDequantizePattern(DequantizeQnnPattern):
    """
    remove rest of dequantize op

    this pass will transform
    IRModule(
        %1 = qnn.dequantize
    )
    to
    IRModule(
    )
    """
    def get_pattern(self):
        self._dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        return self._dequantize

    def dequantize_qnn_expr(self, dequantize_expr, args):
        input_scale = args[1].data.asnumpy()
        input_zero_point = args[2].data.asnumpy()
        # dtype info not available in expr
        span = tvm.relay.SequentialSpan([args[0].span, dequantize_expr.span])
        new_arg = force_set_span(args[0], span)
        self.populate_quantization_info(new_arg, input_zero_point, input_scale)
        return new_arg


class DequantizeQnnResizePattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L742

    this pass will transform
    IRModule(
        %1 = qnn.dequantize
        %2 = image.reisze
        %3 = qnn.quantize
    )
    to
    IRModule(
        %1 = image.reisze
    )
    """
    def get_pattern(self):
        _dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        _resize = is_op('image.resize2d')(_dequantize)
        _quantize = is_op('qnn.quantize')(_resize, is_constant(), is_constant())
        return _quantize

    def dequantize_qnn_expr(self, quantize_expr, args):
        resize = args[0]
        dequantize = resize.args[0]
        new_resize = tvm.relay.image.resize2d(dequantize.args[0], **resize.attrs)
        new_resize = set_span(new_resize, quantize_expr.span)
        return new_resize


class DequantizeQnnSigmoidPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L848

    this pass will transform
    IRModule(
        %1 = qnn.dequantize
        %2 = sigmoid
        %3 = qnn.quantize
    )
    to
    IRModule(
        %1 = sigmoid
    )
    """
    def get_pattern(self):
        _dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        _sigmoid = is_op('sigmoid')(_dequantize)
        _quantize = is_op('qnn.quantize')(_sigmoid, is_constant(), is_constant())
        return _quantize

    def dequantize_qnn_expr(self, quantize_expr, args):
        sigmoid = args[0]
        dequantize = sigmoid.args[0]
        new_sigmoid = tvm.relay.sigmoid(dequantize.args[0])
        new_sigmoid = set_span(new_sigmoid, quantize_expr.span)
        output_scale = args[1].data.asnumpy()
        output_zero_point = args[2].data.asnumpy()
        output_dtype = quantize_expr.attrs['out_dtype']
        self.populate_quantization_info(new_sigmoid, output_zero_point, output_scale, output_dtype)
        return new_sigmoid


class DequantizeQnnSoftmaxPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/68ce1e871cbcd90789f462524ec0943bfee2ff0b/python/tvm/relay/frontend/tflite.py#L874

    this pass will transform
    IRModule(
        %1 = qnn.dequantize
        %2 = nn.softmax
        %3 = qnn.quantize
    )
    to
    IRModule(
        %1 = nn.softmax
    )
    """
    def get_pattern(self):
        _dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        _sigmoid = is_op('nn.softmax')(_dequantize)
        _quantize = is_op('qnn.quantize')(_sigmoid, is_constant(), is_constant())
        return _quantize

    def dequantize_qnn_expr(self, quantize_expr, args):
        softmax = args[0]
        dequantize = softmax.args[0]
        new_softmax = tvm.relay.nn.softmax(dequantize.args[0])
        new_softmax = set_span(new_softmax, quantize_expr.span)
        return new_softmax


class DequantizeQnnQuantizePattern(DequantizeQnnPattern):
    """
    remove rest of quantize/requantize op

    this pass will transform
    IRModule(
        %1 = qnn.quantize
    )
    or
    IRModule(
        %1 = qnn.requantize
    )
    to
    IRModule(
    )
    """
    def get_pattern(self):
        self._quantize = is_op('qnn.quantize')(wildcard(), is_constant(), is_constant())
        self._requantize = is_op('qnn.requantize')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant())
        return self._quantize | self._requantize

    def dequantize_qnn_expr(self, expr, args):
        if len(args) == 3:
            # Quantize
            # args[1]: output_scale
            # args[2]: output_offset
            output_scale = args[1].data.asnumpy()
            output_zero_point = args[2].data.asnumpy()
        else:
            # Requantize
            # args[1]: input_scale
            # args[2]: input_offset
            # args[3]: output_scale
            # args[4]: output_offset
            output_scale = args[3].data.asnumpy()
            output_zero_point = args[4].data.asnumpy()
        span = tvm.relay.SequentialSpan([args[0].span, expr.span])
        new_arg = force_set_span(args[0], span)
        self.populate_quantization_info(new_arg, output_zero_point, output_scale, dtype=expr.attrs['out_dtype'])
        return new_arg


class DequantizeQnnVarPattern(DequantizeQnnPattern):
    """
    dequantize var expr

    this pass will transform
    IRModule(
        %1 = Var(int8 | uint8)
    )
    to
    IRModule(
        %1 = Var(float32)
    )
    """
    def visit_call(self, call):
        # Matching is_var() in call expr is very slow and never matched
        # Override the visit_call function to avoid this case
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_call = tvm.relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        return new_call

    def get_pattern(self):
        _var = is_var().has_dtype('int8') | is_var().has_dtype('uint8')
        return _var

    def dequantize_qnn_expr(self, var_expr, args):
        output_name = var_expr.span.output_names[0]
        output_scale = var_expr.span.output_qparams[output_name].scale
        output_zero_point = var_expr.span.output_qparams[output_name].zero_point
        self.populate_quantization_info(var_expr, output_zero_point, output_scale, dtype=var_expr.type_annotation.dtype)
        self.dtype_dict[var_expr.name_hint] = 'float32'
        new_var_expr = tvm.relay.var(var_expr.name_hint, shape=var_expr.type_annotation.shape, dtype='float32')
        new_var_expr = set_span(new_var_expr, var_expr.span)
        return new_var_expr


@tvm.ir.transform.module_pass(opt_level=3)
class DequantizePass:

    def __init__(self, dtype_dict, span_to_encodings):
        self.dtype_dict = dtype_dict
        # TODO:
        # current workflow dequantizing expr and extracting encoding in same time, which is difficult to maintain
        # should redesign workflow to decouple these two
        self.span_to_encodings = span_to_encodings

    # This function can define a pass.
    def transform_module(self, mod, ctx):
        mod.update_func(mod.get_global_var("main"), DequantizeQnnVarPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        # Pad/ElementwiseBianry pattern must before others to grab output offset/out_dtype
        mod.update_func(mod.get_global_var("main"), DequantizeQnnPadPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnElementwiseBinaryPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))

        mod.update_func(mod.get_global_var("main"), DequantizeQnnAvgPool2dPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnConcatPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnConvPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnConv2dTransposePattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnDensePattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnReluPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnResizePattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnSigmoidPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnSoftmaxPattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))

        # remove rest of quantize/requantize/dequantize op since rest of relay should be in floating format
        # e.g.:
        # input(float32) -> quantize(float32 to uint8) -> some_layers (uint8) -> dequantize(uint8 to float32) -> output (float32)
        # after dequantize transform pass,
        # input(float32) -> quantize(float32 to uint8) -> some layer (float32) -> dequantize(uint8 to float32) -> output (float32), we need to remove the quantize and dequantize
        mod.update_func(mod.get_global_var("main"), DequantizeQnnQuantizePattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnDequantizePattern(self.dtype_dict, self.span_to_encodings).visit(mod['main']))
        return mod
