# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from .onnx_translations import *
from .util import *

from qti.aisw.converters.common import ir_graph


# ------------------------------------------------------------------------------
#   Cast
# ------------------------------------------------------------------------------
class OnnxCastTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Cast', [1, 6, 9, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_warning(code_to_message.get_warning_message("WARNING_CAST_TYPE")(str(src_op.name)))
        input_names = list(map(str, src_op.input))
        params = extract_attributes(src_op,
                                    attr_infos=[('to', 'i', 0)],
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)

        cast_dtype = onnx_to_np_dtype.get(params.to).name
        from_type = converter_context.tensor_to_np_dtype.get(input_names[0])
        if from_type:
            from_type = from_type.name
        # Raise error when cast type is not in list of supported types
        if cast_dtype is None:
            raise ValueError(code_to_message.get_error_message('ERROR_CAST_TYPE_UNSUPPORTED')
                             (str(src_op.name), cast_dtype.name))
        const_op = self.fetch_constant_op(input_names[0], converter_context, dtype=cast_dtype, prunable=False,
                                          fail_if_dynamic=False, fail_if_not_found=True)
        if const_op:
            log_debug1("Node {} with static input(s) is resolved as Constant Op and interpreted during conversion".format(str(src_op.name)))
            const_op.tensor = np.asarray(const_op.tensor, dtype=cast_dtype)
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in input_names])
            converter_context.weights.insert(str(src_op.output[0]), const_op.tensor, was_scalar=was_scalar)
            return None
        # if the input of cast op is scalar, output will also be a scalar
        # adding the output tensor name to the scalar tensors so that the
        # output shapes for the next node can be computed accordingly
        if str(src_op.input[0]) in converter_context.scalar_tensor:
            converter_context.scalar_tensor.add(str(src_op.output[0]))

        if not from_type:
            return op_adapter.CastOp(str(src_op.name), to_type=cast_dtype)
        return op_adapter.CastOp(str(src_op.name), from_type=from_type, to_type=cast_dtype)

    def extract_input_names(self, src_op, converter_context):
        if not converter_context.ir_graph.has_buffer(str(src_op.input[0])) and converter_context.weights.has(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxCastTranslation(),
                                      converter_type('Cast', 'onnx'))


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class OnnxChannelShuffleTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, converter_context):
        # Note: Schema is not used here since this is not a valid Onnx Op
        params = extract_attributes(src_op,
                                    ('groups', 'i'))
        return op_adapter.ChannelShuffleOp(src_op.name, num_groups=params.groups)


OnnxTranslations.register_translation(OnnxChannelShuffleTranslation(),
                                      converter_type('Channel_Shuffle', 'onnx'),
                                      op_adapter.ChannelShuffleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class OnnxClipTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Clip', [1, 6, 11, 12])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())

        if src_op.input[0] not in graph.buffers and converter_context.weights.has(src_op.input[0]):
            input0_op = op_adapter.ConstantOp(src_op.input[0], tensor=converter_context.weights.fetch(src_op.input[0]))
            graph.add(input0_op, [], [src_op.input[0]])
            graph.add_src_op_info(input0_op.name, [], [src_op.input[0]])

        min_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        min_op = self.fetch_constant_op(min_name, converter_context)
        if min_op is None:
            min_val = params.min if 'min' in params else np.finfo(np.float32).min
        else:
            min_val = min_op.tensor.item(0)

        max_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        max_op = self.fetch_constant_op(max_name, converter_context)
        if max_op is None:
            max_val = params.max if 'max' in params else np.finfo(np.float32).max
        else:
            max_val = max_op.tensor.item(0)

        input_names = list(map(str, src_op.input))
        const_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False,
                                          fail_if_dynamic=False, fail_if_not_found=True)
        if const_op:
            log_debug1("Node {} with static input(s) is resolved as Constant Op and interpreted during conversion".format(str(src_op.name)))
            data = const_op.tensor
            clip_data = np.clip(data, min_val, max_val)
            # remove empty string from input_names, since optional inputs may be empty string in Clip
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in input_names if input_name != ''])
            converter_context.weights.insert(str(src_op.output[0]), clip_data, was_scalar=was_scalar)

            # add constant node for later ops
            # without the output buffer from this node, it will cause error in later op
            if src_op.output[0] not in graph.buffers:
                new_const_op = op_adapter.ConstantOp(src_op.output[0], tensor=clip_data)
                output_names = [src_op.output[i] for i in range(len(src_op.output))]
                graph.add(new_const_op, [], output_names)
                graph.add_src_op_info(new_const_op.name, [], output_names)
            return None

        return op_adapter.ElementwiseNeuronOp(src_op.name,
                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                                              min_value=min_val,
                                              max_value=max_val)

    def extract_input_names(self, src_op, converter_context):
        return [list(src_op.input)[0]]


OnnxTranslations.register_translation(OnnxClipTranslation(), converter_type('Clip', 'onnx'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class OnnxConcatTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Concat', [1, 4, 11, 13])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
            for input_name in input_names:
                if not graph.has_buffer(input_name) and converter_context.weights.has(input_name):
                    const_op = self.fetch_constant_op(input_name, converter_context, prunable=False)
                    shape = const_op.tensor.shape
                    # Some ONNX models (saved from pytorch) have empty tensor (one dimension is 0)
                    # Add checking here to remove empty tensor in concat becasue it is meaningless
                    if 0 in shape:
                        input_names.remove(input_name)
                    else:
                        const_node = graph.add(const_op, [], input_name)
                        graph.add_src_op_info(input_name, None, const_node.output_names[0])
                else:
                    # TODO
                    # Short term solution : use this else branch.
                    # Long term solution : use QNN 0-Dim feature, when it is fully ready need refine here.
                    input_buf = graph.get_buffer(input_name)
                    shape = input_buf.shape
                    # Check if there is 0-dim tensor of Concat Op's input.
                    if 0 in shape:
                        input_names.remove(input_name)

        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            return graph.add(op, [], output_names)

        self.add_src_op_info(op.name, src_op, graph)
        return graph.add(op, input_names, output_names)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())

        # static concatenation used for reshaping shape tensors
        if converter_context.weights.has_all(src_op.input):
            data = [converter_context.weights.fetch(input_name) for input_name in src_op.input]
            concat_data = np.concatenate(data, params.axis)
            converter_context.weights.insert(str(src_op.output[0]), concat_data)
            return None

        # handle single input concats
        if len(src_op.input) == 1:
            if converter_context.weights.has_all(src_op.input):
                converter_context.weights.insert(str(src_op.output[0]), converter_context.weights.fetch(src_op.input[0]))
                return None
            return op_adapter.IdentityOp(src_op.name)

        # handle all constant input to concat
        input_names = list(map(str, src_op.input))
        const_input_ops = []
        for input_name in input_names:
            const_input_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_ops.append(const_input_op)
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            data = []
            for const_input_op in const_input_ops:
                data.append(const_input_op.tensor)
            concat_data = np.concatenate(data, params.axis)
            converter_context.weights.insert(str(src_op.output[0]), concat_data)
            return op_adapter.ConstantOp(str(src_op.output[0]), concat_data)

        return op_adapter.ConcatOp(src_op.name, axis=params.axis)

    def extract_input_names(self, src_op, converter_context):
        # If this was translated to a static op don't return input names
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return list(map(str, src_op.input))

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxConcatTranslation(),
                                      converter_type('Concat', 'onnx'),
                                      op_adapter.ConcatOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Constant
# ------------------------------------------------------------------------------
class OnnxConstantTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Constant', [1, 9, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        # attr_infos contain the attribute used for the output of the constant op.
        # for opset 13, it can be one of the following : value, value_float, value_floats, value_int,
        # value_ints, value_string, value_strings, sparse_tensor.
        # for opset 1 and 9, it can be only value.
        # sparse tensor is not currently supported by the converter.
        if src_op.attribute[0].type == 11:
            raise ValueError("SparseTensor attribute type for op {} of type {} is not supported".
                             format(src_op.name, src_op.op_type))
        attr_infos = [(src_op.attribute[0].name, OnnxAttrProtoUtil.enum_to_strCode[src_op.attribute[0].type], "")]
        params = extract_attributes(src_op, schema=self.op_schema(), attr_infos=attr_infos)
        # ONNX return np "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        was_scalar = False
        value = np.array(params.get(attr_infos[0][0]))
        if not value.shape:
            value = value.reshape(1)
            was_scalar = True

        converter_context.weights.insert(src_op.output[0], value, was_scalar)
        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(src_op.output[0], value)

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxConstantTranslation(),
                                      converter_type('Constant', 'onnx'),
                                      op_adapter.ConstantOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConstantOfShape
# ------------------------------------------------------------------------------
class OnnxConstantOfShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ConstantOfShape', [9])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        def extract_default_attributes(name):
            ret = NamedDict()
            ret[name] = np.array([0]).astype(np.float32)
            return ret
        params = extract_attributes(src_op, schema=self.op_schema(), default_attrs=extract_default_attributes("value"))

        input_names = list(map(str, src_op.input))
        was_scalar = False

        # Only support when input is static
        const_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_not_found=True)

        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        shape = const_op.tensor.astype(np.int32)
        tensor_dtype = downcast_dtype_64bit_to_32bit(src_op.name, params.value.dtype)
        data = np.full(shape, params.value[0], dtype=tensor_dtype)
        if not data.shape:
            data = data.reshape(1)
            was_scalar = True
        converter_context.weights.insert(src_op.output[0], data, was_scalar)
        return op_adapter.ConstantOp(src_op.output[0], data)

    def extract_input_names(self, src_op, converter_context):
        return []


OnnxTranslations.register_translation(OnnxConstantOfShapeTranslation(),
                                      converter_type('ConstantOfShape', 'onnx'))

# ------------------------------------------------------------------------------
#   DequantizeLinear
# ------------------------------------------------------------------------------
class OnnxDequantizeLinearTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('DequantizeLinear', [10, 13])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op, enc = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            input_names = []
        else:
            input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, input_names, output_names)

        graph.add_quantization_params(node.op.name,
                                      output_encodings=enc)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        graph = converter_context.ir_graph
        # Three inputs data, scale(s), and zero point(s)
        inputs = src_op.input
        outputs = src_op.output

        log_assert(len(inputs) >= 2,
                   code_to_message.get_error_message("ERROR_QUANTIZE_INVALID_INPUTS")(len(inputs)))

        # Retrieve the scales
        scale_op = self.fetch_constant_op(inputs[1], converter_context, prunable=False, fail_if_dynamic=False)
        if scale_op is not None:
            scale = np.array(scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No scale provided, only static scale or constant is supported for op: {} of type: {}".format(src_op.name, src_op.op_type))

        # Check if zero point provided, otherwise use default of 0
        zp = np.array([0]).astype(np.uint8)
        if len(inputs) > 2:
            zp_op = self.fetch_constant_op(inputs[2], converter_context, prunable=False, fail_if_dynamic=False)
            if zp_op is not None:
                zp = zp_op.tensor
            else:
                raise ValueError("No zero point provided, only static scale or constant is supported for op: {} of type: {}".format(src_op.name, src_op.op_type))

        log_assert(len(scale) == 1 and len(zp) == 1,
                   "Per-channel quantization currently unsupported, len of scale and zero point must be 1")

        # TODO Finish support of per-channel dequant for get_encoding
        if 'axis' in params:
            axis = params.axis
            if axis < 0:
                axis += len(input.shape)

        output_name = str(outputs[0])
        enc = get_encoding(output_name, scale, zp)

        w_op = self.fetch_constant_op(inputs[0], converter_context, prunable=False, fail_if_dynamic=False)
        if w_op is not None:
            # It's quantized parameters, quantize and store
            w = w_op.tensor
            w = (w.astype(scale.dtype) - zp.astype(scale.dtype)) * scale
            converter_context.weights.insert(output_name, w)
            return op_adapter.ConstantOp(output_name, w), enc

        stripped_enc = {k:enc[k] for k in enc if k != 'name'}
        return op_adapter.DequantizeOp(src_op.name, **stripped_enc), enc

    def extract_input_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxDequantizeLinearTranslation(),
                                      converter_type('DequantizeLinear', 'onnx'),
                                      op_adapter.DequantizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Expand
# ------------------------------------------------------------------------------
class OnnxExpandTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Expand', [8, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        src_input_names = list(map(str, src_op.input))

        input_constant_op = self.fetch_constant_op(src_input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        shape_constant_op = self.fetch_constant_op(src_input_names[1], converter_context, dtype=np.int32, fail_if_not_found=True)
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_input_names[1]))
        if shape_constant_op is None:
            raise ValueError("Expand Op {} only support static shape tensor: {}".format(src_op.name, src_input_names[1]))
        shape = shape_constant_op.tensor.tolist()

        if input_constant_op is not None:
            # The output dtype is stored in a dictionary named tensor_to_np_dtype which is filled by reading elem_type
            # field present in value_info of the ONNX operator. It is possible that value_info is not present for this
            # node in the original ONNX framework graph. Hence, we use dtype from the Constant folded input op.
            out_data = np.array((input_constant_op.tensor) * np.ones(shape), dtype=input_constant_op.tensor.dtype)
            was_scalar = converter_context.weights.was_scalar(src_input_names[0])
            converter_context.weights.insert(str(src_op.output[0]), out_data, was_scalar=was_scalar)
            return None

        return op_adapter.ExpandOp(name=src_op.name, shape=shape)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxExpandTranslation(), converter_type('Expand', 'onnx'))


# ------------------------------------------------------------------------------
#   Initializer
# ------------------------------------------------------------------------------
class OnnxInitializerTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, initializer, converter_context):
        params = extract_initializer_tensor(initializer)

        # ONNX return np "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        if not params.shape:
            params = params.reshape(1)

        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(initializer.name, params)

    def extract_input_names(self, src_op, converter_context):
        return []

    def extract_output_names(self, src_op, converter_context):
        return [src_op.name]

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxInitializerTranslation(),
                                      converter_type('Initializer', 'onnx'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class OnnxFlattenTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Flatten', [1, 9, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = params.axis

        # When input is static, ensure they are preprocessed statically.
        input_name = str(src_op.input[0])
        if converter_context.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            w = converter_context.weights.fetch(input_name)
            pre_axes = w.shape[:axis]
            post_axes = w.shape[axis:]
            output_shape = [product(pre_axes), product(post_axes)]
            w = np.reshape(w, output_shape)
            converter_context.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            return None

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        pre_axes = input_shape[:axis]
        post_axes = input_shape[axis:]
        output_shape = [product(pre_axes), product(post_axes)]

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, shape=output_shape)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxFlattenTranslation(), converter_type('Flatten', 'onnx'))


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class OnnxGatherTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gather', [1, 11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        const_input_op, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # const_input_op should only be 1 (either data or indices) or None.
        # if const_input_op = None => either both data and indices are constant or both are dynamic
        if const_input_op and not graph.has_buffer(const_input_op.name):
            node = graph.add(const_input_op, [], const_input_op.name)
            graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when gather op is represented as a constant op
            last_node = graph.add(translated_ops[0], [], output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when gather is represented as gather or gather + reshape
            if len(translated_ops) == 2:
                gather_output_names = [output_names[0] + '_pre_reshape']
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(list(src_op.output)[0], False)
                    if encoding is not None:
                        graph.set_overridden_encoding(gather_output_names[0], encoding, False)
            else:
                gather_output_names = [output_names[0]]

            last_node = graph.add(translated_ops[0], input_names, gather_output_names)
            graph.add_src_op_info(last_node.op.name, None, gather_output_names[0])

            if len(translated_ops) == 2:
                last_node = graph.add(translated_ops[1], gather_output_names, output_names)
                graph.add_src_op_info(last_node.op.name, None, last_node.output_names[0])

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        if graph.has_buffer(input_data_name):
            input_data_buf = graph.get_buffer(input_data_name)
            input_data_rank = input_data_buf.rank()
        else:
            input_data_buf = converter_context.weights.fetch(input_data_name)
            input_data_rank = len(input_data_buf.shape)
        # axis can be negtive in ONNX, but it must not less than 0 in QNN
        axis = params.axis if params.axis >= 0 else (input_data_rank + params.axis)
        if axis < 0 or axis >= input_data_rank:
            raise ValueError(
                code_to_message.get_error_message("ERROR_GATHER_UNSUPPORTED_LAYER_PARAM")("axis", params.axis))

        input_names = list(map(str, src_op.input))
        const_input_ops = []
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, prunable=False, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)
        const_index_op = self.fetch_constant_op(indices_name, converter_context, dtype=np.int32, prunable=False,
                                                fail_if_dynamic=False)
        if const_index_op is not None:
            const_input_ops.append(const_index_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if indices_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(np.int32)

        const_op = const_input_ops[0] if len(const_input_ops) == 1 else None

        # If both input and indices are static then interpret gather and return const op
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            # TODO: deprecate the if condition after 0d tensor is fully supported
            # Constant op will output 1D tensor even if the output is a scalar in onnx,
            # so we need to retrieve the scalar value from tensor.
            was_scalar = converter_context.weights.was_scalar(indices_name)
            if was_scalar:
                indices = indices.item()
            gather_data = np.take(input_data, indices, axis=axis)
            was_result_scalar = False if gather_data.shape else True
            # reshape the output tensor to 1D if it's a scalar
            if was_result_scalar:
                gather_data = gather_data.reshape(1)
            converter_context.weights.insert(str(src_op.output[0]), gather_data, was_scalar=was_result_scalar)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], gather_data))
            return const_op, translated_ops


        translated_ops.append(op_adapter.GatherOp(src_op.name, axis=axis))

        # TODO: deprecate it after 0d tensor is fully supported
        if (converter_context.weights.has(indices_name) and converter_context.weights.was_scalar(indices_name)) or \
                (indices_name in converter_context.scalar_tensor):
            if const_input_op:
                input_buf_shape = const_input_op.tensor.shape
            else:
                input_buf_shape = graph.get_buffer(src_op.input[0]).shape
            output_shape = input_buf_shape[:axis] + input_buf_shape[axis+1:]
            # if Gather output is scalar do not do a post reshape since there is no support
            # for scalar inputs/outputs in IR or backends
            if len(output_shape):
                reshape_op_name = src_op.name
                if src_op.name:
                    reshape_op_name = 'Reshape_post_' + src_op.name
                translated_ops.append(op_adapter.ReshapeOp(reshape_op_name,
                                                           shape=output_shape))
            else:
                # TODO: deprecate it after 0d tensor is fully supported
                converter_context.scalar_tensor.add(str(src_op.output[0]))

        return const_op, translated_ops


OnnxTranslations.register_translation(OnnxGatherTranslation(),
                                      converter_type('Gather', 'onnx'),
                                      op_adapter.GatherOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GatherElements
# ------------------------------------------------------------------------------
class OnnxGatherElementsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GatherElements', [11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # input_ops size should only be 1 or 2 or 0([]).
        # if input_ops = [] => both data and indices are constant or both are dynamic
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops == []:
            last_node = None
        else:
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def _perform_static_gather_elements(self, input_data: np.ndarray, input_indices: np.ndarray, axis: np.int32):
        # validate the inputs
        input_dim = input_data.ndim
        indices_dim = input_indices.ndim
        if input_dim != indices_dim or input_dim < 1:
            raise ValueError(code_to_message.get_error_message(
                "ERROR_GATHER_ELEMENTS_WRONG_RANK")(str(input_dim), str(indices_dim)))
        # internal helper functions
        def _get_element(tensor_data, index_list_format:list):
            for dim in index_list_format:
                tensor_data = tensor_data[dim]
            return np.copy(tensor_data)
        def _set_element(tensor_data, index_list_format:list, data_to_set):
            while len(index_list_format) != 0:
                target_ind = index_list_format.pop(-1)
                tensor_to_set = _get_element(tensor_data, index_list_format)
                tensor_to_set[target_ind] = data_to_set
                data_to_set = tensor_to_set
            return tensor_to_set
        def _get_all_index_in_list_format(curr_indices, tensor, all_index_in_list_format):
            while np.size(tensor[-1]) == 0:
                tensor.pop[-1]
            while len(tensor) > 0:
                last_data = tensor.pop(-1)
                next_indices = curr_indices.copy()
                next_indices.append(len(tensor))
                if type(last_data) == list:
                    _get_all_index_in_list_format(next_indices, last_data, all_index_in_list_format)
                else:
                    all_index_in_list_format.append(next_indices)
            return all_index_in_list_format
        # output has the same shape with input_indices, hence get the indexes of input_indices for output initialize
        all_index=_get_all_index_in_list_format([], np.copy(input_indices).tolist(), [])
        # init the static_gether_elements to input_indices and renew all data in it
        # reference https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements
        static_gether_elements = np.copy(input_indices).astype(input_data.dtype)
        for ind in all_index:
            data_to_set_index = ind.copy()
            data_to_set_index[axis] = _get_element(input_indices, ind)
            data_to_set = _get_element(input_data, data_to_set_index)
            static_gether_elements = _set_element(static_gether_elements, ind, data_to_set)

        return static_gether_elements

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        axis = params.axis

        input_names = list(map(str, src_op.input))
        const_input_ops = []
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, prunable=False, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)
        const_input_op = self.fetch_constant_op(indices_name, converter_context, dtype=np.int32, prunable=False,
                                                fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        # If both input and indices are static then interpret gather and return const op
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            gather_elements_data = self._perform_static_gather_elements(input_data, indices, axis=axis)
            converter_context.weights.insert(str(src_op.output[0]), gather_elements_data)
            return input_ops, translated_ops

        # If only input is stored as weights then create a corresponding const op
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If only indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.int32)
            input_ops.append(op_adapter.ConstantOp(indices_name, indices, quantizable=False))
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(np.int32)

        translated_ops.append(op_adapter.GatherElementsOp(src_op.name, axis=axis))

        return input_ops, translated_ops


OnnxTranslations.register_translation(OnnxGatherElementsTranslation(),
                                      converter_type('GatherElements', 'onnx'),
                                      op_adapter.GatherElementsOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GatherND
# ------------------------------------------------------------------------------
class OnnxGatherNDTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GatherND', [11, 12, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        # input op should only be 1 or 2 or None.
        # if input_op = None => all inputs are constant or all are dynamic
        # When input ops are None, GatherND is a constant op
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when gather_nd op is represented as a constant op i.e input ops is None
            last_node = graph.add(translated_ops[0], [], output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when gather_nd op has one or more dynamic inputs
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(),
                                    attr_infos=[('batch_dims', 'i', 0)])
        batch_dims = params["batch_dims"]

        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])

        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set, inputs are data and indices tensors
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, converter_context, dtype=np.uint32,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        # If all inputs are static, then perform static Gather and return
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor

            gather_nd_data = self._perform_static_gather_nd(input_data, indices, batch_dims)
            converter_context.weights.insert(str(src_op.output[0]), gather_nd_data)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], gather_nd_data))
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.uint32)
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(np.uint32)

        # fetch attribute param batch_dims's real value and reset the init value of it.
        translated_ops.append(op_adapter.GatherNDOp(src_op.name, batch_dims=batch_dims))

        return input_ops, translated_ops

    def _perform_static_gather_nd(self, input_data: np.ndarray, indices: np.ndarray,
                                   batch_dims: np.uint32):
        if batch_dims < 0:
            raise TypeError("Cannot perform static gather_nd. Expected batch_dims should be 0 or positive integer.")

        data = np.copy(input_data)

        # Note the data rank - will be reused multiple times later
        data_rank = len(data.shape)

        # Check input tensors' shape/rank condition
        assert indices.shape[-1] <= data_rank

        #The list of data/indice shape of batch_dims
        batch_dims_shape = []

        #The number of elements in the batch_dims for data/indice array
        batch_dims_size = 1

        # Check the shape of indice and data are identical for batch dims.
        for i in range(batch_dims):
            batch_dims_shape.append(indices.shape[i])
            batch_dims_size *= indices.shape[i]

        # Compute shape of output array
        if (indices.shape[-1] == data_rank - batch_dims):
            output_shape = batch_dims_shape + list(indices.shape)[batch_dims:-1]
        else:
            output_shape = batch_dims_shape + list(indices.shape)[batch_dims:-1] \
                + list(data.shape)[batch_dims + indices.shape[-1]:]

        # Placeholder for output data
        output_data_buffer = []

        # Flatten 'indices' to 2D array
        reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

        # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
        reshaped_data = data.reshape((batch_dims_size, ) + data.shape[batch_dims:])

        # gather each scalar value from 'data'
        for batch_dim in range(reshaped_indices.shape[0]):
            for outer_dim in range(reshaped_indices.shape[1]):
                gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
                output_data_buffer.append(reshaped_data[(batch_dim,) + gather_index])

        return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)

    def extract_input_names(self, src_op, converter_context):
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return super().extract_input_names(src_op, converter_context)

OnnxTranslations.register_translation(OnnxGatherNDTranslation(),
                                      converter_type('GatherND', 'onnx'),
                                      op_adapter.GatherNDOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   HardSwish
# ------------------------------------------------------------------------------
class OnnxHardSwishTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('HardSwish', [14])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(src_op.name,
                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH,
                                              alpha=1/6,
                                              beta=0.5)

    def extract_input_names(self, src_op, converter_context):
        return [list(src_op.input)[0]]

OnnxTranslations.register_translation(OnnxHardSwishTranslation(), converter_type('HardSwish', 'onnx'))


# ------------------------------------------------------------------------------
#   NonZero
# ------------------------------------------------------------------------------
class OnnxNonZeroTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('NonZero', [9, 13])
        self.input_names = None
        self.output_names = None

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return

        intermediate_names = [name + '_intermediate' for name in self.output_names]
        graph.add_src_op_info(op.name, self.input_names, intermediate_names)
        graph.add(op, self.input_names, intermediate_names)

        transpose_op = op_adapter.TransposeOp(name=src_op.name + '_transpose', perm=[1,0])
        graph.add_src_op_info(transpose_op.name, intermediate_names, self.output_names)
        return graph.add(transpose_op, intermediate_names, self.output_names)

    def extract_parameters(self, src_op, converter_context):
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)

        # handle constant input to NonZero
        input_const_op = self.fetch_constant_op(self.input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        if input_const_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_const_data = input_const_op.tensor
            nonzero_output_data = np.array(np.nonzero(input_const_data))
            converter_context.weights.insert(str(src_op.name), nonzero_output_data)
            return None

        return op_adapter.NonZeroOp(name=src_op.name)


OnnxTranslations.register_translation(OnnxNonZeroTranslation(),
                                      converter_type('NonZero', 'onnx'),
                                      op_adapter.NonZeroOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   OneHot
# ------------------------------------------------------------------------------
class OnnxOneHotTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('OneHot', [9, 11])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if len(ops) == 2:
            onehot_output_name = [output_names[0] + '_pre_reshape']
        else:
            onehot_output_name = [output_names[0]]

        last_node = graph.add(ops[0], input_names, onehot_output_name)
        graph.add_src_op_info(last_node.op.name, input_names[0], onehot_output_name[0])

        if len(ops) == 2:
            last_node = graph.add(ops[1], onehot_output_name, output_names)
            graph.add_src_op_info(last_node.op.name, onehot_output_name[0], last_node.output_names[0])

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(map(str, src_op.input))
        ops = []

        depth_const_op = self.fetch_constant_op(input_names[1], converter_context)
        depth = depth_const_op.tensor[0]
        if depth < 0:
            raise ValueError(code_to_message.get_error_message("ERROR_ONEHOT_NEG_DEPTH")(depth))

        values_const_op = self.fetch_constant_op(input_names[2], converter_context)
        values = values_const_op.tensor

        ops.append(op_adapter.OneHotOp(src_op.name, depth=depth, on_value=values[1], off_value=values[0], axis=params.axis))

        # if indices input was a scalar then reshape one_hot output
        if converter_context.weights.has(input_names[0]) and converter_context.weights.was_scalar(input_names[0]):
            output_shape = [depth]
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = 'Reshape_post_' + src_op.name
            ops.append(op_adapter.ReshapeOp(reshape_op_name,
                                            shape=output_shape))

        return ops

    def extract_input_names(self, src_op, converter_context):
        # Filter depth and values from the input
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxOneHotTranslation(),
                                      converter_type('OneHot', 'onnx'))


# ------------------------------------------------------------------------------
#   Pad
# ------------------------------------------------------------------------------
class OnnxPadTranslation(OnnxTranslationBase):
    class OnnxPadMode:
        CONSTANT = 'constant'
        REFLECT = 'reflect'
        EDGE =  'edge'
    supported_modes = {OnnxPadMode.CONSTANT : ir_graph.QNN_OP_PAD_SCHEME_CONSTANT,
                       OnnxPadMode.REFLECT : ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT,
                       OnnxPadMode.EDGE : ir_graph.QNN_OP_PAD_SCHEME_EDGE}

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pad', [1, 2, 11, 13])\
            .register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        pads_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        const_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        pads = None

        if pads_name:
            pads_op = self.fetch_constant_op(pads_name, converter_context, dtype=np.int32)
            if pads_op is not None:
                pads = pads_op.tensor
        elif 'pads' in params:
            pads = params.pads
        elif 'paddings' in params:
            pads = params.paddings

        if pads is None:
            raise ValueError("Failed to retrieve pads value on {} source op {}".format(src_op.op_type,
                                                                                       src_op.name))

        # Pads/paddings need to be translated from r1_begin, r2_begin...r1_end, r2_end, ...
        # to pairs (r1_begin, r1_end), (r2_begin, r2_end)...
        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = len(input_buf.shape)
        log_assert(rank == len(pads) / 2,
                   "Rank of input tensor: {} must equal (# pads/2): {}"
                   .format(rank, int(len(pads) / 2)))

        pad_pairs = []
        for index in range(rank):
            pad_pairs.append([pads[index], pads[index + rank]])
        pad_pairs = np.asarray(pad_pairs, dtype=np.dtype('int32'))

        constant_value = 0
        if const_name:
            const_op = self.fetch_constant_op(const_name, converter_context, dtype=np.int32)
            if const_op is not None:
                constant_value = const_op.tensor[0]
        elif 'value' in params:
            constant_value = params.value

        return op_adapter.PadOp(src_op.name,
                                scheme=self.supported_modes[params.mode],
                                pad_amount=pad_pairs,
                                pad_constant_value=constant_value)

    def extract_input_names(self, src_op, converter_context):
        # Filter if there are any parameters like 'pads' in inputs
        # For example, 'pads' are already handled in extract_parameters
        return [str(src_op.input[0])]

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in OnnxPadTranslation.supported_modes:
                raise ValueError(code_to_message.get_error_message("ERROR_PAD_UNSUPPORTED_MODE")(src_op_mode))


OnnxTranslations.register_translation(OnnxPadTranslation(),
                                      converter_type('Pad', 'onnx'),
                                      op_adapter.PadOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   QuantizeLinear
# ------------------------------------------------------------------------------
class OnnxQuantizeLinearTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('QuantizeLinear', [10, 13])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op, enc = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            input_names = []
        else:
            input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, input_names, output_names)

        graph.add_quantization_params(node.op.name,
                                      output_encodings=enc)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())

        # Three inputs data, scale(s), and zero point(s)
        inputs = src_op.input
        outputs = src_op.output

        log_assert(len(inputs) >= 2,
                   code_to_message.get_error_message("ERROR_QUANTIZE_INVALID_INPUTS")(len(inputs)))

        # Retrieve the scales
        scale_op = self.fetch_constant_op(inputs[1], converter_context, prunable=False ,fail_if_dynamic=False)
        if scale_op is not None:
            scale = np.array(scale_op.tensor).astype(np.float32)
        else:
            raise ValueError("No scale provided, only static scale or constant is supported for op: {} of type: {}".format(src_op.name, src_op.op_type))

        # Check if zero point provided, otherwise use default of 0
        zp = np.array([0]).astype(np.uint8)
        if len(inputs) > 2:
            zp_op = self.fetch_constant_op(inputs[2], converter_context, prunable=False, fail_if_dynamic=False)
            if zp_op is not None:
                zp = zp_op.tensor
            else:
                raise ValueError("No zero point provided, only static scale or constant is supported for op: {} of type: {}".format(src_op.name, src_op.op_type))

        # TODO Finish support of per-channel quant for get_encoding
        if 'axis' in params:
            axis = params.axis
            if axis < 0:
                axis += len(input.shape)

        output_name = str(outputs[0])
        enc = get_encoding(output_name, scale, zp)

        w_op = self.fetch_constant_op(inputs[0], converter_context, prunable=False, fail_if_dynamic=False)
        if w_op is not None:
            # It's quantized parameters, quantize and store
            w = w_op.tensor
            w = np.clip((np.rint(w/scale) + zp), np.iinfo(zp.dtype).min, np.iinfo(zp.dtype).max)
            converter_context.weights.insert(output_name, w)
            return op_adapter.ConstantOp(output_name, w), enc

        stripped_enc = {k:enc[k] for k in enc if k != 'name'}
        return op_adapter.QuantizeOp(src_op.name, **stripped_enc), enc

    def extract_input_names(self, src_op, converter_context):
        # If this was translated to a static op don't return names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxQuantizeLinearTranslation(),
                                      converter_type('QuantizeLinear', 'onnx'),
                                      op_adapter.QuantizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Range
# ------------------------------------------------------------------------------
class OnnxRangeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Range', [11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))
        const_inputs = []

        # Only support when all inputs are static
        for input_name in input_names:
            const_op = self.fetch_constant_op(input_name, converter_context, prunable=False)
            if const_op is not None:
                const_inputs.append(const_op.tensor)

        log_assert(len(const_inputs) == 3,
                   code_to_message.get_error_message("ERROR_RANGE_INVALID_INPUTS")(len(const_inputs)))

        start = const_inputs[0].item(0)
        limit = const_inputs[1].item(0)
        delta = const_inputs[2].item(0)

        # range type is determined by inputs which are expected to be all of same type
        dtype = downcast_dtype_64bit_to_32bit(src_op.output[0],
                                              const_inputs[0].dtype)

        range_output = np.arange(start, limit, delta,dtype=dtype)
        converter_context.weights.insert(str(src_op.output[0]), range_output)
        return op_adapter.ConstantOp(src_op.output[0], range_output)

    def extract_input_names(self, src_op, converter_context):
        return []


OnnxTranslations.register_translation(OnnxRangeTranslation(),
                                      converter_type('Range', 'onnx'))


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class OnnxReshapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Reshape', [1, 5, 13, 14])

    def extract_parameters(self, src_op, converter_context):
        # There are two main versions of ONNX Reshape
        #    1. The old reshape, where shape is provided as an attribute
        #    2. The new reshape, where the shape is provided as a second input
        #
        # Backends and the converter support two versions of Reshape:
        #    1. Dynamic reshaping with a statically provided output shape
        #    2. Static reshaping, performed at conversion time
        #
        # Backends dont support the 2nd ONNX Reshape explicitly, however in the converter we can
        # pass static shape as suppl. attribute of our IR and still allow the network to be resizable for
        # limited cases. In addition, if a Shape' layer provided the shape it will have been saved
        # as static,
        # eg weight data, in the converter and all ops operating on that data will
        # become static ops and will be pruned during the final conversion.
        graph = converter_context.ir_graph
        shape = []
        params = extract_attributes(src_op, schema=self.op_schema())
        if len(src_op.input) > 1:
            shape_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            shape = self.fetch_constant_op(shape_input, converter_context, fail_if_not_found=True,
                                           dtype=np.int32).tensor.tolist()
        else:
            if 'shape' in params:
                shape = params.shape

        if 'allowzero' in params and params.allowzero == 1 :
            raise ValueError("Unsupported value of {} provided as allowzero attribute for Reshape Op {}. Only 0 (default) is supported.".format(1, src_op.name))

        input_name = str(src_op.input[0])
        const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, shape))

            const_input_tensor = const_input_op.tensor
            if len(shape) != 0:
                # Shape attribute of numpy Reshape and onnx Reshape is different, e.g., shape=[0,0,1,-1] is not supported in numpy reshape
                # Recalculating the shape as per numpy standards to avoid error
                shapeNew = ir_graph.ReshapeOp.calculate_shape(src_op.name,list(const_input_tensor.shape), shape)
                const_input_tensor = np.reshape(const_input_tensor, shapeNew)
                converter_context.weights.insert(output_name, const_input_tensor)
            else:
                # Shape (second input) could be an empty shape, which means reshape to a scalar
                # Set was_scalar = True in this case
                converter_context.weights.insert(output_name, const_input_tensor, was_scalar=True)
            return None
        else:
            # dynamic reshape of activations
            input_buff = graph.get_buffer(input_name)
            log_assert(len(shape) != 0 or (input_buff.rank() == 1 and input_buff.shape[0] == 1), "Node {}: Unable to retrieve reshape shape", src_op.name)
            # Keep the output shape as [1] and set the output buff as a scalar tensor
            if len(shape) == 0:
                shape = input_buff.get_buf_dims()
                converter_context.scalar_tensor.add(str(src_op.output[0]))
            return op_adapter.ReshapeOp(src_op.name, shape=shape)

    def extract_input_names(self, src_op, converter_context):
        input_name = str(src_op.input[0])
        if converter_context.weights.consumed(input_name):
            return []
        else:
            return [input_name]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxReshapeTranslation(),
                                      converter_type('Reshape', 'onnx'),
                                      op_adapter.ReshapeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class OnnxResizeTranslation(OnnxTranslationBase):
    SUPPORTED_RESIZE_MODES = ['nearest', 'linear', 'bilinear', 'cubic']
    SUPPORTED_COORD_TRANSFORM_MODES = ['asymmetric', 'align_corners', 'half_pixel', 'tf_half_pixel_for_nn',
                                       'pytorch_half_pixel']
    SUPPORTED_NEAREST_MODES = ['round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil']

    onnx_to_ir_transformation_mode = {
        "asymmetric": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC,
        "align_corners": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS,
        "half_pixel": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
        "tf_half_pixel_for_nn": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
        "pytorch_half_pixel": ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_PYTORCH_HALF_PIXEL,
    }

    onnx_to_ir_interpolation_mode = {
        "nearest": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
        "linear": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        "bilinear": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        "cubic": ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC,
    }

    onnx_to_ir_nearest_mode = {
        "round_prefer_floor": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR,
        "round_prefer_ceil": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL,
        "floor": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_FLOOR,
        "ceil": ir_graph.QNN_OP_RESIZE_NEAREST_MODE_CEIL,
    }

    default_transformation_mode = 'half_pixel'

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('Resize', [10, 11, 13])
        schema_dict.replace_default_values(mode='nearest')
        schema_dict.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        resize_schema = self.op_schema()
        params = extract_attributes(src_op, attr_infos=[('mode', 's', 'nearest'),
                                                        ('coordinate_transformation_mode', 's', self.default_transformation_mode),
                                                        ('cubic_coeff_a', 'f', -0.75),
                                                        ('exclude_outside', 'i', 0),
                                                        ('nearest_mode', 's', 'round_prefer_floor')],
                                    schema=resize_schema, validate=True)
        transformation_mode = params.coordinate_transformation_mode
        ir_cubic_coeff = params.cubic_coeff_a
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        if input_buf.rank() != 5 and input_buf.rank() != 4:
            raise ValueError(code_to_message.get_error_message("ERROR_RESIZE_INPUT_DIMS")(input_buf.shape))

        ir_exclude_outside = False if params.exclude_outside == 0 else True
        ir_transformation_mode = self.onnx_to_ir_transformation_mode.get(transformation_mode)
        ir_interpolation_mode = self.onnx_to_ir_interpolation_mode.get(params.mode)
        ir_nearest_mode = self.onnx_to_ir_nearest_mode.get(params.nearest_mode)
        if not params.mode == "nearest" and transformation_mode == "tf_half_pixel_for_nn":
            raise ValueError(
                code_to_message.get_error_message("ERROR_RESIZE_INVALID_COORDINATE_TRANSFORMATION_MODE_MIX")
                (params.mode, transformation_mode))

        def get_output_dims_from_scales(scales):
            return [int(round(scale * shape)) for scale, shape in zip(scales, input_shape)]

        def get_scales_from_sizes(sizes):
            # Opset 11 has 4th parameter as output sizes,
            # here we are calculating scales from output sizes
            # per onnx spec, scales is float.
            scales = list(map(float, sizes))
            return [scale / shape for scale, shape in zip(scales, input_shape)]

        def get_static_tensor(tensor_name, type, graph, dtype=np.float32):
            """
            :param tensor_name:
            :param type: String for scales or sizes
            :param graph: IrOpGraph
            :param dtype: datatype in numpy type
            :return: numpy tensor or None. Can raise Error
            """
            tensor = None
            if converter_context.weights.has(tensor_name):
                tensor = converter_context.weights.fetch(tensor_name).astype(dtype)
            elif graph.has_buffer(tensor_name):
                if isinstance(graph.get_buffer(tensor_name).producer.op, op_adapter.ConstantOp):
                    tensor = graph.get_buffer(tensor_name).producer.op.tensor
                else:
                    raise TypeError("Resize Op {}: Dynamic {} input ({}) not supported".format(src_op.name,
                                                                                               type,
                                                                                               tensor_name))
            return tensor

        def is_tensor_empty(tensor):
            return tensor.shape == (0,)

        # Handle sizes input if src_op has 4 inputs and scales input is ''
        if len(src_op.input) == 4:
            sizes_name = str(src_op.input[-1])
            scales_name = str(src_op.input[-2])
            scales_tensor = get_static_tensor(scales_name, "scales", graph)

            if scales_tensor is None or is_tensor_empty(scales_tensor):
                # per onnx spec, size is int64. But int32 may be enough.
                sizes_tensor = get_static_tensor(sizes_name, "sizes", graph, dtype=np.int32)
                if sizes_tensor is None or not sizes_tensor.shape:
                    raise ValueError("Resize Op {}: One of scales ({}) or sizes ({}) needs to be provided".format(
                        src_op.name, scales_name, sizes_name
                    ))
                sizes = sizes_tensor.tolist()
                scales = get_scales_from_sizes(sizes)
            else:
                scales = scales_tensor.tolist()
                if scales[0] != 1.0:
                    raise ValueError("Resize Op does not support resize along Batch Dimension")
                if scales[1] != 1.0:
                    raise ValueError("Resize Op does not support resize along Channel Dimension")

                sizes = get_output_dims_from_scales(scales)
                # Calculate scales again to account for align_corners
                scales = get_scales_from_sizes(sizes)
        elif len(src_op.input) > 1:
            scales_name = str(src_op.input[-1])
            scales_tensor = get_static_tensor(scales_name, "scales", graph)
            if scales_tensor is None or not scales_tensor.shape:
                raise ValueError("Resize Op {}: scales ({}) tensor is invalid".format(
                    src_op.name, scales_name
                ))
            scales = scales_tensor.tolist()
            if scales[0] != 1.0:
                raise ValueError("Resize Op does not support resize along Batch Dimension")
            if scales[1] != 1.0:
                raise ValueError("Resize Op does not support resize along Channel Dimension")

            sizes = get_output_dims_from_scales(scales)
            # Calculate scales again to account for align_corners
            scales = get_scales_from_sizes(sizes)
        else:
            # deprecated. Added for Upsample version 7 and below
            scales = extract_attributes(src_op, attr_infos=[('scales', 'lf')], schema=resize_schema, validate=True).scales

        return op_adapter.ResizeOp(src_op.name,
                                   exclude_outside=ir_exclude_outside,
                                   transformation_mode=ir_transformation_mode,
                                   interpolation_mode=ir_interpolation_mode,
                                   nearest_mode=ir_nearest_mode,
                                   cubic_coeff=ir_cubic_coeff,
                                   scale_depth=scales[-3],
                                   scale_height=scales[-2],
                                   scale_width=scales[-1])

    @classmethod
    def validate_attribute_values(cls, src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_RESIZE_MODES:
                raise ValueError(code_to_message.get_error_message("ERROR_RESIZE_UNSUPPORTED_MODE")
                                 (src_op_mode,  cls.SUPPORTED_RESIZE_MODES))
        elif attr_name == 'scales':
            scales = attr_value
            if scales[0] != 1 or scales[1] != 1:
                log_warning(code_to_message.get_warning_message("WARNING_RESIZE"))
        elif attr_name == 'coordinate_transformation_mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_COORD_TRANSFORM_MODES:
                raise ValueError(
                    code_to_message.get_error_message("ERROR_RESIZE_UNSUPPORTED_COORDINATE_TRANSFORMATION_MODE")
                    (src_op_mode, cls.SUPPORTED_COORD_TRANSFORM_MODES))
        elif attr_name == 'nearest_mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_NEAREST_MODES:
                raise ValueError(
                    "nearest mode {} was not supported. Please choose from modes: {}"
                    .format(src_op_mode, cls.SUPPORTED_NEAREST_MODES))

    def extract_input_names(self, src_op, converter_context):
        if len(src_op.input) > 2:
            return [str(src_op.input[0])]
        else:
            return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def infer_output_shapes(self, op, input_shapes):
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, op.output_shape))
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxResizeTranslation(),
                                      converter_type('Resize', 'onnx'),
                                      op_adapter.ResizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ScatterElements
# ------------------------------------------------------------------------------
class OnnxScatterElementsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScatterElements', [11, 13, 16])
        self.register_op_schema('Scatter', [9, 11])
        self.reduction_types = {"none": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE,
                                "add": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD,
                                "mul": ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL}

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # the number of input op should be 1 or 2, or no input ops
        # if input_op = None => inputs are all constant or all dynamic
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when scatter_elements op is represented as a constant op, i.e input ops is None
            last_node = graph.add(translated_ops[-1], [], output_names)
        else:
            # when scatter_elements op has one or more dynamic inputs
            last_node = graph.add(translated_ops[-1], input_names, output_names)
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def _perform_static_scatter_elements(self, input_data: np.ndarray, indices: np.ndarray,
                                   updates: np.ndarray, reduction: str = "none"):
        if reduction not in self.reduction_types:
            raise TypeError("Cannot perform static scatter elements. Expected reduction type"
                            " to be one of: {}, instead got: {}".format(list(self.reduction_types.keys()),
                                                                        reduction))

        # Perform only reduction = none since that is supported in opset 11,13
        # No need to reject other reduction values since the attribute only exists in opset 16
        # TODO: Check for other reduction types once new version is added

        static_scatter_data = np.copy(input_data)
        for idx_tuple in np.ndindex(indices.shape):
            update_value = updates[idx_tuple]
            idx_list = list(idx_tuple)
            idx_list[self.axis] = indices[idx_tuple]
            idx_tuple = tuple(idx_list)
            if self.reduction == "add":
                static_scatter_data[idx_tuple] += update_value
            elif self.reduction == "mul":
                static_scatter_data[idx_tuple] *= update_value
            else:
                static_scatter_data[idx_tuple] = update_value

        return static_scatter_data

    def check_duplicate_indices(self, input_data, indices, op_name):
        # when indices and input_data are constant, we can check whether they have duplicate indices
        if indices is not None and input_data is not None:

            # check to ensure unique indices if reduction is none
            unique_indices = set()
            for idx_tuple in np.ndindex(indices.shape):
                idx_list = list(idx_tuple)
                idx_list[self.axis] = indices[idx_tuple]
                if idx_list[self.axis] < 0:
                    idx_list[self.axis] += input_data.shape[self.axis]
                idx_tuple = tuple(idx_list)
                if idx_tuple not in unique_indices:
                    unique_indices.add(idx_tuple)
                else:
                    log_warning("Duplicate scatter elements indices detected when reduction is set to None for Op {}. "
                                "This is not recommended and may result in inconsistent output and accuracy issues".
                                format(op_name))

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        updates_name = str(src_op.input[2])
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type))
        self.axis = params.axis
        self.reduction = params.get('reduction', 'none')
        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, converter_context,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        const_updates_op = self.fetch_constant_op(updates_name, converter_context,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_updates_op is not None:
            const_input_ops.append(const_updates_op)

        # If all inputs are static, then perform static scatter elements and return constant
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            updates = const_input_ops[2].tensor
            self.check_duplicate_indices(input_data, indices, src_op.name)
            scatter_data = self._perform_static_scatter_elements(input_data, indices, updates)
            converter_context.weights.insert(str(src_op.output[0]), scatter_data)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], scatter_data))
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.int32)
            # canonicalize negative indice value
            if input_data is not None:
                input_shape = input_data.shape
            else:
                input_shape = graph.get_buffer(input_data_name).shape
            indices[indices < 0] += input_shape[self.axis]
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(np.int32)

        self.check_duplicate_indices(input_data, indices, src_op.name)

        # If updates input is stored as weights then create a corresponding const op
        if not graph.has_buffer(updates_name) and converter_context.weights.has(updates_name):
            updates = converter_context.weights.fetch(updates_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(updates_name, updates))

        translated_ops.append(op_adapter.ScatterElementsOp(src_op.name,
                                                           axis=params.axis,
                                                           reduction=self.reduction_types[self.reduction]))

        return input_ops, translated_ops

    def extract_input_names(self, src_op, converter_context):
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return super().extract_input_names(src_op, converter_context)


OnnxTranslations.register_translation(OnnxScatterElementsTranslation(),
                                      converter_type('ScatterElements', 'onnx'),
                                      converter_type('Scatter', 'onnx'))


# ------------------------------------------------------------------------------
#   ScatterND
# ------------------------------------------------------------------------------
class OnnxScatterNDTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScatterND', [11, 13, 16])
        self.reduction_types = {"none": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_NONE,
                                "add": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_ADD,
                                "mul": ir_graph.QNN_OP_SCATTER_ND_REDUCTION_MUL}

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_ops, translated_ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # input op should only be 1 or 2 or None.
        # if input_op = None => all inputs are constant or all are dynamic
        # When input ops are None, scatter ND is a constant op
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when scatter_nd op is represented as a constant op i.e input ops is None
            last_node = graph.add(translated_ops[0], [], output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when scatter nd op has one or more dynamic inputs
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def _perform_static_scatter_nd(self, input_data: np.ndarray, indices: np.ndarray,
                                   updates: np.ndarray, reduction: str = "none"):
        if reduction not in self.reduction_types:
            raise TypeError("Cannot perform static scatter nd. Expected reduction type"
                            " to be one of: {}, instead got: {}".format(list(self.reduction_types.keys()),
                                                                        reduction))

        static_scatter_data = np.copy(input_data)
        update_idx = indices.shape[:-1]
        for idx in np.ndindex(update_idx):
            update_value = updates[idx]
            if reduction == "add":
                static_scatter_data[idx] += update_value
            elif self.reduction == "mul":
                static_scatter_data[idx] *= update_value
            else:
                static_scatter_data[idx] = update_value


        return static_scatter_data

    def extract_parameters(self, src_op, converter_context):
        # Note there are no attributes to extract for versions 11, 13

        graph = converter_context.ir_graph
        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        updates_name = str(src_op.input[2])
        params = extract_attributes(src_op,
                                    attr_infos=[('reduction', 's', "none")],
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        self.reduction = params.reduction

        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set
        const_input_op = self.fetch_constant_op(input_data_name, converter_context, dtype=None, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, converter_context,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        const_updates_op = self.fetch_constant_op(updates_name, converter_context, fail_if_dynamic=False)
        if const_updates_op is not None:
            const_input_ops.append(const_updates_op)

        # If all inputs are static, then perform static scatter and return
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            updates = const_input_ops[2].tensor
            scatter_data = self._perform_static_scatter_nd(input_data, indices, updates, self.reduction)
            converter_context.weights.insert(str(src_op.output[0]), scatter_data)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], scatter_data))
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and converter_context.weights.has(input_data_name):
            input_data = converter_context.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and converter_context.weights.has(indices_name):
            indices = converter_context.weights.fetch(indices_name, prunable=False).astype(np.uint32)
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(np.uint32)

        if indices is not None:
            if np.any(indices < 0):
                if input_data is None:
                    raise ValueError("Cannot resolve constant negative indices for ScatterND indices: "
                                     "{} if input data is not static".format(indices_name))
                else:
                    with np.nditer(indices, op_flags=['readwrite']) as it:
                        for index in it:
                            if index < 0:
                                index += len(input_data.shape)

            # check to ensure unique indices if reduction is none
            # TODO: Change when onnx version is updated as reduction is none for opset version < 16
            update_indices = indices.shape[:-1]
            unique_indices = set()
            for idx in np.ndindex(update_indices):
                # hash to place list value in unique_indices set
                idx_list = tuple(indices[idx].tolist())
                if idx_list not in unique_indices:
                    unique_indices.add(idx_list)
                else:
                    log_warning("Duplicate scatter indices detected when reduction is set to None for Op {}. "
                                "This is not recommended and can result in inconsistent output and accuracy issues".
                                format(src_op.name))

        # If updates is stored as weights then create a corresponding const op
        if not graph.has_buffer(updates_name) and converter_context.weights.has(updates_name):
            updates = converter_context.weights.fetch(updates_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(updates_name, updates))

        translated_ops.append(op_adapter.ScatterNDOp(src_op.name, reduction=self.reduction_types[self.reduction]))

        return input_ops, translated_ops

    def extract_input_names(self, src_op, converter_context):
        if converter_context.weights.has_all(src_op.input):
            return []
        else:
            return super().extract_input_names(src_op, converter_context)


OnnxTranslations.register_translation(OnnxScatterNDTranslation(),
                                      converter_type('ScatterND', 'onnx'),
                                      op_adapter.ScatterNDOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Shape
# ------------------------------------------------------------------------------
class OnnxShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Shape', [1, 13, 15])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        input_name = str(src_op.input[0])

        constant_op = self.fetch_constant_op(input_name, converter_context, dtype=np.int32, fail_if_not_found=True,
                                             fail_if_dynamic=False)
        if constant_op:
            shape = constant_op.tensor.shape
        elif graph.has_buffer(input_name):
            shape = graph.get_buffer(input_name).shape

        start = 0
        end = len(shape)
        schema = self.op_schema(op_type=src_op.op_type)
        if schema.version[0] > 1:
            params = extract_attributes(src_op,
                                    attr_infos=[('start', 'i', 0), ('end', 'i', len(shape))],
                                    schema=schema, validate=True)
            start = params.start
            end = params.end

        output_name = str(src_op.output[0])
        shape = np.asarray(shape, dtype=np.int32)[start:end]
        converter_context.weights.insert(output_name, shape)
        return None

    def extract_input_names(self, src_op, converter_context):
        return []

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxShapeTranslation(),
                                      converter_type('Shape', 'onnx'))


# ------------------------------------------------------------------------------
#   Slice
# ------------------------------------------------------------------------------
class OnnxSliceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Slice', [1, 10, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = [str(x) for x in src_op.input]
        params = extract_attributes(src_op, schema=self.op_schema())
        const_inputs_params = self._fetch_inputs_as_params(src_op, converter_context, params)
        params.update(const_inputs_params)

        log_assert(len(params.starts) == len(params.axes),
                   "Node {}: expected same number of starts as axes",
                   src_op.name)
        log_assert(len(params.ends) == len(params.axes),
                   "Node {}: expected same number of ends as axes",
                   src_op.name)
        log_assert(all(params.steps),
                   "Node {}: expected all steps != 0",
                   src_op.name)

        # Static slicing used for shape tensors
        if converter_context.weights.has(input_names[0]):
            data = converter_context.weights.fetch(input_names[0])
            for i in range(len(params.axes)):
                start, end = self.get_indices(params.starts[i],
                                              params.ends[i],
                                              params.steps[i],
                                              data.shape[params.axes[i]])
                data = data.take(indices=list(range(start, end, params.steps[i])), axis=params.axes[i])
            output_name = str(src_op.output[0])
            converter_context.weights.insert(output_name, data)
            return None

        input_buf = graph.get_buffer(input_names[0])
        rank = input_buf.rank()
        begin = [0] * rank
        end = [0] * rank
        strides = [0] * rank

        for index, axis in enumerate(params.axes):
            begin[axis], end[axis] = self.get_indices(params.starts[index],
                                                      params.ends[index],
                                                      params.steps[index],
                                                      input_buf.shape[axis])
            strides[axis] = params.steps[index]

            # If the input data is dynamic and start is out of bounds with positive steps, i.e., start will be equal to dim
            # Output will be null tensor
            # Still need to add to the weights so that subsequent node tracks properly
            if begin[axis]==input_buf.shape[axis]:
                data=[]
                output_name = str(src_op.output[0])
                converter_context.weights.insert(output_name, data)
                return None

            # add check to find if there is empty data case or out-of-range indices
            log_assert(begin[axis] < end[axis] if strides[axis] > 0 else begin[axis] > end[axis],
                       "Node {}: invalid stride for begin {} and end {} at axis {}",
                       src_op.name, begin[axis], end[axis], axis)
            log_assert(0 <= begin[axis] < input_buf.shape[axis],
                       "Node {}: begin:{} at axis {} is out-of-range",
                       src_op.name, begin[axis])
            log_assert(-1 <= end[axis] <= input_buf.shape[axis],
                       "Node {}: end:{} at axis {} is out-of-range",
                       src_op.name, end[axis], axis)

        for i, stride in enumerate(strides):
            if not stride:
                begin[i], end[i] = 0, input_buf.shape[i]
                strides[i] = 1

        ranges = list(map(list, zip(begin, end, strides)))
        op = op_adapter.StridedSliceOp(name=src_op.name, ranges=ranges)

        # there is null tensor logic in onnx translation so we can not clean out
        # canonicalization logic
        # TODO: clean up canonicalization logic in translation
        op.c_op.set_canonicalized(True)
        return op

    def _fetch_inputs_as_params(self, src_op, converter_context, params):
        # opset 10,11 need handle 5 inputs, fetch constant input and add it to params
        # NOTE: Runtime does not allow dynamic input for starts, ends, axes and steps
        # input indices: data: 0, starts: 1, ends: 2, axes: 3(optional), steps: 4(optional)
        graph = converter_context.ir_graph
        input_names = [str(x) for x in src_op.input]
        rank = 0
        if graph.has_buffer(input_names[0]):
            input_buf = graph.get_buffer(input_names[0])
            rank = input_buf.rank()
        elif converter_context.weights.has(input_names[0]):
            rank = len(converter_context.weights.fetch(input_names[0], prunable=False).shape)
        keys = ['data', 'starts', 'ends', 'axes', 'steps']
        if len(src_op.input) >= 3:
            for key, name in zip(keys[1:], input_names[1:]):
                # ONNX may use empty string as a placeholder
                # So add an and-condition to further check it.
                if name and converter_context.weights.has(name):
                    # handle INT_MAX and INT_MIN case in ONNX spec, require fetch int64 directly
                    # case: INT64_MAX -> cast to float and cast to int64 -> INT64_MIN
                    # case: INT64_MAX -> cast to int32 -> -1
                    params[key] = converter_context.weights.fetch(name, dtype=np.int64, prunable=False).tolist()
                    if key == 'axes':
                        for axis in params['axes']:
                            log_assert(-rank <= axis <= rank-1,
                            "expected axis range from {} to {}, but got {}",
                            -rank, rank-1, axis)
                elif graph.has_buffer(name):
                    raise ValueError(code_to_message.get_error_message('ERROR_SLICE_DYNAMIC_INPUTS')(name))

        if 'axes' not in params or len(params.axes) == 0:
            params['axes'] = list(range(len(params.starts)))
        if 'steps' not in params or len(params.steps) == 0:
            params['steps'] = list([1] * len(params.starts))

        return params

    def extract_input_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        # If this was translated to a static op don't return input names
        if converter_context.weights.has(str(src_op.input[0])):
            return []
        else:
            # Handle constant and initializer cases, do not add them to input_names to avoid prune error.
            actual_input_names = []
            for input_name in map(str, src_op.input):
                if input_name in graph.buffers and not converter_context.weights.has(input_name):
                    actual_input_names.append(input_name)
            return actual_input_names

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.has(str(src_op.input[0])):
            return []
        else:
            return list(map(str, src_op.output))

    @staticmethod
    def get_indices(start, end, step, dim):
        # Negative values mean wrap around, like in python
        if start < 0:
            start = int(start % dim)

        if step < 0:
            # higher than the size, however, means stop at the end - 1.
            start = min(start, dim-1)
            end = max(end, -(dim+1))
        else:
            # If start is out of bounds and step is positive,
            # start will be dim for out of bounds case
            start=min(start,dim)
            end = min(end, dim)

        if end < 0:
            end = end + dim

        return start, end


OnnxTranslations.register_translation(OnnxSliceTranslation(),
                                      converter_type('Slice', 'onnx'),
                                      op_adapter.StridedSliceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class OnnxSplitTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Split', [1, 2, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()

        if schema.version[0] < 13:
            schema.replace_default_values(split=[])
            params = extract_attributes(src_op, schema=schema)
            split_index = ir_graph.SplitOp.convert_sizes_to_indices(params.split)
        else:
            params = extract_attributes(src_op, schema=schema)
            if len(src_op.input) > 1:
                # backend only support split_index as a parameter, so need split input as a constant
                const_split_op = self.fetch_constant_op(str(src_op.input[1]), converter_context,
                                                        fail_if_dynamic=True, fail_if_not_found=True)
                split = const_split_op.tensor
                split_index = ir_graph.SplitOp.convert_sizes_to_indices(split)
            else:
                split_index = []

        const_input_op = self.fetch_constant_op(str(src_op.input[0]), converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            # empty split index for onnx represents equal splitting when the num of output > 1
            if len(split_index) == 0 and len(src_op.output) > 1:
                split_index = len(src_op.output)
            w = const_input_op.tensor
            w = np.array_split(w, split_index, params.axis)
            # To account for multiple consumers
            for i in range(len(w)):
                converter_context.weights.insert(src_op.output[i], w[i])
            return None

        return op_adapter.SplitOp(src_op.name,
                                  axis=params.axis,
                                  split_index=split_index)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxSplitTranslation(),
                                      converter_type('Split', 'onnx'),
                                      op_adapter.SplitOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class OnnxSqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Squeeze', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_name = str(src_op.input[0])
        params = extract_attributes(src_op, schema=self.op_schema())

        axes = []
        if len(src_op.input) > 1:
            axes_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            axes = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32).tensor.tolist()
        elif 'axes' in params:
            axes = params.axes

        const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            # static squeeze of weight parameters
            output_name = str(src_op.output[0])
            w = converter_context.weights.fetch(input_name)
            if not len(axes):
                axes = [i for i, s in enumerate(w.shape) if s == 1]
            output_shape = self._get_squeezed_shape(w.shape, axes)

            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            w = np.reshape(w, output_shape)
            # The w here might be a np "array scalar" whose shape attribute is
            # an empty tuple, which may break backends.
            # So we reshape "array scalar" to exactly an array with shape (1, )
            was_scalar = False
            if not w.shape:
                was_scalar = True
                w = w.reshape(1)
            converter_context.weights.insert(output_name, w, was_scalar)
            return None

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        if not len(axes):
            axes = [i for i, s in enumerate(input_shape) if s == 1]

        if not all(x < len(input_shape) for x in axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIM_GREATER_THAN_RANK")(axes,
                                                                                                      len(input_shape)))
        if not all((input_shape[x] == 1) for x in axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIMS_EQUAL_ONE")(axes,
                                                                                               input_shape))

        output_shape = self._get_squeezed_shape(input_shape, axes)

        log_assert(output_shape,"0D tensors are currently not supported for Squeeze op.")

        return op_adapter.ReshapeOp(src_op.name, shape=output_shape)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    @staticmethod
    def _get_squeezed_shape(input_shape, axes):
        positive_axes=[]
        for i in axes:
            if i < 0:
                # update axes if negative
                positive_axes.insert(i, len(input_shape) + i)
            else:
                positive_axes.insert(i, i)
        output_shape = [s for i, s in enumerate(input_shape) if i not in positive_axes]
        return output_shape

OnnxTranslations.register_translation(OnnxSqueezeTranslation(), converter_type('Squeeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class OnnxTileTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tile', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))

        input_constant_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False,
                                                   fail_if_dynamic=False)
        if input_constant_op:
            input_rank = len(input_constant_op.tensor.shape)
        else:
            input_rank = len(graph.get_buffer(src_op.input[0]).shape)

        if len(input_names) == 3:
            # Represents Tile-1
            tiles = converter_context.weights.fetch(src_op.input[1])
            axis = converter_context.weights.fetch(src_op.input[2])
            repeats = [1] * input_rank
            repeats[axis] = tiles
        elif len(input_names) == 2:
            # Represents Tile-6
            repeats = self.fetch_constant_op(src_op.input[1], converter_context).tensor
        else:
            raise ValueError("Only versions {} of {} node {} are supported".format(self.get_supported_version(),
                                                                                   src_op.op_type, src_op.name))
        if input_constant_op:
            output_tensor = np.tile(input_constant_op.tensor, repeats)
            converter_context.weights.insert(src_op.output[0], output_tensor)
            return None

        return op_adapter.TileOp(src_op.name, multiples=repeats)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTileTranslation(),
                                      converter_type('Tile', 'onnx'),
                                      op_adapter.TileOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class OnnxTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Transpose', [1, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        def extract_default_attributes(name, graph, input_name):
            ret = NamedDict()
            if graph.has_buffer(input_name):
                input_shape = graph.get_buffer(input_name).shape
            else:
                const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False, fail_if_not_found=True)
                input_shape = const_op.tensor.shape
            perm = list(range(len(input_shape))) # creates a list like [0, 1,.., n]
            perm.reverse() # default behaviour of Transpose should reverse the permute [n, n-1, .., 0]
            ret[name] = perm
            return ret
        input_name = str(src_op.input[0])
        params = extract_attributes(src_op, schema=self.op_schema(), default_attrs=extract_default_attributes("perm", graph, input_name))
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False, fail_if_not_found=True)
        if const_op is not None:
            # static permute of weight parameters
            output_name = str(src_op.output[0])
            w = const_op.tensor
            log_debug1('static input: {} to: {}'.format(input_name, w.shape))
            log_debug1('transpose shape to : {}'.format(params.perm))
            w = np.transpose(w, params.perm)
            converter_context.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, w.shape))

            return None

        log_debug1('input: {} to: {}'.format(input_name,graph.get_buffer(input_name).shape))
        log_debug1('transpose shape to : {}'.format(params.perm))
        return op_adapter.TransposeOp(src_op.name, params.perm)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        # return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxTransposeTranslation(),
                                      converter_type('Transpose', 'onnx'),
                                      op_adapter.TransposeOp.TRANSLATION_KEY)


# -----------------------------------------------------------------------------
#   Unsqueeze
# ------------------------------------------------------------------------------
class OnnxUnsqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Unsqueeze', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        axes = []
        if len(src_op.input) > 1:
            axes_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            axes = self.fetch_constant_op(axes_input, converter_context, dtype=np.int32).tensor.tolist()
        elif 'axes' in params:
            axes = params.axes

        if len(set(axes)) != len(axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DUPLICATE_DIMS")(axes))

        input_name = str(src_op.input[0])

        const_input_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            w = const_input_op.tensor
            shape = [] if converter_context.weights.was_scalar(input_name) else w.shape
            output_shape = self._get_unsqueezed_shape(shape, axes)
            w = np.reshape(w, output_shape)
            output_name = str(src_op.output[0])
            converter_context.weights.insert(output_name, w)
            return None

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        new_rank = len(input_shape) + len(axes)
        if not all(x < new_rank for x in axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DIMS_GREATER_THAN_RANK")(axes,
                                                                                                         new_rank))
        output_shape = self._get_unsqueezed_shape(input_shape, axes)

        # Otherwise this is a dynamic unsqueeze so add the unsqueeze/reshape op
        return op_adapter.ReshapeOp(src_op.name, shape=output_shape)

    def extract_input_names(self, src_op, converter_context):
        return [name for name in list(map(str, src_op.input)) if not converter_context.weights.consumed(name)]

    def extract_output_names(self, src_op, converter_context):
        # If this was translated to a static op don't return output names
        if converter_context.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    @staticmethod
    def _get_unsqueezed_shape(org_shape, axes):
        output_shape = list(org_shape)
        for i in sorted(axes):
            # support negative axes since Unsqueeze-11
            if i < 0:
                i += len(output_shape)+1
            output_shape.insert(i, 1)
        return output_shape


OnnxTranslations.register_translation(OnnxUnsqueezeTranslation(), converter_type('Unsqueeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Upsample
# ------------------------------------------------------------------------------
class OnnxUpsampleTranslation(OnnxResizeTranslation):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.default_transformation_mode = 'asymmetric'
        self.register_op_schema('Upsample', [1, 7, 9])\
            .register_method(self.validate_attribute_values)


OnnxTranslations.register_translation(OnnxUpsampleTranslation(),
                                      converter_type('Upsample', 'onnx'))


# ------------------------------------------------------------------------------
#   Where
# ------------------------------------------------------------------------------
class OnnxWhereTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Where', [9, 16])
        self.input_names = None

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if op.type == op_adapter.IdentityOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            return graph.add(op, input_names, output_names)

        for input_name in self.input_names:
            const_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            # Add fetched constant op to graph, if it doesn't exist
            if const_op is not None:
                if not graph.has_buffer(input_name):
                    const_node = graph.add(const_op, [], input_name)
                    graph.add_src_op_info(const_op.name, None, const_node.output_names[0])

        # Add elementwise src op info
        self.add_src_op_info(op.name, src_op, graph)

        return graph.add(op, self.input_names, output_names)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = list(map(str, src_op.input))

        condition_op = self.fetch_constant_op(self.input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
        branch1_op = self.fetch_constant_op(self.input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        branch2_op = self.fetch_constant_op(self.input_names[2], converter_context, prunable=False, fail_if_dynamic=False)

        if condition_op:
            if branch1_op and branch2_op:
                data = np.where(condition_op.tensor, branch1_op.tensor, branch2_op.tensor)
                was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in self.input_names])
                converter_context.weights.insert(str(src_op.output[0]), data, was_scalar=was_scalar)
                return None

            condition_tensor = condition_op.tensor.flatten()
            # Check Identity cases: Either all True yielding a pass-through of input1 or all False
            # yielding a pass-through of input2
            if all(condition for condition in condition_tensor):
                self.input_names = [self.input_names[1]]
                return op_adapter.IdentityOp(src_op.name)
            elif all(not condition for condition in condition_tensor):
                self.input_names = [self.input_names[2]]
                return op_adapter.IdentityOp(src_op.name)

        return op_adapter.ElementwiseTernaryOp(name=src_op.name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SELECT)

    def extract_input_names(self, src_op, converter_context):
        return self.input_names


OnnxTranslations.register_translation(OnnxWhereTranslation(),
                                      converter_type('Where', 'onnx'))
