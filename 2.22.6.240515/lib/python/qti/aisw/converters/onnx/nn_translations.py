# ==============================================================================
#
#  Copyright (c) 2018-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.converter_ir.op_graph import QuantParams

# ------------------------------------------------------------------------------
#   AveragePool, MaxPool, LpPool
# ------------------------------------------------------------------------------
class OnnxPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('AveragePool', [1, 7, 10, 11])
        self.register_op_schema('MaxPool', [1, 8, 10, 11, 12])
        self.register_op_schema('LpPool', [1, 2, 11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_buffer = graph.get_buffer(input_names[0])

        if input_buffer.rank() == 5:
            default_dilations = [1, 1, 1]
            default_strides = [1, 1, 1]
            default_pads = [0, 0, 0, 0, 0, 0]
        else:
            default_dilations = [1, 1]
            default_strides = [1, 1]
            default_pads = [0, 0, 0, 0]

        params = extract_attributes(src_op, attr_infos=
        [('ceil_mode', 'i', 0),  # new attribute since MaxPool-10 and AveragePool-10
         ('storage_order', 'i', 0),  # new attribute since MaxPool-8
         ('dilations', 'li', default_dilations),  # new attribute since MaxPool-10
         ('count_include_pad', 'i', 0),  # new attribute since AveragePool-7
         ('strides', 'li', default_strides),
         # stride default to 1 if not present since MaxPool-11 and AveragePool-11
         ('kernel_shape', 'li'),
         ('pads', 'li', default_pads),
         ('auto_pad', 's', 'NOTSET'),
         ('p', 'i', 2)], # attribute for LpPool
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name, params.ceil_mode,
                                                     translation_utils.pads_righthanded(params.pads))
        if str(src_op.op_type) == 'AveragePool':
            if input_buffer.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_AVG_3D
            else:
                pool_type = ir_graph.QNN_OP_POOL_AVG_2D
        elif str(src_op.op_type) == 'MaxPool':
            if input_buffer.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_MAX_3D
            else:
                pool_type = ir_graph.QNN_OP_POOL_MAX_2D

            log_assert(len(src_op.output) == 1, code_to_message.get_error_message(
                "ERROR_MAXPOOL_OPTIONAL_INDICES_OUTPUT"))
        elif str(src_op.op_type) == 'LpPool':
            pool_type = ir_graph.QNN_OP_L2_POOL_2D

        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis

        if input_buffer.rank() == 3 or len(params.kernel_shape) == 1 :
            if pool_type == ir_graph.QNN_OP_L2_POOL_2D:
                return op_adapter.Pool1dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_x=params.kernel_shape[0],
                                           stride_x=params.strides[0],
                                           padx_before=params.pads[0],
                                           padx_after=params.pads[num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           p=params.p)
            elif pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
                return op_adapter.Pool1dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_x=params.kernel_shape[0],
                                           stride_x=params.strides[0],
                                           dilation_x=params.dilations[0],
                                           padx_before=params.pads[0],
                                           padx_after=params.pads[num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy])
            elif pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
                return op_adapter.Pool1dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_x=params.kernel_shape[0],
                                           stride_x=params.strides[0],
                                           dilation_x=params.dilations[0],
                                           padx_before=params.pads[0],
                                           padx_after=params.pads[num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           count_pad_for_edges=params.count_include_pad)
        elif input_buffer.rank() == 5:
            return op_adapter.Pool3dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_z=params.kernel_shape[0],
                                       size_y=params.kernel_shape[1],
                                       size_x=params.kernel_shape[2],
                                       stride_z=params.strides[0],
                                       stride_y=params.strides[1],
                                       stride_x=params.strides[2],
                                       dilation_z=params.dilations[0],
                                       dilation_y=params.dilations[1],
                                       dilation_x=params.dilations[2],
                                       padz_before=params.pads[0],
                                       padz_after=params.pads[num_input_pads],
                                       pady_before=params.pads[1],
                                       pady_after=params.pads[1 + num_input_pads],
                                       padx_before=params.pads[2],
                                       padx_after=params.pads[2+num_input_pads],
                                       padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                       count_pad_for_edges=params.count_include_pad)
        else:
            log_assert(num_input_pads == 2,
                       code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                       (src_op.name, num_input_pads))
            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            if pool_type == ir_graph.QNN_OP_L2_POOL_2D:
                return op_adapter.Pool2dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_y=params.kernel_shape[0],
                                           size_x=params.kernel_shape[1],
                                           stride_y=params.strides[0],
                                           stride_x=params.strides[1],
                                           pady_before=params.pads[0],
                                           pady_after=params.pads[num_input_pads],
                                           padx_before=params.pads[1],
                                           padx_after=params.pads[1 + num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           p=params.p)
            elif pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
                return op_adapter.Pool2dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_y=params.kernel_shape[0],
                                           size_x=params.kernel_shape[1],
                                           stride_y=params.strides[0],
                                           stride_x=params.strides[1],
                                           dilation_y=params.dilations[0],
                                           dilation_x=params.dilations[1],
                                           pady_before=params.pads[0],
                                           pady_after=params.pads[num_input_pads],
                                           padx_before=params.pads[1],
                                           padx_after=params.pads[1 + num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy])
            elif pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
                return op_adapter.Pool2dOp(src_op.name,
                                           pool_type=pool_type,
                                           size_y=params.kernel_shape[0],
                                           size_x=params.kernel_shape[1],
                                           stride_y=params.strides[0],
                                           stride_x=params.strides[1],
                                           dilation_y=params.dilations[0],
                                           dilation_x=params.dilations[1],
                                           pady_before=params.pads[0],
                                           pady_after=params.pads[num_input_pads],
                                           padx_before=params.pads[1],
                                           padx_after=params.pads[1 + num_input_pads],
                                           padding_size_strategy=op_adapter.IRPaddingStrategies.py_to_c[padding_size_strategy],
                                           count_pad_for_edges=params.count_include_pad)


OnnxTranslations.register_translation(OnnxPoolTranslation(),
                                      converter_type('AveragePool', 'onnx'),
                                      converter_type('MaxPool', 'onnx'),
                                      converter_type('LpPool', 'onnx'),
                                      op_adapter.Pool1dOp.TRANSLATION_KEY,
                                      op_adapter.Pool2dOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   BatchNormalization
# ------------------------------------------------------------------------------
class OnnxBatchNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('BatchNormalization', [1, 6, 7, 9, 14, 15]) \
            .register_method(self.validate_attribute_values)
        self.weight = None
        self.bias = None

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        # In case src_op name is absent, extract parameter is updating quantization_params
        # information with empty '' key. This will throw error later in op_graph_optimization
        # when the bn_params are accessed with python graph node name(generated with naming policy).
        # Thus updating src_op name with same name generated with naming policy.
        bn_op_name = graph.naming_policy.get_op_name_by_type(ir_graph.QNN_OP_BATCHNORM, legacy_translation_key='batchnorm')
        if (len(src_op.name)==0):
            src_op.name = bn_op_name
        else:
            # If framework level name present, use that instead
            bn_op_name = src_op.name
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        node_op.name = bn_op_name
        weights_name = bn_op_name + "_bn_w"
        bias_name = bn_op_name + "_bn_b"
        weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=self.weights)
        bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=self.bias)

        gamma_quant_enc = graph.get_overridden_encoding(list(src_op.input)[1])
        beta_quant_enc = graph.get_overridden_encoding(list(src_op.input)[2])

        # Propagate gamma/beta overrides to weights/bias if provided
        if gamma_quant_enc:
            graph.set_overridden_encoding(weights_name, gamma_quant_enc)
        if beta_quant_enc:
            graph.set_overridden_encoding(bias_name, beta_quant_enc)

        weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
        graph.add_src_op_info(bias_name, None, bias_node.output_names[0])

        input_names.append(weights_name)
        input_names.append(bias_name)
        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema(op_type=src_op.op_type)

        params = extract_attributes(src_op, schema=schema, validate=True)

        if "training_mode" in params and params.training_mode == 1:
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(params.training_mode, "training_mode", src_op.name))

        input_names = list(src_op.input)
        gamma, beta, mu, var = converter_context.weights.fetch(*input_names[1:])

        if graph.has_buffer(input_names[1]):
            converter_context.weights.weight_map[input_names[1]].consumed = False
        if graph.has_buffer(input_names[2]):
            converter_context.weights.weight_map[input_names[2]].consumed = False
        # y = gamma*( (x-mu)/sqrt(var+epsilon) ) + beta
        # weights = gamma/sqrt(var+epsilon)
        self.weights = gamma / np.sqrt(var + params.epsilon)
        # bias = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        self.bias = -mu * self.weights + beta

        # spatial removed starting opset 9
        spatial = bool(params.spatial if 'spatial' in params else 1)
        if not spatial:
            raise ValueError("Batchnorm only supports across spatial. Got spatial = False for op {}"
                             .format(src_op.name))

        # cache gamma, beta bn parameters for HBA quant optimization algorithm
        graph.add_quantization_params(src_op.name, bn_params={"gamma": gamma, "beta": beta})

        return op_adapter.BatchnormOp(src_op.name)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]

    @staticmethod
    def validate_attribute_values(self, attr_name, attr_value):
        # is_test is only supported in test mode, which is_test = 1
        if attr_name == 'is_test':
            log_assert(attr_value, code_to_message.get_error_message('ERROR_BATCHNORM_TEST_ONLY'))


OnnxTranslations.register_translation(OnnxBatchNormalizationTranslation(),
                                      converter_type('BatchNormalization', 'onnx'),
                                      op_adapter.BatchnormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class OnnxConvTranslation(OnnxTranslationBase):

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Conv', [1, 11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        if not graph.has_buffer(input_names[0]) and converter_context.weights.has(input_names[0]):
            input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)
            if input_op and not graph.has_buffer(input_names[0]):
                graph.add(input_op, [], [input_names[0]], axis_formats=[AxisTracker.AxisFormat.NCS])
        input_buffer = graph.get_buffer(input_names[0])

        if input_buffer.rank() == 5:
            default_dilations = [1, 1, 1]
            default_strides = [1, 1, 1]
            default_pads = [0, 0, 0, 0, 0, 0]
        else:
            default_dilations = [1, 1]
            default_strides = [1, 1]
            default_pads = [0, 0, 0, 0]

        params = extract_attributes(src_op, attr_infos=
                                    [('dilations', 'li', default_dilations),
                                     ('strides', 'li', default_strides),
                                     ('kernel_shape', 'li'),
                                     ('pads', 'li', default_pads),
                                     ('auto_pad', 's', 'NOTSET'),
                                     ('group', 'i', 1)], schema=self.op_schema(), validate=True)

        # Extract weights and biases and add them as ConstantOp inputs to the Conv2dOp
        weights_constant_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                           code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            if len(weights_constant_op.tensor.shape) == 3:
                graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            elif len(weights_constant_op.tensor.shape) == 4:
                graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.OIHW])
            elif len(weights_constant_op.tensor.shape) == 5:
                graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.OIDHW])
            else:
                raise NotImplementedError(
                        "Convolutional Tensor doesn't support {}-D translations.".format(len(weights_constant_op.tensor.shape)))
        elif graph.has_buffer(input_names[1]):
            weight_buffer = graph.get_buffer(input_names[1])
            # Properly set the axis format if the buffer already exists in the graph
            if len(weight_buffer.shape) == 3:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif len(weight_buffer.shape) == 4:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.OIHW
            elif len(weight_buffer.shape) == 5:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.OIDHW
            else:
                raise NotImplementedError(
                        "Convolutional Tensor doesn't support {}-D translations.".format(len(weights_constant_op.tensor.shape)))

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract the remaining attributes and calculate the padding size
        padding_size_strategy = IRPaddingStrategies.py_to_c[extract_padding_mode(params.auto_pad, src_op.name)]
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis

        # set the input padding size
        input_shape = graph.get_buffer(input_names[0]).shape
        weights_shape = graph.get_buffer(input_names[1]).shape

        if input_buffer.rank() == 3:
            pad_x = ir_graph.ConvOp.calc_conv_padding_size(input_shape[2],
                                                           weights_shape[2],
                                                           params.dilations[0],
                                                           params.strides[0],
                                                           padding_size_strategy,
                                                           params.pads)

            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_1d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv1d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv1dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                convolution_class = op_adapter.DepthwiseConv1dOp

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[0],
                                     dilationx=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCF)
        elif input_buffer.rank() == 5:
            if num_input_pads != 3:
                raise ValueError("Conv3d the number of pads number shall be 3, instead of {}".format(num_input_pads))

            pad_z = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[2],
                                                             weights_shape[2],
                                                             params.dilations[0],
                                                             params.strides[0],
                                                             padding_size_strategy,
                                                             [params.pads[0], params.pads[num_input_pads]])

            pad_y = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[3],
                                                             weights_shape[3],
                                                             params.dilations[1],
                                                             params.strides[1],
                                                             padding_size_strategy,
                                                             [params.pads[1], params.pads[1 + num_input_pads]])

            pad_x = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[4],
                                                             weights_shape[4],
                                                             params.dilations[2],
                                                             params.strides[2],
                                                             padding_size_strategy,
                                                             [params.pads[2], params.pads[2 + num_input_pads]])

            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_3d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv3d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv3dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                raise ValueError("Depthwise 3D convolution operation is not yet implemented")

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     pady_before=pad_y[0],
                                     pady_after=pad_y[1],
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padz_before=pad_z[0],
                                     padz_after=pad_z[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[2],
                                     stridey=params.strides[1],
                                     stridez=params.strides[0],
                                     dilationx=params.dilations[2],
                                     dilationy=params.dilations[1],
                                     dilationz=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCDHW)
        else:
            log_assert(num_input_pads == 2,
                       code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                       (src_op.name, num_input_pads))
            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            pad_y = ir_graph.ConvOp.calc_conv_padding_size(input_shape[2],
                                                           weights_shape[2],
                                                           params.dilations[0],
                                                           params.strides[0],
                                                           padding_size_strategy,
                                                           [params.pads[0], params.pads[num_input_pads]])

            pad_x = ir_graph.ConvOp.calc_conv_padding_size(input_shape[3],
                                                           weights_shape[3],
                                                           params.dilations[1],
                                                           params.strides[1],
                                                           padding_size_strategy,
                                                           [params.pads[1], params.pads[1 + num_input_pads]])
            # Handle marking this Convolution as a DepthwiseConvolution
            num_input_channels = graph.src_axis_order.extract_2d_spatial_dims(input_shape)[-1]
            num_output_channels = graph.src_axis_order.extract_conv2d_weights_dims(weights_shape)[-1]
            convolution_class = op_adapter.Conv2dOp
            if params.group == num_input_channels and num_input_channels == num_output_channels:
                convolution_class = op_adapter.DepthwiseConv2dOp

            return convolution_class(src_op.name,
                                     bias_op_name=bias_op_name,
                                     pady_before=pad_y[0],
                                     pady_after=pad_y[1],
                                     padx_before=pad_x[0],
                                     padx_after=pad_x[1],
                                     padding_size_strategy=padding_size_strategy,
                                     stridex=params.strides[1],
                                     stridey=params.strides[0],
                                     dilationx=params.dilations[1],
                                     dilationy=params.dilations[0],
                                     groups=params.group,
                                     data_layout=AxisTracker.AxisFormat.NCS)


OnnxTranslations.register_translation(OnnxConvTranslation(),
                                      converter_type('Conv', 'onnx'),
                                      op_adapter.Conv2dOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConvTranspose
# ------------------------------------------------------------------------------
class OnnxConvTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('ConvTranspose', [1, 11])
        schema_dict.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        input_buffer = graph.get_buffer(input_names[0])

        if input_buffer.rank() in [3, 4, 5]:
            spatial_rank = input_buffer.rank() - 2
            default_dilations = [1] * spatial_rank
            default_strides = [1] * spatial_rank
            default_pads = [0, 0] * spatial_rank
            default_output_shape = [0] * spatial_rank
            default_output_padding = [0] * spatial_rank
        else:
            raise NotImplementedError(
                "ConvTranspose doesn't support {}-D translations.".format(input_buffer.rank()))

        params = extract_attributes(src_op,
                                    attr_infos=[('dilations', 'li', default_dilations),
                                                ('strides', 'li', default_strides),
                                                ('kernel_shape', 'li'),
                                                ('pads', 'li', default_pads),
                                                ('auto_pad', 's', 'NOTSET'),
                                                ('group', 'i', 1),
                                                ('output_shape', 'li', default_output_shape),
                                                ('output_padding', 'li', default_output_padding),],
                                    schema=self.op_schema(),
                                    validate=True)

        # Extract weights and biases and add them as ConstantOp inputs to the TransposeConv2dOp
        weights_constant_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                           code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))
            params.kernel_shape = weights_constant_op.tensor.shape[2:]
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            if input_buffer.rank() == 3:
                graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            elif input_buffer.rank() == 4:
                graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.IOHW])
            elif input_buffer.rank() == 5:
                graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.IODHW])
            else:
                raise NotImplementedError(
                    "ConvTranspose doesn't support {}-D translations.".format(input_buffer.rank()))
        elif graph.has_buffer(input_names[1]):
            # Properly set the axis format if the buffer already exists in the graph
            if input_buffer.rank() == 3:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif input_buffer.rank() == 4:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.IOHW
            elif input_buffer.rank() == 5:
                graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.IODHW
            else:
                raise NotImplementedError(
                    "ConvTranspose doesn't support {}-D translations.".format(input_buffer.rank()))

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract the remaining attributes and calculate the padding size
        padding_mode = IRPaddingStrategies.py_to_c[extract_padding_mode(params.auto_pad, src_op.name)]
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis

        # Extract and verify the output padding values
        output_padding = params.output_padding
        if any(output_padding):
            log_assert(output_padding < params.strides,
                       code_to_message.get_error_message(
                           "ERROR_DECONV_OUTPUT_PADDING_NOT_LESS_THAN_STRIDE")
                       (params.strides, output_padding))

        # Align padding strategy with QNN if output shape is explicitly specified
        if any(params.output_shape):
            # padding calculation differ in opset 1 and 11
            # check https://onnx.ai/onnx/operators/text_diff_ConvTranspose_1_11.html
            # we need to change padding mode accordingly to align padding calculation
            schema = self.op_schema()
            if params.auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
                if schema.version[0] == 1:
                    auto_pad_mapping = {
                        'SAME_UPPER': 'SAME_LOWER',
                        'SAME_LOWER': 'SAME_UPPER'
                    }
                    log_warning(f'change auto_pad from {params.auto_pad} to {auto_pad_mapping[params.auto_pad]} to align with QNN padding calculation')
                    padding_mode = IRPaddingStrategies.py_to_c[extract_padding_mode(auto_pad_mapping[params.auto_pad], src_op.name)]
            elif params.auto_pad in ['NOTSET', 'VALID']:
                spatial_shape = input_buffer.get_buf_dims()[2:]
                total_paddings = []
                for idx in range(num_input_pads):
                    output_shape = params.strides[idx] * (spatial_shape[idx] - 1) + \
                                (params.kernel_shape[idx] - 1) * params.dilations[idx] + 1 + \
                                    params.output_padding[idx]
                    total_padding = output_shape - params.output_shape[idx]
                    total_paddings.append(total_padding)
                total_paddings = [max(0, total_padding) for total_padding in total_paddings]
                for idx, total_padding in enumerate(total_paddings):
                    # if pad_before + pad_afer not equal to total_padding, then pads is not save in source model, need to recalculate
                    pads = list(params.pads)
                    if params.pads[idx] + params.pads[idx + num_input_pads] != total_padding:
                        if schema.version[0] == 1:
                            pads[idx] = total_padding // 2
                            pads[idx + num_input_pads] = total_padding - (total_padding // 2)
                        else:
                            pads[idx] = total_padding - (total_padding // 2)
                            pads[idx + num_input_pads] = total_padding // 2
                    params.pads = pads
        weights_shape = graph.get_buffer(input_names[1]).shape
        if input_buffer.rank() == 3:
            pad_x = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][0],
                                                               params.dilations[0],
                                                               params.strides[0],
                                                               padding_mode,
                                                               [params.pads[0], params.pads[num_input_pads]],
                                                               output_padding[0])
            return op_adapter.TransposeConv1dOp(src_op.name,
                                                bias_op_name=bias_op_name,
                                                stridex=params.strides[0],
                                                padx_before=pad_x[0],
                                                padx_after=pad_x[1],
                                                output_paddingx=params.output_padding[0],
                                                padding_size_strategy=padding_mode,
                                                output_width=params.output_shape[0],
                                                groups=params.group)
        elif input_buffer.rank() == 4:
            pad_x = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][1],
                                                               params.dilations[1],
                                                               params.strides[1],
                                                               padding_mode,
                                                               [params.pads[1], params.pads[1+num_input_pads]],
                                                               output_padding[1])
            pad_y = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][0],
                                                               params.dilations[0],
                                                               params.strides[0],
                                                               padding_mode,
                                                               [params.pads[0], params.pads[num_input_pads]],
                                                               output_padding[0])
            # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
            return op_adapter.TransposeConv2dOp(src_op.name,
                                                bias_op_name=bias_op_name,
                                                stridex=params.strides[1],
                                                stridey=params.strides[0],
                                                pady_before=pad_y[0],
                                                pady_after=pad_y[1],
                                                padx_before=pad_x[0],
                                                padx_after=pad_x[1],
                                                output_paddingx=params.output_padding[1],
                                                output_paddingy=params.output_padding[0],
                                                padding_size_strategy=padding_mode,
                                                output_height=params.output_shape[0],
                                                output_width=params.output_shape[1],
                                                groups=params.group)
        elif input_buffer.rank() == 5:
            pad_z, pad_y, pad_x = map(list, zip(params.pads[:3], params.pads[3:]))
            pad_x = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][2],
                                                               params.dilations[2],
                                                               params.strides[2],
                                                               padding_mode,
                                                               pad_x,
                                                               output_padding[2])
            pad_y = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][1],
                                                               params.dilations[1],
                                                               params.strides[1],
                                                               padding_mode,
                                                               pad_y,
                                                               output_padding[1])
            pad_z = ir_graph.TransposeConvOp.calc_padding_size(weights_shape[2:][0],
                                                               params.dilations[0],
                                                               params.strides[0],
                                                               padding_mode,
                                                               pad_z,
                                                               output_padding[0])
            # Note: For pads assumes 3D input where dimensions are NCDHW and DHW are the only spatial dims
            return op_adapter.TransposeConv3dOp(src_op.name,
                                                bias_op_name=bias_op_name,
                                                stridex=params.strides[2],
                                                stridey=params.strides[1],
                                                stridez=params.strides[0],
                                                padz_before=pad_z[0],
                                                padz_after=pad_z[1],
                                                pady_before=pad_y[0],
                                                pady_after=pad_y[1],
                                                padx_before=pad_x[0],
                                                padx_after=pad_x[1],
                                                dilationx=params.dilations[2],
                                                dilationy=params.dilations[1],
                                                dilationz=params.dilations[0],
                                                output_paddingx=params.output_padding[2],
                                                output_paddingy=params.output_padding[1],
                                                output_paddingz=params.output_padding[0],
                                                padding_size_strategy=padding_mode,
                                                output_depth=params.output_shape[0],
                                                output_height=params.output_shape[1],
                                                output_width=params.output_shape[2],
                                                groups=params.group)
        else:
            raise ValueError("Converter only supports 1D, 2D & 3D data for ConvTranspose, but received input rank = {}".format(input_buffer.rank()))

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'output_padding':
            log_assert(len(attr_value) <= 3,
                       code_to_message.get_error_message(
                           "ERROR_DECONV_OUTPUT_PADDING_LENGTH_UNSUPPORTED")
                       (len(attr_value)))


OnnxTranslations.register_translation(OnnxConvTransposeTranslation(),
                                      converter_type('ConvTranspose', 'onnx'),
                                      op_adapter.TransposeConv2dOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   FC
# ------------------------------------------------------------------------------
class OnnxFCTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        fc_op_name = graph.naming_policy.get_op_name(node_op)
        node_op.name = fc_op_name
        weights_name = input_names[1]
        bias_name = input_names[2]
        weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=self.weights)
        bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=self.bias)
        weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
        graph.add_src_op_info(bias_name, None, bias_node.output_names[0])

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)

        # Update weight/bias quantization info if it exists
        src_input_names = list(map(str, src_op.input))
        if graph.has_buffer(src_input_names[1]):
            graph.merge_quantization_params(graph.get_buffer(src_input_names[1]).producer.op.name,
                                            src_op.name, src_input_names[1], 'weights',
                                            encoding_type=QuantParams.PARAM_ENCODINGS)

        if len(src_input_names) > 2 and graph.has_buffer(src_input_names[2]):
                graph.merge_quantization_params(graph.get_buffer(src_input_names[2]).producer.op.name,
                                                src_op.name, src_input_names[2], 'bias',
                                                encoding_type=QuantParams.PARAM_ENCODINGS)

        return node

    def extract_parameters(self, src_op, converter_context):
        # Note: Schema is not used here since this op is not part of the Onnx spec.
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, attr_infos=
        [('axis', 'i', 1),
         ('axis_w', 'i', 1)])
        log_assert(params.axis == 1, code_to_message.get_error_message("ERROR_FC_AXIS_UNSUPPORTED"))
        log_assert(params.axis_w == 1,
                   code_to_message.get_error_message("ERROR_FC_AXIS_W_UNSUPPORTED"))

        input_names = graph.naming_policy.get_input_names(src_op, src_op.input)
        if len(input_names) == 2:
            weights = converter_context.weights.fetch(input_names[1])
            bias = np.zeros(weights.shape[1])
            self.bias_op_name = None
        else:
            weights, bias = converter_context.weights.fetch(*input_names[1:3])
            self.bias_op_name = input_names[2]
        self.weights = weights
        self.bias = bias
        self.weight_name = input_names[1]
        return op_adapter.FullyConnectedOp(src_op.name, bias_op_name=self.bias_op_name)

    def extract_input_names(self, src_op, converter_context):
        input_names = [str(_input) for _input in src_op.input]
        if len(input_names) == 2:
            input_names.append(self.bias_op_name)
        return input_names


OnnxTranslations.register_translation(OnnxFCTranslation(),
                                      converter_type('FC', 'onnx'),
                                      op_adapter.FullyConnectedOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GlobalAveragePool, GlobalMaxPool
# ------------------------------------------------------------------------------
class OnnxGlobalPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GlobalAveragePool', [1])
        self.register_op_schema('GlobalMaxPool', [1])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_buf = graph.get_buffer(str(src_op.input[0]))

        if str(src_op.op_type) == 'GlobalAveragePool':
            if input_buf.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_AVG_3D
            else:
                # Note: pool1d currently shares the same pool type as pool2d,
                # because there is no "QNN_OP_POOL_AVG_1D" in QnnOpDef.h
                pool_type = ir_graph.QNN_OP_POOL_AVG_2D
        else:
            if input_buf.rank() == 5:
                pool_type = ir_graph.QNN_OP_POOL_MAX_3D
            else:
                # Note: pool1d currently shares the same pool type as pool2d,
                # because there is no "QNN_OP_POOL_MAX_1D" in QnnOpDef.h
                pool_type = ir_graph.QNN_OP_POOL_MAX_2D

        if input_buf.rank() == 3:
            return op_adapter.Pool1dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_x=input_buf.shape[2],
                                       stride_x=input_buf.shape[2])
        elif input_buf.rank() == 4:
            return op_adapter.Pool2dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_x=input_buf.shape[3],
                                       size_y=input_buf.shape[2],
                                       stride_x=input_buf.shape[3],
                                       stride_y=input_buf.shape[2])
        elif input_buf.rank() == 5:
            return op_adapter.Pool3dOp(src_op.name,
                                       pool_type=pool_type,
                                       size_x=input_buf.shape[4],
                                       size_y=input_buf.shape[3],
                                       size_z=input_buf.shape[2],
                                       stride_x=input_buf.shape[4],
                                       stride_y=input_buf.shape[3],
                                       stride_z=input_buf.shape[2])
        else:
            raise ValueError("Global pool only supports 3D, 4D and 5D input, "
                             "but received input rank = {}".format(input_buf.rank()))


OnnxTranslations.register_translation(OnnxGlobalPoolTranslation(),
                                      converter_type('GlobalAveragePool', 'onnx'),
                                      converter_type('GlobalMaxPool', 'onnx'))


# ------------------------------------------------------------------------------
#   GridSample
# ------------------------------------------------------------------------------
class OnnxGridSampleTranslation(OnnxTranslationBase):
    SUPPORTED_MODES = ['bilinear', 'nearest']
    SUPPORTED_PADDING_MODES = ['zeros', 'border', 'reflection']

    onnx_to_ir_mode = {
        "bilinear": ir_graph.QNN_OP_GRID_SAMPLE_MODE_BILINEAR,
        "nearest": ir_graph.QNN_OP_GRID_SAMPLE_MODE_NEAREST,
    }

    onnx_to_ir_padding_mode = {
        "zeros": ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS,
        "border": ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER,
        "reflection": ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION,
    }

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GridSample', [16])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        grid_name = input_names[1]

        # handle constant grid to GridSample
        grid_const_op = self.fetch_constant_op(grid_name, converter_context, prunable=False, fail_if_dynamic=False)
        if grid_const_op and not graph.has_buffer(grid_name):
            grid_node = graph.add(grid_const_op, [], [grid_name], axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            graph.add_src_op_info(grid_name, None, grid_node.output_names[0])

        node = graph.add(op, input_names, output_names)
        graph.add_src_op_info(op.name, input_names, output_names)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        log_assert(params.mode in self.SUPPORTED_MODES,
                   "Unsupported mode {} set in GridSampleOp {}.",
                   (params.mode, src_op.name))

        log_assert(params.padding_mode in self.SUPPORTED_PADDING_MODES,
                   "Unsupported padding mode {} set in GridSampleOp {}.",
                   (params.padding_mode, src_op.name))

        return op_adapter.GridSampleOp(src_op.name,
                                       align_corners=params.align_corners,
                                       mode=self.onnx_to_ir_mode[params.mode],
                                       padding_mode=self.onnx_to_ir_padding_mode[params.padding_mode])


OnnxTranslations.register_translation(OnnxGridSampleTranslation(),
                                      converter_type('GridSample', 'onnx'),
                                      op_adapter.GridSampleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   InstanceNormalization
# ------------------------------------------------------------------------------
class OnnxInstanceNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('InstanceNormalization', [1, 6])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        bn_op_name = graph.naming_policy.get_op_name(node_op)
        node_op.name = bn_op_name

        src_input_names = list(map(str, src_op.input))
        weights_name = src_input_names[1]
        bias_name = src_input_names[2]
        weights_constant_op = self.fetch_constant_op(weights_name, converter_context, prunable=False, fail_if_dynamic=False)
        bias_constant_op = self.fetch_constant_op(bias_name, converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(weights_name):
            weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
        if bias_constant_op and not graph.has_buffer(bias_name):
            bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
        input_names.append(weights_name)
        input_names.append(bias_name)

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.InstanceNormOp(src_op.name,
                                         epsilon=params.epsilon,
                                         mode=ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA,
                                         region=ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxInstanceNormalizationTranslation(),
                                      converter_type('InstanceNormalization', 'onnx'),
                                      'instancenorm',
                                      op_adapter.InstanceNormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   LayerNormalization
# ------------------------------------------------------------------------------
class OnnxLayerNormTranslation(OnnxTranslationBase):
    def __init__(self):
        super(OnnxLayerNormTranslation, self).__init__()
        self.register_op_schema('LayerNormalization', [17])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        _, weights_name, bias_name = input_names
        weights_constant_op = self.fetch_constant_op(weights_name, converter_context, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(weights_name):
            weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(weights_name, None, weight_node.output_names[0])

        bias_constant_op = self.fetch_constant_op(bias_name, converter_context, prunable=False, fail_if_dynamic=False)
        if bias_constant_op and not graph.has_buffer(bias_name):
            bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.LayerNormOp(src_op.name,
                                      epsilon=params.epsilon,
                                      axes=[params.axis])


OnnxTranslations.register_translation(OnnxLayerNormTranslation(),
                                      converter_type('LayerNormalization', 'onnx'),
                                      'layernorm',
                                      op_adapter.LayerNormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   MaxRoiPool
# ------------------------------------------------------------------------------
class OnnxMaxRoiPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MaxRoiPool', [1])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        if len(ops) > 1:
            graph.add(ops[0], [], input_names[1])
        last_node = graph.add(ops[-1], input_names, output_names)

        # add src op info for roi pool operation
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        ops = []
        roi_name = input_names[1]

        if not graph.has_buffer(roi_name):
            roi_values = converter_context.weights.fetch(roi_name, prunable=False)
            roi_tensor = roi_values.astype(np.float32)
            roi_shape = roi_tensor.shape
            ops.append(op_adapter.ConstantOp(roi_name, roi_tensor))
        else:
            roi_shape = graph.get_buffer(roi_name).shape

        output_shape = [roi_shape[0],
                        input_buf.shape[1],
                        params.pooled_shape[0],
                        params.pooled_shape[1]]

        ops.append(op_adapter.RoiPoolingOp(src_op.name,
                                           output_shape,
                                           pooled_size_h=params.pooled_shape[0],
                                           pooled_size_w=params.pooled_shape[1],
                                           spatial_scale=params.spatial_scale))
        return ops

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxMaxRoiPoolTranslation(),
                                      converter_type('MaxRoiPool', 'onnx'),
                                      op_adapter.RoiPoolingOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Prelu, LeakyRelu
# ------------------------------------------------------------------------------
# Also handles LeakyRelu as a bonus.
class OnnxPreluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('PRelu', [1, 6, 7, 9])
        self.register_op_schema('LeakyRelu', [1, 6])
        self.coeff = None

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        prelu_op_name = graph.naming_policy.get_op_name(node_op)
        node_op.name = prelu_op_name
        if str(src_op.op_type) == 'LeakyRelu':
            coeff_name = prelu_op_name + "_coeff"
            input_names.append(coeff_name)
        else:
            coeff_name = input_names[1]

        if not graph.has_buffer(coeff_name):
            coeff_constant_op = op_adapter.ConstantOp(coeff_name, tensor=self.coeff)
            coeff_node = graph.add(coeff_constant_op, [], [coeff_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add_src_op_info(coeff_name, None, coeff_node.output_names[0])

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        if str(src_op.op_type) == 'LeakyRelu':
            params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type),
                                        validate=True)
            slope = np.array([params.alpha], dtype=np.float32)
            self.coeff = slope
        else:
            if graph.has_buffer(input_names[1]): # Checking for dynamic alpha input
                self.coeff = None
                dynamic = True
                slope = graph.get_buffer(input_names[1])
            else:
                dynamic = False
                slope = converter_context.weights.fetch(input_names[1])
            if not dynamic and len(slope.shape) == 1:
                slope = np.ones(input_buf.shape[1], dtype=np.float32) * slope[0]
                self.coeff = slope
            else:
                rank_diff = len(slope.shape) - len(input_buf.shape)
                if rank_diff < 0:
                    # Prepending 1's to slope shape and then broadcasting to match input rank
                    slope_shape = [1] * abs(rank_diff) + list(slope.shape)
                    if not dynamic:
                        slope = np.broadcast_to(slope, slope_shape)
                        slope = np.require(slope, dtype=np.float32)
                        self.coeff = slope
                    else:
                        slope.shape = slope_shape

        return op_adapter.PreluOp(src_op.name)


OnnxTranslations.register_translation(OnnxPreluTranslation(),
                                      converter_type('Prelu', 'onnx'),
                                      converter_type('LeakyRelu', 'onnx'),
                                      op_adapter.PreluOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Lrn
# ------------------------------------------------------------------------------
class OnnxLrnTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LRN', [1, 13])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.LrnOp(src_op.name,
                                alpha=params.alpha / params.size,
                                beta=params.beta,
                                bias=params.bias,
                                radius=int((params.size-1)/2),
                                region=ir_graph.QNN_OP_LRN_REGION_ACROSS_CHANNEL)


OnnxTranslations.register_translation(OnnxLrnTranslation(),
                                      converter_type('LRN', 'onnx'),
                                      op_adapter.LrnOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   NonMaxSuppression
# ------------------------------------------------------------------------------
class OnnxNonMaxSuppressionTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('NonMaxSuppression', [10, 11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if params.center_point_box:
            raise ValueError("NonMaxSuppression: Only support minmax representation (y1,x1,y2,x2)")

        # Require static parameters
        input_names = list(map(str, src_op.input))
        if (len(input_names) > 2 and
            any([graph.has_buffer(buf) and not self.fetch_constant_op(buf, converter_context)
                 for buf
                 in input_names[2:]
                ]
               )
           ):
            raise ValueError(
                ('NonMaxSuppression: Only supports static parameters for '
                'max_output_boxes_per_class, iou_threshold, score_threshold')
            )
        max_output_boxes_per_class, iou_threshold, score_threshold = self.fetch_const_input(src_op, converter_context)
        # If max_output_boxes_per_class is negative, it will be set to input number of boxes
        if max_output_boxes_per_class < 0:
            max_output_boxes_per_class = graph.get_buffer(input_names[0]).shape[1]
        return op_adapter.NonMaxSuppressionOp(
            name=src_op.name,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_boxes_selected=max_output_boxes_per_class,
        )

    def fetch_const_input(self, src_op, converter_context):
        # per onnx spec
        max_output_boxes_per_class = 0
        iou_threshold = 0
        score_threshold = 0

        # Handle optional parameters
        input_names = list(map(str, src_op.input))
        if len(input_names) > 2 and input_names[2]:
            max_output_boxes_per_class = self.fetch_constant_op(input_names[2], converter_context).tensor
            max_output_boxes_per_class = int(max_output_boxes_per_class.item(0))
        if len(input_names) > 3 and input_names[3]:
            iou_threshold = self.fetch_constant_op(input_names[3], converter_context).tensor
            iou_threshold = iou_threshold.item(0)
        if len(input_names) > 4 and input_names[4]:
            score_threshold = self.fetch_constant_op(input_names[4], converter_context).tensor
            score_threshold = score_threshold.item(0)

        return max_output_boxes_per_class, iou_threshold, score_threshold

    def extract_input_names(self, src_op, converter_context):
        actual_input_name = []
        for inp in map(str, src_op.input):
            if not (inp and converter_context.weights.has(inp)):
                actual_input_name.append(inp)
        return actual_input_name

    def extract_output_names(self, src_op, converter_context):
        # align with QNN's second output to specify the valid number of selected indices
        output_name = str(src_op.output[0])
        return [output_name, output_name + '_valid_num_selected_indices']


OnnxTranslations.register_translation(OnnxNonMaxSuppressionTranslation(),
                                      converter_type('NonMaxSuppression', 'onnx'),
                                      op_adapter.NonMaxSuppressionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ROI Align
# ------------------------------------------------------------------------------
class OnnxRoiAlignTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RoiAlign', [10, 16])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        if len(ops) > 1:
            batch_indices_node = graph.add(ops[0], [], input_names[2])
            # add src_op info for added constant op
            graph.add_src_op_info(batch_indices_node.op.name, None,
                                  batch_indices_node.output_names[0])

        last_node = graph.add(ops[-1], input_names, output_names)
        # add src op info for roi align operation
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        params = extract_attributes(src_op, schema=schema)
        # coordinate_transformation_mode is a new attribute added in version 16.
        # It's default value is half_pixel but we only provide backward compatibility for now.
        if 'coordinate_transformation_mode' in params and params.coordinate_transformation_mode != "output_half_pixel":
            raise ValueError("ERROR: Unsupported value {} for coordinate_transformation_mode attribute for {} op".format(params.coordinate_transformation_mode, src_op.name))
        ops = []
        indices_name = str(src_op.input[2])
        # If the input is stored as weights we need to create a const node
        if not graph.has_buffer(indices_name):
            indices = self.fetch_constant_op(indices_name, converter_context,
                                             prunable=False, dtype=np.int32)
            ops.append(indices)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op

            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(np.int32)
        log_assert(params.spatial_scale != 0, "Unsupported value of 0 for spatial_scale attribute for {} op.".format(src_op.name))
        ops.append(op_adapter.RoiAlignOp(src_op.name,
                                         pooled_size_h=params.output_height,
                                         pooled_size_w=params.output_width,
                                         spatial_scale=(1.0/params.spatial_scale),
                                         sampling_ratio=params.sampling_ratio,
                                         mode=params.mode))
        return ops


OnnxTranslations.register_translation(OnnxRoiAlignTranslation(),
                                      converter_type('RoiAlign', 'onnx'),
                                      op_adapter.RoiAlignOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   SpaceToDepth
# ------------------------------------------------------------------------------
class OnnxSpaceToDepthTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('SpaceToDepth', [1, 13])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.SpaceToDepthOp(
            name=src_op.name,
            block_size=[params.blocksize] * 2
        )

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxSpaceToDepthTranslation(),
                                      converter_type('SpaceToDepth', 'onnx'),
                                      op_adapter.SpaceToDepthOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class OnnxDepthToSpaceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('DepthToSpace', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        SUPPORTED_DEPTHTOSPACE_MODES = {'DCR': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR,
                                        'CRD': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD}
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if not 'mode' in params:
            mode = 'DCR'
        elif params.mode not in SUPPORTED_DEPTHTOSPACE_MODES:
            raise ValueError("Unsupported depthtospace mode {}".format(params.mode))
        else:
            mode = params.mode
        return op_adapter.DepthToSpaceOp(name=src_op.name,
                                         block_size=[params.blocksize] * 2,
                                         mode=SUPPORTED_DEPTHTOSPACE_MODES[mode])

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxDepthToSpaceTranslation(),
                                      converter_type('DepthToSpace', 'onnx'),
                                      op_adapter.DepthToSpaceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ThresholdedRelu
#   TODO: Add unit test for this op
# ------------------------------------------------------------------------------
class OnnxThresholdedReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ThresholdedRelu', [10])

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        node_op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        node_op.name = graph.naming_policy.get_op_name(node_op)

        node = graph.add(node_op, input_names, output_names)
        self.add_src_op_info(node_op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        return op_adapter.ThresholdedReluOp(src_op.name, alpha=params.alpha)

OnnxTranslations.register_translation(OnnxThresholdedReluTranslation(),
                                      converter_type('ThresholdedRelu', 'onnx'),
                                      op_adapter.ThresholdedReluOp.TRANSLATION_KEY)

