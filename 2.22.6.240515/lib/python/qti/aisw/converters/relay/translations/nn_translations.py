# ==============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.utils.converter_utils import (
    converter_type,
    log_assert,
    log_debug1,
    log_debug2,
    log_debug3,
)
from qti.aisw.converters.common.converter_ir.op_adapter import (
    BatchnormOp,
    ChannelShuffleOp,
    ConstantOp,
    Conv1dOp,
    Conv2dOp,
    Conv3dOp,
    DepthToSpaceOp,
    DepthwiseConv1dOp,
    DepthwiseConv2dOp,
    DetectionOutputOp,
    ElementwiseBinaryOp,
    FullyConnectedOp,
    GroupNormOp,
    IdentityOp,
    InstanceNormOp,
    LayerNormOp,
    LogSoftmaxOp,
    LrnOp,
    MatMulOp,
    ElementwiseNeuronOp,
    PadOp,
    Pool1dOp,
    Pool2dOp,
    Pool3dOp,
    PreluOp,
    ResizeOp,
    ReshapeOp,
    SoftmaxOp,
    SpaceToDepthOp,
    TransposeConv2dOp,
    TransposeOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase, validate_const_name
from qti.aisw.converters.relay.translations import RelayTranslations
from qti.aisw.converters.relay.utils import get_prim_type

import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type


# ------------------------------------------------------------------------------
#   Adaptive Pool Base
# ------------------------------------------------------------------------------
class RelayAdaptivePoolBaseTranslation(RelayTranslationBase):
    def __init__(self, pool_type):
        super(RelayAdaptivePoolBaseTranslation, self).__init__()
        self.pool_type = pool_type

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        adaptive_pool_attr = relay_expr.attrs
        attr_dict['layout'] = adaptive_pool_attr.layout
        attr_dict["output_size"] = adaptive_pool_attr.output_size if hasattr(adaptive_pool_attr, 'output_size') else None

        log_debug3("\tlayout {}", attr_dict['layout'])
        log_debug3("\toutput_size {}", attr_dict['output_size'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, Pool2dOp.TRANSLATION_KEY, Pool2dOp.LEGACY_TRANSLATION_KEY)
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        data_layout = attr_dict['layout']
        output_size = attr_dict['output_size']

        spatial_rank = len(input_shape) - 2

        if len(input_shape) not in [3, 4, 5]:
            raise ValueError("No support {}D Input".format(len(input_shape)))

        if data_layout not in ["NWC", "NHWC", "NDHWC"]:
            raise ValueError("No support {} data layout".format(data_layout))

        x_axis = -2
        stride_x, size_x = self.get_stride_and_size(input_shape, x_axis, output_size)

        if len(input_shape) == 3:
            return Pool1dOp(op_name,
                            pool_type=self.pool_type,
                            stride_x=stride_x,
                            size_x=size_x)

        y_axis = -3
        stride_y, size_y = self.get_stride_and_size(input_shape, y_axis, output_size)

        if len(input_shape) == 4:
            return Pool2dOp(op_name,
                            pool_type=self.pool_type,
                            stride_x=stride_x,
                            stride_y=stride_y,
                            size_x=size_x,
                            size_y=size_y)
        z_axis = -4
        stride_z, size_z = self.get_stride_and_size(input_shape, z_axis, output_size)

        return Pool3dOp(op_name,
                        pool_type=self.pool_type,
                        stride_x=stride_x,
                        stride_y=stride_y,
                        stride_z=stride_z,
                        size_x=size_x,
                        size_y=size_y,
                        size_z=size_z)

    def get_stride_and_size(self, input_shape, axis, output_size):
        if output_size is None:
            stride = 1
            size = input_shape[axis]
        else:
            input_size = input_shape[axis]
            output_size = int(output_size[axis+1])
            stride = int(input_size / output_size)
            size = input_size - (output_size - 1) * stride
        return stride, size


# ------------------------------------------------------------------------------
#   Adaptive Average Pool1d
# ------------------------------------------------------------------------------
class RelayAdaptiveAvgPool1dTranslation(RelayAdaptivePoolBaseTranslation):
    def __init__(self):
        super(RelayAdaptiveAvgPool1dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_AVG_2D)


RelayTranslations.register_translation(RelayAdaptiveAvgPool1dTranslation(),
                                       converter_type('adaptive_avg_pool1d', 'relay'))


# ------------------------------------------------------------------------------
#   Adaptive Max Pool1d
# ------------------------------------------------------------------------------
class RelayAdaptiveMaxPool1dTranslation(RelayAdaptivePoolBaseTranslation):
    def __init__(self):
        super(RelayAdaptiveMaxPool1dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_MAX_2D)


RelayTranslations.register_translation(RelayAdaptiveMaxPool1dTranslation(),
                                       converter_type('adaptive_max_pool1d', 'relay'))


# ------------------------------------------------------------------------------
#   Adaptive Average Pool2d
# ------------------------------------------------------------------------------
class RelayAdaptiveAvgPool2dTranslation(RelayAdaptivePoolBaseTranslation):
    def __init__(self):
        super(RelayAdaptiveAvgPool2dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_AVG_2D)


RelayTranslations.register_translation(RelayAdaptiveAvgPool2dTranslation(),
                                       converter_type('adaptive_avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   Adaptive Max Pool2d
# ------------------------------------------------------------------------------
class RelayAdaptiveMaxPool2dTranslation(RelayAdaptivePoolBaseTranslation):
    def __init__(self):
        super(RelayAdaptiveMaxPool2dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_MAX_2D)


RelayTranslations.register_translation(RelayAdaptiveMaxPool2dTranslation(),
                                       converter_type('adaptive_max_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   BatchMatMul
# ------------------------------------------------------------------------------
class RelayBatchMatMulTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBatchMatMulTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):

        attr_dict = {}
        batch_matmul_attrs = relay_expr.attrs

        attr_dict["transpose_in0"] = batch_matmul_attrs.transpose_a
        attr_dict["transpose_in1"] = batch_matmul_attrs.transpose_b

        log_debug3("\ttranspose_in0 {}", attr_dict["transpose_in0"])
        log_debug3("\ttranspose_in1 {}", attr_dict["transpose_in1"])

        return  attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, MatMulOp.TRANSLATION_KEY, MatMulOp.LEGACY_TRANSLATION_KEY)
        if not quir_graph.has_buffer(input_names[0]):
            const_tensor = relay_params[input_names[0]]
            if isinstance(const_tensor, tvm.runtime.ndarray.NDArray) or isinstance(const_tensor, tvm.runtime.NDArray):
                const_tensor = const_tensor.asnumpy()
            const_op = ConstantOp(input_names[0], tensor=const_tensor)
            self.populate_quantization_params(relay_expr.args[0], converter_context, quir_graph, [input_names[0]], is_param=True)
            quir_graph.add(const_op, [], [input_names[0]], axis_formats=[AxisTracker.AxisFormat.ANY])
        if not quir_graph.has_buffer(input_names[1]):
            const_tensor = relay_params[input_names[1]]
            if isinstance(const_tensor, tvm.runtime.ndarray.NDArray) or isinstance(const_tensor, tvm.runtime.NDArray):
                const_tensor = const_tensor.asnumpy()
            const_op = ConstantOp(input_names[1], tensor=const_tensor)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [input_names[1]], is_param=True)
            quir_graph.add(const_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.ANY])
        ir_op = MatMulOp(op_name,
                         transpose_in0=attr_dict["transpose_in0"],
                         transpose_in1=attr_dict["transpose_in1"])
        return ir_op


RelayTranslations.register_translation(RelayBatchMatMulTranslation(),
                                       converter_type('matmul', 'relay'),
                                       converter_type('batch_matmul', 'relay'))


# ------------------------------------------------------------------------------
#   BatchNorm
# ------------------------------------------------------------------------------
class RelayBatchNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBatchNormTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):

        attr_dict = {}
        batchnorm_attrs = relay_expr.attrs

        attr_dict["epsilon"] = batchnorm_attrs.epsilon if hasattr(batchnorm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = batchnorm_attrs.center if hasattr(batchnorm_attrs, 'center') else True
        attr_dict["scale"] = batchnorm_attrs.scale if hasattr(batchnorm_attrs, 'scale') else True
        attr_dict["axis"] = batchnorm_attrs.axis if hasattr(batchnorm_attrs, 'axis') else 1

        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        log_debug3("\taxis {}", attr_dict["axis"])

        return  attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, BatchnormOp.TRANSLATION_KEY,
                                                BatchnormOp.LEGACY_TRANSLATION_KEY)

        gamma = relay_params[input_names[1]].asnumpy()
        beta = relay_params[input_names[2]].asnumpy()
        moving_mean = relay_params[input_names[3]].asnumpy()
        moving_var = relay_params[input_names[4]].asnumpy()

        log_debug3("\tgamma shape {}", gamma.shape)
        log_debug3("\tbeta shape {}", beta.shape)
        log_debug3("\tmoving_mean shape {}", moving_mean.shape)
        log_debug3("\tmoving_var shape {}", moving_var.shape)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        if attr_dict["axis"] != len(input_shape)-1:
            raise ValueError("Batchnorm channel is dimension {}, got {}".format(len(input_shape)-1, attr_dict["axis"]))

        center = attr_dict["center"]
        scale = attr_dict["scale"]
        epsilon = attr_dict["epsilon"]

        self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [input_names[1]], is_param=True)
        self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
        gamma_quant_enc = quir_graph.get_overridden_encoding(input_names[1])
        beta_quant_enc = quir_graph.get_overridden_encoding(input_names[2])
        if gamma_quant_enc:
            quantized_gamma = translation_utils.quantize_params(gamma, gamma_quant_enc[0])
            gamma = translation_utils.dequantize_params(quantized_gamma, gamma_quant_enc[0])
            # remove gamma encodings since already applied
            quir_graph.remove_overridden_encoding(input_names[1])
        if beta_quant_enc:
            quantized_beta = translation_utils.quantize_params(beta, beta_quant_enc[0])
            beta = translation_utils.dequantize_params(quantized_beta, beta_quant_enc[0])
            # remove beta encodings since already applied
            quir_graph.remove_overridden_encoding(input_names[2])

        # weights = gamma/sqrt(var+epsilon)
        weights = gamma / np.sqrt(moving_var + epsilon)
        # bias = -mu/sqrt(var+epsilon)
        bias = -moving_mean / np.sqrt(moving_var + epsilon)
        if scale:
            # bias = -mu*gamma/sqrt(var+epsilon)
            bias *= gamma
        if center:
            # bias = -mu/sqrt(var+epsilon) + beta or bias = -mu*gamma/sqrt(var+epsilon) + beta
            bias += beta

        weights_name = op_name + "_bn_w"
        bias_name = op_name + "_bn_b"
        weights_constant_op = ConstantOp(weights_name, tensor=weights)
        bias_constant_op = ConstantOp(bias_name, tensor=bias)
        weight_node = quir_graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        bias_node = quir_graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        quir_graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
        quir_graph.add_src_op_info(bias_name, None, bias_node.output_names[0])

        ir_op = BatchnormOp(op_name)

        for name in input_names[1:]:
            input_names.remove(name)
        input_names.append(weights_name)
        input_names.append(bias_name)
        return ir_op


RelayTranslations.register_translation(RelayBatchNormTranslation(),
                                       converter_type('batch_norm', 'relay'))


# ------------------------------------------------------------------------------
#   BiasAdd
# ------------------------------------------------------------------------------
class RelayBiasaddTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBiasaddTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, ir_graph.QNN_OP_ELEMENT_WISE_ADD,
                                                ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD])

        bias = relay_params[input_names[1]]
        if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
            bias = bias.asnumpy().astype(np.float32)

        log_debug3("\tbias shape {}", bias.shape)

        bias_name = op_name + "_const_bias"
        bias_name = validate_const_name(quir_graph, input_names[1], bias_name)
        input_names[1] = bias_name
        if not quir_graph.has_buffer(bias_name):
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [bias_name], is_param=True)
            quir_graph.add(ConstantOp(bias_name, bias), [], [bias_name])

        ir_op = ElementwiseBinaryOp(op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)

        return ir_op


RelayTranslations.register_translation(RelayBiasaddTranslation(),
                                       converter_type('bias_add', 'relay'))


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class RelayChannelShuffleTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayChannelShuffleTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["groups"] = int(relay_expr.attrs.groups)
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ChannelShuffleOp.TRANSLATION_KEY,
                                                ChannelShuffleOp.LEGACY_TRANSLATION_KEY)
        return ChannelShuffleOp(op_name, num_groups=attr_dict["groups"])

RelayTranslations.register_translation(RelayChannelShuffleTranslation(),
                                       converter_type("channel_shuffle", "relay"))


# ------------------------------------------------------------------------------
#   Conv Base
# ------------------------------------------------------------------------------
class RelayConvBaseTranslation(RelayTranslationBase):
    def __init__(self, input_rank):
        self.input_rank = input_rank
        super(RelayConvBaseTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):

        attr_dict = {}

        conv_attrs = relay_expr.attrs

        if len(conv_attrs.data_layout) not in [3, 4, 5]:
            raise ValueError("No support {}D Input".format(len(conv_attrs.data_layout)))

        if conv_attrs.data_layout not in ["NWC", "NHWC", "NDHWC"]:
            raise ValueError("No support {} data layout".format(conv_attrs.data_layout))

        spatial_rank = len(conv_attrs.data_layout) - 2

        attr_dict["kernel_layout"] = conv_attrs.kernel_layout
        log_debug3("\tkernel_layout {}", conv_attrs.kernel_layout)

        # repeat the padding if needed, take Conv2d as an example
        # [1] -> [1, 1, 1, 1]
        # [1, 2] -> [1, 2, 1, 2]
        # [1, 2, 3, 4] -> [1, 2, 3, 4]
        padding = [conv_attrs.padding] if isinstance(conv_attrs.padding, int) else [int(val) for val in conv_attrs.padding]
        padding = padding * (2 * spatial_rank // len(padding))
        log_debug3("\tpadding {}", padding)

        strides = [int(val) for val in conv_attrs.strides]
        log_debug3("\tstrides {}", strides)

        dilation = [int(val) for val in conv_attrs.dilation]
        log_debug3("\tdilation {}", dilation)

        # z -> depth
        # y -> height
        # x -> width
        x_axis = -1
        attr_dict["padx_before"] = int(padding[x_axis-spatial_rank])
        attr_dict["padx_after"] = int(padding[x_axis])
        attr_dict["stridex"] = strides[x_axis]
        attr_dict["dilationx"] = dilation[x_axis]

        if spatial_rank >= 2:
            y_axis = -2
            attr_dict["pady_before"] = int(padding[y_axis-spatial_rank])
            attr_dict["pady_after"] = int(padding[y_axis])
            attr_dict["stridey"] = strides[y_axis]
            attr_dict["dilationy"] = dilation[y_axis]

        if spatial_rank >= 3:
            z_axis = -3
            attr_dict["padz_before"] = int(padding[z_axis-spatial_rank])
            attr_dict["padz_after"] = int(padding[z_axis])
            attr_dict["stridez"] = strides[z_axis]
            attr_dict["dilationz"] = dilation[z_axis]

        attr_dict["padding_size_strategy"] = ir_graph.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        groups = int(conv_attrs.groups)
        log_debug3("\tgroups {}", groups)
        attr_dict["groups"] = groups

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        kernel_layout = attr_dict["kernel_layout"]

        self.validate_kernel_layout(kernel_layout)

        if self.input_rank == 3:
            extract_spatial_dims = quir_graph.src_axis_order.extract_1d_spatial_dims
        elif self.input_rank == 4:
            extract_spatial_dims = quir_graph.src_axis_order.extract_2d_spatial_dims
        elif self.input_rank == 5:
            extract_spatial_dims = quir_graph.src_axis_order.extract_3d_spatial_dims

        num_input_channels = extract_spatial_dims(
            quir_graph.get_buffer(input_names[0]).shape)[-1]
        num_output_channels = run_infer_type(relay_expr.args[1]).checked_type.shape[kernel_layout.find('O')]

        is_depthwise = attr_dict["groups"] == num_input_channels and num_input_channels == num_output_channels

        if self.input_rank == 3:
            convolution_class = DepthwiseConv1dOp if is_depthwise else Conv1dOp
            target_weight_axis_format = AxisTracker.AxisFormat.FIO

        elif self.input_rank == 4:
            convolution_class = DepthwiseConv2dOp if is_depthwise else Conv2dOp
            target_weight_axis_format = AxisTracker.AxisFormat.HWIO

        elif self.input_rank == 5:
            # TODO: DepthwiseConv3dOp hasn't been implemented yet
            convolution_class = Conv3dOp
            target_weight_axis_format = AxisTracker.AxisFormat.DHWIO

        op_name = converter_context.get_op_name(relay_expr, convolution_class.TRANSLATION_KEY,
                                                convolution_class.LEGACY_TRANSLATION_KEY)

        perms = {"WOI": AxisTracker.AxisFormat.FOI_TO_FIO,
                 "HWOI": AxisTracker.AxisFormat.HWOI_TO_HWIO,
                 "DHWOI": AxisTracker.AxisFormat.DHWOI_TO_DHWIO}

        if quir_graph.has_buffer(input_names[1]):
            weights_buf = quir_graph.get_buffer(input_names[1])
            log_debug3("\tweights shape {}", weights_buf.shape)

            if kernel_layout in perms.keys():
                log_debug3("\t{} kernel layout with shape {} detected, "
                        "Transposing the weights to make it out-channel last.".format(kernel_layout, weights_buf.shape))
                if weights_buf.producer.op.type == ConstantOp.TRANSLATION_KEY:
                    weights = weights_buf.producer.op.tensor
                    weights = np.transpose(weights, perms[kernel_layout])
                    weights = np.ascontiguousarray(weights)
                    weights_buf.producer.op.tensor = weights
                    weights_buf.shape = list(weights.shape)
                else:
                    transpose_op_name = input_names[1] + '.' + target_weight_axis_format.lower()
                    transpose_op = TransposeOp(transpose_op_name, perm=perms[kernel_layout])
                    quir_graph.add(transpose_op, [input_names[1]], [transpose_op_name], axis_formats=[target_weight_axis_format])
                    input_names[1] = transpose_op_name

        elif input_names[1] in relay_params:
            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()
            log_debug3("\tweights shape {}", weights.shape)

            if kernel_layout in perms.keys():
                log_debug3("\t{} kernel layout with shape {} detected, "
                        "Transposing the weights to make it out-channel last.".format(kernel_layout, weights.shape))
                weights = np.transpose(weights, perms[kernel_layout])
                weights = np.ascontiguousarray(weights)
                log_debug3("\tTransposed weights to be of shape {}", weights.shape)

            weight_name = op_name + "_const_weight"
            weight_name = validate_const_name(quir_graph, input_names[1], weight_name)
            input_names[1] = weight_name # Update input_names[1] for relay_translations

            weights_op = ConstantOp(input_names[1], tensor=weights)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [weights_op.name], is_param=True)
            quir_graph.add(weights_op, [], [input_names[1]], axis_formats=[target_weight_axis_format])

        else:
            raise ValueError("weights not found!")

        # TODO: DepthwiseConv3dOp hasn't been implemented yet
        if is_depthwise and convolution_class != Conv3dOp:
            weights_buf = quir_graph.get_buffer(input_names[1])
            log_debug3("\tReshaping depthwise convolution weights of shape {}", weights_buf.shape)
            if weights_buf.producer.op.type == ConstantOp.TRANSLATION_KEY:
                weights_buf.producer.op.tensor = np.reshape(weights_buf.producer.op.tensor, (weights_buf.shape[0], weights_buf.shape[1], 1, -1))
                weights_buf.shape = list(weights_buf.producer.op.tensor.shape)
            else:
                reshape_op_name = weights_buf.producer.op.name + '_reshape'
                reshape_op = ReshapeOp(reshape_op_name, shape=(weights_buf.shape[0], weights_buf.shape[1], 1, -1))
                quir_graph.add(reshape_op, [weights_buf.name], [reshape_op_name], axis_formats=[target_weight_axis_format])
                input_names[1] = reshape_op_name
            log_debug3("\tReshaped depthwise convolution weights to shape {}", weights_buf.shape)

        ir_op = convolution_class(op_name,
                                  **attr_dict)

        return ir_op

    def validate_kernel_layout(self, kernel_layout):
        valid_kernel_layout = ["WIO", "WOI",
                               "HWIO", "HWOI",
                               "DHWIO", "DHWOI"]
        if kernel_layout in valid_kernel_layout:
            return
        else:
            raise ValueError("Unsupported kernel layout {}".format(kernel_layout))


# ------------------------------------------------------------------------------
#   Conv1d
# ------------------------------------------------------------------------------
class RelayConv1dTranslation(RelayConvBaseTranslation):
    def __init__(self):
        super(RelayConv1dTranslation, self).__init__(input_rank=3)


RelayTranslations.register_translation(RelayConv1dTranslation(),
                                       converter_type('conv1d', 'relay'))


# ------------------------------------------------------------------------------
#   Conv2d
# ------------------------------------------------------------------------------
class RelayConv2dTranslation(RelayConvBaseTranslation):
    def __init__(self):
        super(RelayConv2dTranslation, self).__init__(input_rank=4)


RelayTranslations.register_translation(RelayConv2dTranslation(),
                                       converter_type('conv2d', 'relay'))


# ------------------------------------------------------------------------------
#   Conv3d
# ------------------------------------------------------------------------------
class RelayConv3dTranslation(RelayConvBaseTranslation):
    def __init__(self):
        super(RelayConv3dTranslation, self).__init__(input_rank=5)


RelayTranslations.register_translation(RelayConv3dTranslation(),
                                       converter_type('conv3d', 'relay'))


# ------------------------------------------------------------------------------
#   Conv2D_Transpose
# ------------------------------------------------------------------------------
class RelayConvTransposeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConvTransposeTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        conv_attrs = relay_expr.attrs

        log_debug3("\tdata layout {}", conv_attrs.data_layout)
        if conv_attrs.data_layout != "NHWC":
            # QUIR expects data to be "NHWC"
            raise ValueError("Unsupported data layout {}".format(conv_attrs.data_layout))

        log_debug3("\tkernel layout {}", conv_attrs.kernel_layout)
        if conv_attrs.kernel_layout not in ["IOHW", "HWIO"]:
            raise ValueError("Unsupported kernel layout {}".format(conv_attrs.kernel_layout))
        attr_dict["kernel_layout"] = conv_attrs.kernel_layout

        log_debug3("\tout layout {}", conv_attrs.out_layout)
        if conv_attrs.out_layout != "":
            # This attribute is not supported, so only empty/default is accepted
            raise ValueError("Unsupported out layout {}".format(conv_attrs.out_layout))

        log_debug3("\tout dtype {}", conv_attrs.out_dtype)
        if conv_attrs.out_dtype not in ["float32", ""]:
            # Only float32 is currently supported
            raise ValueError("Unsupported out dtype {}".format(conv_attrs.out_dtype))

        padding = [int(val) for val in conv_attrs.padding]
        log_debug3("\tpadding {}", padding)
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_left = padding[1]
            pad_bottom = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pad_top"] = pad_top
        attr_dict["pad_bottom"] = pad_bottom
        attr_dict["pad_left"] = pad_left
        attr_dict["pad_right"] = pad_right

        attr_dict["padding_size_strategy"] = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        strides = [int(val) for val in conv_attrs.strides]
        log_debug3("\tstrides {}", strides)
        # y -> height
        # x -> width
        stride_y = strides[0]
        stride_x = strides[1]
        attr_dict["stride_x"] = stride_x
        attr_dict["stride_y"] = stride_y

        dilation = [int(val) for val in conv_attrs.dilation]
        log_debug3("\tdilation {}", dilation)
        # y -> height
        # x -> width
        dilation_y = dilation[0]
        dilation_x = dilation[1]
        attr_dict["dilation_x"] = dilation_x
        attr_dict["dilation_y"] = dilation_y

        groups = int(conv_attrs.groups)
        log_debug3("\tgroups {}", groups)
        attr_dict["groups"] = groups

        output_padding = conv_attrs.output_padding
        log_debug3("\toutput padding {}", conv_attrs.output_padding)
        # FIXME: This attribute can have 1, 2, or 4 numbers.
        # refer to tvm_src/include/tvm/relay/attrs/nn.h, Conv2DTransposeAttrs.
        attr_dict["output_padding_y"] = output_padding[0]
        attr_dict["output_padding_x"] = output_padding[1]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TransposeConv2dOp.TRANSLATION_KEY,
                                                TransposeConv2dOp.LEGACY_TRANSLATION_KEY)

        kernel_layout = attr_dict["kernel_layout"]
        if quir_graph.has_buffer(input_names[1]):
            if kernel_layout == 'IOHW':
                transpose_op_name = input_names[1] + '.hwio'
                if not quir_graph.has_buffer(transpose_op_name):
                    transpose_op = TransposeOp(transpose_op_name, AxisTracker.AxisFormat.IOHW_TO_HWIO)
                    quir_graph.add(transpose_op, [input_names[1]], [transpose_op_name], axis_formats=[AxisTracker.AxisFormat.HWIO])

                input_names[1] = transpose_op_name

                weights_buf = quir_graph.get_buffer(input_names[1])
                log_debug3("\ttransposed deconv weights to {}", weights_buf.shape)
        elif input_names[1] in relay_params:
            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()
            log_debug3("\tweights shape {}", weights.shape)

            if kernel_layout == 'IOHW':
                weights = np.transpose(weights, AxisTracker.AxisFormat.IOHW_TO_HWIO)

            weights = np.ascontiguousarray(weights)
            log_debug3("\ttransposed deconv weights to {}", weights.shape)

            weights_op = ConstantOp(input_names[1], tensor=weights)
            quir_graph.add(weights_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.HWIO])
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [weights_op.name], is_param=True)
        else:
            raise ValueError("weights not found for conv2d_transpose op")

        if len(input_names) > 2:
            if input_names[2] not in relay_params:
                raise ValueError("Unsupported dynamic biases on tensor {}".format(input_names[2]))
            bias = relay_params[input_names[2]]
            if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
                bias = bias.asnumpy().astype(np.float32)
            log_debug3("\tbias shape {}", bias.shape)
            bias_op = ConstantOp(input_names[2], tensor=bias)
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
            quir_graph.add(bias_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        ir_op = TransposeConv2dOp(op_name,
                                  padx_before=attr_dict["pad_left"],
                                  padx_after=attr_dict["pad_right"],
                                  pady_before=attr_dict["pad_top"],
                                  pady_after=attr_dict["pad_bottom"],
                                  stridex=attr_dict["stride_x"],
                                  stridey=attr_dict["stride_y"],
                                  dilationx=attr_dict["dilation_x"],
                                  dilationy=attr_dict["dilation_y"],
                                  output_paddingx=attr_dict["output_padding_x"],
                                  output_paddingy=attr_dict["output_padding_y"],
                                  groups=attr_dict["groups"],
                                  padding_size_strategy=attr_dict["padding_size_strategy"])

        return ir_op


RelayTranslations.register_translation(RelayConvTransposeTranslation(),
                                       converter_type('conv2d_transpose', 'relay'))


# ------------------------------------------------------------------------------
#   Dense
# ------------------------------------------------------------------------------
class RelayDenseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDenseTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        if input_names[1] in relay_params:
            op_name = converter_context.get_op_name(relay_expr, FullyConnectedOp.TRANSLATION_KEY,
                                                    FullyConnectedOp.LEGACY_TRANSLATION_KEY)

            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()

            # Weights has shape [out_units, in_units]
            if not quir_graph.has_buffer(input_names[1]):
                weights_constant_op = ConstantOp(input_names[1], tensor=weights)
                self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [input_names[1]], is_param=True)
                weights_node = quir_graph.add(weights_constant_op, [], [input_names[1]])
                quir_graph.add_src_op_info(input_names[1], None, weights_node.output_names[0])

            bias = np.zeros(weights.shape[-2], dtype=np.float32)
            bias_name = op_name + "_fc_b"
            bias_constant_op = ConstantOp(bias_name, tensor=bias)
            bias_node = quir_graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
            log_debug3("\tweight shape {}", weights.shape)
            log_debug3("\tbias shape {}", bias.shape)

            ir_op = FullyConnectedOp(op_name)
            input_names.append(bias_name)
        else:
            op_name = converter_context.get_op_name(relay_expr, MatMulOp.TRANSLATION_KEY,
                                                    MatMulOp.LEGACY_TRANSLATION_KEY)
            ir_op = MatMulOp(op_name, transpose_in0=False, transpose_in1=True)

        return ir_op


RelayTranslations.register_translation(RelayDenseTranslation(),
                                       converter_type('dense', 'relay'))


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class RelayDepthToSpaceTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDepthToSpaceTranslation, self).__init__()
        self.SUPPORTED_DEPTHTOSPACE_MODES = {'DCR': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR,
                                             'CRD': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD}

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        dts_attrs = relay_expr.attrs

        attr_dict["layout"] = dts_attrs.layout
        attr_dict["mode"] = dts_attrs.mode
        attr_dict["block_size"] = dts_attrs.block_size
        log_debug3("\tDepthToSpaceOp data layout {}, mode {}, block size {}", dts_attrs.layout, dts_attrs.mode, dts_attrs.block_size)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, DepthToSpaceOp.TRANSLATION_KEY,
                                                DepthToSpaceOp.LEGACY_TRANSLATION_KEY)

        log_assert(attr_dict["mode"] in self.SUPPORTED_DEPTHTOSPACE_MODES,
                   "DepthToSpace only support DCR and CRD mode, but got {}", attr_dict["mode"])
        log_assert(attr_dict["layout"] == "NHWC",
                   "DepthToSpace only support NHWC data layout, but got {}", attr_dict["layout"])

        # for normal D2S op, block size is "int"
        # for rectangular D2S op, block size is "list of int"
        if isinstance(attr_dict["block_size"], tvm.ir.container.Array):
            block_size = attr_dict["block_size"][:]
        else:
            block_size = [attr_dict["block_size"]] * 2
        mode = self.SUPPORTED_DEPTHTOSPACE_MODES[attr_dict["mode"]]

        ir_op = DepthToSpaceOp(op_name,
                               block_size=block_size,
                               mode=mode)

        return ir_op


RelayTranslations.register_translation(RelayDepthToSpaceTranslation(),
                                       converter_type('depth_to_space', 'relay'),
                                       converter_type('depth_to_space_rect', 'relay'))


# ------------------------------------------------------------------------------
#   Detecion PostPorcess
# ------------------------------------------------------------------------------
class RelayDetectionPostProcessTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDetectionPostProcessTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = dict(relay_expr.attrs)
        attr_dict['use_bg_in_nms'] = False if attr_dict['use_bg_in_nms'] == 0 else True
        attr_dict['output_background'] =  False if attr_dict['output_background'] == 0 else True
        attr_dict['share_location'] =  False if attr_dict['share_location'] == 0 else True

        log_debug3("\tuse_bg_in_nms {}", attr_dict['use_bg_in_nms'])
        log_debug3("\toutput_background {}", attr_dict['output_background'])
        log_debug3("\tshare_location {}", attr_dict['share_location'])

        for k, v in attr_dict.items():
            attr_dict[k] = get_prim_type(v)
            log_debug3("\t{} {}", k, v)

        return attr_dict


    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, DetectionOutputOp.TRANSLATION_KEY,
                                                DetectionOutputOp.LEGACY_TRANSLATION_KEY)
        if input_names[2] not in relay_params:
            raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[2]))
        self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
        quir_graph.add(ConstantOp(input_names[2], relay_params[input_names[2]].asnumpy()), [], [input_names[2]])

        ir_op = DetectionOutputOp(op_name,
                                  output_dims=attr_dict['output_dims'],
                                  delta_scaling_factors=attr_dict['delta_scaling_factors'],
                                  confidence_threshold=attr_dict['confidence_threshold'],
                                  iou_threshold=attr_dict['iou_threshold'],
                                  nms_type=attr_dict['nms_type'],
                                  background_class_idx=attr_dict['background_class_idx'],
                                  use_bg_in_nms=attr_dict['use_bg_in_nms'],
                                  output_background=attr_dict['output_background'],
                                  share_location=attr_dict['share_location'],
                                  nms_eta=attr_dict['nms_eta'],
                                  detection_limit=attr_dict['detection_limit'])
        return ir_op


RelayTranslations.register_translation(RelayDetectionPostProcessTranslation(),
                                       converter_type('detection_postprocess', 'relay'))


# ------------------------------------------------------------------------------
#   Dropout
# ------------------------------------------------------------------------------
class RelayDropoutTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDropoutTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, IdentityOp.TRANSLATION_KEY,
                                                IdentityOp.LEGACY_TRANSLATION_KEY)
        return IdentityOp(op_name)


RelayTranslations.register_translation(RelayDropoutTranslation(),
                                       converter_type('dropout', 'relay'))


# ------------------------------------------------------------------------------
#   Global Pool2d Base
# ------------------------------------------------------------------------------
class RelayGlobalPool2dBaseTranslation(RelayTranslationBase):
    def __init__(self, pool_type):
        super(RelayGlobalPool2dBaseTranslation, self).__init__()
        self.pool_type = pool_type

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['layout'] = relay_expr.attrs.layout

        log_debug3("\tlayout {}", attr_dict['layout'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, Pool2dOp.TRANSLATION_KEY, Pool2dOp.LEGACY_TRANSLATION_KEY)
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        layout = attr_dict['layout']

        if layout != "NHWC":
            raise ValueError("No support {} data layout".format(layout))

        ir_op = Pool2dOp(op_name,
                         pool_type=self.pool_type,
                         size_y=input_shape[1],
                         size_x=input_shape[2],
                         stride_y=input_shape[1],
                         stride_x=input_shape[2],
                         padding_size_strategy=IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR,
                         count_pad_for_edges=False)

        return ir_op


# ------------------------------------------------------------------------------
#   Global Average Pool2d
# ------------------------------------------------------------------------------
class RelayGlobalAvgPool2dTranslation(RelayGlobalPool2dBaseTranslation):
    def __init__(self):
        super(RelayGlobalAvgPool2dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_AVG_2D)


RelayTranslations.register_translation(RelayGlobalAvgPool2dTranslation(),
                                       converter_type('global_avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   Global Max Pool2d
# ------------------------------------------------------------------------------
class RelayGlobalMaxPool2dTranslation(RelayGlobalPool2dBaseTranslation):
    def __init__(self):
        super(RelayGlobalMaxPool2dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_MAX_2D)


RelayTranslations.register_translation(RelayGlobalMaxPool2dTranslation(),
                                       converter_type('global_max_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   GroupNorm
# ------------------------------------------------------------------------------
class RelayGroupNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGroupNormTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        group_norm_attrs = relay_expr.attrs
        attr_dict["group"] = group_norm_attrs.num_groups
        attr_dict["axis"] = group_norm_attrs.axis if hasattr(group_norm_attrs, 'axis') else 1
        attr_dict["epsilon"] = group_norm_attrs.epsilon if hasattr(group_norm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = group_norm_attrs.center if hasattr(group_norm_attrs, 'center') else True
        attr_dict["scale"] = group_norm_attrs.scale if hasattr(group_norm_attrs, 'scale') else True

        log_debug3("\tgroup {}", attr_dict["group"])
        log_debug3("\taxis {}", attr_dict["axis"])
        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GroupNormOp.TRANSLATION_KEY, GroupNormOp.LEGACY_TRANSLATION_KEY)

        input_shape = quir_graph.get_buffer(input_names[0]).shape
        if attr_dict["axis"] != len(input_shape)-1:
            raise ValueError("GroupNorm's input layout should be channel-last, but axis is {}".format(attr_dict["axis"]))

        # gamma
        if quir_graph.has_buffer(input_names[1]):
            gamma_buf = quir_graph.get_buffer(input_names[1])
            log_debug3("\tgamma shape {}", gamma_buf.shape)
        elif input_names[1] in relay_params:
            gamma = relay_params[input_names[1]].asnumpy()
            log_debug3("\tgamma shape {}", gamma.shape)
            gamma_name = op_name + "_gamma"
            gamma_name = validate_const_name(quir_graph, input_names[1], gamma_name)
            if not attr_dict["scale"]:
                gamma = np.ones(gamma.shape, dtype=np.float32)
                gamma_name += "_noscale"
            input_names[1] = gamma_name
            if not quir_graph.has_buffer(gamma_name):
                self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [gamma_name], is_param=True)
                gamma_node = quir_graph.add(ConstantOp(gamma_name, tensor=gamma), [], [gamma_name], axis_formats=[AxisTracker.AxisFormat.ANY])
                quir_graph.add_src_op_info(gamma_name, None, gamma_node.output_names[0])
        else:
            raise ValueError("gamma not found!")

        # beta
        if quir_graph.has_buffer(input_names[2]):
            beta_buf = quir_graph.get_buffer(input_names[2])
            log_debug3("\tbeta shape {}", beta_buf.shape)
        elif input_names[2] in relay_params:
            beta = relay_params[input_names[2]].asnumpy()
            log_debug3("\tbeta shape {}", beta.shape)
            beta_name = op_name + "_beta"
            beta_name = validate_const_name(quir_graph, input_names[2], beta_name)
            if not attr_dict["center"]:
                beta = np.zeros(beta.shape, dtype=np.float32)
                beta_name += "_nocenter"
            input_names[2] = beta_name
            if not quir_graph.has_buffer(beta_name):
                self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [beta_name], is_param=True)
                beta_node = quir_graph.add(ConstantOp(beta_name, tensor=beta), [], [beta_name], axis_formats=[AxisTracker.AxisFormat.ANY])
                quir_graph.add_src_op_info(beta_name, None, beta_node.output_names[0])
        else:
            raise ValueError("beta not found!")

        ir_op = GroupNormOp(op_name,
                            epsilon=attr_dict["epsilon"],
                            group=attr_dict["group"])
        return ir_op


RelayTranslations.register_translation(RelayGroupNormTranslation(),
                                       converter_type('group_norm', 'relay'))


# ------------------------------------------------------------------------------
#   InstanceNorm
# ------------------------------------------------------------------------------
class RelayInstanceNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayInstanceNormTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        instance_norm_attrs = relay_expr.attrs
        attr_dict["epsilon"] = instance_norm_attrs.epsilon if hasattr(instance_norm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = instance_norm_attrs.center if hasattr(instance_norm_attrs, 'center') else True
        attr_dict["scale"] = instance_norm_attrs.scale if hasattr(instance_norm_attrs, 'scale') else True
        attr_dict["axis"]  = instance_norm_attrs.axis if hasattr(instance_norm_attrs, 'axis') else 1

        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        log_debug3("\taxis {}", attr_dict["axis"])
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr,
                                                InstanceNormOp.TRANSLATION_KEY,
                                                InstanceNormOp.LEGACY_TRANSLATION_KEY)

        input_shape = quir_graph.get_buffer(input_names[0]).shape
        if attr_dict["axis"] not in [len(input_shape)-1, -1]:
            raise ValueError("In channel-last data layout, instancenorm channel should be {} or -1, \
                             got {}".format(len(input_shape)-1, attr_dict["axis"]))

        # gamma
        gamma = relay_params[input_names[1]].asnumpy()
        log_debug3("\tgamma shape {}", gamma.shape)
        gamma_name = op_name + "_gamma"
        gamma_name = validate_const_name(quir_graph, input_names[1], gamma_name)
        if not attr_dict["scale"]:
            gamma = np.ones(gamma.shape, dtype=np.float32)
            gamma_name += "_noscale"
        input_names[1] = gamma_name
        if not quir_graph.has_buffer(gamma_name):
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph,
                                              [gamma_name], is_param=True)
            gamma_node = quir_graph.add(ConstantOp(gamma_name, tensor=gamma), [], [gamma_name],
                                        axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(gamma_name, None, gamma_node.output_names[0])

        # beta
        beta = relay_params[input_names[2]].asnumpy()
        log_debug3("\tbeta shape {}", beta.shape)
        beta_name = op_name + "_beta"
        beta_name = validate_const_name(quir_graph, input_names[2], beta_name)
        if not attr_dict["center"]:
            beta = np.zeros(beta.shape, dtype=np.float32)
            beta_name += "_nocenter"
        input_names[2] = beta_name
        if not quir_graph.has_buffer(beta_name):
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph,
                                              [beta_name], is_param=True)
            beta_node = quir_graph.add(ConstantOp(beta_name, tensor=beta), [], [beta_name],
                                       axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(beta_name, None, beta_node.output_names[0])

        ir_op = InstanceNormOp(op_name,
                               epsilon=attr_dict["epsilon"],
                               mode=ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA,
                               normalize_variance=True,
                               region=ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL)
        return ir_op


RelayTranslations.register_translation(RelayInstanceNormTranslation(),
                                       converter_type('instance_norm', 'relay'))


# ------------------------------------------------------------------------------
#   LayerNorm
# ------------------------------------------------------------------------------
class RelayLayerNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLayerNormTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        layernorm_attrs = relay_expr.attrs
        attr_dict["axis"] = layernorm_attrs.axis if hasattr(layernorm_attrs, 'axis') else -1
        attr_dict["epsilon"] = layernorm_attrs.epsilon if hasattr(layernorm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = layernorm_attrs.center if hasattr(layernorm_attrs, 'center') else True
        attr_dict["scale"] = layernorm_attrs.scale if hasattr(layernorm_attrs, 'scale') else True
        log_debug3("\taxis {}", attr_dict["axis"])
        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, LayerNormOp.TRANSLATION_KEY, LayerNormOp.LEGACY_TRANSLATION_KEY)
        if attr_dict["scale"]:
            gamma = relay_params[input_names[1]].asnumpy()
            log_debug3("\tgamma shape {}", gamma.shape)
            gamma_name = op_name + "_gamma"
            gamma_constant_op = ConstantOp(gamma_name, tensor=gamma)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [gamma_name], is_param=True)
            gamma_node = quir_graph.add(gamma_constant_op, [], [gamma_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(gamma_name, None, gamma_node.output_names[0])
        if attr_dict["center"]:
            beta = relay_params[input_names[2]].asnumpy()
            log_debug3("\tbeta shape {}", beta.shape)
            beta_name = op_name + "_beta"
            beta_constant_op = ConstantOp(beta_name, tensor=beta)
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [beta_name], is_param=True)
            beta_node = quir_graph.add(beta_constant_op, [], [beta_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(beta_name, None, beta_node.output_names[0])
        ir_op = LayerNormOp(op_name,
                            epsilon=attr_dict["epsilon"],
                            axes=[attr_dict["axis"]])
        # update input names
        for name in input_names[1:]:
            input_names.remove(name)
        if attr_dict["scale"]:
            input_names.append(gamma_name)
        if attr_dict["center"]:
            input_names.append(beta_name)
        return ir_op


RelayTranslations.register_translation(RelayLayerNormTranslation(),
                                       converter_type('layer_norm', 'relay'))


# ------------------------------------------------------------------------------
#   LeakyRelu
# ------------------------------------------------------------------------------
class RelayLeakyReluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLeakyReluTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        leaky_relu_attrs = relay_expr.attrs
        attr_dict["alpha"] = leaky_relu_attrs.alpha
        log_debug3("\talpha {}", attr_dict["alpha"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, PreluOp.TRANSLATION_KEY, PreluOp.LEGACY_TRANSLATION_KEY)

        alpha = attr_dict["alpha"]
        coeff = alpha * np.ones(quir_graph.get_buffer(input_names[0]).shape[-1], dtype=np.float32)

        coeff_name = op_name + "_coeff"
        coeff_constant_op = ConstantOp(coeff_name, tensor=coeff)
        coeff_node = quir_graph.add(coeff_constant_op, [], [coeff_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        quir_graph.add_src_op_info(coeff_name, None, coeff_node.output_names[0])
        ir_op = PreluOp(op_name)
        input_names.append(coeff_name)
        return ir_op


RelayTranslations.register_translation(RelayLeakyReluTranslation(),
                                       converter_type('leaky_relu', 'relay'))


# ------------------------------------------------------------------------------
#   LogSoftmax
# ------------------------------------------------------------------------------
class RelayLogSoftmaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLogSoftmaxTranslation, self).__init__()

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
        op_name = converter_context.get_op_name(relay_expr, LogSoftmaxOp.TRANSLATION_KEY, LogSoftmaxOp.LEGACY_TRANSLATION_KEY)

        ir_op = LogSoftmaxOp(op_name, axis=attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelayLogSoftmaxTranslation(),
                                       converter_type('log_softmax', 'relay'))


# ------------------------------------------------------------------------------
#   LRN
# ------------------------------------------------------------------------------
class RelayLRNTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLRNTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        lrn_attrs = relay_expr.attrs
        attr_dict["size"] = lrn_attrs.size
        attr_dict["alpha"] = lrn_attrs.alpha
        attr_dict["beta"] = lrn_attrs.beta
        attr_dict["bias"] = lrn_attrs.bias

        log_debug3("\tsize {}", attr_dict['size'])
        log_debug3("\talpha {}", attr_dict['alpha'])
        log_debug3("\tbeta {}", attr_dict['beta'])
        log_debug3("\tbias {}", attr_dict['bias'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, LrnOp.TRANSLATION_KEY, LrnOp.LEGACY_TRANSLATION_KEY)

        size = attr_dict["size"]
        alpha = attr_dict["alpha"]
        beta = attr_dict["beta"]
        bias = attr_dict["bias"]

        if size % 2 == 0:
            raise ValueError("Qnn LrnOp only supports odd size, but get size:{}".format(size))

        ir_op = LrnOp(op_name,
                      alpha=alpha / size,
                      beta=beta,
                      bias=bias,
                      radius=int((size-1)/2),
                      region=ir_graph.QNN_OP_LRN_REGION_ACROSS_CHANNEL)

        return ir_op


RelayTranslations.register_translation(RelayLRNTranslation(),
                                       converter_type('lrn', 'relay'))


# ------------------------------------------------------------------------------
#   PadOp
# ------------------------------------------------------------------------------
class RelayPadTranslation(RelayTranslationBase):
    class RelayPadMode:
        CONSTANT = 'constant'
        REFLECT = 'reflect'
        EDGE = 'edge'
    def __init__(self):
        super(RelayPadTranslation, self).__init__()
        self.supported_modes = {self.RelayPadMode.CONSTANT : ir_graph.QNN_OP_PAD_SCHEME_CONSTANT,
                                self.RelayPadMode.REFLECT : ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT,
                                self.RelayPadMode.EDGE : ir_graph.QNN_OP_PAD_SCHEME_EDGE}

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        pad_pairs = list()
        for pad in relay_expr.attrs.pad_width:
            pad_pairs.append([int(i) for i in pad])

        attr_dict["pad_pairs"] = pad_pairs
        attr_dict["pad_mode"] = relay_expr.attrs.pad_mode

        # pad value from float, or tvm.relay.Expr, optional, default=0
        # if not in relay_expr.attrs, it will be default value or tvm.relay.Expr
        if hasattr(relay_expr.attrs, 'pad_value'):
            attr_dict["pad_value"] = relay_expr.attrs.pad_value
        else:
            attr_dict["pad_value"] = None

        log_debug3("\tpad_pairs {}", pad_pairs)
        log_debug3("\tpad_mode {}", attr_dict["pad_mode"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PadOp.TRANSLATION_KEY, PadOp.LEGACY_TRANSLATION_KEY)

        pad_pairs = attr_dict["pad_pairs"]
        pad_pairs = np.asarray(pad_pairs, dtype=np.dtype('int32'))
        mode = attr_dict["pad_mode"]
        pad_value = attr_dict["pad_value"]

        if pad_value is None:
            # pad constant value from inputs[1] expr.Constant
            # if no found constant from param, set to zero by default
            pad_value_op_name = input_names[1]
            if pad_value_op_name in relay_params:
                expr_const_pad_value = relay_params[pad_value_op_name]
                pad_value = float(expr_const_pad_value.asnumpy())
            else:
                log_debug2("\tNo Padding value, use default as zero")

                pad_value = 0

        log_debug3("\tpad_value {}", pad_value)

        ir_op = PadOp(op_name,
                        pad_amount=pad_pairs,
                        pad_constant_value=pad_value,
                        scheme=self.supported_modes[mode])

        # Only data input is needed in IR graph. Pad value input is ignored
        for name in input_names[1:]:
            input_names.remove(name)

        return ir_op


RelayTranslations.register_translation(RelayPadTranslation(),
                                       converter_type('pad', 'relay'))


# ------------------------------------------------------------------------------
#   Pool Base
# ------------------------------------------------------------------------------
class RelayPoolBaseTranslation(RelayTranslationBase):
    def __init__(self, pool_type):
        super(RelayTranslationBase, self).__init__()
        self.pool_type = pool_type

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):

        attr_dict = {}
        pool_attrs = relay_expr.attrs

        if len(pool_attrs.layout) not in [3, 4, 5]:
            raise ValueError("No support {}D Input".format(len(pool_attrs.layout)))

        if pool_attrs.layout not in ["NWC", "NHWC", "NDHWC"]:
            raise ValueError("No support {} data layout".format(pool_attrs.layout))

        spatial_rank = len(pool_attrs.layout) - 2
        pool_size = [pool_attrs.pool_size] * spatial_rank if isinstance(pool_attrs.pool_size, int) else [int(val) for val in pool_attrs.pool_size]
        log_debug3("\tpool_size {}", pool_size)

        # repeat the padding if needed, take Pool2d as an example
        # [1] -> [1, 1, 1, 1]
        # [1, 2] -> [1, 2, 1, 2]
        # [1, 2, 3, 4] -> [1, 2, 3, 4]
        padding = [pool_attrs.padding] if isinstance(pool_attrs.padding, int) else [int(val) for val in pool_attrs.padding]
        padding = padding * (2 * spatial_rank // len(padding))
        log_debug3("\tpadding {}", padding)

        strides = [pool_attrs.strides] * spatial_rank if isinstance(pool_attrs.strides, int) else [int(val) for val in pool_attrs.strides]
        log_debug3("\tstrides {}", strides)

        # z -> depth
        # y -> height
        # x -> width
        x_axis = -1
        attr_dict["size_x"] = int(pool_size[x_axis])
        attr_dict["stride_x"] = int(strides[x_axis])

        attr_dict["padx_before"] = int(padding[x_axis-spatial_rank])
        attr_dict["padx_after"] = int(padding[x_axis])

        if spatial_rank >= 2:
            y_axis = -2
            attr_dict["size_y"] = int(pool_size[y_axis])
            attr_dict["stride_y"] = int(strides[y_axis])

            attr_dict["pady_before"] = int(padding[y_axis-spatial_rank])
            attr_dict["pady_after"] = int(padding[y_axis])

        if spatial_rank >= 3:
            z_axis = -3
            attr_dict["size_z"] = int(pool_size[z_axis])
            attr_dict["stride_z"] = int(strides[z_axis])

            attr_dict["padz_before"] = int(padding[z_axis-spatial_rank])
            attr_dict["padz_after"] = int(padding[z_axis])

        ceil_mode = getattr(pool_attrs, "ceil_mode", False)
        if ceil_mode:
            attr_dict["padding_size_strategy"] = ir_graph.PADDING_SIZE_EXPLICIT
        else:
            attr_dict["padding_size_strategy"] = ir_graph.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        attr_dict["count_pad_for_edges"] = getattr(pool_attrs, "count_include_pad", False)
        log_debug3("\tcount_pad_for_edges {}", attr_dict["count_pad_for_edges"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        input_shape = quir_graph.get_buffer(input_names[0]).shape

        if len(input_shape) == 3:
            op_type = Pool1dOp

        elif len(input_shape) == 4:
            op_type = Pool2dOp

        elif len(input_shape) == 5:
            op_type = Pool3dOp

        op_name = converter_context.get_op_name(relay_expr, op_type.TRANSLATION_KEY, op_type.LEGACY_TRANSLATION_KEY)

        ir_op = op_type(op_name,
                        pool_type=self.pool_type,
                        **attr_dict)
        return ir_op


# ------------------------------------------------------------------------------
#   AvgPool1d
# ------------------------------------------------------------------------------
class RelayAvgPool1dTranslation(RelayPoolBaseTranslation):
    def __init__(self):
        super(RelayAvgPool1dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_AVG_2D)


RelayTranslations.register_translation(RelayAvgPool1dTranslation(),
                                       converter_type('avg_pool1d', 'relay'))


# ------------------------------------------------------------------------------
#   MaxPool1d
# ------------------------------------------------------------------------------
class RelayMaxPool1dTranslation(RelayPoolBaseTranslation):
    def __init__(self):
        super(RelayMaxPool1dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_MAX_2D)


RelayTranslations.register_translation(RelayMaxPool1dTranslation(),
                                       converter_type('max_pool1d', 'relay'))


# ------------------------------------------------------------------------------
#   AvgPool2d
# ------------------------------------------------------------------------------
class RelayAvgPool2dTranslation(RelayPoolBaseTranslation):
    def __init__(self):
        super(RelayAvgPool2dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_AVG_2D)


RelayTranslations.register_translation(RelayAvgPool2dTranslation(),
                                       converter_type('avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   MaxPool2d
# ------------------------------------------------------------------------------
class RelayMaxPool2dTranslation(RelayPoolBaseTranslation):
    def __init__(self):
        super(RelayMaxPool2dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_MAX_2D)


RelayTranslations.register_translation(RelayMaxPool2dTranslation(),
                                       converter_type('max_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   AvgPool3d
# ------------------------------------------------------------------------------
class RelayAvgPool3dTranslation(RelayPoolBaseTranslation):
    def __init__(self):
        super(RelayAvgPool3dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_AVG_3D)


RelayTranslations.register_translation(RelayAvgPool3dTranslation(),
                                       converter_type('avg_pool3d', 'relay'))


# ------------------------------------------------------------------------------
#   MaxPool3d
# ------------------------------------------------------------------------------
class RelayMaxPool3dTranslation(RelayPoolBaseTranslation):
    def __init__(self):
        super(RelayMaxPool3dTranslation, self).__init__(pool_type=ir_graph.QNN_OP_POOL_MAX_3D)


RelayTranslations.register_translation(RelayMaxPool3dTranslation(),
                                       converter_type('max_pool3d', 'relay'))


# ------------------------------------------------------------------------------
#   MirrorPadOp
# ------------------------------------------------------------------------------
class RelayMirrorPadTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayMirrorPadTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["mode"] = relay_expr.attrs.mode
        attr_dict["pad_width"] = relay_expr.attrs.pad_width
        log_debug3("\tmode, {}, pad_width {}", attr_dict["mode"], attr_dict["pad_width"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PadOp.TRANSLATION_KEY, PadOp.LEGACY_TRANSLATION_KEY)

        mode = attr_dict["mode"]
        pad_width = attr_dict["pad_width"]

        # The type of pad_width affect the type of output shape,
        # so we translate it to native Python types.
        ir_pad_width = []
        for pad_per_axis in pad_width:
            ir_pad_width.append([int(pad_per_axis[0]), int(pad_per_axis[1])])

        if mode == "SYMMETRIC":
            ir_mode = ir_graph.QNN_OP_PAD_SCHEME_MIRROR_SYMMETRIC
        elif mode == "REFLECT":
            ir_mode = ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT
        else:
            log_assert(False, "Unknown nn.mirror_pad mode: {}", mode)

        ir_op = PadOp(op_name,
                pad_amount=ir_pad_width,
                scheme=ir_mode)

        return ir_op


RelayTranslations.register_translation(RelayMirrorPadTranslation(),
                                       converter_type('mirror_pad', 'relay'))


# ------------------------------------------------------------------------------
#   Prelu
# ------------------------------------------------------------------------------
class RelayPreluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayPreluTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        prelu_attrs = relay_expr.attrs
        attr_dict["axis"] = prelu_attrs.axis
        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, PreluOp.TRANSLATION_KEY, PreluOp.LEGACY_TRANSLATION_KEY)

        channel_axis = attr_dict["axis"]
        slope_input_name = input_names[1]

        input_shape = quir_graph.get_buffer(input_names[0]).shape

        log_assert(channel_axis == len(input_shape)-1,
                   "Expect the channel axis is the last dimension, but got "
                   "channel_axis={} for data_tensor_rank={}",
                   channel_axis, len(input_shape))

        log_assert(slope_input_name in relay_params,
                   "Only support PRelu with constant slope(second input). "
                   "But {} is not in relay_params.",
                   slope_input_name)

        slope = relay_params[slope_input_name]
        if isinstance(slope, (tvm.runtime.ndarray.NDArray, tvm.runtime.NDArray)):
            slope = slope.asnumpy().astype(np.float32)

        coeff_name = slope_input_name
        coeff_constant_op = ConstantOp(coeff_name, tensor=slope)
        self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [coeff_name], is_param=True)
        coeff_node = quir_graph.add(coeff_constant_op, [], [coeff_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        quir_graph.add_src_op_info(coeff_name, None, coeff_node.output_names[0])
        ir_op = PreluOp(op_name)
        return ir_op


RelayTranslations.register_translation(RelayPreluTranslation(),
                                       converter_type('prelu', 'relay'))


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class RelayReluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReluTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ElementwiseNeuronOp.TRANSLATION_KEY, ElementwiseNeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = ElementwiseNeuronOp(op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU)
        return ir_op


RelayTranslations.register_translation(RelayReluTranslation(),
                                       converter_type('relu', 'relay'))


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class RelaySigmoidTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySigmoidTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ElementwiseNeuronOp.TRANSLATION_KEY, ElementwiseNeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = ElementwiseNeuronOp(op_name,
                         operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID,
                         alpha=1.0)
        return ir_op


RelayTranslations.register_translation(RelaySigmoidTranslation(),
                                       converter_type('sigmoid', 'relay'))


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class RelaySoftmaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySoftmaxTranslation, self).__init__()

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
        op_name = converter_context.get_op_name(relay_expr, SoftmaxOp.TRANSLATION_KEY, SoftmaxOp.LEGACY_TRANSLATION_KEY)

        ir_op = SoftmaxOp(op_name, axis=attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelaySoftmaxTranslation(),
                                       converter_type('softmax', 'relay'))


# ------------------------------------------------------------------------------
#   SpaceToDepth
# ------------------------------------------------------------------------------
class RelaySpaceToDepthTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySpaceToDepthTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["block_size"] = relay_expr.attrs.block_size
        attr_dict["layout"] = relay_expr.attrs.layout
        attr_dict["mode"] = relay_expr.attrs.mode

        log_debug3("\tblock_size {}, layout {}, mode {}",
                   attr_dict["block_size"], attr_dict["layout"], attr_dict["mode"])


        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SpaceToDepthOp.TRANSLATION_KEY,
                                                SpaceToDepthOp.LEGACY_TRANSLATION_KEY)

        log_assert(not attr_dict["mode"] or attr_dict["mode"] in ["DCR", "CRD"],
                   "SpaceToDepth only support DCR and CRD mode, but got {}", attr_dict["mode"])
        log_assert(attr_dict["layout"] == "NHWC",
                   "SpaceToDepth only support NHWC layout, but got {}", attr_dict["layout"])

        # for normal S2D op, block_size is "int"
        # for rectangular S2D op, block_size is "list of int"
        if isinstance(attr_dict["block_size"], tvm.ir.container.Array):
            block_size = attr_dict["block_size"][:]
        else:
            block_size = [attr_dict["block_size"]] * 2

        if attr_dict["mode"] == 'CRD':
            mode = ir_graph.QNN_OP_SPACE_TO_DEPTH_MODE_CRD
        else:
            # default mode is DCR mode
            mode = ir_graph.QNN_OP_SPACE_TO_DEPTH_MODE_DCR

        ir_op = SpaceToDepthOp(op_name, block_size=block_size, mode=mode)
        return ir_op


RelayTranslations.register_translation(RelaySpaceToDepthTranslation(),
                                       converter_type('space_to_depth', 'relay'),
                                       converter_type('space_to_depth_rect', 'relay'))


# ------------------------------------------------------------------------------
#   Upsampling
# ------------------------------------------------------------------------------
class RelayUpsamplingTranslation(RelayTranslationBase):

    # scaling method names in relay
    class ScaleModes:
        BICUBIC = "bicubic"
        BILINEAR = "bilinear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    # name mapping from relay to quir
    RELAY_CONSTS_TO_IR = {
        ScaleModes.BILINEAR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        ScaleModes.NEAREST_NEIGHBOR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST
    }

    def __init__(self):
        super(RelayUpsamplingTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        upsampling_attrs = relay_expr.attrs
        attr_dict["scale_h"] = getattr(upsampling_attrs, "scale_h")
        attr_dict["scale_w"] = getattr(upsampling_attrs, "scale_w")
        log_debug3("\tscale_h {}", attr_dict["scale_h"])
        log_debug3("\tscale_w {}", attr_dict["scale_w"])

        attr_dict["layout"] = getattr(upsampling_attrs, "layout")
        log_debug3("\tlayout {}", attr_dict["layout"])

        scale_mode = getattr(upsampling_attrs, "method", self.ScaleModes.NEAREST_NEIGHBOR)
        if scale_mode == self.ScaleModes.BICUBIC:
            raise ValueError("Unsupported scale method {}".format(scale_mode))

        attr_dict["interpolation_mode"] = self.RELAY_CONSTS_TO_IR[scale_mode]
        log_debug3("\tinterpolation_mode mode {}", attr_dict["interpolation_mode"])

        transform_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC
        align_corners = getattr(upsampling_attrs, "align_corners", False)
        if align_corners:
            transform_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS
        attr_dict["transformation_mode"] = transform_mode
        log_debug3("\ttransformation_mode mode {}", attr_dict['transformation_mode'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ResizeOp.TRANSLATION_KEY,
                                                ResizeOp.LEGACY_TRANSLATION_KEY)
        if attr_dict["layout"] != "NHWC":
            raise ValueError("Unsupported data layout {}".format(attr_dict["layout"]))

        ir_op = ResizeOp(op_name,
                         transformation_mode=attr_dict["transformation_mode"],
                         interpolation_mode=attr_dict["interpolation_mode"],
                         scale_height=attr_dict["scale_h"],
                         scale_width=attr_dict["scale_w"])
        return ir_op


RelayTranslations.register_translation(RelayUpsamplingTranslation(),
                                       converter_type('upsampling', 'relay'))
