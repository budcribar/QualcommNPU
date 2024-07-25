# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

'''
This file defines the supported QNN operations
for the different converters.
'''

ONNX            = "Onnx"
TENSORFLOW      = "TensorFlow"
TENSORFLOW_LITE = "TensorFlow Lite"
PYTORCH         = "PyTorch"
converters = [ONNX, TENSORFLOW, TENSORFLOW_LITE, PYTORCH]

# These macros correspond to what is in QnnOpDef.h
# each entry is <qnn_op> : { <source_framework>: <source_op_description> }
# ***Note: Source framework op type should appear in the same format and naming as it appears in corresponding
#          documentation. For Ops that show up with different name during conversion or have multiple versions
#          please specify in bracket(this is especially prevalent in tf. eg: dense, tensordot(MatMul)

COMMAND_LINE = "COMMAND_LINE"
MATCHED = "INFERRED"
NONE = "**---**"

converter_op_support = {
    "QNN_OP_ARGB_TO_RGB": {ONNX: COMMAND_LINE, TENSORFLOW: COMMAND_LINE},
    "QNN_OP_ARGMAX": {ONNX: "ArgMax", TENSORFLOW: "argmax", PYTORCH: "argmax"},
    "QNN_OP_ARGMIN": {ONNX: "ArgMin", TENSORFLOW: "argmin", PYTORCH: "argmin"},
    "QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM": {ONNX: "BBoxTransform(org.pytorch._caffe2) with im_info's img_scale = 1"},
    "QNN_OP_BATCHNORM": {ONNX: "BatchNormalization",
                         TENSORFLOW: "batch_normalization, fused_batch_norm(FusedBatchNorm, FusedBatchNormV3)",
                         TENSORFLOW_LITE: MATCHED, PYTORCH: "BatchNorm2d"},
    "QNN_OP_BATCH_TO_SPACE": {TENSORFLOW: "batch_to_space"},
    "QNN_OP_BOX_WITH_NMS_LIMIT": {ONNX: "BoxWithNMSLimit(org.pytorch._caffe2)"},
    "QNN_OP_CHANNEL_SHUFFLE": {ONNX: MATCHED, TENSORFLOW: MATCHED, PYTORCH: "ChannelShuffle"},
    "QNN_OP_CAST": {ONNX: "Cast", TENSORFLOW: "cast", PYTORCH: "to"},
    "QNN_OP_CONV_2D": {ONNX: "Conv", TENSORFLOW: "conv2d", TENSORFLOW_LITE: "conv_2d", PYTORCH: "Conv2d"},
    "QNN_OP_CONV_3D": {ONNX: "Conv", TENSORFLOW: "conv3d"},
    "QNN_OP_CONCAT": {ONNX: "Concat", TENSORFLOW: "concat(Concat, ConcatV2)",
                      TENSORFLOW_LITE: "concatenation", PYTORCH: "cat"},
    "QNN_OP_CROP_AND_RESIZE": {TENSORFLOW: "crop_and_resize"},
    "QNN_OP_CUMULATIVE_SUM": {ONNX: "CumSum", TENSORFLOW: "cumsum", PYTORCH: "cumsum"},
    "QNN_OP_DEPTH_WISE_CONV_2D": {ONNX: "Conv with 'num_output' == 'input channels' == 'group'", \
                                  TENSORFLOW: "depthwise_conv2d"},
    "QNN_OP_DEPTH_TO_SPACE": {ONNX: "DepthToSpace", TENSORFLOW: "depth_to_space", TENSORFLOW_LITE: "depth_to_space", \
                              PYTORCH: "PixelShuffle"},
    "QNN_OP_DEQUANTIZE":{ONNX: "DequantizeLinear"},
    "QNN_OP_DETECTION_OUTPUT": {TENSORFLOW: MATCHED, TENSORFLOW_LITE: "TfliteDetectionPostProcess"},
    "QNN_OP_TRANSPOSE_CONV_2D": {ONNX: "ConvTranspose", TENSORFLOW: "conv2d_transpose", \
                                 TENSORFLOW_LITE: "transpose_conv", PYTORCH: "ConvTranspose2d"},
    "QNN_OP_TRANSPOSE_CONV_3D": {ONNX: "ConvTranspose"},
    "QNN_OP_ELEMENT_WISE_ABS": {ONNX: "Abs", TENSORFLOW: "abs", TENSORFLOW_LITE: "abs", PYTORCH: "abs"},
    "QNN_OP_ELEMENT_WISE_ADD": {ONNX: "Add, Sum",
                                TENSORFLOW: "add(Add, AddV2, Sum), bias_add", TENSORFLOW_LITE: "add", PYTORCH: "add"},
    "QNN_OP_ELEMENT_WISE_AND": {ONNX: "And", TENSORFLOW: "logical_and", PYTORCH: "logical_and"},
    "QNN_OP_ELEMENT_WISE_ASIN": {ONNX: "Asin", PYTORCH: "asin"},
    "QNN_OP_ELEMENT_WISE_ATAN": {ONNX: "Atan", PYTORCH: "atan"},
    "QNN_OP_ELEMENT_WISE_CEIL": {ONNX: "Ceil", TENSORFLOW: "ceil", PYTORCH: "ceil"},
    "QNN_OP_ELEMENT_WISE_COS": {ONNX: "Cos", PYTORCH: "cos"},
    "QNN_OP_ELEMENT_WISE_DIVIDE": {ONNX: "Div, Reciprocal", TENSORFLOW: "divide, realdiv", TENSORFLOW_LITE: "div", PYTORCH: "div"},
    "QNN_OP_ELEMENT_WISE_EXP": {ONNX: "Exp", TENSORFLOW: "exp", TENSORFLOW_LITE: "exp", PYTORCH: "exp"},
    "QNN_OP_ELEMENT_WISE_EQUAL": {ONNX: "Equal", TENSORFLOW: "equal", PYTORCH: "eq"},
    "QNN_OP_ELEMENT_WISE_FLOOR": {ONNX: "Floor", TENSORFLOW: "floor", TENSORFLOW_LITE: "floor", PYTORCH: "floor"},
    "QNN_OP_ELEMENT_WISE_FLOOR_DIV": {TENSORFLOW: "floordiv", PYTORCH: "floor_divide"},
    "QNN_OP_ELEMENT_WISE_FMOD": {ONNX: "Mod"},
    "QNN_OP_ELEMENT_WISE_GREATER": {ONNX: "Greater", TENSORFLOW: "greater", PYTORCH: "gt"},
    "QNN_OP_ELEMENT_WISE_GREATER_EQUAL": {ONNX: "GreaterOrEqual", TENSORFLOW: "greater_equal", PYTORCH: "ge"},
    "QNN_OP_ELEMENT_WISE_LESS": {ONNX: "Less", TENSORFLOW: "less", PYTORCH: "lt"},
    "QNN_OP_ELEMENT_WISE_LESS_EQUAL": {ONNX: "LessOrEqual", TENSORFLOW: "less_equal", PYTORCH: "le"},
    "QNN_OP_ELEMENT_WISE_LOG": {ONNX: "Log", TENSORFLOW: "log", PYTORCH: "log"},
    "QNN_OP_ELEMENT_WISE_MAXIMUM": {ONNX: "Max", TENSORFLOW: "maximum", \
                                    TENSORFLOW_LITE: "maximum", PYTORCH: "maximum"},
    "QNN_OP_ELEMENT_WISE_MINIMUM": {ONNX: "Min", TENSORFLOW: "minimum", TENSORFLOW_LITE: "minimum", PYTORCH: "minimum"},
    "QNN_OP_ELEMENT_WISE_MOD": {ONNX: "Mod"},
    "QNN_OP_ELEMENT_WISE_MULTIPLY": {ONNX: "Mul", TENSORFLOW: "mul", TENSORFLOW_LITE: "mul", PYTORCH: "mul"},
    "QNN_OP_ELEMENT_WISE_NEG": {ONNX: "Neg", TENSORFLOW: "negative", PYTORCH: "neg"},
    "QNN_OP_ELEMENT_WISE_NOT": {ONNX: "Not", TENSORFLOW: "logical_not", PYTORCH: "logical_not"},
    "QNN_OP_ELEMENT_WISE_NOT_EQUAL": {TENSORFLOW: "not_equal", PYTORCH: "ne"},
    "QNN_OP_ELEMENT_WISE_OR": {ONNX: "Or", TENSORFLOW: "logical_or", PYTORCH: "logical_or"},
    "QNN_OP_ELEMENT_WISE_POWER": {ONNX: "Pow", TENSORFLOW: "pow, square", PYTORCH: "pow"},
    "QNN_OP_ELEMENT_WISE_ROUND": {ONNX: "Round", TENSORFLOW: "round", PYTORCH: "round"},
    "QNN_OP_ELEMENT_WISE_RSQRT": {TENSORFLOW: "rsqrt", PYTORCH: "rsqrt"},
    "QNN_OP_ELEMENT_WISE_SELECT": {TENSORFLOW: "where", ONNX: "Where"},
    "QNN_OP_ELEMENT_WISE_SIGN": {ONNX: "Sign", PYTORCH: "sign"},
    "QNN_OP_ELEMENT_WISE_SIN": {ONNX: "Sin", TENSORFLOW: "sin", PYTORCH: "sin"},
    "QNN_OP_ELEMENT_WISE_SOFTPLUS": {ONNX: "Softplus", PYTORCH: "Softplus"},
    "QNN_OP_ELEMENT_WISE_SQUARE_ROOT": {ONNX: "Sqrt", TENSORFLOW: "sqrt", TENSORFLOW_LITE: "sqrt", PYTORCH: "sqrt"},
    "QNN_OP_ELEMENT_WISE_SUBTRACT": {ONNX: "Sub", TENSORFLOW: "subtract", TENSORFLOW_LITE: "sub", PYTORCH: "sub"},
    "QNN_OP_ELEMENT_WISE_XOR": {ONNX: "Xor", TENSORFLOW: "logical_xor", PYTORCH: "logical_xor"},
    "QNN_OP_ELU": {ONNX: "Elu", TENSORFLOW: "elu"},
    "QNN_OP_EXTRACT_GLIMPSE": {TENSORFLOW: "extract_glimpse"},
    "QNN_OP_EXTRACT_PATCHES": {TENSORFLOW: "extract_patches"},
    "QNN_OP_FULLY_CONNECTED": {ONNX: "MatMul(limited usecase), Gemm(limited usecase)",
                               TENSORFLOW: "dense and tensordot(MatMul)", TENSORFLOW_LITE: "fully_connected", PYTORCH: "Linear"},
    "QNN_OP_GATHER": {ONNX: "Gather", TENSORFLOW: "gather(Gather, GatherV2)"},
    "QNN_OP_GATHER_ELEMENTS": {ONNX: "GatherElements"},
    "QNN_OP_GATHER_ND": {ONNX: "GatherND", TENSORFLOW: "gather_nd"},
    "QNN_OP_GELU": {ONNX: MATCHED, TENSORFLOW: MATCHED, PYTORCH: "GELU"},
    "QNN_OP_GENERATE_PROPOSALS": {ONNX: "GenerateProposals(org.pytorch._caffe2) with im_info's img_scale = 1"},
    "QNN_OP_GRID_SAMPLE": {ONNX: "GridSample"},
    "QNN_OP_GROUP_NORM": {PYTORCH: "GroupNorm"},
    "QNN_OP_HARD_SIGMOID": {ONNX: "HardSigmoid"},
    "QNN_OP_HARD_SWISH": {ONNX: MATCHED, TENSORFLOW: MATCHED, PYTORCH: "Hardswish"},
    "QNN_OP_IMAGE_PROJECTION_TRANSFORM": {TENSORFLOW: "image.transform(ImageProjectiveTransform)"},
    "QNN_OP_INSTANCE_NORM": {ONNX: "InstanceNormalization", TENSORFLOW: MATCHED, PYTORCH: "InstanceNorm2d"},
    "QNN_OP_L2_NORM": {ONNX: "LpNormalization", TENSORFLOW: MATCHED},
    "QNN_OP_LAYERNORM": {ONNX: MATCHED, TENSORFLOW: "LayerNormalization", PYTORCH: "LayerNorm"},
    "QNN_OP_LOG_SOFTMAX": {ONNX: "LogSoftmax", TENSORFLOW: "log_softmax", PYTORCH: "LogSoftmax"},
    "QNN_OP_L2_POOL_2D": {ONNX: "LpPool"},
    "QNN_OP_LRN": {ONNX: "LRN", TENSORFLOW: "local_response_normalization", PYTORCH: "LocalResponseNorm"},
    "QNN_OP_LSTM": {ONNX: "LSTM", TENSORFLOW: MATCHED},
    "QNN_OP_MAT_MUL": {ONNX: "MatMul", TENSORFLOW: "matmul(BatchMatMul, BatchMatMulV2)", PYTORCH: "matmul"},
    "QNN_OP_MOMENTS": {TENSORFLOW: MATCHED},
    "QNN_OP_MULTI_CLASS_NMS": {ONNX: "nms + gather", TENSORFLOW: "nms + gather"},
    "QNN_OP_NON_MAX_SUPPRESSION": {ONNX: "NonMaxSuppression"},
    "QNN_OP_NON_ZERO": {ONNX: "NonZero"},
    "QNN_OP_NV21_TO_RGB": {ONNX: COMMAND_LINE, TENSORFLOW: COMMAND_LINE},
    "QNN_OP_NV12_TO_RGB": {ONNX: COMMAND_LINE, TENSORFLOW: COMMAND_LINE},
    "QNN_OP_ONE_HOT": {ONNX: "OneHot", TENSORFLOW: "one_hot"},
    "QNN_OP_PACK": {TENSORFLOW: "stack(Stack, Pack)", PYTORCH: "stack"},
    "QNN_OP_PAD": {ONNX: "Pad", TENSORFLOW: "pad(Pad, PadV2)", PYTORCH: "ConstantPad"},
    "QNN_OP_POOL_AVG_2D": {ONNX: "AveragePool, GlobalAveragePool", TENSORFLOW: "average_pooling2d", \
                           TENSORFLOW_LITE: "average_pool_2d", PYTORCH: "AvgPool2d"},
    "QNN_OP_POOL_AVG_3D": {ONNX: "AveragePool, GlobalAveragePool"},
    "QNN_OP_POOL_MAX_2D": {ONNX: "MaxPool, GlobalMaxPool", TENSORFLOW: "max_pooling2d", \
                           TENSORFLOW_LITE: "max_pool_2d", PYTORCH: "MaxPool2d"},
    "QNN_OP_POOL_MAX_3D": {ONNX: "MaxPool, GlobalMaxPool"},
    "QNN_OP_PRELU": {ONNX: "PRelu, LeakyRelu", TENSORFLOW: "PReLU", PYTORCH: "PReLU"},
    "QNN_OP_QUANTIZE": {ONNX: "QuantizeLinear"},
    "QNN_OP_REDUCE_MAX": {ONNX: "ReduceMax", TENSORFLOW: "reduce_max", PYTORCH: "max"},
    "QNN_OP_REDUCE_MEAN": {ONNX: "ReduceMean", TENSORFLOW: "reduce_mean", PYTORCH: "mean"},
    "QNN_OP_REDUCE_MIN": {ONNX: "ReduceMin", TENSORFLOW: "reduce_min", PYTORCH: "min"},
    "QNN_OP_REDUCE_PROD": {ONNX: "ReduceProd", TENSORFLOW: "reduce_prod", PYTORCH: "prod"},
    "QNN_OP_REDUCE_SUM": {ONNX: "ReduceSum", TENSORFLOW: "reduce_sum", PYTORCH: "sum"},
    "QNN_OP_RELU": {ONNX: "Relu", TENSORFLOW: "relu", TENSORFLOW_LITE: "relu", PYTORCH: "ReLU"},
    "QNN_OP_RELU6": {TENSORFLOW: "relu6", PYTORCH: "ReLU6"},
    "QNN_OP_RELU_MIN_MAX": {ONNX: "Clip", TENSORFLOW: MATCHED, TENSORFLOW_LITE: "relu6", PYTORCH: "Hardtanh"},
    "QNN_OP_RESHAPE": {ONNX: "Reshape, Flatten, Squeeze, UnSqueeze",
                       TENSORFLOW: "reshape, squeeze, expand_dims", TENSORFLOW_LITE: "reshape", PYTORCH: "reshape"},
    "QNN_OP_RESIZE": {ONNX: "Resize", PYTORCH: "Resize"},
    "QNN_OP_RESIZE_BILINEAR": {ONNX: "Resize", TENSORFLOW: "resize_bilinear", TENSORFLOW_LITE: "resize_bilinear", PYTORCH: "UpsamplingBilinear2d"},
    "QNN_OP_RESIZE_NEAREST_NEIGHBOR": {ONNX: "Resize, ResizeNearest(org.pytorch._caffe2)", TENSORFLOW: "resize_nearest_neighbor"},
    "QNN_OP_ROI_ALIGN": {ONNX: "RoiAlign, RoIAlign(org.pytorch._caffe2)"},
    "QNN_OP_ROI_POOLING": {ONNX: "MaxRoiPool"},
    "QNN_OP_SCATTER_ELEMENTS": {ONNX: "ScatterElements, Scatter (deprecated)"},
    "QNN_OP_SCATTER_ND": {ONNX: "ScatterND"},
    "QNN_OP_SIGMOID": {ONNX: "Sigmoid", TENSORFLOW: "sigmoid", PYTORCH: "sigmoid"},
    "QNN_OP_SPACE_TO_BATCH": {TENSORFLOW: "space_to_batch(SpaceToBatchND)"},
    "QNN_OP_SPACE_TO_DEPTH": {ONNX: "SpaceToDepth", TENSORFLOW: "space_to_depth"},
    "QNN_OP_SOFTMAX": {ONNX: "Softmax", TENSORFLOW: "softmax", TENSORFLOW_LITE: "softmax", PYTORCH: "Softmax"},
    "QNN_OP_SPLIT": {ONNX: "Split", TENSORFLOW: "split(Split, SplitV)", PYTORCH: "split"},
    "QNN_OP_STRIDED_SLICE": {ONNX: "Slice", TENSORFLOW: "strided_slice", TENSORFLOW_LITE: "slice"},
    "QNN_OP_TANH": {ONNX: "Tanh", TENSORFLOW: "tanh", TENSORFLOW_LITE: "tanh", PYTORCH: "tanh"},
    "QNN_OP_TILE": {ONNX: "Tile", TENSORFLOW: "tile"},
    "QNN_OP_TOP_K": {ONNX: "TopK", TENSORFLOW: "top_k", PYTORCH: "topk"},
    "QNN_OP_TRANSPOSE": {ONNX: "Transpose", TENSORFLOW: "transpose", PYTORCH: "transpose"},
    "QNN_OP_UN_PACK": {TENSORFLOW: "unstack", PYTORCH: "unbind"}
}


def converter_supports_op(converter_type, qnn_op_name):
    if converter_type not in converters:
        raise TypeError("Converter must be one of {}".format(converters))
    if qnn_op_name in converter_op_support.keys() \
            and converter_type in converter_op_support[qnn_op_name].keys():
        return True
    return False


def get_source_op(converter_type, qnn_op_name):
    if not converter_supports_op(converter_type, qnn_op_name):
        return NONE
    return converter_op_support[qnn_op_name][converter_type]
