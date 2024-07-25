# =============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import numpy as np

try:
    from . import qnn_definitions
    from . import ir_graph
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that product python lib is in your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common.converter_ir import op_adapter


# ------------------------------------------------------------------------------
#   Module Level utility enum/Functions
# ------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# IR consts to qnn dictionary. This holds the translation between the string constants in IR graph
# to what is defined in qnn_definitions.
# -------------------------------------------------------------------------------------------------
ir_consts_to_qnn = {

    # Eltwise ternary operations
    op_adapter.ElementwiseTernaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SELECT]: ir_graph.QNN_OP_ELEMENT_WISE_SELECT,

    # Eltwise binary operations
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]: ir_graph.QNN_OP_ELEMENT_WISE_ADD,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND]: ir_graph.QNN_OP_ELEMENT_WISE_AND,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE]: ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EQUAL]: ir_graph.QNN_OP_ELEMENT_WISE_EQUAL,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV]: ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FMOD]: ir_graph.QNN_OP_ELEMENT_WISE_FMOD,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER]: ir_graph.QNN_OP_ELEMENT_WISE_GREATER,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL]: ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS]: ir_graph.QNN_OP_ELEMENT_WISE_LESS,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL]: ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_POWER]: ir_graph.QNN_OP_ELEMENT_WISE_POWER,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY]: ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM]: ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM]: ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MOD]: ir_graph.QNN_OP_ELEMENT_WISE_MOD,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL]: ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_OR]: ir_graph.QNN_OP_ELEMENT_WISE_OR,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT]: ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT,
    op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_XOR]: ir_graph.QNN_OP_ELEMENT_WISE_XOR,


    # Eltwise unary operations
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN,
    op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT]: ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT,

    # ExtractGlimpse
    op_adapter.ExtractGlimpseOp.NoiseType.UNIFORM: qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_NOISE_UNIFORM,
    op_adapter.ExtractGlimpseOp.NoiseType.GAUSSIAN: qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_NOISE_GAUSSIAN,
    op_adapter.ExtractGlimpseOp.NoiseType.ZERO: qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_NOISE_ZEROES,

    # Reduce
    op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX]: {"qnn_type": qnn_definitions.QNN_OP_REDUCE_MAX,
                                                                        "axes": qnn_definitions.QNN_OP_REDUCE_MAX_PARAM_AXES,
                                                                        "keep_dims": qnn_definitions.QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS},
    op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN]: {"qnn_type": qnn_definitions.QNN_OP_REDUCE_MEAN,
                                                                         "axes": qnn_definitions.QNN_OP_REDUCE_MEAN_PARAM_AXES,
                                                                         "keep_dims": qnn_definitions.QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS},
    op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN]: {"qnn_type": qnn_definitions.QNN_OP_REDUCE_MIN,
                                                                        "axes": qnn_definitions.QNN_OP_REDUCE_MIN_PARAM_AXES,
                                                                        "keep_dims": qnn_definitions.QNN_OP_REDUCE_MIN_PARAM_KEEP_DIMS},
    op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD]: {"qnn_type": qnn_definitions.QNN_OP_REDUCE_PROD,
                                                                         "axes": qnn_definitions.QNN_OP_REDUCE_PROD_PARAM_AXES,
                                                                         "keep_dims": qnn_definitions.QNN_OP_REDUCE_PROD_PARAM_KEEP_DIMS},
    op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM]: {"qnn_type": qnn_definitions.QNN_OP_REDUCE_SUM,
                                                                        "axes": qnn_definitions.QNN_OP_REDUCE_SUM_PARAM_AXES,
                                                                        "keep_dims": qnn_definitions.QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS},
}

numpy_dtype_to_qnn = {
    # int types
    np.dtype('int8'): ir_graph.QNN_DATATYPE_INT_8,
    np.dtype('int16'): ir_graph.QNN_DATATYPE_INT_16,
    np.dtype('int32'): ir_graph.QNN_DATATYPE_INT_32,
    np.dtype('int64'): ir_graph.QNN_DATATYPE_INT_64,
    np.dtype('uint8'): ir_graph.QNN_DATATYPE_UINT_8,
    np.dtype('uint16'): ir_graph.QNN_DATATYPE_UINT_16,
    np.dtype('uint32'): ir_graph.QNN_DATATYPE_UINT_32,
    np.dtype('uint64'): ir_graph.QNN_DATATYPE_UINT_64,

    # float types
    np.dtype('float16'): ir_graph.QNN_DATATYPE_FLOAT_16,
    np.dtype('float32'): ir_graph.QNN_DATATYPE_FLOAT_32,
    np.dtype('float64'): ir_graph.QNN_DATATYPE_FLOAT_32,

    # bool type
    np.dtype('bool'): ir_graph.QNN_DATATYPE_BOOL_8
}

qnn_to_numpy_dtype = {
    # int types
    ir_graph.QNN_DATATYPE_INT_8: np.dtype('int8'),
    ir_graph.QNN_DATATYPE_INT_16: np.dtype('int16'),
    ir_graph.QNN_DATATYPE_INT_32: np.dtype('int32'),
    ir_graph.QNN_DATATYPE_INT_64: np.dtype('int64'),
    ir_graph.QNN_DATATYPE_UINT_8: np.dtype('uint8'),
    ir_graph.QNN_DATATYPE_UINT_16: np.dtype('uint16'),
    ir_graph.QNN_DATATYPE_UINT_32: np.dtype('uint32'),
    ir_graph.QNN_DATATYPE_UINT_64: np.dtype('uint64'),

    # float types
    ir_graph.QNN_DATATYPE_FLOAT_16: np.dtype('float16'),
    ir_graph.QNN_DATATYPE_FLOAT_32: np.dtype('float32'),

    # bool type
    ir_graph.QNN_DATATYPE_BOOL_8: np.dtype('bool'),

    # qnn quantized types
    ir_graph.QNN_DATATYPE_UFIXED_POINT_8: np.dtype('uint8'),
    ir_graph.QNN_DATATYPE_SFIXED_POINT_8: np.dtype('int8'),
    ir_graph.QNN_DATATYPE_UFIXED_POINT_16: np.dtype('uint16'),
    ir_graph.QNN_DATATYPE_SFIXED_POINT_16: np.dtype('int16'),
    ir_graph.QNN_DATATYPE_UFIXED_POINT_32: np.dtype('uint32'),
    ir_graph.QNN_DATATYPE_SFIXED_POINT_32: np.dtype('int32')
}

qnn_quantized_types = [ir_graph.QNN_DATATYPE_SFIXED_POINT_8, ir_graph.QNN_DATATYPE_SFIXED_POINT_16,
                       ir_graph.QNN_DATATYPE_SFIXED_POINT_32, ir_graph.QNN_DATATYPE_UFIXED_POINT_8,
                       ir_graph.QNN_DATATYPE_UFIXED_POINT_16, ir_graph.QNN_DATATYPE_UFIXED_POINT_32]
