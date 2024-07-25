# ==============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import log_debug2, log_debug3, converter_type, log_assert
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ArgOp,
    ConstantOp,
    CumSumOp,
    ElementwiseBinaryOp,
    ElementwiseTernaryOp,
    ElementwiseUnaryOp,
    ErfOp,
    L2NormOp,
    ElementwiseNeuronOp,
    ReduceOp,
    TopKOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase, validate_const_name
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay

import numpy as np


# ------------------------------------------------------------------------------
#   ArgBase
# ------------------------------------------------------------------------------
class RelayArgBaseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayArgBaseTranslation, self).__init__()
        self.ir_op_type = None

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        axis = relay_expr.attrs.axis
        keepdims = relay_expr.attrs.keepdims
        exclude = relay_expr.attrs.exclude

        if isinstance(axis, tvm.ir.container.Array):
            axis = [int(i) for i in axis]
        elif isinstance(axis, tvm.tir.expr.IntImm):
            axis = int(axis)
        else:
            TypeError("{} axis is of Unsupported datatype {}".format(self.ir_op_type, type(axis)))

        if keepdims:
            keepdims = True
        else:
            keepdims = False

        attr_dict["axis"] = axis
        attr_dict["keepdims"] = keepdims
        attr_dict["exclude"] = exclude

        log_debug3("\taxis {} {}", type(axis), axis)
        log_debug3("\tkeepdims {} {}", type(keepdims), keepdims)
        log_debug3("\texclude {} {}", type(exclude), exclude)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ArgOp.TRANSLATION_KEY, ArgOp.ir_to_legacy_type[self.ir_op_type])
        input_shapes = converter_context.get_input_shapes(relay_expr)

        axis = attr_dict["axis"]
        keepdims = attr_dict["keepdims"]
        exclude = attr_dict["exclude"]

        if exclude:
            axis = [i for i in range(len(input_shapes[0])) if i not in axis]

        if isinstance(axis, list):
            if len(axis) > 1:
                raise ValueError("{} axis only supported as scalar, got list {}".format(self.ir_op_type, axis))
            axis = axis[0]

        ir_op = ArgOp(op_name,
                      arg_type = self.ir_op_type,
                      axis=axis,
                      keep_dims=keepdims)

        return ir_op


# ------------------------------------------------------------------------------
#   ArgMax
# ------------------------------------------------------------------------------
class RelayArgMaxTranslation(RelayArgBaseTranslation):
    def __init__(self):
        super(RelayArgMaxTranslation, self).__init__()
        self.ir_op_type = ir_graph.QNN_OP_ARGMAX


RelayTranslations.register_translation(RelayArgMaxTranslation(),
                                       converter_type('argmax', 'relay'))


# ------------------------------------------------------------------------------
#   ArgMin
# ------------------------------------------------------------------------------
class RelayArgMinTranslation(RelayArgBaseTranslation):
    def __init__(self):
        super(RelayArgMinTranslation, self).__init__()
        self.ir_op_type = ir_graph.QNN_OP_ARGMIN


RelayTranslations.register_translation(RelayArgMinTranslation(),
                                       converter_type('argmin', 'relay'))


# ------------------------------------------------------------------------------
#   CumulativeSum
# ------------------------------------------------------------------------------
class RelayCumulativeSumTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCumulativeSumTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attrs = relay_expr.attrs
        attr_dict['axis'] = attrs.axis
        if attr_dict['axis'] is None:
            attr_dict['axis'] = 0
        attr_dict['exclusive'] = attrs.exclusive
        if attr_dict['exclusive'] is None:
            attr_dict['exclusive'] = False

        log_debug3("\axis {}", attr_dict['axis'])
        log_debug3("\exclusive {}", attr_dict['exclusive'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, CumSumOp.TRANSLATION_KEY, CumSumOp.LEGACY_TRANSLATION_KEY)
        axis_attrs = attr_dict.get('axis')
        if isinstance(axis_attrs, tvm.tir.IntImm):
            axis_attrs = int(axis_attrs)
        ir_op = CumSumOp(name=op_name,
                         axis=axis_attrs,
                         exclusive=attr_dict.get('exclusive'))
        return ir_op


RelayTranslations.register_translation(RelayCumulativeSumTranslation(),
                                       converter_type('cumsum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseTernary
# ------------------------------------------------------------------------------
class RelayElementwiseTernaryBaseTranslation(RelayTranslationBase):
    def __init__(self, ir_op_type):
        super(RelayElementwiseTernaryBaseTranslation, self).__init__()
        self.ir_op_type = ir_op_type

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, self.ir_op_type,
                                                ElementwiseTernaryOp.ir_to_legacy_type[self.ir_op_type])

        for idx, input_name in enumerate(input_names):
            if not quir_graph.has_buffer(input_name):
                # Op has input as Constant tensor
                input_tensor = relay_params[input_name]
                log_debug3("\tconst tensor type {} shape {}", type(input_tensor), input_tensor.shape)
                if isinstance(input_tensor, tvm.runtime.ndarray.NDArray) or \
                        isinstance(input_tensor, tvm.runtime.NDArray):
                    input_tensor = input_tensor.asnumpy()

                if not input_tensor.shape:
                    input_tensor = np.reshape(input_tensor.data, (1,))
                log_debug3("\tconst tensor after type {} shape {}", type(input_tensor), input_tensor.shape)
                constant_output_name = op_name + "_const_" + str(idx)
                constant_output_name = validate_const_name(quir_graph, input_name, constant_output_name)
                input_names[idx] = constant_output_name

                log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
                self.populate_quantization_params(relay_expr.args[idx], converter_context, quir_graph, [constant_output_name], is_param=True)
                quir_graph.add(ConstantOp(constant_output_name, input_tensor), [], [constant_output_name])

        ir_op = ElementwiseTernaryOp(op_name, eltwise_type=self.ir_op_type)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseBinaryBase
# ------------------------------------------------------------------------------
class RelayElementwiseBinaryBaseTranslation(RelayTranslationBase):
    def __init__(self, ir_op_type, operation):
        super(RelayElementwiseBinaryBaseTranslation, self).__init__()
        self.ir_op_type = ir_op_type
        self.operation = operation

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, self.ir_op_type,
                                                ElementwiseBinaryOp.ir_to_legacy_type[self.ir_op_type])

        for idx, input_name in enumerate(input_names):
            if not quir_graph.has_buffer(input_name):
                # Op has input as Constant tensor
                input_tensor = relay_params[input_name]
                log_debug3("\tconst tensor type {} shape {}", type(input_tensor), input_tensor.shape)
                if isinstance(input_tensor, tvm.runtime.ndarray.NDArray) or \
                        isinstance(input_tensor, tvm.runtime.NDArray):
                    input_tensor = input_tensor.asnumpy()

                if not input_tensor.shape:
                    input_tensor = np.reshape(input_tensor.data, (1,))
                log_debug3("\tconst tensor after type {} shape {}", type(input_tensor), input_tensor.shape)

                constant_output_name = op_name + "_const_" + str(idx)
                constant_output_name = validate_const_name(quir_graph, input_name, constant_output_name)
                input_names[idx] = constant_output_name

                log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
                self.populate_quantization_params(relay_expr.args[idx], converter_context, quir_graph, [constant_output_name], is_param=True)
                quir_graph.add(ConstantOp(constant_output_name, input_tensor), [], [constant_output_name])

        ir_op = ElementwiseBinaryOp(op_name, operation=self.operation)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseAnd
# ------------------------------------------------------------------------------
class RelayElementwiseAndTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseAndTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_AND,
                                                             ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND)

RelayTranslations.register_translation(RelayElementwiseAndTranslation(),
                                       converter_type('logical_and', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseDiv
# ------------------------------------------------------------------------------
class RelayElementwiseDivTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseDivTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE,
                                                             ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE)

RelayTranslations.register_translation(RelayElementwiseDivTranslation(),
                                       converter_type('divide', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseEqual
# ------------------------------------------------------------------------------
class RelayElementwiseEqualTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseEqualTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_EQUAL,
                                                               ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL)

RelayTranslations.register_translation(RelayElementwiseEqualTranslation(),
                                       converter_type('equal', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseFloorDiv
# ------------------------------------------------------------------------------
class RelayElementwiseFloorDivTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseFloorDivTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV,
                                                                  ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV)

RelayTranslations.register_translation(RelayElementwiseFloorDivTranslation(),
                                       converter_type('floor_divide', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseGreater
# ------------------------------------------------------------------------------
class RelayElementwiseGreaterTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseGreaterTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_GREATER,
                                                                 ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER)

RelayTranslations.register_translation(RelayElementwiseGreaterTranslation(),
                                       converter_type('greater', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseGreaterEqual
# ------------------------------------------------------------------------------
class RelayElementwiseGreaterEqualTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseGreaterEqualTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL,
                                                                      ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL)

RelayTranslations.register_translation(RelayElementwiseGreaterEqualTranslation(),
                                       converter_type('greater_equal', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseLess
# ------------------------------------------------------------------------------
class RelayElementwiseLessTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseLessTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_LESS,
                                                              ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS)

RelayTranslations.register_translation(RelayElementwiseLessTranslation(),
                                       converter_type('less', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseLessEqual
# ------------------------------------------------------------------------------
class RelayElementwiseLessEqualTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseLessEqualTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL,
                                                                   ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL)

RelayTranslations.register_translation(RelayElementwiseLessEqualTranslation(),
                                       converter_type('less_equal', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseMaximum
# ------------------------------------------------------------------------------
class RelayElementwiseMaximumTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseMaximumTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM,
                                                                 ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM)

RelayTranslations.register_translation(RelayElementwiseMaximumTranslation(),
                                       converter_type('maximum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseMinimum
# ------------------------------------------------------------------------------
class RelayElementwiseMinimumTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseMinimumTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM,
                                                                 ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM)

RelayTranslations.register_translation(RelayElementwiseMinimumTranslation(),
                                       converter_type('minimum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseNotEqual
# ------------------------------------------------------------------------------
class RelayElementwiseNotEqualTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseNotEqualTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL,
                                                                  ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL)

RelayTranslations.register_translation(RelayElementwiseNotEqualTranslation(),
                                       converter_type('not_equal', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseOr
# ------------------------------------------------------------------------------
class RelayElementwiseOrTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseOrTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_OR,
                                                            ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR)

RelayTranslations.register_translation(RelayElementwiseOrTranslation(),
                                       converter_type('logical_or', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwisePower
# ------------------------------------------------------------------------------
class RelayElementwisePowerTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwisePowerTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_POWER,
                                                               ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER)

RelayTranslations.register_translation(RelayElementwisePowerTranslation(),
                                       converter_type('power', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseProd
# ------------------------------------------------------------------------------
class RelayElementwiseProdTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseProdTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                              ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)

RelayTranslations.register_translation(RelayElementwiseProdTranslation(),
                                       converter_type('multiply', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSub
# ------------------------------------------------------------------------------
class RelayElementwiseSubTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSubTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT,
                                                             ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT)

RelayTranslations.register_translation(RelayElementwiseSubTranslation(),
                                       converter_type('subtract', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSum
# ------------------------------------------------------------------------------
class RelayElementwiseSumTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSumTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_ADD,
                                                             ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)

RelayTranslations.register_translation(RelayElementwiseSumTranslation(),
                                       converter_type('add', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSelect
# ------------------------------------------------------------------------------
class RelayElementwiseSelectTranslation(RelayElementwiseTernaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSelectTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_SELECT)

RelayTranslations.register_translation(RelayElementwiseSelectTranslation(),
                                       converter_type('where', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseXor
# ------------------------------------------------------------------------------
class RelayElementwiseXorTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseXorTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_XOR,
                                                             ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR)

RelayTranslations.register_translation(RelayElementwiseXorTranslation(),
                                       converter_type('logical_xor', 'relay'))


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class RelayTanhTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTanhTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ElementwiseNeuronOp.TRANSLATION_KEY,
                                                ElementwiseNeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = ElementwiseNeuronOp(op_name,
                         operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
                         alpha=1.0,
                         beta=1.0)
        return ir_op


RelayTranslations.register_translation(RelayTanhTranslation(),
                                       converter_type('tanh', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryBase
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryBaseTranslation(RelayTranslationBase):
    def __init__(self, operation):
        super(RelayElementwiseUnaryBaseTranslation, self).__init__()
        self.operation = operation

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr,
                                                ElementwiseUnaryOp.operation_to_legacy[self.operation],
                                                ElementwiseUnaryOp.operation_to_legacy[self.operation])
        ir_op = ElementwiseUnaryOp(op_name, operation=self.operation)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseUnaryAbs
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryAbsTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryAbsTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS)

RelayTranslations.register_translation(RelayElementwiseUnaryAbsTranslation(),
                                       converter_type('abs', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryAsin
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryAsinTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryAsinTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN)

RelayTranslations.register_translation(RelayElementwiseUnaryAsinTranslation(),
                                       converter_type('asin', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryAtan
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryAtanTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryAtanTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN)

RelayTranslations.register_translation(RelayElementwiseUnaryAtanTranslation(),
                                       converter_type('atan', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryCeil
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryCeilTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryCeilTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL)

RelayTranslations.register_translation(RelayElementwiseUnaryCeilTranslation(),
                                       converter_type('ceil', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryCos
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryCosTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryCosTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS)

RelayTranslations.register_translation(RelayElementwiseUnaryCosTranslation(),
                                       converter_type('cos', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryExp
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryExpTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryExpTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP)

RelayTranslations.register_translation(RelayElementwiseUnaryExpTranslation(),
                                       converter_type('exp', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryFloor
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryFloorTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryFloorTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR)

RelayTranslations.register_translation(RelayElementwiseUnaryFloorTranslation(),
                                       converter_type('floor', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryLog
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryLogTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryLogTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG)

RelayTranslations.register_translation(RelayElementwiseUnaryLogTranslation(),
                                       converter_type('log', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryNeg
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryNegTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryNegTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG)

RelayTranslations.register_translation(RelayElementwiseUnaryNegTranslation(),
                                       converter_type('negative', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryNot
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryNotTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryNotTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT)

RelayTranslations.register_translation(RelayElementwiseUnaryNotTranslation(),
                                       converter_type('logical_not', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryRound
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryRoundTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryRoundTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND)

RelayTranslations.register_translation(RelayElementwiseUnaryRoundTranslation(),
                                       converter_type('round', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryRsqrt
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryRsqrtTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryRsqrtTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT)

RelayTranslations.register_translation(RelayElementwiseUnaryRsqrtTranslation(),
                                       converter_type('rsqrt', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnarySign
# ------------------------------------------------------------------------------
class RelayElementwiseUnarySignTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnarySignTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN)

RelayTranslations.register_translation(RelayElementwiseUnarySignTranslation(),
                                       converter_type('sign', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnarySin
# ------------------------------------------------------------------------------
class RelayElementwiseUnarySinTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnarySinTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN)

RelayTranslations.register_translation(RelayElementwiseUnarySinTranslation(),
                                       converter_type('sin', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnarySqrt
# ------------------------------------------------------------------------------
class RelayElementwiseUnarySqrtTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnarySqrtTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT)

RelayTranslations.register_translation(RelayElementwiseUnarySqrtTranslation(),
                                       converter_type('sqrt', 'relay'))


# ------------------------------------------------------------------------------
#   Erf
# ------------------------------------------------------------------------------
class RelayErfTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayErfTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ErfOp.TRANSLATION_KEY, ErfOp.LEGACY_TRANSLATION_KEY)

        ir_op = ErfOp(op_name)
        return ir_op


RelayTranslations.register_translation(RelayErfTranslation(),
                                       converter_type('erf', 'relay'))


# ------------------------------------------------------------------------------
#   L2Normalization
# ------------------------------------------------------------------------------
class RelayL2NormalizationTranslation(RelayTranslationBase, ABC):
    def __init__(self):
        super(RelayL2NormalizationTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['eps'] = float(relay_expr.attrs.eps)
        if relay_expr.attrs.axis is None:
            attr_dict['axes'] = None
        else:
            attr_dict['axes'] = [int(axis) for axis in relay_expr.attrs.axis]

        log_debug3("\teps {}", attr_dict['eps'])
        log_debug3("\taxes {}", attr_dict['axes'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, L2NormOp.TRANSLATION_KEY, L2NormOp.LEGACY_TRANSLATION_KEY)

        input_shapes = converter_context.get_input_shapes(relay_expr)
        input_rank = len(input_shapes[0])

        eps = attr_dict['eps']
        if attr_dict['axes'] is None:
            axes = list(range(input_rank))
        else:
            axes = [axis % input_rank for axis in attr_dict['axes']]

        ir_op = L2NormOp(op_name,
                         epsilon=eps,
                         axes=axes)

        return ir_op

RelayTranslations.register_translation(RelayL2NormalizationTranslation(),
                                       converter_type('l2_normalize', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceOp
# ------------------------------------------------------------------------------
class RelayReduceBaseTranslation(RelayTranslationBase, ABC):
    def __init__(self, reduce_type):
        super(RelayReduceBaseTranslation, self).__init__()
        self.reduce_type = reduce_type

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['keep_dims'] = relay_expr.attrs.keepdims
        axis_attrs = relay_expr.attrs.axis
        if isinstance(axis_attrs, tvm.ir.container.Array):
            attr_dict['axis'] = [int(i) for i in axis_attrs]
        elif isinstance(axis_attrs, tvm.tir.IntImm):
            attr_dict['axis'] = [int(axis_attrs)]
        else:
            attr_dict['axis'] = list()
        attr_dict['exclude'] = relay_expr.attrs.exclude

        log_debug3("\taxis {}", attr_dict['axis'])
        log_debug3("\tkeep_dims {}", attr_dict['keep_dims'])
        log_debug3("\texclude {}", attr_dict['exclude'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, self.reduce_type,
                                                ReduceOp.ir_to_legacy_type[self.reduce_type])
        axis = attr_dict['axis']
        exclude = attr_dict['exclude']
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        input_dim = len(input_shape)
        if len(axis) == 0:
            axis = [i for i in range(input_dim)]

        if exclude:
            axis = [i for i in range(input_dim) if i not in axis]

        keep_dims = attr_dict['keep_dims']
        ir_op = ReduceOp(op_name,
                         reduce_type=self.reduce_type,
                         axes=axis,
                         keep_dims=keep_dims)

        return ir_op


# ------------------------------------------------------------------------------
#   ReduceMaxOp
# ------------------------------------------------------------------------------
class RelayReduceMaxTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceMaxTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_MAX)

RelayTranslations.register_translation(RelayReduceMaxTranslation(),
                                       converter_type('max', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceMinOp
# ------------------------------------------------------------------------------
class RelayReduceMinTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceMinTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_MIN)

RelayTranslations.register_translation(RelayReduceMinTranslation(),
                                       converter_type('min', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceMeanOp
# ------------------------------------------------------------------------------
class RelayReduceMeanTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceMeanTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_MEAN)

RelayTranslations.register_translation(RelayReduceMeanTranslation(),
                                       converter_type('mean', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceProdOp
# ------------------------------------------------------------------------------
class RelayReduceProdTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceProdTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_PROD)

RelayTranslations.register_translation(RelayReduceProdTranslation(),
                                       converter_type('prod', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceSumOp
# ------------------------------------------------------------------------------
class RelayReduceSumTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceSumTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_SUM)

RelayTranslations.register_translation(RelayReduceSumTranslation(),
                                       converter_type('sum', 'relay'))


# ------------------------------------------------------------------------------
#   TopKOp
# ------------------------------------------------------------------------------
class RelayTopKTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTopKTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        k = int(relay_expr.attrs.k)
        axis = int(relay_expr.attrs.axis)
        is_ascend = relay_expr.attrs.is_ascend
        attr_dict["k"] = k
        attr_dict["axis"] = axis
        attr_dict["is_ascend"] = is_ascend
        attr_dict["sorted"] = True
        log_debug3("\tk {} {}", type(k), k)
        log_debug3("\taxis {} {}", type(axis), axis)
        log_debug3("\tis_ascend {} {}", type(is_ascend), is_ascend)
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TopKOp.TRANSLATION_KEY, TopKOp.LEGACY_TRANSLATION_KEY)
        k = attr_dict['k']
        input_shapes = converter_context.get_input_shapes(relay_expr)
        axis = attr_dict['axis']
        input_rank = len(input_shapes[0])
        if axis < 0:
            axis += input_rank
        if axis != input_rank - 1:
            raise ValueError("TopK axis only supported as input_rank-1, got axis {}".format(axis))
        if k < 1:
        #TVM topk define:k (int or relay.Expr, optional) – Number of top elements to select. Return all elements if k < 1.
            k = input_shapes[0][axis]
        if k > input_shapes[0][axis]:
            raise ValueError("ERROR_TOPK_K_INVALID:{}".format(k))
        is_ascend = attr_dict['is_ascend']
        is_sorted = attr_dict['sorted']
        largest = not is_ascend
        log_assert(largest, "TopK Op {}, attribute largest must be True", op_name)
        log_assert(is_sorted, "TopK Op {}, attribute sorted must be True", op_name)
        log_assert(input_rank >= 1, "ERROR_TOPK_INPUT_TENSOR_RANK:input_rank should >= 1,but got {}", input_rank)

        ir_op = TopKOp(op_name, k=k)

        return ir_op

RelayTranslations.register_translation(RelayTopKTranslation(),
                                       converter_type('topk', 'relay'))
