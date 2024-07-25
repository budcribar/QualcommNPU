# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


from .onnx_translations import *

import distutils
from distutils import version
import packaging
from enum import Enum

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import *

# ------------------------------------------------------------------------------
#   Abs
# ------------------------------------------------------------------------------
class OnnxAbsTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Abs', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS
        self.numpy_op = np.abs


OnnxTranslations.register_translation(OnnxAbsTranslation(),
                                      converter_type('Abs', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS])


# ------------------------------------------------------------------------------
#   Add
# ------------------------------------------------------------------------------
class OnnxAddTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Add', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD
        self.numpy_op = np.add


OnnxTranslations.register_translation(OnnxAddTranslation(),
                                      converter_type('Add', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD])


# ------------------------------------------------------------------------------
#   And
# ------------------------------------------------------------------------------
class OnnxAndTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('And', [1, 7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND
        self.numpy_op = np.logical_and


OnnxTranslations.register_translation(OnnxAndTranslation(),
                                      converter_type('And', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND])


# ------------------------------------------------------------------------------
#   ArgMax, ArgMin
# ------------------------------------------------------------------------------
class OnnxArgOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ArgMax', [1, 11, 12, 13])
        self.register_op_schema('ArgMin', [1, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        schema = self.op_schema(op_type=src_op.op_type)
        # these parameters depends on src_op.op_type(ArgMax/ArgMin)
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type), validate=True)

        if "select_last_index" in params and params.select_last_index == 1:
            log_warning("select_last_index is set to 1 for the op - {}. Only 0(default value) is supported. This may cause accuracy issues.".format(src_op.name))

        if str(src_op.op_type) == 'ArgMax':
            arg_type = ir_graph.QNN_OP_ARGMAX
        elif str(src_op.op_type) == 'ArgMin':
            arg_type = ir_graph.QNN_OP_ARGMIN

        return op_adapter.ArgOp(str(src_op.name),
                                arg_type = arg_type,
                                axis=params.axis,
                                keep_dims=params.keepdims)


OnnxTranslations.register_translation(OnnxArgOpTranslation(),
                                      converter_type('ArgMax', 'onnx'),
                                      converter_type('ArgMin', 'onnx'),
                                      op_adapter.ArgOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Asin
# ------------------------------------------------------------------------------
class OnnxAsinTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Asin', [7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN
        self.numpy_op = np.arcsin


OnnxTranslations.register_translation(OnnxAsinTranslation(),
                                      converter_type('Asin', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN])


# ------------------------------------------------------------------------------
#   Atan
# ------------------------------------------------------------------------------
class OnnxAtanTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Atan', [7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN
        self.numpy_op = np.arctan


OnnxTranslations.register_translation(OnnxAtanTranslation(),
                                      converter_type('Atan', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN])


# ------------------------------------------------------------------------------
#   Ceil
# ------------------------------------------------------------------------------
class OnnxCeilTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Ceil', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL
        self.numpy_op = np.ceil


OnnxTranslations.register_translation(OnnxCeilTranslation(),
                                      converter_type('Ceil', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL])


# ------------------------------------------------------------------------------
#   Cos
# ------------------------------------------------------------------------------
class OnnxCosTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Cos', [7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS
        self.numpy_op = np.cos


OnnxTranslations.register_translation(OnnxCosTranslation(),
                                      converter_type('Cos', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS])


# ------------------------------------------------------------------------------
#   CumSum
# ------------------------------------------------------------------------------
class OnnxCumSumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('CumSum', [11, 14])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()

        # extract axis param
        const_op = self.fetch_constant_op(input_names[1], converter_context, fail_if_dynamic=True)
        axis = const_op.tensor.astype(np.int32).item(0)

        # check for axis to be in [-r,r-1]
        if axis not in range(-input_rank, input_rank):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(axis, input_names[1], src_op.op_type))
        if axis < 0:
            axis += input_rank

        # extract reverse and exclusive params
        reverse = params.reverse if 'reverse' in params else 0
        if reverse not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(reverse, "reverse", src_op.op_type))
        exclusive = params.exclusive if 'exclusive' in params else 0
        if exclusive not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(exclusive, "exclusive", src_op.op_type))

        # axis received as input, but added as param in our IR graph
        return op_adapter.CumSumOp(str(src_op.name),
                                   axis=axis,
                                   reverse=bool(reverse),
                                   exclusive=bool(exclusive))

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]

OnnxTranslations.register_translation(OnnxCumSumTranslation(),
                                      converter_type('CumSum', 'onnx'),
                                      op_adapter.CumSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Div
# ------------------------------------------------------------------------------
class OnnxDivTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Div', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE
        self.numpy_op = np.divide


OnnxTranslations.register_translation(OnnxDivTranslation(),
                                      converter_type('Div', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE])


# ------------------------------------------------------------------------------
#   Einsum
# ------------------------------------------------------------------------------
class OnnxEinsumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Einsum', [12])
        self.transform_pattern = ""
        self.perm=[]

    class Pattern(Enum):
        MATMUL = 1
        MATMUL_TRANSPOSE_a = 2 # matmul(transpose(a), b).
        MATMUL_TRANSPOSE_b = 3 # matmul(a, transpose(b)).
        TRANSPOSE = 4

    def post_process_equation(self, equation):
        operand_str = []
        # check equation is explicit or implicit
        # only can support explicit equation, will enable implicit next step.
        if '->' not in equation:
            raise ValueError('Converter can not support implicit equation. Got {}.'.format(equation))
        # explicit equation
        else:
            lhs_equ, rhs_equ = equation.split('->')
            # TODO: limitations will be enabled next step.
            # can not support "..." in equation.
            # can not support inputs more then 2 tensors.
            # can not support outputs more then 1 tensor.
            if lhs_equ.count("...") >= 1 or rhs_equ.count("...") >= 1:
                raise ValueError('Converter can not support "..." broadcast in equation. Got {}.'.format(equation))
            elif len(lhs_equ.split(",")) > 2:
                raise ValueError('Converter can not support more then two inputs. Got {}.'.format(equation))
            elif len(rhs_equ.split(",")) > 1:
                raise ValueError('Converter can not support more then one output. Got {}.'.format(equation))

        # Einsum may have two or only one input.
        # when there are two inputs just split lhs_equ with ',' otherwise keep lhs_equ as the only one input.
        # invalid check and produce operand_str.
        # numpy reference https://np.org/doc/stable/reference/generated/np.einsum.html
        for lhs_item in lhs_equ.split(","):
            lhs_item = lhs_item.strip()
            if not lhs_item.isalpha():
                raise ValueError("equation {} has invalid symbol that is not alphabet.".format(equation))
            operand_str.append(lhs_item)
        operand_str.append(rhs_equ)

        # operand_str include inputs and output in equation
        return operand_str

    def extract_transform_pattern(self, operand_str, origin_equation):
        # scenario : two inputs, one output
        if len(operand_str) == 3:
            lhs_equ_in_first = operand_str[0]
            lhs_equ_in_second = operand_str[1]
            rhs_equ = operand_str[2]

            # extract transform pattern
            # pattern: matmul(a,b)
            # eg: "hbwij, hbwjc->hbwic" to "matmul"
            if lhs_equ_in_first[-1] == lhs_equ_in_second[-2] \
                and rhs_equ[-1] == lhs_equ_in_second[-1] \
                and rhs_equ[-2] == lhs_equ_in_first[-2]:
                    self.transform_pattern = self.Pattern.MATMUL
            # pattern: matmul(transpose(a),b)
            # eg: "hbwpc,hbwpq->hbwcq" to "transopse(a) + matmul"
            elif lhs_equ_in_first[-2] == lhs_equ_in_second[-2] \
                and rhs_equ[-1] == lhs_equ_in_second[-1] \
                and rhs_equ[-2] == lhs_equ_in_first[-1]:
                    self.transform_pattern = self.Pattern.MATMUL_TRANSPOSE_a
            # pattern: matmul(a,transpose(b))
            # eg: "hbwpc,hbwqc->hbwpq" to "transopse(b) + matmul"
            elif lhs_equ_in_first[-1] == lhs_equ_in_second[-1] \
                and rhs_equ[-1] == lhs_equ_in_second[-2] \
                and rhs_equ[-2] == lhs_equ_in_first[-2]:
                    self.transform_pattern = self.Pattern.MATMUL_TRANSPOSE_b
            else:
                # TODO: More patterns will be supported next step.
                raise ValueError('Converter can not support equation {}.'.format(origin_equation))
        # scenario : one input, one output
        elif len(operand_str) == 2:
            lhs_equ_in_first = operand_str[0]
            rhs_equ = operand_str[1]
            # extract transform pattern
            # pattern: transpose(a)
            # eg: "mnkj->mknj" to "transpose" with perm (0,2,1,3)
            if len(rhs_equ) == len(lhs_equ_in_first) \
                and set(rhs_equ) == set(lhs_equ_in_first):
                    # calculate perm parameter and set transform pattern.
                    self.perm = [lhs_equ_in_first.index(s) for s in rhs_equ]
                    self.transform_pattern = self.Pattern.TRANSPOSE
            else:
                # TODO: More patterns will be supported next step.
                raise ValueError('Converter can not support equation {}.'.format(origin_equation))
        else:
            raise ValueError('Got invalid equation {}'.format(origin_equation))

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = list(map(str, src_op.input))
        output_names = list(map(str, src_op.output))
        params = extract_attributes(src_op, schema=self.op_schema())

        # remove redundant spaces in equation.
        # eg: " hbwpc,  hbwqc ->  hbwpq " to "hbwpc,hbwqc->hbwpq"
        origin_equation = params.equation.replace(" ", "")

        # check inputs number match or not between operand str and Einsum Op.
        operand_str = self.post_process_equation(origin_equation)
        equ_input_num = len(operand_str) - 1
        op_input_num = len(input_names)
        if equ_input_num != op_input_num:
            raise ValueError("inputs number {} in equation not match inputs number {} of Einsum Op." \
                .format(equ_input_num, op_input_num))

        # extract transform pattern with inputs and output string in the valid equation.
        self.extract_transform_pattern(operand_str, origin_equation)

        # Transform Einsum to other Ops according to equation.
        # pattern 1: Transform Einsum to Matmul.
        if self.transform_pattern == self.Pattern.MATMUL:
            # If input A is static, add a Constant Op
            if converter_context.weights.has(input_names[0]):
                static_input_name = input_names[0]
                tensor = converter_context.weights.fetch(static_input_name, prunable=False)
                if not graph.has_buffer(static_input_name):
                    static_input_op = op_adapter.ConstantOp(static_input_name, tensor=tensor)
                    static_input_node = graph.add(static_input_op, [], [static_input_name])
                    graph.add_src_op_info(static_input_node.op.name, [], [static_input_name])

            # If input B is constant, add a Constant Op
            if converter_context.weights.has(input_names[1]):
                weight_input_name = input_names[1]
                weights = converter_context.weights.fetch(weight_input_name, prunable=False)
                if not graph.has_buffer(weight_input_name):
                    weights_constant_op = op_adapter.ConstantOp(weight_input_name, tensor=weights)
                    weight_node = graph.add(weights_constant_op, [], [weight_input_name])
                    graph.add_src_op_info(weight_node.op.name, [], [weight_input_name])

            return op_adapter.MatMulOp(name=str(src_op.name), transpose_in0=False, transpose_in1=False)

        # pattern 2: Transform Einsum to Matmul(transpose(in0), in1)
        elif self.transform_pattern == self.Pattern.MATMUL_TRANSPOSE_a:
            matmul_op = op_adapter.MatMulOp(src_op.name, transpose_in0=True, transpose_in1=False)
            graph.add_src_op_info(src_op.name, input_names, output_names)
            graph.add(matmul_op, input_names, output_names)

            return None

        # pattern 3: Transform Einsum to Matmul(in0, transpose(in1))
        elif self.transform_pattern == self.Pattern.MATMUL_TRANSPOSE_b:
            matmul_op = op_adapter.MatMulOp(src_op.name, transpose_in0=False, transpose_in1=True)
            graph.add_src_op_info(src_op.name, input_names, output_names)
            graph.add(matmul_op, input_names, output_names)

            return None

        # pattern 4: Transform Einsum to Transpose.
        elif self.transform_pattern == self.Pattern.TRANSPOSE:
            const_op = self.fetch_constant_op(input_names[0], converter_context, fail_if_dynamic=False, fail_if_not_found=True)
            if const_op is not None:
                # static permute of weight parameters
                output_name = str(src_op.output[0])
                w = const_op.tensor
                log_debug1('static input: {} to: {}'.format(input_names[0], w.shape))
                log_debug1('transpose shape to : {}'.format(params.perm))
                w = np.transpose(w, params.perm)
                converter_context.weights.insert(output_name, w)
                log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_names[0], output_name, w.shape))

                return None

            log_debug1('input: {} to: {}'.format(input_names[0], graph.get_buffer(input_names[0]).shape))
            log_debug1('transpose shape to : {}'.format(params.perm))

            return op_adapter.TransposeOp(src_op.name, params.perm)

        else:
            # TODO: More patterns will be supported next step.
            raise ValueError('No pattern matched, converter can not support equation {}.'.format(origin_equation))


OnnxTranslations.register_translation(OnnxEinsumTranslation(),
                                      converter_type('Einsum', 'onnx'))


# ------------------------------------------------------------------------------
#   Elu
# ------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Elu', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        # these parameters belong to Elu
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU,
                                              alpha=params.alpha)


OnnxTranslations.register_translation(OnnxEluTranslation(),
                                      converter_type('Elu', 'onnx'))


# ------------------------------------------------------------------------------
#   Equal
# ------------------------------------------------------------------------------
class OnnxEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Equal', [1, 7, 11, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL
        self.numpy_op = np.equal


OnnxTranslations.register_translation(OnnxEqualTranslation(),
                                      converter_type('Equal', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EQUAL])


# ------------------------------------------------------------------------------
#   Erf
# ------------------------------------------------------------------------------
class OnnxErfTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Erf', [9, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ErfOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxErfTranslation(),
                                      converter_type('Erf', 'onnx'),
                                      op_adapter.ErfOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Exp
# ------------------------------------------------------------------------------
class OnnxExpTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Exp', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP
        self.numpy_op = np.exp


OnnxTranslations.register_translation(OnnxExpTranslation(),
                                      converter_type('Exp', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP])


# ------------------------------------------------------------------------------
#   Floor
# ------------------------------------------------------------------------------
class OnnxFloorTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Floor', [1, 6])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR
        self.numpy_op = np.floor


OnnxTranslations.register_translation(OnnxFloorTranslation(),
                                      converter_type('Floor', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR])


# ------------------------------------------------------------------------------
#   GEMM
# ------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gemm', [1, 6, 7, 9, 11, 13])
        self.params = None

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_warning(code_to_message.get_warning_message("WARNING_GEMM"))
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))

        # In newer opset versions, bias is made an optional parameter
        # in the Gemm operator. Default value of bias in this case is 0
        # Get weights
        if not graph.has_buffer(input_names[1]):
            # If the weight is a constant, add the constant op to graph
            if converter_context.weights.has(input_names[1]):
                weights = converter_context.weights.fetch(input_names[1])
                weights_constant_op = op_adapter.ConstantOp(input_names[1], tensor=weights)
                weight_node = graph.add(weights_constant_op, [], [input_names[1]])
                graph.add_src_op_info(input_names[1], None, weight_node.output_names)
            # If the weight is neither in weight nor in graph, raise an error
            else:
                raise ValueError("Op {}'s weight is neither in converter_context.weights nor in graph.buffer".format(str(src_op.name)))

        # Add the multiplied op to graph for both dynamic and constant weights
        # Transpose input if transB is given, Weights = weights'
        if self.params.transB:
            trans_weight_name = input_names[1] + '_permute'
            if not graph.has_buffer(trans_weight_name):
                permute_op = op_adapter.TransposeOp(trans_weight_name, perm=[1, 0])
                graph.add(permute_op, [input_names[1]], [trans_weight_name])
                graph.add_src_op_info(permute_op.name, [input_names[1]], [trans_weight_name])
        else:
            trans_weight_name = input_names[1]

        # Weights = alpha * weights'
        if self.params.alpha != 1:
            alpha = self.params.alpha
            # Use input[0]'s name in case the alpha is different
            alpha_constant_name = input_names[0] + "_alpha"
            if not graph.has_buffer(alpha_constant_name):
                alpha_constant_op = op_adapter.ConstantOp(alpha_constant_name, tensor=alpha)
                graph.add(alpha_constant_op, [], [alpha_constant_name])
                graph.add_src_op_info(alpha_constant_name, None, [alpha_constant_name])

            alpha_mul_name = alpha_constant_name + "_multiply"
            if not graph.has_buffer(alpha_mul_name):
                alpha_mul_op = op_adapter.ElementwiseBinaryOp(alpha_mul_name,
                                                              operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                graph.add(alpha_mul_op, [trans_weight_name, alpha_constant_name], [alpha_mul_name])
                graph.add_src_op_info(alpha_mul_name, [trans_weight_name, alpha_constant_name], [alpha_mul_name])
            self.weights_name = alpha_mul_name
        else:
            self.weights_name = trans_weight_name

        # Get bias
        if len(src_op.input) >= 3:
            # If the bias is a constant, add the constant op to graph
            if not graph.has_buffer(input_names[2]):
                if converter_context.weights.has(input_names[2]):
                    # squeezing the bias to remove all single dimensional entries
                    bias = np.atleast_1d(np.squeeze(converter_context.weights.fetch(input_names[2])))
                    bias_constant_op = op_adapter.ConstantOp(input_names[2], tensor=bias)
                    graph.add(bias_constant_op, [], [input_names[2]])
                    graph.add_src_op_info(input_names[2], None, input_names[2])
                else:
                    raise ValueError("Op {}'s bias is neither in converter_context.weights nor in graph.buffer".format(str(src_op.name)))
            # Add the multiplied op to graph for both dynamic and constant bias
            # Bias = beta * bias
            if self.params.beta != 1:
                beta = self.params.beta
                # Use input[0]'s name in case the beta is different
                beta_constant_name = input_names[0] + "_beta"
                if not graph.has_buffer(beta_constant_name):
                    beta_constant_op = op_adapter.ConstantOp(beta_constant_name, tensor=beta)
                    graph.add(beta_constant_op, [], [beta_constant_name])
                beta_mul_name = beta_constant_name + "_multiply"
                if not graph.has_buffer(beta_mul_name):
                    beta_mul_op = op_adapter.ElementwiseBinaryOp(beta_mul_name,
                                                                 operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                    graph.add(beta_mul_op, [input_names[2], beta_constant_name], [beta_mul_name])
                    graph.add_src_op_info(beta_mul_name, [input_names[2], beta_constant_name], [beta_mul_name])
                self.bias_name = beta_mul_name
            else:
                self.bias_name = input_names[2]
        else:
            # If the bias is not given, create a zero tensor for bias
            self.bias_name = input_names[0] + '_b'
            bias = np.zeros((graph.get_buffer(self.weights_name).shape[1]))
            if not graph.has_buffer(self.bias_name):
                bias_constant_op = op_adapter.ConstantOp(self.bias_name, tensor=bias)
                graph.add(bias_constant_op, [], [self.bias_name])
                graph.add_src_op_info(self.bias_name, None, self.bias_name)

        # Transpose input if transA is given
        if self.params.transA:
            permute_op = op_adapter.TransposeOp(input_names[0] + '_permute', perm=[1, 0])
            graph.add(permute_op, [input_names[0]], [input_names[0] + '_permute'])
            graph.add_src_op_info(permute_op.name, [input_names[0]], [input_names[0] + '_permute'])

        return op_adapter.FullyConnectedOp(str(src_op.name),
                                           bias_op_name=self.bias_name)

    def extract_input_names(self, src_op, converter_context):
        if self.params.transA:
            return [str(src_op.input[0]) + '_permute', self.weights_name, self.bias_name]
        return [src_op.input[0], self.weights_name, self.bias_name]


OnnxTranslations.register_translation(OnnxGemmTranslation(), converter_type('Gemm', 'onnx'))


# ------------------------------------------------------------------------------
#   Greater
# ------------------------------------------------------------------------------
class OnnxGreaterTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Greater', [1, 7, 9, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER
        self.numpy_op = np.greater


OnnxTranslations.register_translation(OnnxGreaterTranslation(),
                                      converter_type('Greater', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER])


# ------------------------------------------------------------------------------
#   GreaterOrEqual
# ------------------------------------------------------------------------------
class OnnxGreaterOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GreaterOrEqual', [12])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL
        self.numpy_op = np.greater_equal


# GreaterOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) >= packaging.version.Version("1.7.0"):
    OnnxTranslations.register_translation(OnnxGreaterOrEqualTranslation(),
                                          converter_type('GreaterOrEqual', 'onnx'),
                                          op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL])


# ------------------------------------------------------------------------------
#   HardSigmoid
# ------------------------------------------------------------------------------
class OnnxHardSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('HardSigmoid', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID,
                                   alpha=params.alpha,
                                   beta=params.beta)


OnnxTranslations.register_translation(OnnxHardSigmoidTranslation(),
                                      converter_type('HardSigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Identity
# ------------------------------------------------------------------------------
class OnnxIdentityTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Identity', [1, 13, 14, 16])

    def extract_parameters(self, src_op, converter_context):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op. Otherwise the identity op is a no-op that
        # gets squashed later.
        graph = converter_context.ir_graph
        if not graph.has_buffer(src_op.input[0]):
            const_input = converter_context.weights.fetch(str(src_op.input[0]))
            converter_context.weights.insert(str(src_op.output[0]), const_input)
            return op_adapter.ConstantOp(src_op.output[0], const_input)

        return op_adapter.IdentityOp(str(src_op.name))

    def extract_input_names(self, src_op, converter_context):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op which do not need an input name.
        if not converter_context.ir_graph.has_buffer(src_op.input[0]):
            return []
        return str(src_op.input[0])


OnnxTranslations.register_translation(OnnxIdentityTranslation(),
                                      converter_type('Identity', 'onnx'))


# ------------------------------------------------------------------------------
#   Less
# ------------------------------------------------------------------------------
class OnnxLessTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Less', [1, 7, 9, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS
        self.numpy_op = np.less


OnnxTranslations.register_translation(OnnxLessTranslation(),
                                      converter_type('Less', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS])


# ------------------------------------------------------------------------------
#   LessOrEqual
# ------------------------------------------------------------------------------
class OnnxLessOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LessOrEqual', [12])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL
        self.numpy_op = np.less_equal


# LessOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) >= packaging.version.Version("1.7.0"):
    OnnxTranslations.register_translation(OnnxLessOrEqualTranslation(),
                                          converter_type('LessOrEqual', 'onnx'),
                                          op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL])


# ------------------------------------------------------------------------------
#   LpNormalization
# ------------------------------------------------------------------------------
class OnnxLpNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LpNormalization', [1])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())

        if params.p != 2:
            raise ValueError("Only the L2-Norm is supported. "
                             "Found order of {}".format(params.p))

        # we use the default value of epsilon here
        return op_adapter.L2NormOp(src_op.name,
                                   axis=params.axis)


OnnxTranslations.register_translation(OnnxLpNormalizationTranslation(),
                                      converter_type('LpNormalization', 'onnx'),
                                      op_adapter.L2NormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Log
# ------------------------------------------------------------------------------
class OnnxLogTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Log', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG
        self.numpy_op = np.log


OnnxTranslations.register_translation(OnnxLogTranslation(),
                                      converter_type('Log', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG])


# ------------------------------------------------------------------------------
#   LogSoftmax
# ------------------------------------------------------------------------------
class OnnxLogSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LogSoftmax', [1, 11])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = getattr(params, "axis", 1)
        return op_adapter.LogSoftmaxOp(str(src_op.name),
                                       axis=axis)


OnnxTranslations.register_translation(OnnxLogSoftmaxTranslation(),
                                      converter_type('LogSoftmax', 'onnx'),
                                      op_adapter.LogSoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Matmul
# ------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MatMul', [1, 9, 13])
        self.input_names = []
        self.output_names = []

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        ops = self.extract_parameters(src_op, converter_context)
        if isinstance(ops, list):
            # we need to insert post reshape behind matmul
            # since we have expanded A or B in extract_parameters()
            matmul_op, reshape_op = ops
            matmul_output_names = [self.output_names[0] + '_pre_reshape']
            matmul_node = graph.add(matmul_op, self.input_names, matmul_output_names)
            graph.add_src_op_info(matmul_node.op.name, self.input_names, matmul_output_names)
            reshape_node = graph.add(reshape_op, matmul_output_names, self.output_names)
            graph.add_src_op_info(reshape_node.op.name, matmul_output_names, self.output_names)
            return reshape_node

        # normal flow for matmul op
        # No need to insert post reshape behind
        op = ops

        if not op:
            # Return last node
            return graph.list_nodes()[-1]

        node = graph.add(op, self.input_names, self.output_names)
        self.add_src_op_info(node.op.name, src_op, graph)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)

        # If input A is static, add a Constant Op
        if converter_context.weights.has(self.input_names[0]):
            static_input_name = self.input_names[0]
            tensor = converter_context.weights.fetch(static_input_name, prunable=False)
            if not graph.has_buffer(static_input_name):
                static_input_op = op_adapter.ConstantOp(static_input_name, tensor=tensor)
                static_input_node = graph.add(static_input_op, [], [static_input_name])
                graph.add_src_op_info(static_input_node.op.name, [], [static_input_name])

        # If input A is an 1-D tensor [n], unsqueeze to [1, n]
        first_input = graph.buffers[self.input_names[0]]
        if len(first_input.shape) == 1:
            first_unsqueeze_name = str(src_op.name) + "_first_unsqueeze"
            first_unsqueeze_output_name = first_input.name + ".unsqueeze"
            first_unsqueeze_op = op_adapter.ReshapeOp(first_unsqueeze_name, shape=[1, first_input.shape[0]])
            first_unsqueeze_node = graph.add(first_unsqueeze_op, [first_input.name], [first_unsqueeze_output_name])
            graph.add_src_op_info(first_unsqueeze_node.op.name, [first_input.name], [first_unsqueeze_output_name])
            self.input_names[0] = first_unsqueeze_output_name

        # If input B is constant, add a Constant Op
        if converter_context.weights.has(self.input_names[1]):
            weight_input_name = self.input_names[1]
            weights = converter_context.weights.fetch(weight_input_name, prunable=False)
            if not graph.has_buffer(weight_input_name):
                weights_constant_op = op_adapter.ConstantOp(weight_input_name, tensor=weights)
                weight_node = graph.add(weights_constant_op, [], [weight_input_name])
                graph.add_src_op_info(weight_node.op.name, [], [weight_input_name])

        # If input B is an 1-D tensor [n], unsqueeze to [n, 1]
        second_input = graph.buffers[self.input_names[1]]
        if len(second_input.shape) == 1:
            second_unsqueeze_name = str(src_op.name) + "_second_unsqueeze"
            second_unsqueeze_output_name = second_input.name + ".unsqueeze"
            second_unsqueeze_op = op_adapter.ReshapeOp(second_unsqueeze_name, shape=[second_input.shape[0], 1])
            second_unsqueeze_node = graph.add(second_unsqueeze_op, [second_input.name], [second_unsqueeze_output_name])
            graph.add_src_op_info(second_unsqueeze_node.op.name, [second_input.name], [second_unsqueeze_output_name])
            self.input_names[1] = second_unsqueeze_output_name

        if len(first_input.shape) == 1 and len(second_input.shape) == 1:
            # if src input A and src input B are both 1D tensor, matmul src output shape will be 0D tensor
            # which is not supported in current backend
            raise ValueError("ERROR: Unsupported 0d output tensor for {} op {}".format(src_op.op_type, src_op.name))
        elif len(first_input.shape) != 1 and len(second_input.shape) != 1:
            # No post reshape op is needed, return one matmul op
            # Since ONNX Matmul does not support matrix transpose,
            # both transpose_in0 and transpose_in1 are set False
            return op_adapter.MatMulOp(name=str(src_op.name),
                                       transpose_in0=False,
                                       transpose_in1=False)
        else:
            # Post reshape op is needed after matmul op because we have expanded 1D input tensor
            # Since ONNX Matmul does not support matrix transpose,
            # both transpose_in0 and transpose_in1 are set False
            matmul_op = op_adapter.MatMulOp(name=str(src_op.name),
                                            transpose_in0=False,
                                            transpose_in1=False)
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = 'Reshape_post_' + src_op.name

            if len(first_input.shape) == 1 and len(second_input.shape) != 1:
                # We have prepended 1 for A input, so post reshape is needed after matmul
                # Example: A is [3], B is [1,4,3,7] --> output shape is [1,4,7]
                output_shape = second_input.shape[:-2] + second_input.shape[-1:]
            elif len(first_input.shape) != 1 and len(second_input.shape) == 1:
                # We have appended 1 for B input, so post reshape is needed after matmul
                # Example: A is [1,4,5,3], B is [3] --> output shape is [1,4,5]
                output_shape = first_input.shape[:-1]

            reshape_op = op_adapter.ReshapeOp(reshape_op_name,
                                              shape=output_shape)
            return [matmul_op, reshape_op]


OnnxTranslations.register_translation(OnnxMatMulTranslation(), converter_type('MatMul', 'onnx'))


# ------------------------------------------------------------------------------
#   Max
# ------------------------------------------------------------------------------
class OnnxMaxTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Max', [1, 6, 8, 12, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM
        self.numpy_op = np.maximum


OnnxTranslations.register_translation(OnnxMaxTranslation(),
                                      converter_type('Max', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM])


# ------------------------------------------------------------------------------
#   Min
# ------------------------------------------------------------------------------
class OnnxMinTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Min', [1, 6, 8, 12, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM
        self.numpy_op = np.minimum


OnnxTranslations.register_translation(OnnxMinTranslation(),
                                      converter_type('Min', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM])


# ------------------------------------------------------------------------------
#   Mod
# ------------------------------------------------------------------------------
class OnnxModTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mod', [10, 13])
        self.numpy_op = np.mod
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        fmod = getattr(params, "fmod", 0)
        if fmod:
            self.numpy_op = np.fmod
            self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD

        # reuse super function to perform constant folding and return op
        return super().extract_parameters(src_op, converter_context)


OnnxTranslations.register_translation(OnnxModTranslation(),
                                      converter_type('Mod', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MOD],
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FMOD])


# ------------------------------------------------------------------------------
#   Mul
# ------------------------------------------------------------------------------
class OnnxMulTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mul', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY
        self.numpy_op = np.multiply


OnnxTranslations.register_translation(OnnxMulTranslation(),
                                      converter_type('Mul', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY])


# ------------------------------------------------------------------------------
#   Neg
# ------------------------------------------------------------------------------
class OnnxNegTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Neg', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG
        self.numpy_op = np.negative


OnnxTranslations.register_translation(OnnxNegTranslation(),
                                      converter_type('Neg', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG])


# ------------------------------------------------------------------------------
#   Not
# ------------------------------------------------------------------------------
class OnnxNotTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Not', [1])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT
        self.numpy_op = np.logical_not


OnnxTranslations.register_translation(OnnxNotTranslation(),
                                      converter_type('Not', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT])


# ------------------------------------------------------------------------------
#   Or
# ------------------------------------------------------------------------------
class OnnxOrTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Or', [1, 7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR
        self.numpy_op = np.logical_or


OnnxTranslations.register_translation(OnnxOrTranslation(),
                                      converter_type('Or', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_OR])


# ------------------------------------------------------------------------------
#   Pow
# ------------------------------------------------------------------------------
class OnnxPowTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pow', [1, 7, 12, 13, 15])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER
        self.numpy_op = np.power


OnnxTranslations.register_translation(OnnxPowTranslation(),
                                      converter_type('Pow', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_POWER])


# ------------------------------------------------------------------------------
#   Reciprocal
# ------------------------------------------------------------------------------
class OnnxReciprocalTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Reciprocal', [1, 6])
        self.input_names = []
        self.numpy_op = np.reciprocal

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)

        const_input_data = []
        for input_name in self.input_names:
            const_input_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_data.append(const_input_op.tensor)
        if len(const_input_data) == len(self.input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            out_dtype = converter_context.tensor_to_np_dtype.get(str(src_op.output[0]))
            data = self.numpy_op(*const_input_data).astype(out_dtype)
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in self.input_names])
            converter_context.weights.insert(str(src_op.output[0]), data, was_scalar=was_scalar)
            return None
        else:
            # if input is not const, map reciprocal to divide
            input_buf = graph.get_buffer(self.input_names[0])
            const_op_name = src_op.name + "_coeff"
            const_op = op_adapter.ConstantOp(const_op_name, np.ones(input_buf.shape, np.float32))
            const_node = graph.add(const_op, [], [const_op_name])
            graph.add_src_op_info(const_node.op.name, [], [const_op_name])
            self.input_names.insert(0, const_op_name)
            return op_adapter.ElementwiseBinaryOp(str(src_op.name), operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE)

OnnxTranslations.register_translation(OnnxReciprocalTranslation(),
                                      converter_type('Reciprocal', 'onnx'))


# ------------------------------------------------------------------------------
#   ReduceBase
# ------------------------------------------------------------------------------
class OnnxReduceBaseTranslation(OnnxTranslationBase):
    def __init__(self, reduce_type):
        OnnxTranslationBase.__init__(self)
        self.reduce_type = reduce_type

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        input_name = src_op.input[0]
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_op is None:
            input_buf = graph.get_buffer(input_name)
            rank = input_buf.rank()
        else:
            rank = len(const_op.tensor.shape)

        # TODO: reducesum in opset13 is different with opset11
        # and other reduce types have a same attributes and inputs operation when in opset18.
        if (schema.version[0] < 13 and self.reduce_type =="ReduceSum") or \
            (schema.version[0] < 18 and self.reduce_type !="ReduceSum"):
            schema.replace_default_values(axes=range(rank))
            params = extract_attributes(src_op, schema=schema)
            axes = params.axes
        else:
            schema.replace_default_values(noop_with_empty_axes=0)
            params = extract_attributes(src_op, schema=schema)
            # if noop_with_empty_axes is true and axes is empty, input tensor will not be reduced.
            if params.noop_with_empty_axes and len(src_op.input) == 1:
                return op_adapter.IdentityOp(str(src_op.name))
            else:
                if len(src_op.input) != 1:
                    axes_op = self.fetch_constant_op(str(src_op.input[1]), converter_context, fail_if_dynamic=False)
                    if axes_op is not None:
                        axes = axes_op.tensor
                    else:
                        raise ValueError("Dynamic axes provided, only static axes is supported for op: {} of type: {}".format(src_op.name, src_op.op_type))
                else:
                    # when axes is None,noop_with_empty_axes with 'false' is to reduce all axes.
                    axes = list(range(rank))

        # handle static input case with numpy logic
        if const_op:
            data = const_op.tensor
            if self.reduce_type == "ReduceL2":
                reduce_data = np.sqrt(np.sum(a=np.square(data), axis=tuple(axes), keepdims=bool(params.keepdims)))
            elif self.reduce_type == "ReduceMax":
                reduce_data = np.maximum.reduce(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceMean":
                reduce_data = np.mean(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceMin":
                reduce_data = np.minimum.reduce(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceProd":
                reduce_data = np.prod(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            elif self.reduce_type == "ReduceSum":
                reduce_data = np.sum(data, axis=tuple(axes), keepdims=bool(params.keepdims))
            converter_context.weights.insert(str(src_op.output[0]), reduce_data)
            return None

        return op_adapter.ReduceOp(str(src_op.name),
                                   reduce_type=self.reduce_type,
                                   axes=axes,
                                   keep_dims=params.keepdims)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


# ------------------------------------------------------------------------------
#   ReduceL2
# ------------------------------------------------------------------------------
class OnnxReduceL2Translation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.IR_OP_REDUCE_L2)
        self.register_op_schema('ReduceL2', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceL2Translation(),
                                      converter_type('ReduceL2', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.IR_OP_REDUCE_L2])


# ------------------------------------------------------------------------------
#   ReduceMax
# ------------------------------------------------------------------------------
class OnnxReduceMaxTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MAX)
        self.register_op_schema('ReduceMax', [1, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMaxTranslation(),
                                      converter_type('ReduceMax', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX])


# ------------------------------------------------------------------------------
#   ReduceMean
# ------------------------------------------------------------------------------
class OnnxReduceMeanTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MEAN)
        self.register_op_schema('ReduceMean', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMeanTranslation(),
                                      converter_type('ReduceMean', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN])


# ------------------------------------------------------------------------------
#   ReduceMin
# ------------------------------------------------------------------------------
class OnnxReduceMinTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MIN)
        self.register_op_schema('ReduceMin', [1, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMinTranslation(),
                                      converter_type('ReduceMin', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN])


# ------------------------------------------------------------------------------
#   ReduceProd
# ------------------------------------------------------------------------------
class OnnxReduceProdTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_PROD)
        self.register_op_schema('ReduceProd', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceProdTranslation(),
                                      converter_type('ReduceProd', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD])


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_SUM)
        self.register_op_schema('ReduceSum', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceSumTranslation(),
                                      converter_type('ReduceSum', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM])


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumSquareTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ReduceSumSquare', [1, 11, 13, 18])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        const_op, power_op, reduce_sum_op = self.extract_parameters(src_op, converter_context)

        const_output_name = input_names[0] + '_exponents'
        const_node = graph.add(const_op, [], [const_output_name])
        graph.add_src_op_info(const_node.op.name, [], [const_output_name])

        power_output_name = input_names[0] + '_power'
        power_input_names = [input_names[0], const_output_name]
        power_node = graph.add(power_op, power_input_names, [power_output_name])
        graph.add_src_op_info(power_node.op.name, power_input_names, [power_output_name])

        reduce_sum_node = graph.add(reduce_sum_op, [power_output_name], output_names)
        graph.add_src_op_info(reduce_sum_node.op.name, [power_output_name], output_names)

        return reduce_sum_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        input_name = src_op.input[0]
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_op is None:
            input_buf = graph.get_buffer(input_name)
            rank = input_buf.rank()
        else:
            rank = len(const_op.tensor.shape)

        if schema.version[0] < 18:
            schema.replace_default_values(axes=range(rank))
            params = extract_attributes(src_op, schema=schema)
            axes = params.axes
        else:
            schema.replace_default_values(noop_with_empty_axes=0)
            params = extract_attributes(src_op, schema=schema)
            # if noop_with_empty_axes is true and axes is empty, input tensor will not be reduced.
            if params.noop_with_empty_axes and len(src_op.input) == 1:
                return op_adapter.IdentityOp(str(src_op.name))
            else:
                if len(src_op.input) != 1:
                    axes_op = self.fetch_constant_op(str(src_op.input[1]), converter_context, fail_if_dynamic=False)
                    if axes_op is not None:
                        axes = axes_op.tensor
                    else:
                        raise ValueError("Dynamic axes provided, only static axes is supported for op: {} of type: {}".format(src_op.name, src_op.op_type))
                else:
                    # when axes is None,noop_with_empty_axes with 'false' is to reduce all axes.
                    axes = list(range(rank))

        elementwise_power = op_adapter.ElementwiseBinaryOp(str(src_op.name) + "_power", 
                                                           operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER)
        reduce_sum = op_adapter.ReduceOp(str(src_op.name) + "_reduced_sum",
                                                    reduce_type="ReduceSum",
                                                    axes=axes,
                                                    keep_dims=params.keepdims)

        input_shape = np.array(graph.get_buffer(src_op.input[0]).shape)
        reduce_dimensions = [input_shape[axis] if axis in axes else 1 for axis in range(len(input_shape))]
        exponent_tensor = np.zeros(reduce_dimensions) + 2
        constant_op = op_adapter.ConstantOp(str(src_op.name) + "_exponent",
                                            tensor=exponent_tensor)
        converter_context.weights.insert(constant_op.name, exponent_tensor)

        return [constant_op, elementwise_power, reduce_sum]

OnnxTranslations.register_translation(OnnxReduceSumSquareTranslation(),
                                      converter_type('ReduceSumSquare', 'onnx'))


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Relu', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        input_names = self.extract_input_names(src_op, converter_context)
        constant_input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)

        if constant_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            # ReLU computation
            data = numpy.maximum(constant_input_op.tensor, 0)
            was_scalar = converter_context.weights.was_scalar(input_names[0])
            converter_context.weights.insert(str(src_op.output[0]), data, was_scalar=was_scalar)
            return None

        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                 operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU)


OnnxTranslations.register_translation(OnnxReluTranslation(),
                                      converter_type('Relu', 'onnx'),
                                      op_adapter.ElementwiseNeuronOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Round
# ------------------------------------------------------------------------------
class OnnxRoundTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Round', [11])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND
        self.numpy_op = np.round


OnnxTranslations.register_translation(OnnxRoundTranslation(),
                                      converter_type('Round', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND])


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sigmoid', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID, alpha=1.0)


OnnxTranslations.register_translation(OnnxSigmoidTranslation(), converter_type('Sigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Sign
# ------------------------------------------------------------------------------
class OnnxSignTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sign', [13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN
        self.numpy_op = np.sign


OnnxTranslations.register_translation(OnnxSignTranslation(),
                                      converter_type('Sign', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN])


# ------------------------------------------------------------------------------
#   Sin
# ------------------------------------------------------------------------------
class OnnxSinTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sin', [7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN
        self.numpy_op = np.sin


OnnxTranslations.register_translation(OnnxSinTranslation(),
                                      converter_type('Sin', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN])


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softmax', [1, 11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        translated_ops = self.extract_parameters(src_op, converter_context)
        if len(translated_ops) == 3:
            # translate_ops = [reshape, softmax, reshape]
            # add pre reshape
            pre_reshape_output_names = [input_names[0] + '_reshape']
            pre_reshape_node = graph.add(translated_ops[0], input_names, pre_reshape_output_names)
            graph.add_src_op_info(pre_reshape_node.op.name, input_names, pre_reshape_output_names)

            # add softmax
            softmax_output_names = [input_names[0] + '_softmax']
            softmax_node = graph.add(translated_ops[1], pre_reshape_node.output_names, softmax_output_names)
            graph.add_src_op_info(softmax_node.op.name, pre_reshape_node.output_names, softmax_output_names)

            # add post reshape
            input_buf = graph.get_buffer(input_names[0])
            post_reshape_node = graph.add(translated_ops[2], softmax_node.output_names, output_names,
                                          axis_formats=[input_buf.axis_format])
            graph.add_src_op_info(post_reshape_node.op.name, softmax_node.output_names, output_names)
            last_node = post_reshape_node
            # copy quantization_overrides info to each new op
            rename_user_quantization_overrides(graph, output_names[0], softmax_output_names[0])

        else:
            # translate_ops = [softmax]
            softmax_node = graph.add(translated_ops[0], input_names, output_names)
            graph.add_src_op_info(softmax_node.op.name, input_names, output_names)
            last_node = softmax_node

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        params = extract_attributes(src_op, schema=schema)
        axis = getattr(params, "axis", 1)

        origin_shape = graph.get_buffer(src_op.input[0]).shape

        # only insert reshape if input rank larger than 2,
        # the reason is that in such case adding reshape won't influence result
        if schema.version[0] < 13 and len(origin_shape) > 2:
            # for softmax with older version, it will flatten dimension after axis(include) and calculate softmax on it
            shape = [*origin_shape[:axis], np.prod(origin_shape[axis:])]

            # flatten dimension after axis (include)
            pre_reshape_node_name = str(src_op.input[0]) + '_flatten'
            pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_node_name, shape=shape)

            # softmax on axis
            softmax_op = op_adapter.SoftmaxOp(str(src_op.name), axis=axis)

            # reshape to origin shape
            post_reshape_node_name = str(src_op.output[0]) + '_reshape'
            post_reshape_op = op_adapter.ReshapeOp(name=post_reshape_node_name, shape=origin_shape)

            translated_ops = [pre_reshape_op, softmax_op, post_reshape_op]
        else:
            translated_ops = [op_adapter.SoftmaxOp(str(src_op.name), axis=axis)]
        return translated_ops


OnnxTranslations.register_translation(OnnxSoftmaxTranslation(),
                                      converter_type('Softmax', 'onnx'),
                                      op_adapter.SoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Softplus
# ------------------------------------------------------------------------------
class OnnxSoftplusTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softplus', [1])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(str(src_op.name), operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS)


OnnxTranslations.register_translation(OnnxSoftplusTranslation(),
                                      converter_type('Softplus', 'onnx'))


# ------------------------------------------------------------------------------
#   Sub
# ------------------------------------------------------------------------------
class OnnxSubTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sub', [1, 6, 7, 13, 14])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT
        self.numpy_op = np.subtract


OnnxTranslations.register_translation(OnnxSubTranslation(),
                                      converter_type('Sub', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT])


# ------------------------------------------------------------------------------
#   Sum
# ------------------------------------------------------------------------------
class OnnxSumTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sum', [1, 6, 8, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD
        self.numpy_op = np.add


OnnxTranslations.register_translation(OnnxSumTranslation(), converter_type('Sum', 'onnx'))


# ------------------------------------------------------------------------------
#   Sqrt
# ------------------------------------------------------------------------------
class OnnxSqrtTranslation(ElementwiseUnaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sqrt', [1, 6, 13])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT
        self.numpy_op = np.sqrt


OnnxTranslations.register_translation(OnnxSqrtTranslation(),
                                      converter_type('Sqrt', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT])


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tanh', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
                                   alpha=1.0,
                                   beta=1.0)


OnnxTranslations.register_translation(OnnxTanhTranslation(),
                                      converter_type('Tanh', 'onnx'))


# ------------------------------------------------------------------------------
#   ScaledTanh
# ------------------------------------------------------------------------------
class OnnxScaledTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScaledTanh', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                   operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
                                   alpha=params.alpha,
                                   beta=params.beta)


# scaledtanh is removed in ONNX release v1.5.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) < packaging.version.Version("1.5.0"):
    OnnxTranslations.register_translation(OnnxScaledTanhTranslation(),
                                          converter_type('ScaledTanh', 'onnx'))


# ------------------------------------------------------------------------------
#   TopK
# ------------------------------------------------------------------------------
class OnnxTopKTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('TopK', [1, 10, 11])

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op, post_permute_ops = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)

        if len(post_permute_ops) != 0:
            src_op.output[0] = src_op.output[0] + '_permute'
            src_op.output[1] = src_op.output[1] + '_permute'
            output_names = self.extract_output_names(src_op, context)
            node = graph.add(op, input_names, output_names)
            self.add_src_op_info(node.op.name, src_op, graph)
            post_permute_node_0 = graph.add(post_permute_ops[0], [output_names[0]], [output_names[0].replace('_permute', '')])
            graph.add_src_op_info(post_permute_ops[0].name, [output_names[0]], [output_names[0].replace('_permute', '')])
            post_permute_node_1 = graph.add(post_permute_ops[1], [output_names[1]], [output_names[1].replace('_permute', '')])
            graph.add_src_op_info(post_permute_ops[1].name, [output_names[1]], [output_names[1].replace('_permute', '')])
            return post_permute_node_0

        output_names = self.extract_output_names(src_op, context)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        output_names = list(src_op.output)
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()
        input_dims = input_buf.get_buf_dims()
        post_permute_ops = list()

        # extract K as input in versions 10, 11 and as parameter in version 1
        if len(input_names) == 2:
            const_op = self.fetch_constant_op(input_names[1], converter_context)
            log_assert(const_op is not None,
                       "Input tensor {} of node {} could not be extracted.".format(input_names[1], src_op.name))
            k = const_op.tensor.astype(np.int64).item(0)
        else:
            k = params.k

        largest = params.largest if 'largest' in params else 1
        sorted = params.sorted if 'sorted' in params else 1
        if largest != 1:
            log_assert(largest != 1, "TopK Op {}, attribute largest not supported", src_op.name)
        if sorted != 1:
            log_assert(sorted != 1, "TopK Op {}, attribute sorted not supported", src_op.name)
        axis = params.axis

        if axis < 0:
            axis += input_rank

        log_assert(input_rank >= 1,
                   code_to_message.get_error_message("ERROR_TOPK_INPUT_TENSOR_RANK")(input_rank))
        if axis != input_rank - 1:
            permutation = [i for i in range(input_rank)]
            permutation[axis], permutation[len(permutation)-1] = permutation[len(permutation)-1], permutation[axis]
            permute_op = op_adapter.TransposeOp(input_names[0] + '_permute', perm=permutation)
            graph.add(permute_op, [input_names[0]], [input_names[0] + '_permute'])
            graph.add_src_op_info(permute_op.name, [input_names[0]], [input_names[0] + '_permute'])
            src_op.input[0] = input_names[0] + '_permute'
            post_permute_ops.append(op_adapter.TransposeOp(output_names[0] + '_permute', perm=permutation))
            post_permute_ops.append(op_adapter.TransposeOp(output_names[1] + '_permute', perm=permutation))
            axis = input_rank - 1

        if k < 0 or input_dims[axis] < k:
            raise ValueError(
                code_to_message.get_error_message("ERROR_TOPK_K_INVALID")(k, input_dims[axis]))

        return op_adapter.TopKOp(src_op.name, k=k), post_permute_ops

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTopKTranslation(),
                                      converter_type('TopK', 'onnx'),
                                      op_adapter.TopKOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Xor
# ------------------------------------------------------------------------------
class OnnxXorTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Xor', [1, 7])
        self.operation = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR
        self.numpy_op = numpy.logical_xor


OnnxTranslations.register_translation(OnnxXorTranslation(),
                                      converter_type('Xor', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_XOR])
