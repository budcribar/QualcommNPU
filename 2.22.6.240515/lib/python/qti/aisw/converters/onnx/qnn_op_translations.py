# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from .util import *
from qti.aisw.converters.common.converter_ir.op_adapter import OpAdapterMap
from qti.aisw.converters.common.converter_ir import op_adapter
# ------------------------------------------------------------------------------
#   QNN Op
# ------------------------------------------------------------------------------
class OnnxQnnOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.qnn_op = None

    def extract_input_names(self, src_op, converter_context):
        return [str(input) for input in src_op.input]

    def extract_output_names(self, src_op, converter_context):
        return [str(output) for output in src_op.output]

    def extract_parameters(self, src_op, converter_context, **kwargs):

        # extract the parameter of the source ops
        param_dict = {}
        for attr in src_op.attribute:
            code = OnnxAttrProtoUtil.enum_to_strCode[attr.type]
            attr_value = extract_onnx_type(code, attr)
            param_dict[attr.name] = attr_value

        neuron_types = ['ElementWiseNeuron'] # relu_min_max

        # First check whether the optype is supported or not. If yes, call the
        # respective op constructor using the translation map.
        # Sub op types (like Neuron) needs to be handled separately
        # since qnn doesn't have these sub op concept.
        if src_op.op_type in neuron_types and \
            src_op.operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX:
            return op_adapter.ElementwiseNeuronOp(str(src_op.name),
                                                  operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                                                  **param_dict)
        elif src_op.op_type in OpAdapterMap.translations.keys():
            return OpAdapterMap.translations[src_op.op_type](str(src_op.name), **param_dict)
        else:
            raise ValueError("Undefined QNN op type {} for node name {}".format(src_op.op_type, src_op.name))


    def add_op(self, src_op, context, **kwargs):
        def add_static_tensor_as_constant_op(node_name):
            if not graph.has_node(node_name):
                constant_op = self.fetch_constant_op(node_name, context)
                graph.add(
                    op=constant_op,
                    input_names=[],
                    output_names=[node_name])

        def is_static_input(input_name):
            return context.weights.has(input_name)

        def add_static_inputs_to_graph(input_names):
            static_input_names = [name for name in input_names if is_static_input(name)]
            for static_input_name in static_input_names:
                add_static_tensor_as_constant_op(static_input_name)

        graph = context.ir_graph
        op = self.extract_parameters(src_op, context, **kwargs)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        add_static_inputs_to_graph(input_names)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

OnnxTranslations.register_translation(OnnxQnnOpTranslation(),
                                      converter_type('qnn', 'onnx'))
