# =============================================================================
#
#  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import MatMulOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class MatMulLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, matmul_op, transpose_a=False, transpose_b=False, output_names=None):
            super(MatMulLayerResolver.Descriptor, self).__init__('MatMul', name, nodes, output_names=output_names)
            self.matmul_op = matmul_op
            self.transpose_a = transpose_a
            self.transpose_b = transpose_b

        def is_input_op(self, op):
            return op == self.matmul_op

    def __init__(self):

        sequence = GraphSequence([
            ConverterSequenceNode('matmul_op', ['MatMul', 'BatchMatMul', 'BatchMatMulV2']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence.set_inputs('matmul_op', ['inputs', 'weights'])
        sequence.set_outputs(['matmul_op'])

        self.sequences = [sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                matmul_op = match['matmul_op']
                weights_op = match['weights']
                weights_shape = graph_helper.get_op_output_shape(weights_op)
                if len(weights_shape) == 2 and (
                        weights_op.type in ['Identity', 'Const', 'Split', 'FakeQuantWithMinMaxVars'] or \
                        graph_helper.check_op_const_origin(weights_op)[0]
                        ):
                    # This can be handled by Fully Connected Layer
                    continue

                consumed_nodes = match.consumed_nodes
                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]

                if matmul_op.type in ['MatMul']:
                    transpose_a = matmul_op.get_attr('transpose_a')
                    transpose_b = matmul_op.get_attr('transpose_b')
                else:
                    transpose_a = matmul_op.get_attr('adj_x')
                    transpose_b = matmul_op.get_attr('adj_y')
                descriptors.append(
                    MatMulLayerResolver.Descriptor(str(matmul_op.name), consumed_nodes,
                                                   matmul_op,
                                                   transpose_a=transpose_a,
                                                   transpose_b=transpose_b,
                                                   output_names=output_op_nodes_names))

        return descriptors


class MatMulLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FullyConnectedLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(MatMulOp(name=descriptor.layer_name,
                                     transpose_in0=descriptor.transpose_a,
                                     transpose_in1=descriptor.transpose_b),
                            input_names,
                            output_name)
