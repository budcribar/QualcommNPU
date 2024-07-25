# =============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import CumSumOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class CumsumLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, reverse=False, exclusive=False, output_names=None):
            super(CumsumLayerResolver.Descriptor, self).__init__('Cumsum', name, nodes, output_names=output_names)
            self.axis = axis
            self.reverse = reverse
            self.exclusive = exclusive

    def __init__(self):
        sequence = GraphSequence([
            ConverterSequenceNode('cumsum', ['Cumsum']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('axis', ['?'])
        ])
        sequence.set_inputs('cumsum', ['input','axis'])
        sequence.set_outputs(['cumsum'])
        self.sequences = [sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                cumsum_op = match['cumsum']
                axis_op = match['axis']
                axis = int(graph_helper.evaluate_tensor_output(axis_op.outputs[0]))

                # check for axis to be in [-r,r-1]
                input_op = match['input']
                input_rank = len(graph_helper.get_op_output_shape(input_op))
                if axis not in range(-input_rank, input_rank):
                    raise ValueError("ERROR: Invalid value {} for axis attribute for Cumsum op".format(axis))
                if axis < 0:
                    axis += input_rank

                reverse = bool(cumsum_op.get_attr('reverse'))
                exclusive = bool(cumsum_op.get_attr('exclusive'))
                consumed_nodes = match.consumed_nodes

                descriptors.append(
                    CumsumLayerResolver.Descriptor(str(cumsum_op.name),
                                                   consumed_nodes,
                                                   axis=axis,
                                                   reverse=reverse,
                                                   exclusive=exclusive,
                                                   output_names=[str(cumsum_op.outputs[0].name)]))

        return descriptors


class CumsumLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: CumsumLayerResolver.Descriptor
        :rtype: OpNode
        """
        # input names are input and axis, pass input as input_names and axis is the parameter in the ir graph
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        return ir_graph.add(CumSumOp(name=descriptor.layer_name,
                                     axis=descriptor.axis,
                                     reverse=descriptor.reverse,
                                     exclusive=descriptor.exclusive),
                            input_names=[input_names[0]],
                            output_names=descriptor.output_names)