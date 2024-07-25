# =============================================================================
#
#  Copyright (c) 2015-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.op_adapter import SplitOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import TensorNotFoundError


class SplitLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, split_sizes):
            super(SplitLayerResolver.Descriptor, self).__init__('Split', name, nodes)
            self.axis = axis
            self.split_sizes = split_sizes

        def is_input_tensor(self, op, tensor):
            # resolver supports the attribute inputs only when Const, hence the non-const is the actual input
            if "Const" in [tensor.op.type, GraphHelper.get_none_identity_input(tensor)[0].op.type]:
                return False
            return True

        @property
        def output_names(self):
            return [str(t.name) for t in self.child_ops[-1].outputs]

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Split', 'SplitV'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        potential_descriptors = []
        for match in matches:
            split_op = match['root']
            split_axis, split_sizes = self.get_split_axis_and_sizes(graph_helper, split_op)
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                SplitLayerResolver.Descriptor(str(split_op.name), consumed_nodes,
                                              split_axis,
                                              split_sizes))
        return potential_descriptors

    @classmethod
    def get_split_axis_and_sizes(cls, graph_helper, split_op):
        try:
            _, split_sizes, split_axis = GraphHelper.get_op_input_tensors(split_op, ('?', 'Const', 'Const'))
            split_sizes = list(graph_helper.evaluate_tensor_output(split_sizes))
        except TensorNotFoundError:
            split_axis, _ = GraphHelper.get_op_input_tensors(split_op, ('Const', '?'))
            split_sizes = []

        split_axis = int(graph_helper.evaluate_tensor_output(split_axis))
        return split_axis, split_sizes


class SplitLayerBuilder(LayerBuilder):
    def build_layer(self, graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: SplitLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        split_index = ir_graph.SplitOp.convert_sizes_to_indices(descriptor.split_sizes)
        return graph.add(SplitOp(name=descriptor.layer_name,
                                 axis=descriptor.axis,
                                 split_index=split_index),
                         input_names=input_name,
                         output_names=descriptor.output_names)
