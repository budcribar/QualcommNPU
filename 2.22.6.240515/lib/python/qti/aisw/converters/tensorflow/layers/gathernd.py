# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils.code_to_message import get_error_message
from qti.aisw.converters.common.converter_ir.op_adapter import GatherNDOp, ReshapeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerBuilder, LayerResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from abc import ABCMeta
from abc import abstractmethod
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import TensorNotFoundError, ConverterError


class GatherndLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_names, output_names=None):
            super(GatherndLayerResolver.Descriptor, self).__init__('GatherNd', name, nodes, output_names=output_names)
            self.input_names = input_names

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0] or tensor == op.inputs[1]

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['GatherNd']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('indices', ['?'])
        ])
        self.sequence.set_inputs('root', ['params', 'indices'])
        self.sequence.set_outputs(['root'])



    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
                gather_op = match['root']
                consumed_nodes = match.consumed_nodes
                params, indices = self.get_tensors(graph_helper, gather_op)
                params, const_params_consumed_ops = graph_helper.get_none_identity_input(params)
                indices, const_indices_consumed_ops = graph_helper.get_none_identity_input(indices)
                input_names = []

                input_names.extend([GraphHelper.indexed_tensor_name(params.op.name),
                                    GraphHelper.indexed_tensor_name(indices.op.name)])
                descriptor = GatherndLayerResolver.Descriptor(str(gather_op.name), consumed_nodes,
                                                            input_names, [gather_op.outputs[0].name])

                descriptors.append(descriptor)

                if indices.op.type == 'Const':
                    const_indices_shape = GraphHelper.get_tensor_output_shape(indices)
                    const_indices_val = graph_helper.evaluate_tensor_output(indices).astype('int32')
                    const_indices_descriptor = ConstantLayerResolver.Descriptor(str(indices.op.name),
                                                                                const_indices_consumed_ops,
                                                                                const_indices_val, const_indices_shape,
                                                                                descriptor)
                    descriptors.append(const_indices_descriptor)

                if params.op.type == 'Const':
                    const_shape = GraphHelper.get_tensor_output_shape(params)
                    const_val = graph_helper.evaluate_tensor_output(params)
                    const_descriptor = ConstantLayerResolver.Descriptor(str(params.op.name),
                                                                        const_params_consumed_ops,
                                                                        const_val, const_shape,
                                                                        descriptor)
                    descriptors.append(const_descriptor)
        return descriptors

    @classmethod
    def get_tensors(cls, graph_helper, GatherNDOp):

        params, indices = GraphHelper.get_op_input_tensors(GatherNDOp, ('?', '?'))

        if (indices.dtype.name == 'float32' or indices.dtype.name == 'float64') \
                and (params.dtype.name == 'int32' or params.dtype.name == 'int64'):
            indices, params = params, indices
        return params, indices


class GatherndLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ScatterLayerBuilder.Descriptor
        :rtype: int
        """
        gathernd_output_name = descriptor.output_names[0]
        gatherndnode = ir_graph.add(GatherNDOp(descriptor.layer_name),
                                    descriptor.input_names,
                                    gathernd_output_name)
        return gatherndnode
