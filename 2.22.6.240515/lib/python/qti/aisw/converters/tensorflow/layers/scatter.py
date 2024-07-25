# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from qti.aisw.converters.common.utils.code_to_message import get_error_message
from qti.aisw.converters.common.converter_ir.op_adapter import ScatterNDOp , ReshapeOp, ConstantOp
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
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker


class ScatterLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_names, output_shape, output_names=None):
            super(ScatterLayerResolver.Descriptor, self).__init__('Scatter', name, nodes, output_names=output_names)
            self.input_names = input_names
            self.output_shape = output_shape

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['ScatterNd']),
            NonConsumableConverterSequenceNode('indices', ['?']),
            NonConsumableConverterSequenceNode('updates', ['?']),
            NonConsumableConverterSequenceNode('shape', ['?'])
        ])
        self.sequence.set_inputs('root', ['indices', 'updates', 'shape'])
        self.sequence.set_outputs(['root'])


    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []

        for match in graph_matcher.match_sequence(self.sequence):
            ScatterNDOp = match['root']
            consumed_nodes = match.consumed_nodes
            indices, updates, shape_op = self.get_tensors(graph_helper, ScatterNDOp)
            output_shape = list(graph_helper.evaluate_tensor_output(shape_op))
            updates, const_updates_consumed_ops = graph_helper.get_none_identity_input(updates)
            indices, const_indices_consumed_ops = graph_helper.get_none_identity_input(indices)
            shapes, const_shapes_consumed_ops = graph_helper.get_none_identity_input(shape_op)
            input_names = []
            scatter_inp_name = str(ScatterNDOp.name) + '_data'
            input_names.extend([scatter_inp_name,
                                GraphHelper.indexed_tensor_name(indices.op.name),
                                GraphHelper.indexed_tensor_name(updates.op.name)])
            descriptor = ScatterLayerResolver.Descriptor(str(ScatterNDOp.name), consumed_nodes,
                                                         input_names, output_shape, [ScatterNDOp.outputs[0].name])

            descriptors.append(descriptor)

            if indices.op.type == 'Const':
                const_indices_shape = GraphHelper.get_tensor_output_shape(indices)
                const_indices_val = graph_helper.evaluate_tensor_output(indices).astype('int32')
                const_indices_descriptor = ConstantLayerResolver.Descriptor(str(indices.op.name),
                                                                            const_indices_consumed_ops,
                                                                            const_indices_val, const_indices_shape,
                                                                            descriptor)
                descriptors.append(const_indices_descriptor)

            if updates.op.type == 'Const':
                const_updates_shape = GraphHelper.get_tensor_output_shape(updates)
                const_updates_val = graph_helper.evaluate_tensor_output(updates)
                const_updates_descriptor = ConstantLayerResolver.Descriptor(str(updates.op.name),
                                                                            const_updates_consumed_ops,
                                                                            const_updates_val, const_updates_shape,
                                                                            descriptor)
                descriptors.append(const_updates_descriptor)

            if shapes.op.type == 'Const':
                const_shapes_shape = GraphHelper.get_tensor_output_shape(shapes)
                const_shapes_val = graph_helper.evaluate_tensor_output(shapes)
                const_shapes_descriptor = ConstantLayerResolver.Descriptor(str(updates.op.name),
                                                                           const_shapes_consumed_ops,
                                                                           const_shapes_val, const_shapes_shape,
                                                                           descriptor)
                descriptors.append(const_shapes_descriptor)


        return descriptors

    @classmethod
    def get_tensors(cls, graph_helper, ScatterNDOp):

        indices, updates, shape = GraphHelper.get_op_input_tensors(ScatterNDOp, ('?', '?', '?'))

        if (indices.dtype.name == 'float32' or indices.dtype.name == 'float64') \
                and (updates.dtype.name == 'int32' or updates.dtype.name == 'int64'):
            indices, updates = updates, indices
        return indices, updates, shape


class ScatterLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ScatterLayerBuilder.Descriptor
        :rtype: int
        """
        scatter_output_name = descriptor.output_names[0]

        scatter_inp_name = descriptor.input_names[0]
        scatter_data_tensor = np.zeros(descriptor.output_shape, dtype=np.float32)
        scatter_inp_name_op = ConstantOp(scatter_inp_name, tensor=scatter_data_tensor)
        scatter_data_node = ir_graph.add(scatter_inp_name_op, [], [scatter_inp_name], [AxisTracker.AxisFormat.NONTRIVIAL])
        ir_graph.add_src_op_info(scatter_inp_name, None, scatter_data_node.output_names[0])

        scatter_node = ir_graph.add(ScatterNDOp(descriptor.layer_name),
                                    descriptor.input_names,
                                    scatter_output_name)
        return scatter_node
