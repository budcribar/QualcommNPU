# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.op_adapter import ResizeOp
from qti.aisw.converters.common.utils.converter_utils import log_assert
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class ResizeBilinearLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'
    TF_ATTRIBUTE_HALF_PIXEL_CENTERS = 'half_pixel_centers'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, resize_op, resize_input_op,
                     transformation_mode, scale_height, scale_width, output_names=None):
            super(ResizeBilinearLayerResolver.Descriptor, self).__init__('Resize', name, nodes, output_names=output_names)
            self.transformation_mode = transformation_mode
            self.input_tensor_shape = input_tensor_shape
            self.interpolation_mode = ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR
            self.resize_op = resize_op
            self.resize_input_op = resize_input_op
            self.scale_height = scale_height
            self.scale_width = scale_width

        def is_input_op(self, op):
            return len(op.inputs) and op.inputs[0].op == self.resize_input_op

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        sequence_resize = GraphSequence([ConverterSequenceNode('root', ['ResizeBilinear'])])
        sequence_resize.set_outputs(['root'])

        sequence_shape_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeBilinear'])])

        sequence_shape_stridedslice_resize.set_inputs('shape', ['input'])
        sequence_shape_stridedslice_resize.set_inputs('stridedSlice', ['shape',
                                                                       'const_stridedSlice_1',
                                                                       'const_stridedSlice_2',
                                                                       'const_stridedSlice_3'])
        sequence_shape_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_shape_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_shape_stridedslice_resize.set_outputs(['root'])

        sequence_resize_for_decode = GraphSequence([
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('strided_slice_1', ['StridedSlice']),
            ConverterSequenceNode('strided_slice', ['StridedSlice']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('size', ['Pack']),
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('root', ['ResizeBilinear']),
            ConverterSequenceNode('stub_8', ['?']),
            ConverterSequenceNode('stub_9', ['?']),
            ConverterSequenceNode('stub_10', ['?']),
            ConverterSequenceNode('stub_11', ['?']),
            ConverterSequenceNode('stub_12', ['?']),
            ConverterSequenceNode('stub_13', ['?']),
            ConverterSequenceNode('stub_14', ['?']),
            ConverterSequenceNode('stub_15', ['?']),
        ])
        sequence_resize_for_decode.set_inputs('mul_1', ['strided_slice_1', 'stub_14'])
        sequence_resize_for_decode.set_inputs('root', ['input', 'size'])
        sequence_resize_for_decode.set_inputs('shape', ['input'])
        sequence_resize_for_decode.set_inputs('mul', ['strided_slice', 'stub_15'])
        sequence_resize_for_decode.set_inputs('strided_slice', ['shape', 'stub_11', 'stub_12', 'stub_13'])
        sequence_resize_for_decode.set_inputs('size', ['mul', 'mul_1'])
        sequence_resize_for_decode.set_inputs('strided_slice_1', ['shape', 'stub_8', 'stub_9', 'stub_10'])
        sequence_resize_for_decode.set_outputs(['root'])

        self.sequences = [sequence_resize, sequence_shape_stridedslice_resize, sequence_resize_for_decode]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                resize_op = match['root']
                transformation_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC
                if resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS):
                    transformation_mode =ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS
                try:
                    # only newer version of TF have half_pixel_center
                    if resize_op.get_attr(self.TF_ATTRIBUTE_HALF_PIXEL_CENTERS):
                        transformation_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL
                except ValueError:
                    pass
                resize_input_op = match['input'] if 'input' in match else resize_op.inputs[0].op
                input_tensor_shape = graph_helper.get_op_output_shape(resize_input_op)
                output_tensor_shape = graph_helper.get_op_output_shape(resize_op.outputs[0].op)
                log_assert(len(input_tensor_shape) in [3, 4] and len(output_tensor_shape) in [3, 4],
                           "Unsupported rank for input/output for resize {}. Expected either rank 3 or 4"
                           .format(resize_op.name))
                scale_height = output_tensor_shape[-3] / input_tensor_shape[-3]
                scale_width = output_tensor_shape[-2] / input_tensor_shape[-2]
                consumed_nodes = match.consumed_nodes


                descriptors.append(
                    ResizeBilinearLayerResolver.Descriptor(str(resize_op.name),
                                                           consumed_nodes,
                                                           input_tensor_shape,
                                                           resize_op,
                                                           resize_input_op,
                                                           transformation_mode,
                                                           scale_height,
                                                           scale_width,
                                                           output_names=[str(resize_op.outputs[0].name)]))

        return descriptors


class ResizeNearestNeighborLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'
    TF_ATTRIBUTE_HALF_PIXEL_CENTERS = 'half_pixel_centers'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, resize_op, resize_input_op, transformation_mode,
                     scale_height, scale_width, output_names=None):
            super(ResizeNearestNeighborLayerResolver.Descriptor, self).__init__('ResizeNearestNeighbor', name, nodes,
                                                                                output_names=output_names)
            self.transformation_mode = transformation_mode
            self.input_tensor_shape = input_tensor_shape
            self.interpolation_mode = ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST
            self.resize_op = resize_op
            self.resize_input_op = resize_input_op
            self.scale_height = scale_height
            self.scale_width = scale_width

        def is_input_op(self, op):
            return len(op.inputs) and op.inputs[0].op == self.resize_input_op

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        sequence_resize = GraphSequence([ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])
        sequence_resize.set_outputs(['root'])

        sequence_shape_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])

        sequence_shape_stridedslice_resize.set_inputs('shape', ['input'])
        sequence_shape_stridedslice_resize.set_inputs('stridedSlice', ['shape',
                                                                       'const_stridedSlice_1',
                                                                       'const_stridedSlice_2',
                                                                       'const_stridedSlice_3'])
        sequence_shape_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_shape_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_shape_stridedslice_resize.set_outputs(['root'])

        sequence_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('const_stridedSlice_4', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])

        sequence_stridedslice_resize.set_inputs('stridedSlice', ['const_stridedSlice_1',
                                                                 'const_stridedSlice_2',
                                                                 'const_stridedSlice_3',
                                                                 'const_stridedSlice_4'])
        sequence_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_stridedslice_resize.set_outputs(['root'])

        # sequence for nearest neighbour resize without using tf resize op. Eg: seen in Mobilenetv1-FPN-SSD
        sequence_reshape_mul_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('Shape', ['Shape']),
            ConverterSequenceNode('Reshape/shape', ['Pack']),
            ConverterSequenceNode('strided_slice', ['StridedSlice']),
            ConverterSequenceNode('Reshape', ['Reshape']),
            ConverterSequenceNode('Reshape_1/shape', ['Pack']),
            ConverterSequenceNode('scale_mul', ['Mul']),
            ConverterSequenceNode('root', ['Reshape']),  # root here is the reshape layer for getting back
                                                         # to input shape
            ConverterSequenceNode('stub_1', ['?']),
            ConverterSequenceNode('stub_2', ['?']),
            ConverterSequenceNode('stub_3', ['?']),
            ConverterSequenceNode('stub_4', ['?']),
            ConverterSequenceNode('stub_5', ['?']),
            ConverterSequenceNode('stub_6', ['?']),
            ConverterSequenceNode('stub_7', ['?']),
            ConverterSequenceNode('stub_8', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('stub_10', ['?']),
            ConverterSequenceNode('stub_11', ['?']),
            ConverterSequenceNode('stub_12', ['?'])
            ])
        sequence_reshape_mul_resize.set_inputs('Shape', ['input'])
        sequence_reshape_mul_resize.set_inputs('strided_slice', ['Shape', 'stub_1', 'stub_2', 'stub_3'])
        sequence_reshape_mul_resize.set_inputs('Reshape/shape', ['strided_slice', 'stub_4', 'stub_5', 'stub_6', 'stub_7', 'stub_8'])
        sequence_reshape_mul_resize.set_inputs('Reshape', ['input','Reshape/shape'])
        sequence_reshape_mul_resize.set_inputs('scale_mul', ['Reshape', 'mul_const'])
        sequence_reshape_mul_resize.set_inputs('Reshape_1/shape', ['strided_slice', 'stub_10', 'stub_11', 'stub_12'])
        sequence_reshape_mul_resize.set_inputs('root', ['scale_mul', 'Reshape_1/shape'])
        sequence_reshape_mul_resize.set_outputs(['root'])

        self.sequences = [sequence_resize, sequence_shape_stridedslice_resize, sequence_stridedslice_resize, sequence_reshape_mul_resize]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                resize_op = match['root']
                tranformation_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC
                try:
                    if resize_op.get_attr(self.TF_ATTRIBUTE_HALF_PIXEL_CENTERS):
                        tranformation_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL
                except ValueError:
                    pass
                try:
                    # Model where resize is done without calling tf resize nearest neighbor by just
                    # using reshape slice and mul. Hence have a default of False for align corners
                    align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
                    if align_corners_bool:
                        tranformation_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS
                except ValueError:
                    pass
                # if a tf resize op is not used, input should be the first input to the pattern matching
                resize_input_op = match['input'] if 'input' in match else resize_op.inputs[0].op
                input_tensor_shape = graph_helper.get_op_output_shape(resize_input_op)
                output_tensor_shape = graph_helper.get_op_output_shape(resize_op.outputs[0].op)
                log_assert(len(input_tensor_shape) in [3, 4] and len(output_tensor_shape) in [3, 4],
                           "Unsupported rank for input/output for resize {}. Expected either rank 3 or 4"
                           .format(resize_op.name))
                scale_height = output_tensor_shape[-3] / input_tensor_shape[-3]
                scale_width = output_tensor_shape[-2] / input_tensor_shape[-2]
                consumed_nodes = match.consumed_nodes

                descriptors.append(
                    ResizeNearestNeighborLayerResolver.Descriptor(str(resize_op.name),
                                                                  consumed_nodes,
                                                                  input_tensor_shape,
                                                                  resize_op,
                                                                  resize_input_op,
                                                                  tranformation_mode,
                                                                  scale_height,
                                                                  scale_width,
                                                                  output_names=[str(resize_op.outputs[0].name)]))

        return descriptors


class ResizeLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return ir_graph.add(ResizeOp(descriptor.output_names[0],
                                     transformation_mode=descriptor.transformation_mode,
                                     interpolation_mode=descriptor.interpolation_mode,
                                     scale_height=descriptor.scale_height,
                                     scale_width=descriptor.scale_width),
                            input_names=input_name,
                            output_names=descriptor.output_names[0])
