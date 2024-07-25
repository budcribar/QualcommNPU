# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.op_adapter import ExtractPatchesOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class ExtractPatchesLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, size, stride, rate, padding,
                     output_names=None):
            super(ExtractPatchesLayerResolver.Descriptor, self).__init__('ExtractPatches', name,
                                                                         operations,
                                                                         output_names=output_names)
            self.size = size
            self.stride = stride
            self.rate = rate
            self.padding = padding

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('extract_patches', ['ExtractImagePatches'])])
        self.sequence.set_outputs(['extract_patches'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)

        # Nothing matched
        if len(matches) == 0:
            return []

        potential_descriptors = []
        for match in matches:
            extract_patches = match['extract_patches']

            # sizes
            sizes_value = extract_patches.get_attr('ksizes')
            if len(sizes_value) != 4:
                raise ConverterError('Cannot resolve extract patches layer, sizes must be exactly 4 values.')
            sizes = sizes_value[1:3]

            # strides
            strides_value = extract_patches.get_attr('strides')
            if len(strides_value) != 4:
                raise ConverterError('Cannot resolve extract patches layer, strides must be exactly 4 values.')
            strides = strides_value[1:3]

            # rates
            rates_value = extract_patches.get_attr('rates')
            if len(rates_value) != 4:
                raise ConverterError('Cannot resolve extract patches layer, rates must be exactly 4 values.')
            rates = rates_value[1:3]

            # padding
            padding = extract_patches.get_attr('padding')
            if padding.decode().upper() == "VALID":
                padding = ir_graph.QNN_OP_EXTRACT_PATCHES_PADDING_VALID
            elif padding.decode().upper() == "SAME":
                padding = ir_graph.QNN_OP_EXTRACT_PATCHES_PADDING_SAME
            else:
                raise ConverterError('Got unsupported padding: {}.'.format(padding.decode()))

            output_op_nodes_names = [str(extract_patches.outputs[0].name)]
            consumed_nodes = match.consumed_nodes

            extract_patches_descriptor = ExtractPatchesLayerResolver.Descriptor(str(extract_patches.name), consumed_nodes,
                                                                                size=sizes,
                                                                                stride=strides,
                                                                                rate=rates,
                                                                                padding=padding,
                                                                                output_names=output_op_nodes_names)
            potential_descriptors.append(extract_patches_descriptor)
        return potential_descriptors


class ExtractPatchesLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ExtractPatchesLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ExtractPatchesOp(name=descriptor.layer_name,
                                             size=descriptor.size,
                                             stride=descriptor.stride,
                                             rate=descriptor.rate,
                                             padding=descriptor.padding),
                            input_names=input_names,
                            output_names=output_name)
