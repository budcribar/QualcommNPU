# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from .onnx_translations import *
from .util import *

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker

# ------------------------------------------------------------------------------
#   Sparse Convolution
# ------------------------------------------------------------------------------
class OnnxSparseConvTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.reuse_sparse_indices = True

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        params = extract_attributes(src_op, attr_infos=
                                    [('stride', 'li', [1, 1, 1]),
                                     ('padding', 'li', [1, 1, 1]),
                                     ('dilation', 'li', [1, 1, 1]),
                                     ('kernel_size', 'li'),
                                     ('subm', 'i', 1)])

        self.reuse_sparse_indices = params.subm
        weights_constant_op = self.fetch_constant_op(input_names[1], converter_context, prunable=False, fail_if_dynamic=False)
        weights_constant_op.tensor = np.transpose(weights_constant_op.tensor, (0, 4, 1, 2, 3))
        weights_constant_op.tensor = np.ascontiguousarray(weights_constant_op.tensor.astype(np.float32))
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_size:
                log_assert(tuple(params.kernel_size) == weights_constant_op.tensor.shape[2:],
                           code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.
                      AxisFormat.OIDHW])
        elif graph.has_buffer(input_names[1]):
            raise ValueError("Unsupported data format for weight tensors")

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], converter_context, prunable=False, fail_if_dynamic=False)
            bias_constant_op.tensor = np.ascontiguousarray(bias_constant_op.tensor.astype(np.float32))
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract the remaining attributes and calculate the padding size
        padding_size_strategy = IRPaddingStrategies.py_to_c[extract_padding_mode('NOTSET', src_op.name)]

        # set the input padding size
        input_shape = graph.get_buffer(input_names[0]).shape
        weights_shape = graph.get_buffer(input_names[1]).shape

        pad_z = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[2],
                                                         weights_shape[0],
                                                         params.dilation[0],
                                                         params.stride[0],
                                                         padding_size_strategy,
                                                         [params.padding[0],
                                                          params.padding[0]])

        pad_y = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[3],
                                                         weights_shape[1],
                                                         params.dilation[1],
                                                         params.stride[1],
                                                         padding_size_strategy,
                                                         [params.padding[1],
                                                          params.padding[1]])

        pad_x = ir_graph.Conv3dOp.calc_conv_padding_size(input_shape[4],
                                                         weights_shape[2],
                                                         params.dilation[2],
                                                         params.stride[2],
                                                         padding_size_strategy,
                                                         [params.padding[2],
                                                          params.padding[2]])

        op = op_adapter.Conv3dOp(src_op.name,
                                 bias_op_name=bias_op_name,
                                 pady_before=pad_y[0],
                                 pady_after=pad_y[1],
                                 padx_before=pad_x[0],
                                 padx_after=pad_x[1],
                                 padz_before=pad_z[0],
                                 padz_after=pad_z[1],
                                 padding_size_strategy=padding_size_strategy,
                                 stridex=params.stride[2],
                                 stridey=params.stride[1],
                                 stridez=params.stride[0],
                                 dilationx=params.dilation[2],
                                 dilationy=params.dilation[1],
                                 dilationz=params.dilation[0],
                                 groups=1,
                                 data_layout=AxisTracker.AxisFormat.NCDHW,
                                 reuse_sparse_indicies=self.reuse_sparse_indices)
        return op

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        params = extract_attributes(src_op, attr_infos=
                                    [('ndim', 'i'),
                                     ('output_bound', 'i'),
                                     ('input_spatial_shape', 'li'),
                                     ('activation', 's', 'None'),
                                     ('in_channels', 'i')])

        coo_info = ir_graph.IrSparseHybridCooInfo(params.output_bound, params.ndim)
        sparse_params = ir_graph.IrSparseParamsInfo(coo_info)

        if len(src_op.input) == 4:
            shape = [1, params.in_channels, *params.input_spatial_shape]
            create_sparse_op = op_adapter.CreateSparseOp("CreateSparse", shape=shape)
            create_sparse_input = src_op.input[:2]

            for input in create_sparse_input:
                if graph.has_buffer(input):
                    input_buffer = graph.get_buffer(input)
                    if len(input_buffer.get_consumers()) == 0:
                        input_buffer.set_buf_dims([params.output_bound, input_buffer.get_buf_dims()[1]])
                        producer_op = graph.get_producer_op(input_buffer.name)
                        producer_op.shape = [params.output_bound, input_buffer.get_buf_dims()[1]]
                else:
                    raise ValueError("Graph does not contain input buffer {} for node: {}"
                                     .format(input, src_op.name))

            create_sparse_output = ["create_sparse_output_1"]
            create_sparse_node = graph.add(create_sparse_op, create_sparse_input, create_sparse_output,
                                           axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL], sparse_params=sparse_params)
            graph.add_src_op_info(create_sparse_node.op.name, create_sparse_input, create_sparse_output)
            del src_op.input[:2]
            src_op.input.insert(0, create_sparse_output[0])

        op = self.extract_parameters(src_op, context)
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)

        if params.activation != "None":
            neuron_op_name = op.name + "_activation"
            conv_op_output = op.name + "_out"
            neuron_type = ir_graph.ElementwiseNeuronOp.extract_neuron_type(params.activation)
            neuron_op = op_adapter.ElementwiseNeuronOp(neuron_op_name,
                                                       operation=op_adapter.ElementwiseNeuronOp.neuron_to_operation[neuron_type])
            node = graph.add(op, input_names, [conv_op_output], axis_formats=[AxisTracker.AxisFormat.NCDHW], sparse_params=sparse_params)
            neuron_node = graph.add(neuron_op, [conv_op_output], output_names,
                                    axis_formats=[AxisTracker.AxisFormat.NCDHW], sparse_params=sparse_params)
            graph.add_src_op_info(neuron_node.op.name, [conv_op_output], output_names)
            src_op.output.remove(output_names[0])
            src_op.output.insert(0, conv_op_output)
        else:
            node = graph.add(op, input_names, output_names, axis_formats=[AxisTracker.AxisFormat.NCDHW], sparse_params=sparse_params)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

OnnxTranslations.register_translation(OnnxSparseConvTranslation(),
                                      converter_type('SparseConvolution', 'spconv'))

# ------------------------------------------------------------------------------
#   ScatterDense
# ------------------------------------------------------------------------------
class OnnxScatterDenseTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.SparseToDenseOp(src_op.name)

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op = self.extract_parameters(src_op, context)
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        node = graph.add(op, input_names, output_names, axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

OnnxTranslations.register_translation(OnnxScatterDenseTranslation(),
                                      converter_type('ScatterDense', 'spconv'))