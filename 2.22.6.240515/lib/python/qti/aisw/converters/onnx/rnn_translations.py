# ==============================================================================
#
#  Copyright (c) 2018-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common import ir_graph
from .onnx_translations import *
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.utils import code_to_message

# ------------------------------------------------------------------------------
#  RNNTranslationBase
# ------------------------------------------------------------------------------
OPTIONAL_INPUTS = NamedDict(initial_h='', initial_c='', norm_weights='', cell_state_weights='', proj_weights='', proj_bias='')
RNN_INPUT_TYPES = ('_initial_h', '_initial_c')
RNN_OUTPUT_TYPES = ('_all_hidden', '_final_hidden', '_final_cell')


class OnnxRnnTranslationsBase(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.weights = []
        self.rec_weights = []
        self.bias = None
        self.cell_state_weights = None
        self.params = NamedDict()
        self.no_of_gates = 1
        self.direction = ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD
        self.output_names = []
        self.num_directions = 1
        self.timesteps = []

    def extract_params_for_type(self, src_op, converter_context, rnn_type='Rnn'):
        graph = converter_context.ir_graph

        self.input_names = list(map(str, src_op.input))
        self.output_names = self.extract_output_names(src_op, converter_context)
        self.params.direction = str(self.params.direction).lower()
        self.direction = ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD if self.params.direction == 'forward' else ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE
        if rnn_type == 'Gru':
            self.direction = ir_graph.QNN_OP_GRU_DIRECTION_FORWARD if self.params.direction == 'forward' else ir_graph.QNN_OP_GRU_DIRECTION_REVERSE
        self.num_directions = 1

        # Ensure weights or rec_weights are not passed in dynamically
        weights_input_name = self.input_names[1]
        if not converter_context.weights.has(weights_input_name) or \
            (graph.has_buffer(weights_input_name) and not isinstance(graph.get_producer_op(weights_input_name), op_adapter.ConstantOp)):
                raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(weights_input_name,
                                                                                                  src_op.name))
        self.weights = converter_context.weights.fetch(weights_input_name)

        rec_weights_input_name = self.input_names[2]
        if not converter_context.weights.has(rec_weights_input_name) or \
            (graph.has_buffer(rec_weights_input_name) and not isinstance(graph.get_producer_op(rec_weights_input_name), op_adapter.ConstantOp)):
                raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(rec_weights_input_name,
                                                                                                  src_op.name))
        self.rec_weights = converter_context.weights.fetch(rec_weights_input_name)

        # ONNX may use empty string as a placeholder
        # So add an and-condition to further check it.
        if len(self.input_names) >= 4 and self.input_names[3]:
            bias_input_name = self.input_names[3]
            if not converter_context.weights.has(bias_input_name) or \
                (graph.has_buffer(bias_input_name) and not isinstance(graph.get_producer_op(bias_input_name), op_adapter.ConstantOp)):
                    raise ValueError("Unsupported dynamic bias for input {} of RNN node {}".format(bias_input_name,
                                                                                                   src_op.name))
            self.bias = converter_context.weights.fetch(bias_input_name)

        if len(self.input_names) >= 5 and self.input_names[4]:
            self.timesteps = int(converter_context.weights.fetch(self.input_names[4]))

        # Fetch cell_state_weights by source op input names if provided
        self.cell_state_weights = None

        # If it is bi-directional check that weights and rec_weights
        # have the right shape
        if self.params.direction == "bidirectional":
            log_assert(self.weights.shape[0] == 2 and self.rec_weights.shape[0] == 2,
                       "Node {}: Bidirectional input requires two sets of weights and recurrent "
                       "weights each. Got only {} set of weights",
                       src_op.name, self.weights.shape[0])

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):

        no_of_gates = self.no_of_gates

        if bias is None:
            bias = np.zeros((no_of_gates, hidden_size), dtype=np.float32)
        else:
            # for probably vendor specific reasons, ONNX defines GRU bias to
            # be separated into forward and recurrent parts, that are always
            # added together (unless linear_before_reset is false, but we
            # don't support that). So we will always combine.
            # We need to reshape bias which is in (2*no_of_gates*hidden_size)
            # into (2, no_of_gates * hidden_size).
            bias = np.reshape(bias, (2, no_of_gates * hidden_size))
            new_bias = np.empty((no_of_gates * hidden_size), dtype=np.float32)
            # Elements are stored in [weights, rec_weights] where each column
            # represents the gate and the number of rows is the hidden size
            np.add(bias[0, :], bias[1, :], out=new_bias[:])
            bias = new_bias.reshape(no_of_gates, hidden_size)

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size)respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def prepare_params_as_constants(self, graph, params):
        for param_name, tensor in params.items():
            if not graph.has_buffer(param_name):
                graph.add(op_adapter.ConstantOp(name=param_name, tensor=tensor),
                          input_names=[], output_names=[param_name],
                          axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                graph.add_src_op_info(param_name, [], [param_name])
            elif graph.get_producer_op(param_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("lstm requires weights and bias to be constant, got dynamic tensor from {}".format(
                    graph.get_producer_op(param_name).name))

    def extract_output_names(self, src_op, converter_context):
        onnx_output_names = [output for i, output in enumerate(src_op.output) if output]
        onnx_output_names_len = len(onnx_output_names)
        has_initial_state = bool(OPTIONAL_INPUTS.initial_h and OPTIONAL_INPUTS.initial_c)
        if onnx_output_names_len < 3 and not has_initial_state:
            return onnx_output_names
        else:
            # ONNX LSTM output order: Y, Y_h, Y_c
            # SNPE LSTM output order: h, c_T, h_T
            if has_initial_state and len(onnx_output_names) < 3:
                # add dummy output names if initial states are present
                output_names = [onnx_output_names[0], onnx_output_names[0] + "_cell_output_dummy",
                                onnx_output_names[0] + "_hidden_output_dummy"]
            else:
                output_names = [onnx_output_names[0], onnx_output_names[2], onnx_output_names[1]]
            for i in range(3, onnx_output_names_len):
                output_names.append(onnx_output_names[i])

            # update output names
            self.output_names = output_names

            return output_names

    def create_rnn(self, src_op, converter_context, create_unidirectional_func, create_bidirectional_func, rnn_type='Rnn'):
        graph = converter_context.ir_graph

        if self.params.direction == "bidirectional":
            create_bidirectional_func(src_op, converter_context)
        else:
            # set up naming so that the buffers are all different and tagged correctly
            input_names = self.extract_input_names(src_op, converter_context)
            output_names = self.extract_output_names(src_op, converter_context)
            # QNN GruOp has two mandatory outputs. If the model has only one output then add
            # an additional output which signifies the final hidden output.
            if rnn_type == 'Gru' and input_names[1] and len(output_names) == 1:
                final_hidden_output_name = output_names[0] + '_unidirection'
                output_names.append(final_hidden_output_name)
            output_names_len = len(output_names)
            module_name = src_op.name if len(src_op.name) != 0 else \
                output_names[0] if output_names_len != 0 else \
                    src_op.op_type

            # set up rnn ops
            rnn_op = create_unidirectional_func(graph, module_name, input_names)

            input_x_name = input_names[0]
            if graph.has_buffer(input_x_name):
                input_x_buf = graph.get_buffer(input_x_name)
                input_x_buf.set_axis_format(AxisTracker.AxisFormat.TNF)
            else:
                raise KeyError("Graph has no buffer {} for {}Op input x", input_x_name, rnn_type)

            for idx, input_name in enumerate(input_names[1:3]):
                # GruOp only has initial_h.
                if rnn_type == 'Gru' and idx == 1:
                    break
                # Add constant op if one of the initial c/h inputs is Initializer and not added to graph
                if converter_context.weights.has(input_name) and not graph.has_buffer(input_name):
                    tensor = converter_context.weights.fetch(input_name, prunable=False)
                    graph.add(op_adapter.ConstantOp(input_name, tensor),
                              [],
                              input_name,
                              axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                elif graph.has_buffer(input_name):
                    # set axis format as NONTRIVIAL for dynamic c/h inputs
                    input_buf = graph.get_buffer(input_name)
                    input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

            # set up reshape ops
            reshape_ops = []
            input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
            batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
            hidden_size = self.rec_weights.shape[-1]

            output_shapes = []
            for i in range(0, output_names_len):
                if i == 0:
                    output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
                else:
                    output_shapes.append([self.num_directions, batch_size, hidden_size])
            for i in range(0, output_names_len):
                # the outputs are reshaped to be ONNX format
                reshape_ops.append(op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape',
                                                        shape=output_shapes[i]))

            rnn_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_rnn"
                                for i, name in enumerate(output_names)]

            reshape_output_names = [str(name) for name in output_names]

            self.add_src_op_info(rnn_op.name, src_op, graph)
            rnn_node = graph.add(rnn_op, input_names, rnn_output_names)

            for i in reversed(range(0, output_names_len)):
                # add reshape for outputs to make shape to be ONNX's format
                log_debug("Adding reshape op {} while creating unidirectional RNN unit".format(reshape_ops[i].name))
                graph.add(reshape_ops[i], [rnn_output_names[i]], [reshape_output_names[i]])
                graph.add_src_op_info(reshape_ops[i].name, [rnn_output_names[i]], [reshape_output_names[i]])

            return rnn_node

    def create_bidirectional_module(self, src_op, converter_context, weights, rec_weights, bias, params,
                                    create_rnn_type):
        graph = converter_context.ir_graph
        # set up naming so that the buffers are all different and tagged correctly
        src_op_inputs = list(src_op.input)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        module_name = src_op.name if len(src_op.name) != 0 else output_names[0]
        initial_h_c_inputs_present = True if OPTIONAL_INPUTS.initial_h and OPTIONAL_INPUTS.initial_c else False

        src_op_inputs_len = len(src_op_inputs)
        if src_op_inputs_len < 3 or src_op_inputs_len > 8:
            raise ValueError("Unsupported number of inputs for source LSTM op {}. Expected number of inputs between 3 and 8, got {}".
                             format(src_op.name, src_op_inputs_len))

        output_names_len = len(output_names)
        if output_names_len > 3:
            raise ValueError("Unsupported number of outputs for source bidirectional LSTM op {}. Expected number of outputs between 0 and 3, got {}".
                             format(src_op.name, output_names_len))

        # bidirectional lstm shares the same source weights name of forward and backward, so we need to specify them in graph
        forward_op_input_names = input_names[:]
        backward_op_input_names = input_names[:]
        for i in range(3, len(input_names)):
            curr_input_name = input_names[i]
            if curr_input_name:
                forward_op_input_names[i] = curr_input_name + '_forward'
                backward_op_input_names[i] = curr_input_name + '_backward'

        h_0_forward_split_name, h_0_backward_split_name = '', ''
        c_0_forward_split_name, c_0_backward_split_name = '', ''
        # set up split ops for initial_h and initial_c along `num_directions` dimension if present
        if initial_h_c_inputs_present:
            src_h_c_input_names = src_op_inputs[5:7]
            h_c_split_output_names = list()
            for i, name in enumerate(src_h_c_input_names):
                # add initial_h and initial_c to graph
                if converter_context.weights.has(name) and not graph.has_buffer(name):
                    tensor = converter_context.weights.fetch(name, prunable=False)
                    graph.add(op_adapter.ConstantOp(name, tensor),
                              [],
                              name,
                              axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                elif graph.has_buffer(name):
                    input_buf = graph.get_buffer(name)
                    input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

                split_op_name = f"{module_name}_{name}_{RNN_INPUT_TYPES[i]}_split"
                # If split_index is not given, split equally along the specified axis
                split_op = op_adapter.SplitOp(split_op_name, axis=0)
                split_output_names = [f"{split_op_name}_forward",
                                      f"{split_op_name}_backward"]
                h_c_split_output_names.append(split_output_names)
                log_debug("Adding split op {} while creating bidirectional RNN unit".format(split_op.name))
                graph.add(split_op, [name], split_output_names)
                graph.add_src_op_info(split_op.name, [name], split_output_names)

            h_0_forward_split_name, h_0_backward_split_name = h_c_split_output_names[0]
            c_0_forward_split_name, c_0_backward_split_name = h_c_split_output_names[1]

        # set up forward op
        forward_op = create_rnn_type(graph,
                                     module_name + '_forward',
                                     forward_op_input_names,
                                     weights=weights[0, :, :],
                                     rec_weights=rec_weights[0, :, :],
                                     bias=bias[0, :] if bias is not None else bias,
                                     hidden_size=params.hidden_size,
                                     direction=ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD,
                                     h_0_input_name=h_0_forward_split_name,
                                     c_0_input_name=c_0_forward_split_name)

        # set up backward op
        backward_op = create_rnn_type(graph,
                                      module_name + '_backward',
                                      backward_op_input_names,
                                      weights=weights[1, :, :],
                                      rec_weights=rec_weights[1, :, :],
                                      bias=bias[1, :] if bias is not None else bias,
                                      hidden_size=params.hidden_size,
                                      direction=ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE,
                                      h_0_input_name=h_0_backward_split_name,
                                      c_0_input_name=c_0_backward_split_name)

        # set up reshape ops
        reshape_ops = []
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
        hidden_size = params.hidden_size
        output_shapes = []
        for i in range(0, output_names_len):
            # The shape is for unidirectional lstm, so we set num_directions to 1 here
            # Outputs from forward and backward lstm will then be concatenated together
            if i == 0:
                output_shapes.append([time_steps, 1, batch_size, hidden_size])
            else:
                output_shapes.append([1, batch_size, hidden_size])
        for i in range(0, output_names_len):
            # the outputs are reshaped to be ONNX format
            reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape_forward',
                                                     shape=output_shapes[i]),
                                op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape_backward',
                                                     shape=output_shapes[i])])

        # set up concat ops
        concat_ops = []
        for i in range(0, output_names_len):
            # The first output is used to concat all hidden output values at last axis
            axis = 1 if i == 0 else 0
            concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

        forward_output_names = [module_name + str(name) + RNN_OUTPUT_TYPES[i] + "_forward" for i, name in
                                enumerate(output_names)]
        backward_output_names = [module_name + str(name) + RNN_OUTPUT_TYPES[i] + "_backward" for i, name in
                                 enumerate(output_names)]
        reshape_output_names = [[module_name + str(name) + RNN_OUTPUT_TYPES[i] + "_reshape_forward",
                                 module_name + str(name) + RNN_OUTPUT_TYPES[i] + "_reshape_backward"]
                                for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]
        forward_input_names = [input_names[0], h_0_forward_split_name, c_0_forward_split_name] + forward_op_input_names[3:]
        backward_input_names = [input_names[0], h_0_backward_split_name, c_0_backward_split_name] + backward_op_input_names[3:]

        log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
        graph.add(forward_op, forward_input_names, forward_output_names)
        graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)

        log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
        graph.add(backward_op, backward_input_names, backward_output_names)
        graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for i in range(0, output_names_len):
            # add reshape for outputs to make shape to be ONNX's format
            log_debug("Adding reshape op {} and {} while creating bidirectional RNN unit".format(reshape_ops[i][0].name, reshape_ops[i][1].name))
            graph.add(reshape_ops[i][0], [forward_output_names[i]], [reshape_output_names[i][0]])
            graph.add(reshape_ops[i][1], [backward_output_names[i]], [reshape_output_names[i][1]])
            graph.add_src_op_info(reshape_ops[i][0].name, [forward_output_names[i]], [reshape_output_names[i][0]])
            graph.add_src_op_info(reshape_ops[i][1].name, [backward_output_names[i]], [reshape_output_names[i][1]])

            log_debug("Adding concat op {} while creating bidirectional RNN unit".format(concat_ops[i].name))
            graph.add(concat_ops[i], [reshape_output_names[i][0], reshape_output_names[i][1]], [concat_output_names[i]])
            graph.add_src_op_info(concat_ops[i].name, [reshape_output_names[i][0], reshape_output_names[i][1]], [concat_output_names[i]])


# ------------------------------------------------------------------------------
#  GRU
# ------------------------------------------------------------------------------

class OnnxGruTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('GRU', [1, 7, 14], [['clip',
                                                               'activation_alpha',
                                                               'activation_beta',
                                                               'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid','Tanh'])
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 3
        self.linear_before_reset = 0

    def extract_parameters(self, src_op, converter_context):
        self.params = extract_attributes(src_op, schema=self.op_schema())
        self.extract_params_for_type(src_op, converter_context, rnn_type='Gru')

        if len(self.input_names) >= 6:
            # check if initial_h is included
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

        self.linear_before_reset = self.params.linear_before_reset

    def extract_input_names(self, src_op, converter_context):
        input_weights_name = self.input_names[1]
        rec_weights_name = self.input_names[2]
        # Onnx may use empty string as a placeholder, so add an and-condition to further check it
        bias_name = self.input_names[3] if len(self.input_names) >= 4 and self.input_names[3] else 'gru_zero_bias'
        formatted_input_names = [self.input_names[0], OPTIONAL_INPUTS.initial_h, input_weights_name,
                                 rec_weights_name, bias_name]
        return formatted_input_names

    def extract_output_names(self, src_op, converter_context):
        onnx_output_names = [output for i, output in enumerate(src_op.output) if output]
        if len(onnx_output_names) < 2 :
            final_hidden_output_name = onnx_output_names[0] + '_hidden_output'
            onnx_output_names.append(final_hidden_output_name)

        return onnx_output_names

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):
        if bias is not None:
            # ONNX defines GRU bias to be separated into forward and recurrent parts
            # We need to reshape bias which is in (2*no_of_gates*hidden_size)
            # into (2, no_of_gates * hidden_size).
            bias = np.reshape(bias, (2, self.no_of_gates * hidden_size))

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size) respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (self.no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (self.no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.extract_parameters(src_op, converter_context)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, converter_context, self.create_unidirectional_gru,
                               self.create_bidirectional_gru, rnn_type='Gru')

    def create_unidirectional_gru(self, graph, name, input_name_list, **kwargs):

        if kwargs:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(kwargs['weights'],
                                                                       kwargs['rec_weights'],
                                                                       kwargs['bias'],
                                                                       kwargs['hidden_size'])
            h_0_input_name = kwargs['h_0_input_name']
        else:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(self.weights,
                                                                       self.rec_weights,
                                                                       self.bias,
                                                                       self.params.hidden_size)
            h_0_input_name = OPTIONAL_INPUTS.initial_h

        # gru specific organization into separate gates
        activations = self.params.activations

        # MergedWeightsGru specific organization into gate format
        gate_bias = bias.reshape(-1, )
        gate_rec_weights = rec_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
        gate_weights = weights.reshape(self.no_of_gates * self.params.hidden_size, -1)

        params_dict = {}
        # Extract the input names for weights and bias.
        if input_name_list is not None:
            input_weights_name, rec_weights_name, bias_name = input_name_list[2:]
            params_dict[input_weights_name] = gate_weights
            params_dict[rec_weights_name] = gate_rec_weights
            params_dict[bias_name] = gate_bias
            self.prepare_params_as_constants(graph, params_dict)

        return op_adapter.MergedWeightsGruOp(name,
                                             activation=activations[0],
                                             gate_activation=activations[0],
                                             rec_gate_activation=activations[1],
                                             h_0_input_name=h_0_input_name,
                                             direction = kwargs['direction'] if kwargs else self.direction,
                                             hidden_size=self.params.hidden_size,
                                             linear_before_reset = self.params.linear_before_reset,
                                             time_major=True)

    def create_bidirectional_gru(self, src_op, converter_context):
        graph = converter_context.ir_graph

        src_op_inputs = list(src_op.input)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # If the initial_h input is not present. QNN GruOp has two mandatory outputs.
        # If the model has only one output then add an additional output which signifies the final hidden output.
        if input_names[1] and len(output_names) == 1:
            final_hidden_output_name = output_names[0] + '_bidirection'
            output_names.append(final_hidden_output_name)
        module_name = src_op.name if len(src_op.name) != 0 else output_names[0]

        src_op_inputs_len = len(src_op_inputs)
        if src_op_inputs_len < 6:
            raise ValueError("Unsupported number of inputs on source op {}. Expected >= 6, got {}".
                             format(src_op.name, src_op_inputs_len))

        input_names_len = len(input_names)
        if input_names_len > 5:
            raise ValueError("Unsupported number of inputs for bidirectional unit. Expected <= 5, got {}".
                             format(input_names_len))

        output_names_len = len(output_names)
        if output_names_len > 2:
            raise ValueError("Unsupported number of outputs for bidirectional unit. Expected < 2, got {}".
                             format(output_names_len))

        forward_op_input_names = input_names[:]
        backward_op_input_names = input_names[:]
        for i in range(2, input_names_len):
            curr_input_name = input_names[i]
            if curr_input_name:
                forward_op_input_names[i] = curr_input_name + '_forward'
                backward_op_input_names[i] = curr_input_name + '_backward'

        # set up split ops
        split_ops = []
        split_ops.append(op_adapter.SplitOp(str(input_names[1]) + RNN_INPUT_TYPES[0] + "_split", axis=0))
        # set up forward op
        forward_op = self.create_unidirectional_gru(graph,
                                                    module_name + '_forward',
                                                    forward_op_input_names,
                                                    weights=self.weights[0, :, :],
                                                    rec_weights=self.rec_weights[0, :, :],
                                                    bias=self.bias[0, :] if self.bias is not None else self.bias,
                                                    hidden_size=self.params.hidden_size,
                                                    direction=ir_graph.QNN_OP_GRU_DIRECTION_FORWARD,
                                                    h_0_input_name=module_name + str(input_names[1]) + RNN_INPUT_TYPES[0] + "_forward_split" if
                                                    input_names_len > 1 else OPTIONAL_INPUTS.initial_h)

        # set up backward op
        backward_op = self.create_unidirectional_gru(graph,
                                                     module_name + '_backward',
                                                     backward_op_input_names,
                                                     weights=self.weights[1, :, :],
                                                     rec_weights=self.rec_weights[1, :, :],
                                                     bias=self.bias[1, :] if self.bias is not None else self.bias,
                                                     hidden_size=self.params.hidden_size,
                                                     direction=ir_graph.QNN_OP_GRU_DIRECTION_REVERSE,
                                                     h_0_input_name=module_name + str(input_names[1]) + RNN_INPUT_TYPES[0] + "_backward_split" if
                                                     input_names_len > 1 else OPTIONAL_INPUTS.initial_h)

        # set up reshape ops
        reshape_ops = []
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
        hidden_size = self.params.hidden_size
        output_shapes = []
        for i in range(0, output_names_len):
            if i == 0:
                output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
            else:
                output_shapes.append([self.num_directions, batch_size, hidden_size])
        for i in range(0, output_names_len):
            # the outputs are reshaped to be ONNX format
            reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape_forward', shape=output_shapes[i]),
                                op_adapter.ReshapeOp(output_names[i] + RNN_OUTPUT_TYPES[i] + '_reshape_backward', shape=output_shapes[i])])
        # set up concat ops
        concat_ops = []
        for i in range(0, output_names_len):
            # The first output is used to concat all hidden output values at last axis
            axis = 1 if i == 0 else 0
            concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

        for input_name in input_names[1:2]:
            # Add constant op if one of the initial h input is Initializer and not added to graph
            if converter_context.weights.has(input_name) and not graph.has_buffer(input_name):
                tensor = converter_context.weights.fetch(input_name, prunable=False)
                graph.add(op_adapter.ConstantOp(input_name, tensor),
                          [],
                          input_name,
                          axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
            elif graph.has_buffer(input_name):
                # set axis format as NONTRIVIAL for dynamic h input
                input_buf = graph.get_buffer(input_name)
                input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        split_output_names = [[str(input_names[1]) + RNN_INPUT_TYPES[0] + "_forward_split",
                               str(input_names[1]) + RNN_INPUT_TYPES[0] + "_backward_split"]]
        forward_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_forward" for i, name in
                                enumerate(output_names)]
        backward_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_backward" for i, name in
                                 enumerate(output_names)]
        reshape_output_names = [[str(name) + RNN_OUTPUT_TYPES[i] + "_reshape_forward",
                                 str(name) + RNN_OUTPUT_TYPES[i] + "_reshape_backward"]
                                for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]

        # add split ops to the graph, each split op should have two outputs, one going into each RNN
        # inputs[i] where i >= 1 must be split along num_directions dimension in ONNX
        split_src_op_inputs = src_op_inputs[5:]
        split_input_names = [input_names[1]]

        if len(split_src_op_inputs) < len(split_input_names):
            raise ValueError("Source op {} requires initial_h.".format(src_op.name))

        for i in range(0, len(split_input_names)):
            log_debug("Adding split op {} while creating bidirectional RNN unit".format(split_ops[i].name))
            graph.add(split_ops[i], [split_input_names[i]], split_output_names[i])
            graph.add_src_op_info(split_ops[i].name, split_src_op_inputs[i], split_output_names[i])

        # Modify input names to be different according to split
        forward_input_names = [input_names[0]] + [split_output_names[i][0] for i in range(len(split_input_names))] + forward_op_input_names[2:]
        backward_input_names = [input_names[0]] + [split_output_names[i][1] for i in range(len(split_input_names))] + backward_op_input_names[2:]

        log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
        graph.add(forward_op, forward_input_names, forward_output_names)
        graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)

        log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
        graph.add(backward_op, backward_input_names, backward_output_names)
        graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for i in range(0, output_names_len):
            # add reshape for outputs to make shape to be ONNX's format
            log_debug("Adding reshape op {} and {} while creating bidirectional RNN unit".format(reshape_ops[i][0].name, reshape_ops[i][1].name))
            graph.add(reshape_ops[i][0], [forward_output_names[i]], [reshape_output_names[i][0]])
            graph.add(reshape_ops[i][1], [backward_output_names[i]], [reshape_output_names[i][1]])
            graph.add_src_op_info(reshape_ops[i][0].name, [forward_output_names[i]], [reshape_output_names[i][0]])
            graph.add_src_op_info(reshape_ops[i][1].name, [backward_output_names[i]], [reshape_output_names[i][1]])

            log_debug("Adding concat op {} while creating bidirectional RNN unit".format(concat_ops[i].name))
            graph.add(concat_ops[i], [reshape_output_names[i][0], reshape_output_names[i][1]], [concat_output_names[i]])
            graph.add_src_op_info(concat_ops[i].name, [reshape_output_names[i][0], reshape_output_names[i][1]], [concat_output_names[i]])

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'linear_before_reset':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)
        elif attr_name == "activations":
            gru_supported_activations = ["Sigmoid", "Tanh"]
            if len(attr_value) == 4:
                # bidirectional case has double the number of activations
                gru_supported_activations = gru_supported_activations*2
            log_assert(attr_value == gru_supported_activations,
                       "Received activations list: {} differs from GRU supported activations list: {}",
                       attr_value, gru_supported_activations)



OnnxTranslations.register_translation(OnnxGruTranslation(),
                                      converter_type('GRU', 'onnx'),
                                      op_adapter.GruOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#  LSTM
# ------------------------------------------------------------------------------
class OnnxLSTMTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('LSTM', [1, 7, 14], [['clip',
                                                                'activation_alpha',
                                                                'activation_beta',
                                                                'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid', 'Tanh', 'Tanh'])
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 4
        self.peephole_weights = []

    def extract_parameters(self, src_op, converter_context):
        schema = self.op_schema(op_type=src_op.op_type)
        self.params = extract_attributes(src_op, schema=schema, validate=True)

        if "layout" in self.params and self.params.layout == 1:
            raise ValueError("ERROR: Unsupported value - 1 for layout attribute for {} op".format(src_op.name))

        # set parameters
        self.extract_params_for_type(src_op, converter_context)

        if len(self.input_names) >= 7:
            # check if initial_h and initial_c are included
            # snpe requires that if they are included, then all
            # 3 outputs will be returned.
            OPTIONAL_INPUTS.initial_c = self.input_names[6]
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

    def extract_input_names(self, src_op, converter_context):
        input_weights_name = self.input_names[1]
        rec_weights_name = self.input_names[2]
        # Onnx may use empty string as a placeholder, so add an and-condition to further check it
        bias_name = self.input_names[3] if len(self.input_names) >= 4 and self.input_names[3] else 'lstm_dummy_bias_all_zeros'

        formatted_input_names = [self.input_names[0], OPTIONAL_INPUTS.initial_h, OPTIONAL_INPUTS.initial_c,
                                 input_weights_name, rec_weights_name, bias_name,
                                 OPTIONAL_INPUTS.norm_weights, OPTIONAL_INPUTS.cell_state_weights,
                                 OPTIONAL_INPUTS.proj_weights, OPTIONAL_INPUTS.proj_bias]
        return formatted_input_names

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.extract_parameters(src_op, converter_context)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, converter_context, self.create_unidirectional_lstm,
                               self.create_bidirectional_lstm)

    def create_unidirectional_lstm(self, graph, op_name, input_name_list, **kwargs):
        direction = kwargs['direction'] if kwargs else self.direction
        h_0_input_name = kwargs['h_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_h
        c_0_input_name = kwargs['c_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_c

        # For constant weights, they could be shared across multiple RolledLstm nodes, so
        # add a check here to avoid post-processing the weights again
        skip_prepare_weights = all([graph.has_buffer(name) and isinstance(graph.get_producer_op(name), op_adapter.ConstantOp)
                                    for name in input_name_list[3:] if name])
        if not skip_prepare_weights:
            if kwargs:
                [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(kwargs['weights'],
                                                                                          kwargs['rec_weights'],
                                                                                          kwargs['bias'],
                                                                                          kwargs['hidden_size'])
            else:
                [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(self.weights,
                                                                                          self.rec_weights,
                                                                                          self.bias,
                                                                                          self.params.hidden_size)

            # transform from iofc to ifoc
            if self.no_of_gates == 4:
                self.params.hidden_size = int(gate_bias.size / self.no_of_gates)

                new_gate_bias = np.empty(gate_bias.shape, dtype=np.float32)
                for new, old in enumerate([0, 2, 1, 3]):
                    new_gate_bias[new, :] = gate_bias[old, :]
                gate_bias = new_gate_bias

                new_gate_rec_weights = np.empty(gate_rec_weights.shape, dtype=np.float32)
                for new, old in enumerate([0, 2, 1, 3]):
                    new_gate_rec_weights[new, :, :] = gate_rec_weights[old, :, :]
                gate_rec_weights = new_gate_rec_weights

                new_gate_weights = np.empty(gate_weights.shape, dtype=np.float32)
                for new, old in enumerate([0, 2, 1, 3]):
                    new_gate_weights[new, :, :] = gate_weights[old, :, :]
                gate_weights = new_gate_weights

            # LSTM specific organization into gate format
            gate_bias = gate_bias.reshape(-1, )
            gate_rec_weights = gate_rec_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
            gate_weights = gate_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)

            input_weights_name, rec_weights_name, bias_name = input_name_list[3:6]
            norm_weights_name, cell_state_weights_name, proj_weights_name, proj_bias_name = input_name_list[6:]

            params_dict = {}
            params_dict[input_weights_name] = gate_weights
            params_dict[rec_weights_name] = gate_rec_weights
            params_dict[bias_name] = gate_bias
            # cell state weights is the only optional input that can be captured in Onnx lstm
            if cell_state_weights_name:
                params_dict[cell_state_weights_name] = self.cell_state_weights

            self.prepare_params_as_constants(graph, params_dict)

        return op_adapter.RolledLstmOp(op_name,
                                       direction=direction,
                                       time_major=True,
                                       c_0_input_name=c_0_input_name,
                                       h_0_input_name=h_0_input_name,
                                       # if c_0 and h_0 exist, reset_state_at_time_step_0 will be False
                                       reset_state_at_time_step_0=False if h_0_input_name and c_0_input_name else True,
                                       hidden_size=self.params.hidden_size)

    def create_bidirectional_lstm(self, src_op, converter_context):
        return self.create_bidirectional_module(src_op, converter_context, self.weights, self.rec_weights,
                                                self.bias,
                                                self.params, self.create_unidirectional_lstm)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'input_forget':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)
        elif attr_name == "activations":
            lstm_supported_activations = ["Sigmoid", "Tanh", "Tanh"]
            if len(attr_value) == 6:
                # bidirectional case has double the number of activations
                lstm_supported_activations = lstm_supported_activations*2
            log_assert(attr_value == lstm_supported_activations,
                       "Received activations list: {} differs from supported activations list: {}",
                       attr_value, lstm_supported_activations)


OnnxTranslations.register_translation(OnnxLSTMTranslation(),
                                      converter_type('LSTM', 'onnx'),
                                      op_adapter.LstmOp.TRANSLATION_KEY)
