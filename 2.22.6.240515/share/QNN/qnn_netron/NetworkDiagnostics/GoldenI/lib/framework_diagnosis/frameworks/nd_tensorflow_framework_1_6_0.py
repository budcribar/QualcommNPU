# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.framework_diagnosis.frameworks.nd_base_framework import BaseFramework
from lib.utils.nd_exceptions import FrameworkError
from lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message

import tensorflow as tf
import tensorflow.contrib.graph_editor
import tensorflow.contrib.image
import logging
from collections import OrderedDict


class TensorFlowFramework_1_6_0(BaseFramework):
    __VERSION__ = '1.6.0'

    def __init__(self, logger):
        super(TensorFlowFramework_1_6_0, self).__init__(logger)
        self._graph = None

    @property
    def graph(self):
        return self._graph

    def load_model(self, model_path):
        # Import graph and save to instance variable
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            self._graph = tf.get_default_graph()

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_TENSORFLOW_MISMATCH_INPUTS"))

        with tf.Session(graph=self._graph) as sess:
            for data, tensor_name in zip(input_data, input_tensor_names):
                tensor = sess.graph.get_tensor_by_name(tensor_name)

                # Check dimension of input is compatible with input tensor
                if tensor.get_shape().dims is not None:
                    for tensor_dimension, data_dimension in zip(tensor.get_shape().as_list(), data.shape):
                        if tensor_dimension != data_dimension and tensor_dimension is not None:
                            raise FrameworkError(get_message("ERROR_FRAMEWORK_TENSORFLOW_MISMATCH_INPUT_DIMENSIONS"))
                else:
                    logging.warning(get_warning_message("WARNING_FRAMEWORK_TENSORFLOW_DIMENSION_UNSPECIFIED"))

            input_dict = {name: data for name, data in zip(input_tensor_names, input_data)
                          if self._graph.is_feedable(self._graph.get_tensor_by_name(name))}

            output_tensors = [sess.graph.get_tensor_by_name(tensor_name) for tensor_name in output_tensor_names]

            result = {}
            for output_tensor in output_tensors:
                try:
                    result[output_tensor.name] = output_tensor.eval(input_dict, sess)
                except tf.errors.OutOfRangeError as exc:
                    logging.debug(get_debugging_message("DEBUG_FRAMEWORK_TENSORFLOW_TENSOR_NOT_EVALUATED")(str(exc)))
                except tf.errors.OpError as exc:
                    logging.warning(get_warning_message("WARNING_FRAMEWORK_TENSORFLOW_TENSOR_NOT_EVALUATED")(str(exc)))

            return result

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        input_ops = [self._graph.get_tensor_by_name(tensor_name).op for tensor_name in input_tensors]
        output_ops = [self._graph.get_tensor_by_name(tensor_name).op for tensor_name in output_tensors]

        ops = tf.contrib.graph_editor.get_walks_intersection_ops(input_ops, output_ops, forward_inclusive=False,
                                                                 backward_inclusive=True, control_inputs=False)

        tensor_pairs = []

        for op in ops:
            inputs = []
            outputs = []
            for input_tensor in op.inputs:
                if input_tensor.op in ops or input_tensor.op in input_ops:
                    inputs.append(input_tensor.name)
            for output_tensor in op.outputs:
                if output_tensor.op in ops:
                    outputs.append(output_tensor.name)

            tensor_pairs.append((inputs, outputs))

        tensor_pairs = self.filter_blocks(tensor_pairs, 'map')

        return tensor_pairs

    def filter_blocks(self, tensor_pairs, block):
        ops = self._graph.get_operations()
        ops_in_block = [op.name for op in ops if block in op.name]

        scope_to_remove = []
        for op in ops_in_block:
            scope = op.split(block, 1)[0] + block
            if scope not in scope_to_remove:
                scope_to_remove.append(scope)

        for scope in scope_to_remove:
            tensor_pairs = TensorFlowFramework_1_6_0.filter_block(tensor_pairs, scope)

        return tensor_pairs

    @staticmethod
    def filter_block(tensor_pairs, remove_scope):
        inputs_to_map_block = []
        outputs_to_map_block = []
        for input_tensors, output_tensors in tensor_pairs:
            block_inputs = [tensor_name for tensor_name in input_tensors
                            if any(remove_scope in tensor for tensor in output_tensors) and
                            remove_scope not in tensor_name]
            block_outputs = [tensor_name for tensor_name in input_tensors
                             if any(remove_scope not in tensor for tensor in output_tensors) and
                             remove_scope in tensor_name]

            inputs_to_map_block = inputs_to_map_block + list(set(block_inputs) - set(inputs_to_map_block))
            outputs_to_map_block = outputs_to_map_block + list(set(block_outputs) - set(outputs_to_map_block))

        tensor_pairs = list(filter((lambda pair: not any(remove_scope in name for name in pair[0]) or
                                                 not any(remove_scope in name for name in pair[1])), tensor_pairs))

        tensor_pairs.append((inputs_to_map_block, outputs_to_map_block))
        return tensor_pairs

    def get_dimensions(self, tensor_name):
        with tf.Session() as sess:
            tensor = sess.graph.get_tensor_by_name(tensor_name)
            if tensor.get_shape().dims is None:
                return None
            else:
                return tensor.get_shape().as_list()

    def get_graph_structure(self):
        op_dict = OrderedDict()
        for op in self._graph.get_operations():
            input_dict = {input_tensor.name: self.get_dimensions(input_tensor.name) for input_tensor in op.inputs}
            output_dict = {output_tensor.name: self.get_dimensions(output_tensor.name) for output_tensor in op.outputs}

            op_dict[op.name] = (input_dict, output_dict)

        return op_dict

    def _find_act_quant_node(self, node_name):
        for tensor in tf.get_default_graph().as_graph_def().node:
            if len(tensor.input) > 0 and tensor.input[0] == node_name and tensor.op == "FakeQuantWithMinMaxVars":
                return tensor.name
        return node_name

    def get_mapping_for_qnn_node(self, qnn_output):
        for tensor in tf.get_default_graph().as_graph_def().node:
            tensor_name = tensor.name + "_0"
            tensor_replace = tensor_name.replace(".", "_")
            tensor_replace = tensor_replace.replace("/", "_")
            if tensor_replace[0].isdigit():
                tensor_replace = '_' + tensor_replace
            if tensor_replace == qnn_output:
                return self._find_act_quant_node(tensor.name) + ":0"

        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_TENSORFLOW_MISMATCH_TENSOR")(qnn_output))
        return " "

    def get_version(self):
        return tf.__version__
