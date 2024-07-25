# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.framework_diagnosis.frameworks.nd_base_framework import BaseFramework
from lib.utils.nd_exceptions import FrameworkError
from lib.utils.nd_errors import get_message, get_warning_message
from lib.utils.nd_logger import setup_logger

import tensorflow as tf
import absl.logging
import logging
import tflite.Model
import flatbuffers


class TFLiteFramework_2_2_0(BaseFramework):
    __VERSION__ = '2.2.0'

    def __init__(self, logger):
        super(TFLiteFramework_2_2_0, self).__init__(logger)
        self._interpreter = None
        self._tensor_map = None
        self._model_buffer = None
        self._model = None
        self._tensor_to_op_map = None

        # Work around TensorFlow logging bug by re-making logger
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
        self.logger = setup_logger(logging.INFO)

    def load_model(self, model_path):
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._model_buffer = open(model_path, "rb").read()
        self._model = tflite.Model.GetRootAsModel(bytearray(self._model_buffer), 0)
        self._interpreter.allocate_tensors()

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        # type: (numpy.ndarray, List[str], List[str]) -> Dict[str, numpy.ndarray]
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_MISMATCH_INPUTS"))

        input_tensor_indices = [self._tensor_map[name] for name in input_tensor_names]
        output_tensor_indices = [self._tensor_map[name] for name in output_tensor_names]

        results = {}

        for i, output_index in enumerate(output_tensor_indices):
            self._model_buffer = \
                self.buffer_change_output_tensor_to(output_index)
            self._interpreter = tf.lite.Interpreter(model_content=self._model_buffer)
            self._interpreter.allocate_tensors()
            for j, input_index in enumerate(input_tensor_indices):
                self._interpreter.set_tensor(input_index, input_data[j])
            self._interpreter.invoke()
            results[output_tensor_names[i]] = self._interpreter.get_tensor(output_index)

        return results

    def OutputsOffset(self, subgraph, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
        if o !=0:
            a = subgraph._tab.Vector(o)
            return a + flatbuffers.number_types.UOffsetTFlags.py_type(j*4)
        return 0

    def buffer_change_output_tensor_to(self, new_tensor_i):
        """
        Reads model_buffer as a proper flatbuffer file and gets the offset programatically.
        Set subgraph 0's output(s) to new_tensor_i.
        """

        # Custom added function (OutputsOffset) to return the file offset to this vector :
        try:
            # output_tensor_index_offset = self._model.Subgraphs(0).OutputsOffset(0)
            output_tensor_index_offset = self.OutputsOffset(self._model.Subgraphs(0), 0)
        except:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_CUSTOM_FUNCTION_NOT_ADDED"))
        # Flatbuffer scalars are stored in little-endian.
        new_tensor_i_bytes = bytes([
          new_tensor_i & 0x000000FF, \
          (new_tensor_i & 0x0000FF00) >> 8, \
          (new_tensor_i & 0x00FF0000) >> 16, \
          (new_tensor_i & 0xFF000000) >> 24 \
        ])
        # Replace the 4 bytes corresponding to the first output tensor index
        return self._model_buffer[:output_tensor_index_offset] + \
            new_tensor_i_bytes + \
            self._model_buffer[output_tensor_index_offset + 4:]

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        # type: (List[str], List[str]) -> List[Tuple[List[str]]]

        # tensor_details = self._interpreter.get_tensor_details()
        output_found = [False for i in range(len(output_tensors))]

        input_details = self._interpreter.get_input_details()
        model_input_names = [tensor['name'] for tensor in input_details]

        # Build up tensor_map (dictionary of tensor names to indices):
        self._tensor_map = {}
        self._tensor_to_op_map = {}
        tensor_list = self._interpreter.get_tensor_details()
        for tensor in tensor_list:
            self._tensor_map[tensor['name']] = tensor['index']
            self._tensor_to_op_map[tensor['index']] = -1

        # All input_tensors must be part of the model's input details!:
        for name in input_tensors:
            if (name not in model_input_names):
                raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_INPUT_TENSOR")
                                     (input_tensors, model_input_names))
        # All output tensors must be a valid tensor:
        for name in output_tensors:
            if (name not in self._tensor_map):
                raise FrameworkError(get_message
                        ("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_OUTPUT_TENSOR")(name))

        # Let us see what inputs are not provided in input_tensors, but are in the model's
        # input details:
        # We may not need these inputs depending on the model structure and output tensors given:
        neglected_inputs = []
        for name in model_input_names:
            if (name not in input_tensors):
                neglected_inputs.append(name)

        # Build up tensor_to_op_map (dictionary of tensor names to lists of ops):
        # In tensor_to_op_map, the keys are the tensor index
        # Each value in the dict is the op that outputs the corresponding tensor key
        for op in range(self._model.Subgraphs(0).OperatorsLength()):
            operator = self._model.Subgraphs(0).Operators(op)
            for k in range(operator.OutputsLength()):
                self._tensor_to_op_map[operator.Outputs(k)] = op

        # After this, if the tensor_to_op_map value is still -1, then the tensor is an input tensor

        op_list = self.operator_list(input_tensors, output_tensors)
        tensor_pairs = []
        for i in op_list:
            inputs = []
            outputs = []
            operator = self._model.Subgraphs(0).Operators(i)
            for j in range(operator.InputsLength()):
                name = tensor_list[operator.Inputs(j)]['name']
                inputs.append(name)
                # If a neglected input is encountered, the user didn't give enough inputs:
                if (name in neglected_inputs):
                    raise FrameworkError(get_message
                            ("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_INPUT_TENSOR")
                            (input_tensors, model_input_names))
            for k in range(operator.OutputsLength()):
                name = tensor_list[operator.Outputs(k)]['name']
                outputs.append(name)
                if name in output_tensors:
                    output_found[output_tensors.index(name)] = True
            tensor_pairs.append((inputs, outputs))

        if (not all(output_found)):  # If we didn't find some outputs, something went wrong
            for i in range(len(output_found)):
                if (not output_found[i]):
                    raise FrameworkError(get_message
                                            ("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_OUTPUT_TENSOR")
                                            (output_tensors[i]))

        return tensor_pairs

    def operator_list(self, input_tensors, output_tensors):
        # type: (List[str], List[str]) -> List[int]
        """
        Calls DFS on model starting from each operation that outputs an element from
        output_tensors.
        Returns a list of operator indices that were encountered in this process

        input_tensors: Inputted list of input tensor names
        output_tensors: Inputted list of output tensor names

        op_list: Outputted list of operator indices that were encountered in DFS
        """

        op_list = []
        output_indices = [self._tensor_map[name] for name in output_tensors]
        # Start from output node:
        for i, output_idx in enumerate(output_indices):
            output_op = self._tensor_to_op_map[output_idx]
            if (output_op == -1):
                # This tensor is not outputted from any ops, meaning it is an input, constant, etc.
                # As such, it should not be labelled as an output tensor!:
                raise FrameworkError(get_message
                                            ("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_OUTPUT_TENSOR")
                                            (output_tensors[i]))
            if (output_op not in op_list):
                self.dfs_operator_list_stack(output_op, op_list)

        return op_list

    def dfs_operator_list_stack(self, op_idx, op_list):
        # type: (int, List[int]) -> List[int]
        """
        Performs DFS on the model starting from operator index op_idx, using
        stack based implementation.

        op_idx: Inputted index where DFS starts from
        op_list: Inputted list of operator indices visited so far. Is appended to in this function,
        and since Lists are mutable, changes in it are seen by the calling function.
        """

        stack = []
        stack.append(op_idx)
        while (len(stack) != 0):
            curr_idx = stack.pop()
            if (curr_idx not in op_list):
                op_list.append(curr_idx)
                # Find adjacent Nodes:
                # Adjacent nodes (ops) are ones where the outputs of the adjacent op
                # include the input of the current op:
                operator = self._model.Subgraphs(0).Operators(curr_idx)
                for i in range(operator.InputsLength()):
                    # Use tensor_to_op_map to see which ops have the current input
                    # as an output:
                    adj_op = self._tensor_to_op_map[operator.Inputs(i)]
                    if (adj_op != -1):
                        stack.append(adj_op)

    def get_mapping_for_qnn_node(self, qnn_output):
        raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_MISMATCH_TENSOR")(qnn_output))
        return None

    def get_version(self):
        # type: () -> str
        return tf.__version__
