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


class TFLiteFramework_1_14_0(BaseFramework):
    __VERSION__ = '1.14.0'

    def __init__(self, logger):
        super(TFLiteFramework_1_14_0, self).__init__(logger)
        self._interpreter = None
        self._tensor_map = None

        # Work around TensorFlow 1.14.0 logging bug by re-making logger
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
        self.logger = setup_logger(logging.INFO)
        self.logger.warning(get_warning_message("WARNING_FRAMEWORK_TFLITE_NO_INTERMEDIATE_TENSORS"))

    def load_model(self, model_path):
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        # type: (numpy.ndarray, List[str], List[str]) -> Dict[str, numpy.ndarray]
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_MISMATCH_INPUTS"))

        input_tensor_indices = [self._tensor_map[name] for name in input_tensor_names]
        output_tensor_indices = [self._tensor_map[name] for name in output_tensor_names]

        for i, index in enumerate(input_tensor_indices):
            self._interpreter.set_tensor(index, input_data[i])

        self._interpreter.invoke()

        results = {}

        for i, index in enumerate(output_tensor_indices):
            results[output_tensor_names[i]] = self._interpreter.get_tensor(index)

        return results

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        # type: (List[str], List[str]) -> List[Tuple[List[str]]]

        # tensor_details = self._interpreter.get_tensor_details()
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        input_tensor_names = [tensor['name'] for tensor in input_details]
        output_tensor_names = [tensor['name'] for tensor in output_details]

        input_tensors_copy = input_tensors.copy()
        input_tensor_names_copy = input_tensor_names.copy()

        input_tensors_copy.sort()
        input_tensor_names_copy.sort()

        if not input_tensors_copy == input_tensor_names_copy:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_INPUT_TENSOR")
                                 (input_tensors, input_tensor_names))

        for output in output_tensors:
            if output not in output_tensor_names:
                raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_OUTPUT_TENSOR")
                                     (output))

        tensor_pairs = [(input_tensor_names, output_tensors)]

        self._tensor_map = {}

        for tensor in input_details:
            self._tensor_map[tensor['name']] = tensor['index']

        for tensor in output_details:
            self._tensor_map[tensor['name']] = tensor['index']

        return tensor_pairs

    def get_mapping_for_qnn_node(self, qnn_output):
        raise FrameworkError(get_message("ERROR_FRAMEWORK_TFLITE_MISMATCH_TENSOR")(qnn_output))
        return None

    def get_version(self):
        # type: () -> str
        return tf.__version__
