# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from collections import OrderedDict
from typing import List, Text, Tuple, Dict, Callable, Optional
from qti.aisw.converters.common.utils.framework_utils import TensorInfo
from qti.aisw.converters.common.qnn_runtime_base import IModelRuntime
from qti.aisw.converters.common.utils.framework_utils import generate_test_data
from qti.aisw.converters.common.utils.converter_utils import log_warning


ERROR_FUNC_TYPE = Callable[[np.ndarray, np.ndarray], float]


class ValidationResult:
    def __init__(self, name1: Text, name2: Text, error_info: Dict[Text, float]) -> None:
        """
        Validation Results Structure.

        :param Text name1: Name of first runtime instance.
        :param Text name2: Name of runtime instance to compare with,
        :param Dict[Text,float] error_info: Mapping of error type and its value.
        """
        self.name1 = name1
        self.name2 = name2
        self.error_info = error_info

    def __str__(self) -> str:
        """
        Convert validation Results to string format.

        :return str: Validation in result in string format
        """
        error_str = f"Validation result : {self.name1} v/s {self.name2} :"
        trim_idx = len(error_str)
        for error_type, value in self.error_info.items():
            error_str = error_str + " {} : {:.2f}% ,".format(error_type, value)
            trim_idx = -1
        return error_str[:trim_idx]  # Removing last ','.

    def __repr__(self) -> str:
        """
        ValidationResult representation in string format.

        :return str: Validation in result in string format
        """
        return str(self)

    def format(self,*args) -> str:
        """
        Adding a placeholder API to work with
        our existing logger APIs.
        """
        return str(self).format(*args)


class Validator:
    def __init__(self) -> None:
        """
        Initializing Validator Object.
        """
        ## Runtime objects name to instance mapping.
        self.runtime_objects_info = OrderedDict()

    def add_runtime_sessions(self, name: Text, runtime_obj: IModelRuntime):
        """
        Adds the model runtime session objects to be compare for validation.

        :param Text name: name of the object to run validation.
        :param IModelRuntime runtime_obj: Runtime object to be used for validation.
        """
        assert isinstance(
            runtime_obj, IModelRuntime
        ), f"Unsupported runtime object of type {type(runtime_obj)}"
        self.runtime_objects_info[name] = runtime_obj

    def __relative_error(self, output_a: np.ndarray, output_b: np.ndarray) -> Tuple:
        """
        Calculate the relative error between two tensors.

        :param output_a (np.ndarray): Native output tensors
        :param output_b (np.ndarray): Prepared output tensors.
        :return Tuple: Average and 90 percentile error between the two tensors.
        """
        assert (
            output_a.shape == output_b.shape
        ), "Shapes of output_a and output_b shell be same."
        output_a_abs = (np.abs(output_a)).astype(float)
        output_a_abs[output_a_abs == 0] = output_a_abs[output_a_abs == 0] + 1e-5
        abs_diff = np.abs(output_a - output_b).astype(float)
        percent = np.sort(((100 * abs_diff) / (output_a_abs)).flatten())
        percentile_error = percent[int(len(percent) * 0.9)]
        avg_error = np.mean(percent)
        return avg_error, percentile_error

    def __calculate_error(
        self,
        output_a: Dict[Text, np.ndarray],
        output_b: Dict[Text, np.ndarray],
        session_name1: Text,
        session_name2: Text,
        error_funcs: Optional[Dict[Text, ERROR_FUNC_TYPE]],
    ) -> ValidationResult:
        """
        Calculate error between the runtime object's output.

        :param Dict[Text,np.ndarray] output_a: First runtime object's output under comparison.
        :param Dict[Text,np.ndarray] output_b: Object to compare with.
        :param Text session_name1: name of the first session.
        :param Text session_name2: name of the other session.
        :param Dict[Text, ERROR_FUNC_TYPE] error_funcs: Name to function mappings,
          Which will used for calculating the error between two output tensors.
        :return ValidationResult :Returns the error between two tensors.
        """

        error_info = OrderedDict()
        AVG_ERROR = "Avg Error"
        PERCENTILE_ERROR = "90% Error"

        error_info[AVG_ERROR] = float("-inf")
        error_info[PERCENTILE_ERROR] = float("-inf")
        default_response = ValidationResult(session_name1, session_name2, error_info)

        if (not output_a) and (not output_b):
            log_warning(f"Inference for {session_name1} and {session_name2} failed.")
            return default_response

        if not output_a:
            log_warning(f"Inference for {session_name1} failed.")
            return default_response

        if not output_b:
            log_warning(f"Inference for {session_name2} failed.")
            return default_response

        tensors_to_exclude = ", ".join(set(output_a.keys()) ^ set(output_b.keys()))

        if tensors_to_exclude:
            log_warning(
                f"tensors : {tensors_to_exclude} are not common between the {session_name1}"
                f"and {session_name2}. Skipping comparison for these tensors."
            )

        error_info = OrderedDict()
        if not error_funcs:
            error_info[AVG_ERROR] = []
            error_info[PERCENTILE_ERROR] = []

        for key_a, values_a in output_a.items():
            if key_a not in output_b:
                continue
            values_b = output_b[key_a]

            if not error_funcs:
                avg_error, percentile_error = self.__relative_error(values_a, values_b)
                error_info[AVG_ERROR].append(avg_error)
                error_info[PERCENTILE_ERROR].append(percentile_error)
                continue

            for func_name in error_funcs:
                if func_name not in error_info:
                    error_info[func_name] = []
                func = error_funcs[func_name]
                error_info[func_name].append(func(values_a, values_b))

        error_info.update(
            {
                tensor_name: np.mean(error_info[tensor_name])
                for tensor_name in error_info.keys()
            }
        )
        return ValidationResult(session_name1, session_name2, error_info)

    def validate(
        self, error_funcs: Optional[Dict[Text, ERROR_FUNC_TYPE]] = {}
    ) -> List[ValidationResult]:
        """
        Compare the error between the added runtime objects.

        :param Optional[Dict[Text, Callable]] error_funcs: Name to function mappings,
          Which will used for calculating the error between two output tensors.
        :return List[ValidationResult] : Returns the list of ValidationResult
        """

        assert (
            len(self.runtime_objects_info) > 1
        ), "Please add at least two runtime session objects before validate."

        # We are generating test data according to the first runtime instance,
        # In order input data to be valid for all the runtime instance we are
        # comparing first input info with all the subsequent runtime instance inputs.

        input_names = list(self.runtime_objects_info.keys())
        input_info = self.runtime_objects_info[input_names[0]].get_input_info()
        test_data = generate_test_data(input_info)

        # TODO : Here we are only comparing between optimized and original
        # model, Once we add support for the graph runtime we need to
        # revisit the logic how the comparisons performed.

        results = []
        for i in range(len(input_names) - 1):
            inp_name1 = input_names[i]
            inp_name2 = input_names[i + 1]
            if input_info != self.runtime_objects_info[inp_name1].get_input_info():
                log_warning(
                    f"Input mismatch between {input_names[0]} and {inp_name1}, skipping validation."
                )
                continue

            if input_info != self.runtime_objects_info[inp_name2].get_input_info():
                log_warning(
                    f"Input mismatch between {input_names[0]} and {inp_name2}, skipping validation."
                )
                continue

            output_a = self.runtime_objects_info[inp_name1].execute_inference(test_data)
            output_b = self.runtime_objects_info[inp_name2].execute_inference(test_data)

            result = self.__calculate_error(
                output_a, output_b, inp_name1, inp_name2, error_funcs
            )
            results.append(result)

        return results
