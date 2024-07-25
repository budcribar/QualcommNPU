# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Dict, Tuple
from abc import abstractmethod

from lib.inference_engine.converters.nd_converter import Converter
from lib.utils.nd_exceptions import InferenceEngineError
from lib.utils.nd_errors import get_message


class SNPEConverter(Converter):
    def __init__(self, context):
        super(SNPEConverter, self).__init__(context)
        self.executable = context.executable
        self.model_path_flags = context.arguments["model_path_flags"].copy()
        self.input_tensor_flag = context.arguments["input_tensor_flag"]
        self.output_tensor_flag = context.arguments["output_tensor_flag"]
        self.output_path_flag = context.arguments["output_path_flag"]
        self.flags = context.arguments["flags"].copy()

    def build_convert_command(self, model_path, input_tensors, output_tensors, output_path):
        # type: (str, Dict[str][str], Tuple[str], str) -> str
        model_paths = model_path.split(",")

        formatted_input_tensors = self.format_input_tensors(input_tensors)
        formatted_output_tensors = self.format_output_tensors(output_tensors)

        if len(model_paths) != len(self.model_path_flags):
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_MISMATCH_MODEL_PATH_INPUTS"))

        convert_command_list = [self.executable, self.output_path_flag, output_path] + self.flags
        for path_flag, user_input in zip(self.model_path_flags, model_paths):
            convert_command_list.extend([path_flag, user_input])
        if self.input_tensor_flag:
            for input_tensor, dimension in formatted_input_tensors.items():
                convert_command_list.extend([self.input_tensor_flag, '"{}"'.format(input_tensor), dimension])
        if self.output_tensor_flag:
            for output_tensor in formatted_output_tensors:
                convert_command_list.extend([self.output_tensor_flag, output_tensor])
        convert_command_str = ' '.join(convert_command_list)

        return convert_command_str

    @abstractmethod
    def format_input_tensors(self, input_tensors):
        pass

    @abstractmethod
    def format_output_tensors(self, output_tensors):
        pass
