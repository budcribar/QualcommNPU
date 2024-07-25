# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.inference_engine import inference_engine_repository
from lib.utils.nd_constants import ComponentType, Framework, Engine
from lib.inference_engine.converters.nd_SNPE_converter import SNPEConverter
from lib.utils.nd_exceptions import InferenceEngineError
from lib.utils.nd_errors import get_message

@inference_engine_repository.register(cls_type=ComponentType.converter,
                                      framework=Framework.caffe,
                                      engine=Engine.SNPE,
                                      engine_version="1.51.0")
class SNPECaffeConverter(SNPEConverter):
    def __init__(self, context):
        super(SNPECaffeConverter, self).__init__(context)
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

        if len(model_paths) != 2:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_MISMATCH_MODEL_PATH_INPUTS"))

        convert_command_list = [self.executable, self.output_path_flag, output_path] + self.flags
        for path_flag, user_input in zip(self.model_path_flags, model_paths):
            convert_command_list.extend([path_flag, user_input])

        if self.output_tensor_flag:
            for output_tensor in formatted_output_tensors:
                convert_command_list.extend([self.output_tensor_flag, output_tensor])

        convert_command_str = ' '.join(convert_command_list)

        return convert_command_str

    def format_input_tensors(self, input_tensors):
        return input_tensors

    def format_output_tensors(self, output_tensors):
        return output_tensors