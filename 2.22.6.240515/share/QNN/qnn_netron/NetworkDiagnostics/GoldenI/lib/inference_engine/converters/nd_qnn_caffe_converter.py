# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.inference_engine import inference_engine_repository
from lib.utils.nd_constants import ComponentType, Framework, Engine
from lib.inference_engine.converters.nd_qnn_converter import QNNConverter


@inference_engine_repository.register(cls_type=ComponentType.converter,
                                      framework=Framework.caffe,
                                      engine=Engine.QNN,
                                      engine_version="1.1.0.22262")
class QNNCaffeConverter(QNNConverter):
    def __init__(self, context):
        super(QNNCaffeConverter, self).__init__(context)
        # Instantiate lib generator fields from context
        self.executable = context.executable
        self.model_path_flags = context.arguments["model_path_flag"].copy()
        self.output_tensor_flag = context.arguments["output_tensor_flag"]
        self.output_path_flag = context.arguments["output_path_flag"]
        self.flags = context.arguments["flags"].copy()

    def build_convert_command(self, model_path, input_tensors, output_tensors, output_path, input_list_txt,
                              quantization_overrides, param_quantizer ,act_quantizer, weight_bw, bias_bw, act_bw,
                              algorithms, ignore_encodings, per_channel_quantization):
        model_paths = model_path.split(",")

        convert_command = [self.executable, self.output_path_flag, output_path] + self.flags

        for tensor in output_tensors:
            convert_command += [self.output_tensor_flag, "\"" + tensor + "\""]

        for path_flag, user_input in zip(self.model_path_flags, model_paths):
            convert_command += [path_flag, user_input]

        if input_list_txt:
            convert_command += self.quantization_command(input_list_txt, quantization_overrides, param_quantizer, act_quantizer,
                                weight_bw, bias_bw, act_bw, algorithms, ignore_encodings, per_channel_quantization)

        convert_command_str = ' '.join(convert_command)

        return convert_command_str