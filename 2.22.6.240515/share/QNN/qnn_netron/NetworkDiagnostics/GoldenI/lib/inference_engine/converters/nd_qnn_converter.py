# =============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import abc

from lib.inference_engine.converters.nd_converter import Converter
from lib.utils.nd_exceptions import InferenceEngineError
from lib.utils.nd_errors import get_message

class QNNConverter(Converter):
    def __init__(self, context):
        super(QNNConverter, self).__init__(context)
        # Instantiate lib generator fields from context
        self.executable = context.executable
        self.model_path_flag = context.arguments["model_path_flag"]
        self.output_path_flag = context.arguments["output_path_flag"]

        quantization_flag = context.arguments["quantization_flag"]
        self.input_list_flag = quantization_flag["input_list_flag"]
        self.quantization_overrides_flag = quantization_flag["quantization_overrides_flag"]
        self.param_quantizer_flag = quantization_flag["param_quantizer_flag"]
        self.act_quantizer_flag = quantization_flag["act_quantizer_flag"]
        self.weight_bw_flag = quantization_flag["weight_bw_flag"]
        self.bias_bw_flag = quantization_flag["bias_bw_flag"]
        self.act_bw_flag = quantization_flag["act_bw_flag"]
        self.algorithms_flag = quantization_flag["algorithms_flag"]
        self.ignore_encodings_flag = quantization_flag["ignore_encodings_flag"]
        self.use_per_channel_quantization_flag = quantization_flag["use_per_channel_quantization_flag"]

    def quantization_command(self, input_list_txt, quantization_overrides, param_quantizer ,act_quantizer,
                            weight_bw, bias_bw, act_bw, algorithms, ignore_encodings, per_channel_quantization):
        convert_command = []
        if quantization_overrides and ignore_encodings:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_QNN_QUANTIZATION_FLAG_INPUTS"))

        convert_command += [self.input_list_flag, input_list_txt]
        if param_quantizer:
            convert_command += [self.param_quantizer_flag, param_quantizer]
        if act_quantizer:
            convert_command += [self.act_quantizer_flag, act_quantizer]

        if weight_bw:
            convert_command += [self.weight_bw_flag, str(weight_bw)]
        if bias_bw:
            convert_command += [self.bias_bw_flag, str(bias_bw)]
        if act_bw:
            convert_command += [self.act_bw_flag, str(act_bw)]

        if algorithms:
            convert_command += [self.algorithms_flag, algorithms]
        if quantization_overrides:
            convert_command += [self.quantization_overrides_flag, quantization_overrides]
        if ignore_encodings:
            convert_command += [self.ignore_encodings_flag]
        if per_channel_quantization:
            convert_command += [self.use_per_channel_quantization_flag]

        return convert_command

    @abc.abstractmethod
    def build_convert_command(self, model_path, input_tensors, output_tensors, output_path, input_list_txt,
                              quantization_overrides, param_quantizer, act_quantizer, weight_bw, bias_bw, act_bw,
                              algorithms, ignore_encodings, per_channel_quantization):
        # type: (str, Dict[str][str], Tuple[str], str) -> str
        """
        Build command (using converter tools) to convert model to QNN Graph

        model_path: Path to model file (the .pb or .caffe)
        input_tensors: Names and dimensions of input tensors
        output_tensors: Names of output tensors for network
        output_path: Output directory for QNN .cpp and .bin

        return value: String command using converter tool (tensorflow-to-qnn or caffe-to-qnn)
        that is used to generate QNN .cpp and .bin files
        """
        pass
