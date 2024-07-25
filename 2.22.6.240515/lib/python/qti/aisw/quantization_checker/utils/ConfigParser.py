#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import json
from typing import Dict

def extractConfigParams(config_path) -> Dict:
    if config_path is None:
        return {}
    configInfosJson = {}
    configParams = None
    try:
        with open(config_path) as configFile:
            configInfosJson = json.load(configFile)
        configParams = dict()
        if "RUN_IN_REPAIR_MODE" in configInfosJson:
            configParams["repair_mode"] = configInfosJson["RUN_IN_REPAIR_MODE"]
        if "MODEL_PATH" in configInfosJson:
            configParams["model"] = configInfosJson["MODEL_PATH"]
        if "INPUT_LIST_PATH" in configInfosJson:
            configParams["input_list"] = configInfosJson["INPUT_LIST_PATH"]
        if "QNN_SDK_ROOT" in configInfosJson:
            configParams["sdk_dir"] = configInfosJson["QNN_SDK_ROOT"]
        if "ACTIVATION_WIDTH" in configInfosJson:
            configParams["activation_width"] = configInfosJson["ACTIVATION_WIDTH"]
        if "BIAS_WIDTH" in configInfosJson:
            configParams["bias_width"] = configInfosJson["BIAS_WIDTH"]
        if "WEIGHT_WIDTH" in configInfosJson:
            configParams["weight_width"] = configInfosJson["WEIGHT_WIDTH"]
        if "INPUT_LAYOUT" in configInfosJson:
            configParams["input_layout"] = configInfosJson["INPUT_LAYOUT"]
        if "OUTPUT_DIR_PATH" in configInfosJson:
            configParams["output_dir"] = configInfosJson["OUTPUT_DIR_PATH"]
        if "UNQUANTIZED_WEIGHT_COMPARISON_ALGORITHMS" in configInfosJson:
            configParams["unquantized_weight_comparison_algorithms"] = configInfosJson["UNQUANTIZED_WEIGHT_COMPARISON_ALGORITHMS"]
        if "UNQUANTIZED_BIAS_COMPARISON_ALGORITHMS" in configInfosJson:
            configParams["unquantized_bias_comparison_algorithms"] = configInfosJson["UNQUANTIZED_BIAS_COMPARISON_ALGORITHMS"]
        if "WEIGHT_COMPARISON_ALGORITHMS" in configInfosJson:
            configParams["weight_comparison_algorithms"] = configInfosJson["WEIGHT_COMPARISON_ALGORITHMS"]
        if "BIAS_COMPARISON_ALGORITHMS" in configInfosJson:
            configParams["bias_comparison_algorithms"] = configInfosJson["BIAS_COMPARISON_ALGORITHMS"]
        if "ACT_COMPARISON_ALGORITHMS" in configInfosJson:
            configParams["act_comparison_algorithms"] = configInfosJson["ACT_COMPARISON_ALGORITHMS"]
        if "INPUT_DATA_ANALYSIS_ALGORITHMS" in configInfosJson:
            configParams["input_data_analysis_algorithms"] = configInfosJson["INPUT_DATA_ANALYSIS_ALGORITHMS"]
        if "OUTPUT_CSV"  in configInfosJson:
            configParams["output_csv"] = configInfosJson["OUTPUT_CSV"]
        if "GENERATE_HISTOGRAM" in configInfosJson:
            configParams["generate_histogram"] = bool(configInfosJson["GENERATE_HISTOGRAM"])
        if "PER_CHANNEL_HISTOGRAM" in configInfosJson:
            configParams["per_channel_histogram"] = bool(configInfosJson["PER_CHANNEL_HISTOGRAM"])
        if "QUANTIZATION_OVERRIDES" in configInfosJson:
            configParams["quantization_overrides"] = configInfosJson["QUANTIZATION_OVERRIDES"]
        if "QUANTIZATION_VARIATIONS" in configInfosJson:
            configParams["quantization_variations"] = configInfosJson["QUANTIZATION_VARIATIONS"]
        if "QUANTIZATION_ALGORITHMS" in configInfosJson:
            configParams["quantization_algorithms"] = configInfosJson["QUANTIZATION_ALGORITHMS"]
    except FileNotFoundError:
        print('User defined configuration file not found. Please verify the configuration file path given. Exiting...')
        exit(-1)

    return configParams

def extractEnvironmentConfigParams(config_path) -> Dict:
    if config_path is None:
        return {}
    configInfosJson = {}
    generatorConfigParams = None
    try:
        with open(config_path) as configFile:
            configInfosJson = json.load(configFile)
        generatorConfigParams = dict()
        if "CLANG_PATH" in configInfosJson:
            generatorConfigParams["CLANG_PATH"] = configInfosJson["CLANG_PATH"]
        if "BASH_PATH" in configInfosJson:
            generatorConfigParams["BASH_PATH"] = configInfosJson["BASH_PATH"]
        if "PY3_PATH" in configInfosJson:
            generatorConfigParams["PY3_PATH"] = configInfosJson["PY3_PATH"]
        if "BIN_PATH" in configInfosJson:
            generatorConfigParams["BIN_PATH"] = configInfosJson["BIN_PATH"]
        if "TENSORFLOW_HOME" in configInfosJson:
            generatorConfigParams["TENSORFLOW_HOME"] = configInfosJson["TENSORFLOW_HOME"]
        if "TFLITE_HOME" in configInfosJson:
            generatorConfigParams["TFLITE_HOME"] = configInfosJson["TFLITE_HOME"]
        if "ONNX_HOME" in configInfosJson:
            generatorConfigParams["ONNX_HOME"] = configInfosJson["ONNX_HOME"]
    except FileNotFoundError:
        print('User defined configuration file not found. Please verify the configuration file path given. Exiting...')
        exit(-1)

    return generatorConfigParams
