# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

error_codes_to_messages = {
    ###########################################################################
    # CONFIG ERROR CODES
    ###########################################################################

    "ERROR_CONFIG_FRAMEWORK_NOT_FOUND": "Framework '{}' not found.",
    "ERROR_CONFIG_FRAMEWORK_VERSION_NOT_FOUND": "Version '{}' for framework '{}' not found.",

    "ERROR_CONFIG_ENGINE_NOT_FOUND": "Engine '{}' not found.",
    "ERROR_CONFIG_ENGINE_VERSION_NOT_FOUND": "Version '{}' for engine '{}' not found.",

    "ERROR_CONFIG_CONFIG_NOT_FOUND": "No wrapper configurations found in {}",

    ###########################################################################
    # ENVIRONMENT ERROR CODES
    ###########################################################################
    "ERROR_FRAMEWORK_ENVIRONMENT_ACTIVATION_FAILED": "Environment activation failed for framework '{}' version '{}'",
    "ERROR_ENGINE_ENVIRONMENT_ACTIVATION_FAILED": "Environment activation failed for engine '{}' version '{}'",
    "ERROR_VERIFIER_ENVIRONMENT_ACTIVATION_FAILED": "Environment activation failed for verifier",

    ###########################################################################
    # FRAMEWORK ERROR CODES
    ###########################################################################

    "ERROR_FRAMEWORK_VERSION_MISMATCH": "Framework mismatch: wanted '{}', but found '{}'.",
    "ERROR_FRAMEWORK_FAILED_CONFIGURATION": "Failed to load configuration for '{}'.",

    "ERROR_FRAMEWORK_RUNNER_INPUT_TENSOR_LENGHT_ERROR": "The args.input_tensors should have 3 or 4 part for FrameworkRunner.",

    "ERROR_FRAMEWORK_TENSORFLOW_MISMATCH_INPUTS": "Mismatched number of input data points to input tensors.",
    "ERROR_FRAMEWORK_TENSORFLOW_MISMATCH_INPUT_DIMENSIONS": "Input data does not match input tensor dimensions.",
    "ERROR_FRAMEWORK_TENSORFLOW_MISMATCH_TENSOR": "Mismatched tf tensor to qnn output tensor: {}.",

    "ERROR_FRAMEWORK_CAFFE_MODEL_INPUTS":"The caffe model input need prototxt and caffemodel two parts",
    "ERROR_FRAMEWORK_CAFFE_MISMATCH_INPUTS": "Mismatched number of input data points to input tensors.",
    "ERROR_FRAMEWORK_CAFFE_MISMATCH_INPUT_DIMENSIONS": "Input data does not match input tensor dimensions.",
    "ERROR_FRAMEWORK_CAFFE_MISMATCH_OUTPUTS": "Invalide  output tensors name.",

    "ERROR_FRAMEWORK_CAFFE2_MISMATCH_INPUTS": "Mismatched number of input data points to input tensors.",
    "ERROR_FRAMEWORK_CAFFE2_DUPLICATE_INPUTS_OR_OUTPUTS": "Input or output tensor names must be unique.",
    "ERROR_FRAMEWORK_CAFFE2_MISMATCH_INPUT_DIMENSIONS": "Input data does not match input tensor dimensions.",
    "ERROR_FRAMEWORK_CAFFE2_NON-EXISTENT_BLOBS": "Unable to find all intermediate tensors.",
    "ERROR_FRAMEWORK_CAFFE2_PREDICT_NET_INITIALIZATION_FAILED": "Failed to initialize predict_net.",
    "ERROR_FRAMEWORK_CAFFE2_FAILED_TO_GET_DIMENSIONS": "Unable to find shape for blob '{}'.",
    "ERROR_FRAMEWORK_CAFFE2_UNINITIALIZED_PREDICT_NET": "Predict_net not initialized, unable to fetch blob shape.",

    "ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_INPUT_TENSOR": "Provided input tensors {} do not match model's input tensor details {}.",
    "ERROR_FRAMEWORK_TFLITE_UNSUPPORTED_OUTPUT_TENSOR": "Provided output tensor {}, is not a part of model's output tensor details.",
    "ERROR_FRAMEWORK_TFLITE_MISMATCH_INPUTS": "Mismatched number of input data points to input tensors.",
    "ERROR_FRAMEWORK_TFLITE_CUSTOM_FUNCTION_NOT_ADDED": "Custom function OutputsOffset not added to Subgraphs.py in tflite library.",

    "ERROR_FRAMEWORK_NO_VALID_CONFIGURATIONS": "No valid framework configurations found",

    ###########################################################################
    # DEPENDENCY ERROR CODES
    ###########################################################################

    "ERROR_VIRTUALENV_INSTALLATION_FAILURE": "Could not install virtualenv package to Python user directory. Ensure Python --user directory is writable.",
    "ERROR_VIRTUALENVAPI_IMPORT_FAILURE": "Error importing virtualenvapi. Ensure virtualenvapi is present in {}",
    "ERROR_LIBCPP_SHARED_SO_FILE_NOT_FOUND": "Error finding libc++_shared.so file. User did not provide the file. File should be present in the same folder as libqnn_model.so",

    "ERROR_DEPENDENCY_INVALID_NAME": "Invalid dependency name '{}' could not be installed",

    ###########################################################################
    # DEVICE ERROR CODES
    ###########################################################################

    "ERROR_DEVICE_MANAGER_X86_NON_EXISTENT_PATH": "The path '{}' does not exist.",
    "ERROR_ADB_MISSING_DEVICES": "No devices connected.",
    "ERROR_ADB_NOT_INSTALLED": "No ADB installation detected. Either add ADB to PATH or enter the ADB location in the .ini config file.",
    "ERROR_ADB_PATH_INVALID": "ADB location specified in the .ini config file is invalid.",
    "ERROR_ADB_DEVICE_ID_REQUIRED": "More than one device connected, need to specify device id in the .ini config file.",

    ###########################################################################
    # INFERENCE ENGINE ERROR CODES
    ###########################################################################

    "ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND": "Engine '{}' not found.",
    "ERROR_INFERENCE_ENGINE_MISSING_ENGINE_VERSION": "Engine version missing.",
    "ERROR_INFERENCE_ENGINE_ENGINE_VERSION_NOT_SUPPORTED": "Engine version '{}' cannot be supported.",
    "ERROR_INFERENCE_ENGINE_FRAMEWORK_NOT_FOUND": "Framework '{}' not found.",
    "ERROR_INFERENCE_ENGINE_RUNTIME_NOT_FOUND": "Runtime '{}' not found.",
    "ERROR_INFERENCE_ENGINE_MISMATCH_MODEL_PATH_INPUTS": "Mismatched number of model paths to model path flags.",
    "ERROR_INFERENCE_ENGINE_FINE_GRAINED_NOT_SUPPORTED": "Fine grained mode is currently not supported.",
    "ERROR_INFERENCE_ENGINE_CONVERSION_FAILED": "Failed to convert model.",
    "ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED": "Failed to do initial conversion of model.",
    "ERROR_INFERENCE_ENGINE_INFERENCE_FAILED": "Failed to execute inference. Reason {}",
    "ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED": "Failed to push input data to target device with following error '{}'.",
    "ERROR_INFERENCE_ENGINE_PATH_INVALID": "Engine path provided '{}' is unrecognized. Only '.zip' and extracted directories are acceptable.",
    "ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE": "Failed to push binaries to device.",
    "ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE": "Failed to push base dlc to device.",
    "ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED": "Failed to pull inference results from target device to host. Reason {}",
    "ERROR_INFERENCE_ENGINE_REMOVE_RESULTS_FAILED": "Failed to remove inference results from the target device.",
    "ERROR_INFERENCE_ENGINE_MKDIR_FAILED": "Unable to make directory on target with following error '{}.'",
    "ERROR_INFERENCE_ENGINE_UNSUPPORTED_FRAMEWORK_IN_CONVERTER": "Framework '{}' not supported in converter config.",
    "ERROR_INFERENCE_ENGINE_INVALID_EXECUTABLE": "Specified framework version {} has no executable. Valid executables are: {}",
    "ERROR_INFERENCE_ENGINE_CHMOD_FAILED": "Failed to change permissions of executable on device.",
    "ERROR_INFERENCE_ENGINE_DISABLE_ACCELERATION_FAILED": "Failed to disable '{}' using command: {}",
    "ERROR_INFERENCE_ENGINE_ENABLE_ACCELERATION_FAILED": "Failed to enable '{}' using command: {}",
    "ERROR_INFERENCE_ENGINE_PROCESS_SEARCH_FAILED": "Failed to search device processes.",
    "ERROR_INFERENCE_ENGINE_PROCESS_NOT_FOUND": "Process '{}' not found.",
    "ERROR_INFERENCE_ENGINE_KILL_PROCESS_FAILED": "Failed to kill process '{}'",
    "ERROR_INFERENCE_ENGINE_RUNTIME_INVALID": "Runtime '{}' is invalid for inference engine '{}'.",
    "ERROR_INFERENCE_ENGINE_CUSTOM_FUNCTION_NOT_ADDED": "Custom function OutputsOffset not added to Subgraphs.py in tflite library.",
    "ERROR_INFERENCE_ENGINE_UNSUPPORTED_INPUT_TENSOR": "Provided input tensors {} do not match model's input tensor details {}.",
    "ERROR_INFERENCE_ENGINE_UNSUPPORTED_OUTPUT_TENSOR": "Provided output tensor {}, is not a part of model's output tensor details.",
    "ERROR_INFERENCE_ENGINE_MODEL_FILE_DOES_NOT_EXIST": "The intermediate tensor {} does not have a corresponding .tflite file.",
    "ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED": "The arch {} model binaries failed to be created. Reason: {}",
    "ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED": "The context binary failed to be created. Reason {}",
    "ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION": "target or host device {} not configured in device config json.",
    "ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED": "Failed to convert the float DLC models into quantized DLC models.",

    "ERROR_INFERENCE_ENGINE_QNN_QUANTIZATION_FLAG_INPUTS": "Cannot use --ignore_encodings with --quantization_overrides",

    ###########################################################################
    # PROFILER ERROR CODES
    ###########################################################################
    "ERROR_PROFILER_DATA_EXTRACTION_FAILED": "The profiling data faield to be extracted",


    ###########################################################################
    # VERIFIER ERROR CODES
    ###########################################################################

    "ERROR_VERIFIER_INVALID_VERIFIER_NAME": "No verifier found for '{}'.",
    "ERROR_VERIFIER_NON_EXISTENT_INFERENCE_DIR": "Inference output directory '{}' could not be found"
                                                 " or does not exist.",
    "ERROR_VERIFIER_NON_EXISTENT_FRAMEWORK_DIR": "Framework output directory '{}' could not be found or does not exist.",
    "ERROR_VERIFIER_USE_MULTI_VERIFY_AND_CONFIG": "Verification does not support use multi_verifier and specific verifier together.",

    "ERROR_VERIFIER_RTOL_ATOL_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 1.",
    "ERROR_VERIFIER_TOPK_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 1.",
    "ERROR_VERIFIER_MEAN_IOU_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 2",
    "ERROR_VERIFIER_L1ERROR_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 1.",
    "ERROR_VERIFIER_MSE_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 1.",
    "ERROR_VERIFIER_SQNR_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 1.",
    "ERROR_VERIFIER_COSINE_SIMILARITY_INCORRECT_INPUT_SIZE": "Golden_output or inference_output is not length 1.",

    "ERROR_VERIFIER_RTOL_ATOL_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                                 "Golden tensor size: {}, Inference tensor size: {}",
    "ERROR_VERIFIER_TOPK_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                            "Golden tensor size: {}, Inference tensor size: {}",
    "ERROR_VERIFIER_MEAN_IOU_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                                "Golden boxes size: {}, Golden classifications size: {}, "
                                                "Inference boxes size: {},  Inference classifications size: {}.",
    "ERROR_VERIFIER_L1ERROR_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                                 "Golden tensor size: {}, Inference tensor size: {}",
    "ERROR_VERIFIER_MSE_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                                 "Golden tensor size: {}, Inference tensor size: {}",
    "ERROR_VERIFIER_SQNR_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                                 "Golden tensor size: {}, Inference tensor size: {}",
    "ERROR_VERIFIER_COSINE_SIMILARITY_DIFFERENT_SIZE": "Size of golden and inference data are not compatible. "
                                                 "Golden tensor size: {}, Inference tensor size: {}",
    "ERROR_VERIFIER_SCALED_DIFF_MISSING_OUTPUT_ENCODING": "Missing output encoding for ScaledDiffVerifier",
    "ERROR_VERIFIER_CANNOT_USE_SCALEDDIFF_VERIFIR": "Can't use ScaledDiff verifier given encodings are all zeros",

    ###########################################################################
    # DEEP ANALYZER ERROR CODES
    ###########################################################################

    "ERROR_DEEP_ANALYZER_INVALID_ANALYZER_NAME":"No Analyzer found for '{}'.",
    "ERROR_DEEP_ANALYZER_INVALID_VERIFIER_NAME":"No Verifier found for '{}'.",
    "ERROR_DEEP_ANALYZER_NON_EXISTENT_PATH":"Path directory '{}' could not be found"
                                                " or does not exist."
}

warning_codes_to_messages = {
    ###########################################################################
    # FRAMEWORK WARNING CODES
    ###########################################################################
    "WARNING_FRAMEWORK_API_VERSION_VS_ENV_LOADED_LIB_MISMATCH": "Mismatched currently used framework API version  : '{}' vs env loaded libs version: '{}',"
                                                    "Continuing with the mismatched versions but this may run into issues,"
                                                    "For better accuracy, please load the correct framework version into your environment and rerun.",
    "WARNING_FRAMEWORK_CAFFE_MISMATCH_TENSOR": "Mismatched tf tensor to qnn output tensor: {}.",
    "WARNING_FRAMEWORK_TENSORFLOW_DIMENSION_UNSPECIFIED": "Tensor dimension from tensor.get_shape() is None. "
                                                          "Continuing with inputs.",
    "WARNING_FRAMEWORK_TENSORFLOW_TENSOR_NOT_EVALUATED": "Tensor could not be evaluated: '{}'. "
                                                         "Continuing with evaluation on next tensor",
    "WARNING_FRAMEWORK_TENSORFLOW_MISMATCH_TENSOR": "Mismatched tf tensor to qnn output tensor: {}.",

    "WARNING_FRAMEWORK_CAFFE2_FAILED_TO_FETCH_BLOB": "Could not fetch blob: {}.",
    "WARNING_FRAMEWORK_CAFFE2_ENCOUNTERED_BYTES_OBJECT": "Unable to evaluate blob '{}', moving onto next blob.",

    "WARNING_FRAMEWORK_CAFFE_FAILED_TO_FETCH_BLOB": "Could not fetch blob: {}.",
    "WARNING_FRAMEWORK_CAFFE_ENCOUNTERED_BYTES_OBJECT": "Unable to evaluate blob '{}', moving onto next blob.",

    "WARNING_FRAMEWORK_TFLITE_NO_INTERMEDIATE_TENSORS": "TFlite host interpreter does not provide functionality to produce intermediate tensor results.",

    "WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR": "Mismatched tf tensor to qnn output tensor: {}.",
    ###########################################################################
    # UTILS WARNING CODES
    ###########################################################################

    "WARNING_UTILS_CANNOT_CREATE_SYMLINK": "Cannot create symlink '{}' to '{}'.",

    ###########################################################################
    # VERIFICATION WARNING CODES
    ###########################################################################

    "WARNING_VERIFIER_MISSING_INFERENCE_TENSOR_DATA": "No inference data found for tensor(s): {}. Continuing with "
                                                      "evaluation on next tensor.",
    "WARNING_VERIFIER_MISSING_GOLDEN_TENSOR_DATA": "No golden data found for tensor(s): {}. Continuing with "
                                                   "evaluation on next tensor."

}

debug_codes_to_messages = {
    "DEBUG_FRAMEWORK_TENSORFLOW_TENSOR_NOT_EVALUATED": "Tensor could not be evaluated: '{}'. "
                                                       "Continuing with evaluation on next tensor."
}

progress_codes_to_messages = {
    ###########################################################################
    # FRAMEWORK PROGRESS CODES
    ###########################################################################

    "PROGRESS_FRAMEWORK_VERSION_VALIDATION": "Validating framework configuration: {} {}.",
    "PROGRESS_FRAMEWORK_VERSION_AUTOMATIC": "No version specified for {}, defaulting to version {}.",
    "PROGRESS_FRAMEWORK_INSTANCE_VALIDATION": "Verifying instantiated framework instance matches configuration.",
    "PROGRESS_FRAMEWORK_INSTANTIATION": "Creating an instance of {}.",
    "PROGRESS_FRAMEWORK_GENERATE_OUTPUTS": "Generating intermediate tensors, outputs are being written into {}.",
    "PROGRESS_FRAMEWORK_GENERATED_INTERMEDIATE_TENSORS": "Intermediate tensors successfully generated from {} {}.",

    ###########################################################################
    # UTILS PROGRESS CODES
    ###########################################################################

    "PROGRESS_UTILS_CREATE_SYMLINK": "Creating symlink from '{}' to '{}'."

}


def _wrapper_(error_code, message_table):
    try:
        message = message_table[error_code]
    except KeyError:
        message = ""

    def _formatter_(*args):
        if message.count('{}') == len(args):
            return "{}: {}".format(error_code, message.format(*[str(arg) for arg in args]))
        else:
            return "{}: N/A".format(error_code)

    if message.count('{}') > 0:
        return _formatter_
    else:
        return '{}: {}'.format(error_code, message)


def get_message(error_code):
    return _wrapper_(error_code, error_codes_to_messages)


def get_warning_message(error_code):
    return _wrapper_(error_code, warning_codes_to_messages)


def get_debugging_message(error_code):
    return _wrapper_(error_code, debug_codes_to_messages)


def get_progress_message(error_code):
    return _wrapper_(error_code, progress_codes_to_messages)
