# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from datetime import datetime
from distutils.version import StrictVersion
from importlib import import_module
from logging import Logger
from typing import List, Tuple

from lib.framework_diagnosis.nd_framework_objects import get_available_frameworks
from lib.utils.nd_errors import get_message, get_progress_message, get_warning_message
from lib.utils.nd_exceptions import FrameworkError
from lib.utils.nd_framework_utility import save_outputs, load_inputs


class FrameworkRunner(object):
    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self.parsed_args = args
        self.framework = args.framework.lower()
        self.version = args.version
        self.version_original = args.version
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        self.available_frameworks = get_available_frameworks()
        self.framework_instance = None
        self._logger = logger

    def _validate_framework(self):  # type: () -> None
        def max_version(framework):
            versions = self.available_frameworks.get(framework, {})
            return max(versions.keys(), key=lambda x: StrictVersion(x))

        self._logger.info(get_progress_message('PROGRESS_FRAMEWORK_VERSION_VALIDATION')(self.framework, self.version))

        if self.framework not in self.available_frameworks:
            raise FrameworkError(get_message('ERROR_FRAMEWORK_FAILED_CONFIGURATION')(self.framework))

        #TODO: if version specified is higher than the highest API runner we have we should note a warning and fall back to the highest supported api runner.
        if self.version is None:
            self.version = max_version(self.framework)
            self._logger.info(
                get_progress_message('PROGRESS_FRAMEWORK_VERSION_AUTOMATIC')(self.framework, self.version))

        if self.version not in self.available_frameworks[self.framework]:
            raise FrameworkError(get_message('ERROR_FRAMEWORK_FAILED_CONFIGURATION')(self.framework))

    def _validate_framework_instance(self):
        self._logger.info(get_progress_message('PROGRESS_FRAMEWORK_INSTANCE_VALIDATION'))

        if not self.version == self.framework_instance.get_version():
            self._logger.warning(
                get_warning_message("WARNING_FRAMEWORK_API_VERSION_VS_ENV_LOADED_LIB_MISMATCH")\
                (self.version, self.framework_instance.get_version()))

    def load_framework_instance(self):
        module, framework = self.available_frameworks[self.framework][self.version]

        self._logger.info(get_progress_message('PROGRESS_FRAMEWORK_INSTANTIATION')(framework))
        try:
            framework_type = getattr(import_module(module), framework)
        except ImportError as exc:
            self._logger.exception(exc)
            raise FrameworkError(get_message('ERROR_FRAMEWORK_FAILED_CONFIGURATION')(self.framework))

        self.framework_instance = framework_type(self._logger)

    def load_framework(self):  # type: () -> None
        self._validate_framework()
        self.load_framework_instance()

        if not self.version == self.framework_instance.get_version() and self.version_original is None and self.framework !="tflite":
            self.version = self.framework_instance.get_version()
            self._logger.info(
                get_progress_message('PROGRESS_FRAMEWORK_VERSION_AUTOMATIC')(self.framework, self.version))
            self.load_framework_instance()

        #tflite's get_version() is the tensorflow's version, so tflite'version is alway check failed
        if self.framework !="tflite":
            self._validate_framework_instance()
        self.framework_instance.load_model(self.model_path)

    def generate_intermediate_outputs(self, output_dir):  # type: (str) -> None
        data_path = os.path.join( output_dir, '{}{}')

        #validation check for input_tensor/output_tensor
        try:
            output_names = self.parsed_args.output_tensor
            input_tensor = self.parsed_args.input_tensor
        except Exception:
            raise FrameworkError(get_message('ERROR_FRAMEWORK_RUNNER_NO_INPUT_TENSOR_OR_NO_OUTPUT_TENSOR'))

        if output_names == []:
            raise FrameworkError(get_message('ERROR_FRAMEWORK_RUNNER_NO_INPUT_TENSOR_OR_NO_OUTPUT_TENSOR'))

        #get the input tensor
        in_list = list(zip(*input_tensor))
        if len(in_list) == 4:
            (in_names, in_dims, in_data_paths, in_types) = in_list
        elif len(in_list) == 3:
            (in_names, in_dims, in_data_paths) = in_list
            in_types = None
        else:
            raise FrameworkError(get_message('ERROR_FRAMEWORK_RUNNER_INPUT_TENSOR_LENGHT_ERROR'))
        input_names = list(in_names)
        input_dims = [[int(x) for x in dim.split(',')] for dim in in_dims]
        input_data_paths = in_data_paths
        input_types = in_types

        tensor_pairs = self.framework_instance.get_intermediate_tensors(input_names, output_names)

        self._logger.info(get_progress_message('PROGRESS_FRAMEWORK_GENERATE_OUTPUTS')(data_path.format('', '')))

        if not input_types:
            self._logger.error('ERROR_PROGRESS_FRAMEWORK_GENERATE_OUTPUTS_NO_INPUT_TYPES')

        input_data = [load_inputs(file, data_type, dim) for file, data_type, dim in zip(input_data_paths,
                                                                                        input_types,
                                                                                        input_dims)]

        for _, output_tensor_names in tensor_pairs:
            result = self.framework_instance.run_inference(input_data,
                                                           input_names,
                                                           output_tensor_names)
            for output_tensor, data in result.items():
                file_path = data_path.format(output_tensor, '.raw')
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                #TODO:// there might be a need to customize specify output type
                save_outputs(data, file_path, "float32")
        self._logger.info(get_progress_message('PROGRESS_FRAMEWORK_GENERATED_INTERMEDIATE_TENSORS')
                          (self.framework, self.framework_instance.get_version()))

    def load_framework_for_tensor_mapping(self):
        self.load_framework()

    def run(self):
        self.load_framework()
        self.generate_intermediate_outputs(self.output_dir)
