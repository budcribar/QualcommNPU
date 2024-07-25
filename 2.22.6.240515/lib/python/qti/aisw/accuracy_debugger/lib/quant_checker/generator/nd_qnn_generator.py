# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import retrieveQnnSdkDir
from qti.aisw.accuracy_debugger.lib.quant_checker.generator.nd_base_generator import BaseGenerator
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_utils import verify_path

import os
import json
import subprocess
import zipfile
import copy


class QnnGenerator(BaseGenerator):

    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        # set engine path if none specified
        if args.engine_path is None:
            args.engine_path = retrieveQnnSdkDir()
        self._initialize()

    def generate_all_quant_models(self):
        '''
        Generate all the quantized models for corresponding quant scheme
        '''
        # Create the path where quantized models will be stored
        quantized_variations_path = verify_path(self._args.output_dir, 'quantization_variations')

        # loop over all possible quantization schemes, generate the model and dump them
        for quantization_variation, cle, pcq, variation_name in self._all_quantization_schemes():
            quantized_variation_model_dir = verify_path(quantized_variations_path, variation_name)
            self._quantized_variation_model_dir_map[variation_name] = quantized_variation_model_dir
            self._generate_quantized_model(quantized_variation_model_dir, variation_name,
                                           quantization_variation, cle, pcq)

    def _set_host_environment(self):
        """This helper function sets up the QNN execution environment on host
        x86 device."""
        sdk_tools_root = self._config_data['inference_engine']['sdk_tools_root']
        env_variables = self._config_data['inference_engine']['environment_variables']
        # Get file paths:
        sdk_tools_root = sdk_tools_root.format(engine_path=self._engine_path)

        for var in env_variables:
            env_variables[var] = env_variables[var].format(sdk_tools_root=sdk_tools_root)

        # set environment:
        _host_env = {}
        # set environment variables depending on host device architecture
        if self._args.host_device in ["x86_64-windows-msvc", "wos"]:
            for var in env_variables:
                _host_env[var] = env_variables[var] + os.pathsep
        else:
            for var in env_variables:
                _host_env[var] = env_variables[var] + os.pathsep + '$' + var
        # Add   path to PATH:
        _host_env['QNN_SDK_ROOT'] = self._engine_path
        self._host_env = _host_env

    def _generate_quantized_model(self, quantized_variation_model_dir, variation_name,
                                  quantization_variation, cle, pcq):
        '''
        Given a quant Scheme generates the quantized model
        '''
        # Path for qnn model.cpp
        qnn_model_cpp = os.path.join(quantized_variation_model_dir, variation_name + '.cpp')

        if variation_name == 'unquantized':
            # convert command for dumping fp32 model
            convert_command = self._converter.build_convert_command(
                model_path=self._args.model_path, input_tensors=self._args.input_tensor,
                output_tensors=self._args.output_tensor, output_path=qnn_model_cpp,
                input_list_txt=None, quantization_overrides=None, param_quantizer=None,
                act_quantizer=None, weight_bw=None, bias_bw=None, act_bw=None, float_bias_bw=None,
                restrict_quantization_steps=None, algorithms=None, ignore_encodings=None,
                per_channel_quantization=None, act_quantizer_calibration=None,
                param_quantizer_calibration=None, act_quantizer_schema=None,
                param_quantizer_schema=None, percentile_calibration_value=None,
                extra_converter_args=None)
        else:
            # Quant scheme specific convert command
            convert_command = self._converter.build_convert_command(
                model_path=self._args.model_path, input_tensors=self._args.input_tensor,
                output_tensors=self._args.output_tensor, output_path=qnn_model_cpp,
                input_list_txt=self._args.input_list,
                quantization_overrides=self._args.quantization_overrides,
                param_quantizer=quantization_variation, act_quantizer=None,
                weight_bw=self._args.weight_width, bias_bw=self._args.bias_width, act_bw=None,
                float_bias_bw=None, restrict_quantization_steps=None, algorithms=cle,
                ignore_encodings=None, per_channel_quantization=pcq, act_quantizer_calibration=None,
                param_quantizer_calibration=None, act_quantizer_schema=None,
                param_quantizer_schema=None, percentile_calibration_value=None,
                extra_converter_args=self._args.extra_converter_args)
        try:
            self._logger.debug('Model converter command : {}'.format(convert_command))
            code, _, err = self._host_device.execute(commands=[convert_command],
                                                     cwd=self._engine_path, env=self._host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))
            self._logger.info(get_progress_message("PROGRESS_INFERENCE_ENGINE_CONVERSION_FINISHED"))
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))
