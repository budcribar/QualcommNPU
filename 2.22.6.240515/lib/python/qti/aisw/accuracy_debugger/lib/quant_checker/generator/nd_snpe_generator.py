# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import retrieveSnpeSdkDir
from qti.aisw.accuracy_debugger.lib.quant_checker.generator.nd_base_generator import BaseGenerator
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_utils import verify_path

import os
import sys
import json
import subprocess
import zipfile
import copy
import importlib


class SnpeGenerator(BaseGenerator):

    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        # set engine path if none specified
        if args.engine_path is None:
            args.engine_path = retrieveSnpeSdkDir()
        self._CONVERTER_LOCATION = self._config_data['inference_engine']['converter_location']
        self._snpe_lib_python = self._config_data['inference_engine']['snpe_lib_python']
        self._snpe_dlc_utils_package = self._config_data['inference_engine'][
            'snpe_dlc_utils_package']
        self._snpe_quantizer_config = self._config_data['snpe_quantizer']
        self._initialize()
        self._unquantized_dlc_path = None

    def generate_all_quant_models(self):
        '''
        Generate all the quantized models for corresponding quant scheme
        '''
        # Create the path where quantized models will be stored
        quantized_variations_path = verify_path(self._args.output_dir, 'quantization_variations')

        # loop over all possible quantization schemes, generate the model and dump them
        for quantization_variation, cle, _, variation_name in self._all_quantization_schemes():
            quantized_variation_model_dir = verify_path(quantized_variations_path, variation_name)
            status = self._generate_quantized_model(quantized_variation_model_dir, variation_name,
                                                    quantization_variation, cle)
            if status == 0:
                self._quantized_variation_model_dir_map[
                    variation_name] = quantized_variation_model_dir
        self._logger.info(get_message(str(self._quantized_variation_model_dir_map.keys())))

    def _set_host_environment(self):
        """This helper function sets up the SNPE execution environment on host
        x86 device."""
        self._CONVERTER_LOCATION = self._CONVERTER_LOCATION.format(engine_path=self._engine_path,
                                                                   host_arch=self._args.host_device)
        self._host_env = {}
        self._host_env["PATH"] = self._CONVERTER_LOCATION + os.pathsep \
                                + ('' if self._args.host_device in ["x86_64-windows-msvc", "wos"] else '$PATH')
        self._host_env['PYTHONPATH'] = os.path.join(
            self._engine_path, self._snpe_lib_python) + os.pathsep + (
                '' if self._args.host_device in ["x86_64-windows-msvc","wos"] else '$PYTHONPATH')
        print(self._host_env)
        sys.path.insert(0, os.path.join(self._engine_path, self._snpe_lib_python))
        sys.path.insert(0, os.path.join(self._engine_path, self._snpe_dlc_utils_package))
        # self.snpe_dlc = importlib.import_module('snpe_dlc_utils')

    def _generate_quantized_model(self, quantized_variation_model_dir, variation_name,
                                  quantization_variation, cle):
        '''
        Given a quant Scheme generates the quantized model
        '''
        snpe_model_dlc_path = os.path.join(quantized_variation_model_dir, variation_name + '.dlc')
        conversion_inputs = {input_name: dim for input_name, dim, _ in self._args.input_tensor}
        if variation_name == 'unquantized':
            self._unquantized_dlc_path = snpe_model_dlc_path
            return self._create_dlc(snpe_model_dlc_path, conversion_inputs,
                                    self._args.output_tensor)
        else:
            return self._execute_dlc_quantization(snpe_model_dlc_path, quantization_variation, cle)

    def _create_dlc(self, dlc_path, inputs, outputs):
        """Convert a model into a dlc.

        :param dlc_path: Path to save the new dlc
        :param inputs: Input names and dimensions to the model
        :param outputs: Output names of the model
        """
        convert_command = self._converter.build_convert_command(self._args.model_path, inputs,
                                                                outputs, dlc_path,
                                                                self._args.quantization_overrides,
                                                                self._args.extra_converter_args)
        log_string = 'Starting conversion with: ' + \
                     'Inputs: ' + str(list(inputs.keys())) + ' ' + \
                     'Outputs: ' + str(outputs)
        self._logger.info(log_string)

        print(self._host_env)

        try:
            code, _, err = self._host_device.execute(commands=[convert_command], env=self._host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))

            self._logger.info('Model converted successfully')
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))
        return code

    def _execute_dlc_quantization(self, quantized_dlc_path, quantization_variation, cle):
        """Execute DLC quantization.

        :param quantized_dlc_path: Path to the converter dlc result
        :param quantization_variation: either of tf, enhanced, adjusted, symmetric
        :param cle: If not None, perform cross layer equalization
        """
        try:
            if (self._host_device.device in ['x86_64-windows-msvc', 'wos']):
                self._snpe_quantizer_config["executable"] = self._snpe_quantizer_config[
                    "windows_executable"]
            convert_command = [
                self._snpe_quantizer_config["executable"],
                self._snpe_quantizer_config["arguments"]["dlc_path"], self._unquantized_dlc_path
            ]
            if self._args.input_list:
                convert_command += [
                    self._snpe_quantizer_config["arguments"]["input_list"], self._args.input_list
                ]
            else:
                raise InferenceEngineError(
                    "snpe dlc quantization should be input the inputlist, but you miss it!")

            convert_command += [
                self._snpe_quantizer_config["arguments"]["bias_bitwidth"] + "=" +
                str(self._args.bias_width)
            ]
            convert_command += [
                self._snpe_quantizer_config["arguments"]["output_path"], quantized_dlc_path
            ]
            convert_command += [
                self._snpe_quantizer_config["arguments"]["weights_bitwidth"] + "=" +
                str(self._args.weight_width)
            ]
            if quantization_variation == 'symmetric':
                convert_command += [
                    self._snpe_quantizer_config["arguments"]["use_symmetric_quantize_weights"]
                ]
            elif quantization_variation == 'adjusted':
                convert_command += [
                    self._snpe_quantizer_config["arguments"]["use_adjusted_weights_quantizer"]
                ]
            elif quantization_variation == 'enhanced':
                convert_command += [
                    self._snpe_quantizer_config["arguments"]["use_enhanced_quantizer"]
                ]
            if self._args.quantization_overrides:
                convert_command += [self._snpe_quantizer_config["arguments"]["override_params"]]
            if cle:
                convert_command += ['--optimizations cle']
            convert_command += self._snpe_quantizer_config["arguments"]["flags"]
            convert_command_str = ' '.join(convert_command)

            log_string = 'Running DLC quantize with: ' + \
                        'Inputs: ' + str(self._unquantized_dlc_path) + ' ' + \
                        'Outputs: ' + str(quantized_dlc_path)
            self._logger.info(log_string)
            code, _, err = self._host_device.execute(commands=[convert_command_str],
                                                     cwd=self._engine_path, env=self._host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))
            self._logger.info('DLC model quantized successfully')
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(
                get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))
        return code
