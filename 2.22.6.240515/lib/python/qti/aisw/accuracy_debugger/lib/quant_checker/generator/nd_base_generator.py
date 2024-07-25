# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, Framework
from qti.aisw.accuracy_debugger.lib.inference_engine.configs import CONFIG_PATH
from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, Framework, Devices_list
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.device.nd_device_factory import DeviceFactory

import os
import json
from abc import ABC, abstractmethod
import copy
import subprocess
import zipfile


class BaseGenerator:

    def __init__(self, args, logger=None):
        self._args = args
        self._logger = setup_logger(args.verbose, args.output_dir) if logger is None else logger
        self._engine_type = Engine(args.engine)
        self._framework = Framework(args.framework)
        file_path = os.path.join(CONFIG_PATH, self._engine_type.value, 'config.json')
        with open(file_path, 'r') as file:
            self._config_data = json.load(file)
        self._registry = inference_engine_repository
        self._converter = None
        self._host_env = None
        self._host_device = None
        self._engine_path = None
        self._quantized_variation_model_dir_map = dict()

    def _initialize(self):
        '''
        Initialize the generator
        '''
        self._set_engine_path()
        self._set_host_environment()
        self._get_coverter()
        self._create_host_device()

    @abstractmethod
    def _set_host_environment(self):
        pass

    def _all_quantization_schemes(self):
        '''
        Permutate all quant Schemes
        '''
        schemes = []
        schemes.append((None, None, None, 'unquantized'))
        ValidQuantizationVariations = ['tf', 'enhanced', 'adjusted', 'symmetric']
        ValidQuantizationAlgorithms = ["None", 'cle'] + (['pcq', 'cle_pcq'] if
                                                         self._engine_type.value == 'QNN' else [])
        for quantization_variation in self._args.quantization_variations:
            if quantization_variation in ValidQuantizationVariations:
                variation_name = quantization_variation
                #(quantization_variation, cle, pcq/bc, variation_name)
                for quantization_algorithm in self._args.quantization_algorithms:
                    if quantization_algorithm in ValidQuantizationAlgorithms:
                        variation_name = quantization_variation + (
                            '_' +
                            quantization_algorithm if quantization_algorithm != "None" else '')
                        if quantization_algorithm == 'cle':
                            schemes.append((quantization_variation, quantization_algorithm, None,
                                            variation_name))
                        elif quantization_algorithm == 'pcq':
                            schemes.append((quantization_variation, None, quantization_algorithm,
                                            variation_name))
                        elif quantization_algorithm == "None":
                            schemes.append((quantization_variation, None, None, variation_name))
                        else:
                            algo1, algo2 = quantization_algorithm.split('_')
                            schemes.append((quantization_variation, algo1, algo2, variation_name))

        return schemes

    def _set_engine_path(self):
        '''
        Sets the engine(QNN SDK) pathS
        '''
        if self._args.engine_path.endswith(".zip"):
            self._engine_path = self._unzip_engine_path()
        else:
            self._engine_path = self._args.engine_path

    def _unzip_engine_path(self):
        """This helper function unzips engine_zip and sets the engine_path to
        the correct path."""
        host_output_dir = self._args.output_dir
        engine_zip_path = self._args.engine_path
        #Zipfile is breaking the symlink while extracting. So using subprocess for extracting
        try:
            subprocess.run(['unzip', '-q', engine_zip_path, '-d', host_output_dir],
                           stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            print("ERROR: Extracting SDK with the following error: ", err.returncode)
        with zipfile.ZipFile(engine_zip_path, 'r') as f:
            filelists = f.namelist()
            for file in filelists:
                os.chmod(os.path.join(host_output_dir, file), 0o755)
        if './' in filelists[0]:
            engine_path = os.path.join(host_output_dir, os.path.dirname(filelists[1]))
        else:
            engine_path = os.path.join(host_output_dir, os.path.dirname(filelists[0]))

        return engine_path

    def _create_host_device(self):
        '''
        Create x86 host device
        '''
        valid_host_devices = self._config_data[ComponentType.devices.value]['host']
        if self._args.host_device and self._args.host_device not in valid_host_devices:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_BAD_DEVICE")(self._args.host_device))
        if self._args.host_device is not None:
            if self._args.host_device not in Devices_list.devices.value:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")(
                        self._args.host_device))
            device_setup_path = self._args.engine_path if self._args.host_device in ["x86_64-windows-msvc", "wos"] else None

            self._host_device = DeviceFactory.factory(self._args.host_device, self._args.deviceId,
                                                      self._logger,
                                                      device_setup_path=device_setup_path)

    def _get_coverter(self):
        '''
        Get the converter class object
        '''
        converter_config = self._config_data[ComponentType.converter.value][self._framework.value]
        converter_cls = self._registry.get_converter_class(self._framework, self._engine_type, None)
        self._converter = converter_cls(self._get_context(converter_config))

    def _get_context(self, data=None):
        '''
        Returns the context variable
        '''
        args_ = copy.deepcopy(self._args)
        config = vars(args_)
        config.update(data)
        return Namespace(config, logger=self._logger, context_binary_generator_config=None,
                         snpe_quantizer_config=None)

    def get_quantized_variation_model_dir_map(self):
        '''
        Returns the map of quantization variations and corresponding model dump directory
        '''
        return self._quantized_variation_model_dir_map
