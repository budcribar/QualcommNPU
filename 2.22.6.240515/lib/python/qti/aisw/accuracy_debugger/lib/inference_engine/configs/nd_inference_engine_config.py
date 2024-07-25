# =============================================================================
#
#  Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import os
from configparser import ConfigParser
from packaging.version import Version

from pathlib2 import Path
from qti.aisw.accuracy_debugger.lib.device.nd_device_factory import DeviceFactory
from qti.aisw.accuracy_debugger.lib.inference_engine.configs import CONFIG_PATH
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Framework, ComponentType, Runtime, Devices_list, X86_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace


class InferenceEngineConfig:

    def __init__(self, args, registry, logger):

        def _validate_params():
            if not hasattr(Engine, str(args.engine)):
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND")(args.engine))

            if not hasattr(Runtime, str(args.runtime)):
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_RUNTIME_NOT_FOUND")(args.runtime))

        _validate_params()

        self.config = args
        self._engine_type = Engine(args.engine)
        self._engine_version = Version(
            args.engine_version) if args.engine_version is not None else None
        self._framework = Framework(args.framework) if args.framework is not None else None
        self._runtime = Runtime(args.runtime)
        self.registry = registry
        self.logger = logger
        self.snpe_quantizer_config = None

    @staticmethod
    def load_json(*paths):
        path = os.path.join(*paths)

        with open(path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def load_ini(*paths):
        path = os.path.join(*paths)
        config = ConfigParser()
        config.read(path)
        return config

    def _get_configs(self):

        def get_converter_config(config):
            if self._framework is None or ComponentType.converter.value not in config:
                return None
            if self._framework.value not in config[ComponentType.converter.value]:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_NO_FRAMEWORK_IN_CONVERTER")(
                        self._framework.value))
            return config[ComponentType.converter.value][self._framework.value]

        def get_executor_config(config):
            return config[ComponentType.executor.value]

        def get_snpe_quantizer_config(config):
            return config[ComponentType.snpe_quantizer.value]

        def get_inference_engine_config(config):
            engine_config = config[ComponentType.inference_engine.value]
            target_device, host_device = self.create_devices(config[ComponentType.devices.value])
            engine_config["target_device"], engine_config["host_device"] = target_device, \
                                                                           host_device
            return engine_config

        def get_context_binary_generator_config(config):
            # Config for context binary generator is different for x86_64_windows_msvc
            if "x86_64-windows-msvc" in self.config.lib_target:
                return config[ComponentType.x86_64_windows_context_binary_generator.value]
            elif "wos" in self.config.lib_target:
                return config[ComponentType.wos_context_binary_generator.value]
            else:
                return config[ComponentType.context_binary_generator.value]

        config_data = self.load_json(CONFIG_PATH, self._engine_type.value, 'config.json')

        converter_config = get_converter_config(config_data)

        if self.config.offline_prepare and self._engine_type == Engine.QNN:
            self.context_binary_generator_config = get_context_binary_generator_config(config_data)
        else:
            self.context_binary_generator_config = None

        executor_config = get_executor_config(config_data)

        inference_engine_config = get_inference_engine_config(config_data)

        if self._engine_type == Engine.SNPE:
            self.snpe_quantizer_config = get_snpe_quantizer_config(config_data)

        return converter_config, executor_config, inference_engine_config

    def create_devices(self, valid_devices):
        valid_host_devices, valid_target_devices = valid_devices["host"], valid_devices["target"]

        if self.config.target_device not in valid_target_devices:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_BAD_DEVICE")(self.config.target_device))

        if self.config.host_device and self.config.host_device not in valid_host_devices:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_BAD_DEVICE")(self.config.host_device))

        if self.config.target_device not in Devices_list.devices.value:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")(
                    self.config.target_device))

        dev_ip = self.config.remote_server
        dev_username = self.config.remote_username
        dev_password = self.config.remote_password
        device_setup_path = self.config.engine_path if self.config.target_device in ["x86_64-windows-msvc", "wos"] else None
        target_device = DeviceFactory.factory(self.config.target_device,
                                            self.config.deviceId,
                                            self.logger,
                                            device_ip = dev_ip,
                                            device_username = dev_username,
                                            device_password = dev_password,
                                            device_setup_path = device_setup_path)

        host_device = None
        if self.config.host_device is not None:
            if self.config.host_device not in Devices_list.devices.value:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")(
                        self.config.host_device))

            device_setup_path = self.config.engine_path if self.config.host_device in ["x86_64-windows-msvc", "wos"] else None
            host_device = DeviceFactory.factory(self.config.host_device,
                                                self.config.deviceId,
                                                self.logger,device_ip = dev_ip,
                                                device_username = dev_username,
                                                device_password = dev_password,
                                                device_setup_path = device_setup_path)

        return target_device, host_device

    def load_inference_engine_from_config(self):
        converter_config, executor_config, inference_engine_config = self._get_configs()

        inference_engine_cls = self.registry.get_inference_engine_class(
            None, self._engine_type, self._engine_version)

        executor_cls = self.registry.get_executor_class(self._framework, self._engine_type,
                                                        self._engine_version)
        executor = executor_cls(self.get_context(executor_config))
        if converter_config:
            converter_cls = self.registry.get_converter_class(self._framework, self._engine_type,
                                                              self._engine_version)
            converter = converter_cls(self.get_context(converter_config))
            return inference_engine_cls(self.get_context(inference_engine_config), converter,
                                        executor)
        else:
            return inference_engine_cls(self.get_context(inference_engine_config), None, executor)

    def get_context(self, data=None):
        config = vars(self.config)
        config.update(data)
        return Namespace(config, logger=self.logger,
                         context_binary_generator_config=self.context_binary_generator_config,
                         snpe_quantizer_config=self.snpe_quantizer_config)
