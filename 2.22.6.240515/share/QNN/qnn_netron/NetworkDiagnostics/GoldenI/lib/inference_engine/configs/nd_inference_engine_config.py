# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import os
from configparser import ConfigParser
from distutils.version import LooseVersion

from pathlib2 import Path
from lib.device.nd_device_factory import DeviceFactory
from lib.inference_engine.configs import CONFIG_PATH
from lib.utils.nd_constants import Engine, Framework, ComponentType, Runtime, Devices_list
from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import InferenceEngineError
from lib.utils.nd_namespace import Namespace


class InferenceEngineConfig:

    def __init__(self, args, registry, logger):
        def _validate_params():
            if not hasattr(Engine, str(args.engine)):
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND")
                                           (args.engine))

            if not hasattr(Runtime, str(args.runtime)):
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_RUNTIME_NOT_FOUND")
                                           (args.runtime))

        _validate_params()
        self.config = args
        self._engine_type = Engine(args.engine)
        self._engine_version = LooseVersion(args.engine_version) if args.engine_version is not None else None
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

    def load_latest_version(self, engine_type, engine_version, versions=None):
        # type: (Engine, LooseVersion, list[LooseVersion]) -> LooseVersion
        config_versions = [Path(config_path).stem for config_path in
                           os.listdir(os.path.join(CONFIG_PATH, engine_type.value))]

        if versions is not None:
            config_versions = versions

        config_versions = [LooseVersion(version) for version in config_versions]
        if engine_version is not None:
            config_versions = [version for version in config_versions if version <= engine_version]
        else:
            latest_version = sorted(config_versions)[-1]
            self.logger.info("No version specified for " + engine_type.value +
                             " Inference Engine, defaulting to version " + str(latest_version))

        return sorted(config_versions).pop() if config_versions else None

    def _get_configs(self):
        def get_converter_config(config):
            if self._framework is None or ComponentType.converter.value not in config:
                return None
            if self._framework.value not in config[ComponentType.converter.value]:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_NO_FRAMEWORK_IN_CONVERTER")
                    (self._framework.value))
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
            return config[ComponentType.context_binary_generator.value]

        latest_version = self.load_latest_version(self._engine_type, self._engine_version)

        if latest_version is None:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_ENGINE_VERSION_NOT_SUPPORTED")
                                       (self._engine_version))

        config_data = self.load_json(CONFIG_PATH, self._engine_type.value, str(latest_version) +
                                     '.json')

        converter_config = get_converter_config(config_data)

        if self.config.offline_prepare and self._engine_type == Engine.QNN:
            self.context_binary_generator_config = get_context_binary_generator_config(config_data)
        else:
            self.context_binary_generator_config = None

        executor_config = get_executor_config(config_data)

        inference_engine_config = get_inference_engine_config(config_data)

        if self._engine_type == Engine.SNPE:
            self.snpe_quantizer_config = get_snpe_quantizer_config(config_data)

        return converter_config, executor_config , inference_engine_config

    def create_devices(self, valid_devices):
        valid_host_devices, valid_target_devices = valid_devices["host"], valid_devices["target"]

        if self.config.target_device not in valid_target_devices:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_BAD_DEVICE")
                                       (self.config.target_device))

        if self.config.host_device and self.config.host_device not in valid_host_devices:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_BAD_DEVICE")
                                       (self.config.host_device))

        if self.config.target_device not in Devices_list.devices.value:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")
                                       (self.config.target_device))

        target_device = DeviceFactory.factory(self.config.target_device,
                                               self.config.deviceId, self.logger)

        host_device = None
        if self.config.host_device is not None:
            if self.config.host_device not in Devices_list.devices.value:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")
                                           (self.config.host_device))

            host_device = DeviceFactory.factory(self.config.host_device,
                                                 self.config.deviceId, self.logger)

        return target_device, host_device

    def load_inference_engine_from_config(self):
        converter_config, executor_config, inference_engine_config= self._get_configs()

        inference_engine_cls = self.registry.get_inference_engine_class(None, self._engine_type,
                                                                        self._engine_version)

        executor_cls = self.registry.get_executor_class(
            self._framework, self._engine_type, self._engine_version)
        executor = executor_cls(self.get_context(executor_config))
        if converter_config:
            converter_cls = self.registry.get_converter_class(
                self._framework, self._engine_type, self._engine_version)
            converter = converter_cls(self.get_context(converter_config))
            return inference_engine_cls(self.get_context(inference_engine_config), converter,
                                        executor)
        else:
            return inference_engine_cls(self.get_context(inference_engine_config), None, executor)

    def get_context(self, data=None):
        config = vars(self.config)
        config.update(data)
        return Namespace(config, logger=self.logger, context_binary_generator_config=self.context_binary_generator_config,
                         snpe_quantizer_config=self.snpe_quantizer_config)
