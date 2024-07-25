# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from __future__ import absolute_import
import json
import os
import shutil
import sys
from .bm_jsonkeys import *
from .bm_config_restrictions import *
import datetime
from subprocess import check_output
from common_utils.adb import Adb
from common_utils.exceptions import ConfigError, AdbShellCmdFailedException
from .error import Error
import logging

logger = logging.getLogger(__name__)


def load_json(cfgfile):
    try:
        with open(cfgfile, 'r') as cfg:
            try:
                json_data = json.load(cfg)
                return json_data
            except ValueError as e:
                logger.error("error parsing JSON file: " + cfgfile)
                return []
    except Exception as e:
        logger.error("Error opening file: " + cfgfile + " " + repr(e))
        return []


class DnnModel(object):
    def __init__(self, config, host_artifacts, product , outputbasedir, htp_serialized):
        self._product = product
        self._name = config[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_NAME_SUBKEY]
        self._model = DnnModel.__default_path(
            config[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_SUBKEY])
        self._dev_root_dir = '/'.join([config[product.CONFIG_DEVICE_PATH_KEY], self._name])
        self._host_artifacts = host_artifacts
        self._host_tmp_dir = product.HOST_TMP_DIR

        if outputbasedir:
            self._host_tmp_dir = os.path.join(os.path.abspath(outputbasedir), self._host_tmp_dir)

        if os.path.isdir(self._host_tmp_dir):
            shutil.rmtree(self._host_tmp_dir)

        os.makedirs(self._host_tmp_dir)
        subkeys = []
        for sub_key in config[product.CONFIG_MODEL_KEY]:
            subkeys.append(sub_key)
        if "InputList" in subkeys:
            self._input_list_name = os.path.basename(
                config[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_INPUTLIST_SUBKEY])
            self._artifacts = []
            self._artifacts.append([DnnModel.__default_path(config[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_INPUTLIST_SUBKEY]),
                                    self._dev_root_dir])

            for data in config[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_DATA_SUBKEY]:
                _abs_data_path = DnnModel.__default_path(data)
                self._artifacts.append([_abs_data_path, '/'.join([
                    self._dev_root_dir, os.path.basename(data)])])
        self._artifacts.append([self._model, self._dev_root_dir])

    @staticmethod
    def __default_path(artifact_path):
        _abs_path = artifact_path
        if '.so' in _abs_path or '.bin' in _abs_path:
            if 'MODELZOO' in os.environ:
                _abs_path = os.path.join(os.environ['MODELZOO'], _abs_path)
                return _abs_path
        if not os.path.isabs(artifact_path):
            # relative to current directory
            _abs_path = os.path.abspath(artifact_path)
        if not os.path.exists(_abs_path):
            raise ConfigError(artifact_path + " does not exist")
        return _abs_path

    @property
    def name(self):
        return self._name

    @property
    def input_list_name(self):
        return self._input_list_name

    @property
    def host_dir(self):
        return self._host_tmp_dir

    @property
    def model(self):
        return self._model

    @property
    def device_rootdir(self):
        return self._dev_root_dir

    @property
    def artifacts(self):
        return self._artifacts


class Config(object):
    def __init__(self, cfg_file, cfg_from_json, outputbasedir, devicelist, hostname, deviceostype,
                 userbuffer_mode, backend_config, perfprofile, iterations, profilinglevel,
                 product, dsp_type, htp_serialized, arm_prepare, use_signed_skel,cache, use_shared_buffer,
                 discard_output, test_duration):
        config_prefix = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..')
        self._cfg_from_json = cfg_from_json
        self._product = product
        self._dsp_type = dsp_type
        self._htp_serialized = htp_serialized
        self.device_os_type = deviceostype
        self._use_signed_skel = use_signed_skel
        self.discard_output = discard_output
        self.test_duration = test_duration
        if self._htp_serialized:
            self.GRAPH_PREPARE_TYPE = 'X86 PREPARE'
        elif arm_prepare:
            self.GRAPH_PREPARE_TYPE = 'ARM PREPARE'
        elif cache:
            self.GRAPH_PREPARE_TYPE = 'BACKEND PREPARE CACHE'
        else:
            self.GRAPH_PREPARE_TYPE = 'BACKEND PREPARE'

        if deviceostype and deviceostype not in CONFIG_VALID_DEVICEOSTYPES:
            raise ConfigError(
                'Device OS Type not valid.  Only specify one of %s' %
                CONFIG_VALID_DEVICEOSTYPES)
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_ANDROID_AARCH64:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_ANDROID_AARCH64_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_MSVC
            self._platform_os = self._product.PLATFORM_OS_WINDOWS
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_WINDOWS_AARCH64_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_QNX_AARCH64:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_QNX
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_QNX_AARCH64_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC112:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_AARCH64_LINUX_OE_GCC112_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC93:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_AARCH64_LINUX_OE_GCC93_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC82:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_AARCH64_LINUX_OE_GCC82_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_OE_GCC75:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_AARCH64_UBUNTU_OE_GCC75_ARTIFACTS_JSON))
        elif deviceostype == self._product.CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_OE_GCC94:
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_AARCH64_UBUNTU_OE_GCC94_ARTIFACTS_JSON))
        elif deviceostype:
            # presumed to be Android aarch64
            self._architectures = [self._product.ARCH_AARCH64]
            self._compiler = self._product.COMPILER_CLANG90
            self._platform_os = self._product.PLATFORM_OS_ANDROID
            self._stl_library = self._product.STL_LIBCXX_SHARED
            self._artifacts_config = load_json(
                os.path.join(
                    config_prefix,
                    product.PRODUCT_BM,
                    self._product.BENCH_ANDROID_AARCH64_ARTIFACTS_JSON))
        self.__override_cfgfile__(outputbasedir, devicelist, hostname, iterations)
        self.__quick_verify__()
        self._dnnmodel = DnnModel(
            self._cfg_from_json,
            self.host_artifacts,
            self._product, outputbasedir, htp_serialized)
        self.userbuffer_mode = userbuffer_mode
        self.use_shared_buffer = use_shared_buffer
        self.SHARED_BUFFER = 'Yes' if  self.use_shared_buffer else 'No'

        try:
            self._hostname = self._cfg_from_json[self._product.CONFIG_HOST_NAME_KEY]
        except KeyError:
            self._cfg_from_json[self._product.CONFIG_HOST_NAME_KEY] = 'localhost'

        try:
            if not self._cfg_from_json[self._product.CONFIG_PERF_PROFILE_KEY] is 'null':
                self.perfprofile = self._cfg_from_json[self._product.CONFIG_PERF_PROFILE_KEY]
        except KeyError as ke:
            self.perfprofile = perfprofile

        try:
            if not self._cfg_from_json[self._product.CONFIG_PROFILING_LEVEL_KEY] is 'null':
                self.profilinglevel = self._cfg_from_json[self._product.CONFIG_PROFILING_LEVEL_KEY]
        except KeyError as ke:
            self.profilinglevel = profilinglevel

        try:
            self.buffertypes = self._cfg_from_json[self._product.CONFIG_BUFFERTYPES_KEY]
        except KeyError:
            self.buffertypes = "All"

        self.backend_config = backend_config
        # Required to set it to default 'high_performance' in case
        # perf_profile is not provided as runtime argument or in the config
        # file
        if self.perfprofile == '' and perfprofile == '':
            self.perfprofile = 'high_performance'
        # Required to give preference to runtime argument in case
        # perf_profile is provided both as runtime argument and in the config
        # file
        elif perfprofile != '':
            self.perfprofile = perfprofile

        if self.profilinglevel == '' and profilinglevel == '':
            self.profilinglevel = 'basic'
        elif profilinglevel != '':
            self.profilinglevel = profilinglevel

    def __override_cfgfile__(self, outputbasedir, devicelist, hostname, iterations):
        # Override output base dir if the host paths are relative paths
        # after override, it will become an absolute path
        if outputbasedir is not None:
            logger.debug(
                'Overriding output base dir to %s, instead of %s' %
                (outputbasedir, os.getcwd()))
            for _key in [self._product.CONFIG_HOST_ROOTPATH_KEY,
                         self._product.CONFIG_HOST_RESULTSDIR_KEY]:
                _value = self._cfg_from_json[_key]
                if not os.path.isabs(_value):
                    _abs_path = os.path.abspath(outputbasedir + '/' + _value)
                    self._cfg_from_json[_key] = _abs_path
        # Override device id if one's supplied
        if devicelist is not None:
            _prevDevices = self._cfg_from_json.get(
                self._product.CONFIG_DEVICES_KEY, None)
            self._cfg_from_json[self._product.CONFIG_DEVICES_KEY] = devicelist
            logger.debug('Overriding device id to %s, instead of %s from config file' % (
                self._cfg_from_json[self._product.CONFIG_DEVICES_KEY], _prevDevices))
        # Override Host Name is one's supplied
        if hostname is not None:
            self._cfg_from_json[self._product.CONFIG_HOST_NAME_KEY] = hostname
            logger.debug('Overriding host name to %s' %
                        (self._cfg_from_json[self._product.CONFIG_HOST_NAME_KEY]))

        # Override number of runs if supplied
        if iterations is not None:
            self._cfg_from_json[self._product.CONFIG_RUNS_KEY] = iterations
            logger.debug('Overriding number of runs to %s' %
                        (self._cfg_from_json[self._product.CONFIG_RUNS_KEY]))

    def __quick_verify__(self):
        try:
            # Check that we have at least all top level keys
            # note that this will give an exception on any key that
            # isn't present
            # NOTE: We do not validate the values provided for that key
            for _key in self._product.CONFIG_JSON_ROOTKEYS:
                # check optional keys
                if not _key is self._product.CONFIG_PERF_PROFILE_KEY and not _key is self._product.CONFIG_HOST_NAME_KEY \
                    and not _key is self._product.CONFIG_BUFFERTYPES_KEY and not _key is self._product.CONFIG_PROFILING_LEVEL_KEY \
                        and not _key is self._product.CONFIG_USE_SHARED_BUFFER:
                    if self._cfg_from_json[_key] is 'null':
                        raise ConfigError("Missing value for " + _key)
            # Check for no foreign top level keys
            for _key in self._cfg_from_json.keys():
                if _key not in self._product.CONFIG_JSON_ROOTKEYS:
                    raise ConfigError("Found invalid top level key: " + _key)
        except KeyError as ke:
            raise ConfigError('Missing key in config file: ' + repr(ke))

        subkeys = self._product.CONFIG_JSON_MODEL_COMMON_SUBKEYS + \
            self._product.CONFIG_JSON_MODEL_DEFINED_INPUT_SUBKEYS
        for _key in subkeys:
            if not _key in self._cfg_from_json[self._product.CONFIG_MODEL_KEY]:
                raise ConfigError(
                    "No " +
                    self._product.CONFIG_MODEL_KEY +
                    ":" +
                    _key +
                    " found")

        # All relative paths are relative to the current directory
        for _key in [self._product.CONFIG_HOST_ROOTPATH_KEY,
                     self._product.CONFIG_HOST_RESULTSDIR_KEY]:
            _value = self._cfg_from_json[_key]
            if not os.path.isabs(_value):
                _abs_path = os.path.abspath(_value)
                if os.path.isfile(_abs_path):
                    raise ConfigError(_key + " is not a directory")
                self._cfg_from_json[_key] = os.path.abspath(_value)

        # Check artifacts paths, relative path is not supported
        # currently hardcoded to ARM as target architecture
        for _arch in self.architectures:
            for _compiler, _artifacts in list(
                    self._artifacts_config[self._product.CONFIG_ARTIFACTS_KEY].items()):
                if (not _compiler.startswith(_arch)) and (not _compiler.startswith(
                        self._product.ARCH_X86) and (not _compiler.startswith(self._product.ARCH_DSP))):
                    continue
                for _artifact_path in _artifacts:
                    if (not os.path.isabs(_artifact_path)
                            ) and os.path.dirname(_artifact_path):
                        raise ConfigError(
                            "{0} does not support relative path".format(
                                self._product.CONFIG_ARTIFACTS_KEY))
                    elif os.path.isabs(_artifact_path) and (not os.path.exists(_artifact_path)):
                        raise ConfigError(
                            "{0} does not exist".format(_artifact_path))
                    elif not os.path.dirname(_artifact_path):
                        if not os.path.exists(self.__default_artifact_path(
                                _compiler, _artifact_path, self._product)):
                            if _artifact_path != "libsymphony-cpu.so":
                                logger.debug(
                                    "Could not find {0} for {1}, path used {2}".format(
                                        _artifact_path, _compiler, self.__default_artifact_path(
                                            _compiler, _artifact_path, self._product)))
        # at the moment, despite devices takes in a list, benchmark does not
        # work correctly when multiple devices are specified
        device_list = self._cfg_from_json.get(
            self._product.CONFIG_DEVICES_KEY, None)
        if len(device_list) == 0 or not device_list[0]:
            raise ConfigError('Benchmark does not have any device specified')
        elif len(device_list) != 1:
            raise ConfigError(
                'Benchmark does not yet support more than 1 device')
        # Measurements allowed are "timing" and "mem"
        if 0 == len(self._cfg_from_json.get(
                self._product.CONFIG_MEASUREMENTS_KEY, None)):
            raise ConfigError('Benchmark does not specify what to measure')
        else:
            for measurement in self._cfg_from_json.get(
                    self._product.CONFIG_MEASUREMENTS_KEY, None):
                if measurement not in [
                        self._product.MEASURE_TIMING, self._product.MEASURE_MEM]:
                    raise ConfigError(
                        '"%s" is unknown measurement' %
                        measurement)

    def __default_artifact_path(self, compiler, artifact, product):
        if product.SDK_ROOT not in os.environ:
            raise ConfigError(
                "Environment variables " +
                product.SDK_ROOT +
                "is not defined, absolute path is needed for " +
                artifact +
                " in " +
                product.BM_ARTIFACTS)
        _sdk_root = os.environ[product.SDK_ROOT]
        _base_name = os.path.basename(artifact)
        if "hexagon" in compiler:
            if self._use_signed_skel:
                return os.path.join(_sdk_root, 'lib', compiler, 'signed', artifact)
            else:
                return os.path.join(_sdk_root, 'lib', compiler, 'unsigned', artifact)
        elif _base_name.endswith(".so") or _base_name.endswith(".dll") or _base_name.startswith("lib"):
            return os.path.join(_sdk_root, 'lib', compiler, artifact)
        else:
            return os.path.join(_sdk_root, 'bin', compiler, artifact)

    def measurement_types_are_valid(self):
        for item in self.measurements:
            if item not in CONFIG_VALID_MEASURMENTS:
                return False
        return True

    @property
    def csvrows(self):
        _csvrows = []
        _csvrows.append([self._product.CONFIG_NAME_KEY] + [self.name])
        _csvrows.append(
            [self._product.CONFIG_HOST_ROOTPATH_KEY] + [self.host_rootpath])
        _csvrows.append(
            [self._product.CONFIG_HOST_RESULTSDIR_KEY] + [self.host_resultspath])
        _csvrows.append(
            [self._product.CONFIG_DEVICE_PATH_KEY] + [self.device_path])
        _csvrows.append([self._product.CONFIG_DEVICES_KEY] +
                        [','.join(self.devices)])
        _csvrows.append([self._product.CONFIG_HOST_NAME_KEY] + [self.hostname])
        _csvrows.append([self._product.CONFIG_GRAPH_PREPARE_TYPE] + [self.GRAPH_PREPARE_TYPE])
        _csvrows.append([self._product.CONFIG_RUNS_KEY] + [self.iterations])
        _csvrows.append([self._product.CONFIG_MODEL_KEY +
                         ":" +
                         self._product.CONFIG_MODEL_NAME_SUBKEY] +
                        [self.dnn_model.name])
        _csvrows.append([self._product.CONFIG_MODEL_KEY +
                         ":" +
                         self._product.CONFIG_MODEL_SUBKEY] +
                        [self.dnn_model.model])
        _csvrows.append([self._product.CONFIG_MODEL_KEY +
                         ":" +
                         self._product.CONFIG_MODEL_INPUTLIST_SUBKEY] +
                        [self.dnn_model.input_list_name])
        for data in self._cfg_from_json[self._product.CONFIG_MODEL_KEY][self._product.CONFIG_MODEL_DATA_SUBKEY]:
            _csvrows.append([self._product.CONFIG_MODEL_KEY + ":" +
                             self._product.CONFIG_MODEL_DATA_SUBKEY] + [data])
        _csvrows.append([self._product.CONFIG_RUNTIMES_KEY] +
                        [','.join(self.runtime_flavors)])
        _csvrows.append([self._product.CONFIG_ARCHITECTURES_KEY] +
                        [','.join(self.architectures)])
        _csvrows.append([self._product.CONFIG_COMPILER_KEY] + [self.compiler])
        _csvrows.append(
            [self._product.CONFIG_STL_LIBRARY_KEY] + [self.stl_library])
        _csvrows.append([self._product.CONFIG_MEASUREMENTS_KEY] +
                        [','.join(self.measurements)])
        _csvrows.append(
            [self._product.CONFIG_PERF_PROFILE_KEY] + [self.perfprofile])
        _csvrows.append(
            [self._product.CONFIG_PROFILING_LEVEL_KEY] + [self.profilinglevel])
        _csvrows.append([self._product.CONFIG_USE_SHARED_BUFFER] + [self.SHARED_BUFFER])
        _csvrows.append(['Date'] + [datetime.datetime.now()])
        return _csvrows

    @property
    def jsonrows(self):
        _jsonrows = {}
        _jsonrows.update({self._product.CONFIG_NAME_KEY: self.name})
        _jsonrows.update(
            {self._product.CONFIG_HOST_ROOTPATH_KEY: self.host_rootpath})
        _jsonrows.update(
            {self._product.CONFIG_HOST_RESULTSDIR_KEY: self.host_resultspath})
        _jsonrows.update(
            {self._product.CONFIG_DEVICE_PATH_KEY: self.device_path})
        _jsonrows.update(
            {self._product.CONFIG_DEVICES_KEY: ','.join(self.devices)})
        _jsonrows.update({self._product.CONFIG_HOST_NAME_KEY: self.hostname})
        _jsonrows.update({self._product.CONFIG_GRAPH_PREPARE_TYPE: self.GRAPH_PREPARE_TYPE})
        _jsonrows.update({self._product.CONFIG_RUNS_KEY: self.iterations})
        _jsonrows.update({self._product.CONFIG_MODEL_KEY + ":" +
                          self._product.CONFIG_MODEL_NAME_SUBKEY: self.dnn_model.name})
        _jsonrows.update({self._product.CONFIG_MODEL_KEY + ":" +
                          self._product.CONFIG_MODEL_SUBKEY: self.dnn_model.model})
        _jsonrows.update(
            {
                self._product.CONFIG_MODEL_KEY +
                ":" +
                self._product.CONFIG_MODEL_INPUTLIST_SUBKEY: self.dnn_model.input_list_name})
        for data in self._cfg_from_json[self._product.CONFIG_MODEL_KEY][self._product.CONFIG_MODEL_DATA_SUBKEY]:
            _jsonrows.update({self._product.CONFIG_MODEL_KEY +
                              ":" + self._product.CONFIG_MODEL_DATA_SUBKEY: data})
        _jsonrows.update(
            {self._product.CONFIG_RUNTIMES_KEY: ','.join(self.runtime_flavors)})
        _jsonrows.update(
            {self._product.CONFIG_ARCHITECTURES_KEY: ','.join(self.architectures)})
        _jsonrows.update({self._product.CONFIG_COMPILER_KEY: self.compiler})
        _jsonrows.update(
            {self._product.CONFIG_STL_LIBRARY_KEY: self.stl_library})
        _jsonrows.update(
            {self._product.CONFIG_MEASUREMENTS_KEY: ','.join(self.measurements)})
        _jsonrows.update(
            {self._product.CONFIG_PERF_PROFILE_KEY: self.perfprofile})
        _jsonrows.update(
            {self._product.CONFIG_PROFILING_LEVEL_KEY: self.profilinglevel})
        _jsonrows.update(
            {self._product.CONFIG_USE_SHARED_BUFFER: self.SHARED_BUFFER})
        _jsonrows.update(
            {'Date': datetime.datetime.now().strftime("%Y-%m-%d")})

        return _jsonrows

    @property
    def name(self):
        return self._cfg_from_json[self._product.CONFIG_NAME_KEY]

    @property
    def host_rootpath(self):
        return self._cfg_from_json[self._product.CONFIG_HOST_ROOTPATH_KEY]

    @property
    def host_resultspath(self):
        return self._cfg_from_json[self._product.CONFIG_HOST_RESULTSDIR_KEY]

    @property
    def devices(self):
        return self._cfg_from_json[self._product.CONFIG_DEVICES_KEY]

    @property
    def hostname(self):
        return self._cfg_from_json[self._product.CONFIG_HOST_NAME_KEY]

    @property
    def device_path(self):
        return self._cfg_from_json[self._product.CONFIG_DEVICE_PATH_KEY]

    @property
    def iterations(self):
        return self._cfg_from_json[self._product.CONFIG_RUNS_KEY]

    @property
    def dnn_model(self):
        return self._dnnmodel

    @property
    def runtimes(self):
        return self._cfg_from_json[self._product.CONFIG_RUNTIMES_KEY]

    @property
    def userbuffer_mode(self):
        return self.userbuffer_mode

    def userbuffer_mode(self, value):
        self.userbuffer_mode = value

    @property
    def perfprofile(self):
        return self._cfg_from_json[self._product.CONFIG_PERF_PROFILE_KEY]

    def perfprofile(self, value):
        self.perfprofile = value

    @property
    def profilinglevel(self):
        return self._cfg_from_json[self._product.CONFIG_PROFILING_LEVEL_KEY]

    def profilinglevel(self, value):
        self.profilinglevel = value

    @property
    def architectures(self):
        return self._architectures

    @property
    def compiler(self):
        return self._compiler

    @property
    def platform(self):
        return self._platform_os

    def set_platform(self, platform_os):
        self._platform_os = platform_os

    @property
    def stl_library(self):
        return self._stl_library

    @property
    def measurements(self):
        return self._cfg_from_json[self._product.CONFIG_MEASUREMENTS_KEY]

    '''
       Support to add optional field "BufferTypes" : ["float","ub_float","ub_tf8"]
       when BufferTypes is not present in json, it runs with defaut behaviour i.e., adds all possible runtimes.
       "BufferTypes" : when given, runs for all given buffer types.
    '''

    def return_valid_run_flavors(self):
        # '' corresponds to no userbuffer mode
        flavors = {i: [] for i in self._product.RUNTIMES}
        # give higher precedence to command line arg
        if self.userbuffer_mode != '':
            if self.userbuffer_mode in self._product.BUFFER_MODES:
                for runtime in flavors:
                    flavors[runtime].append(self.userbuffer_mode)
            else:
                raise ConfigError(
                    'Wrong userbuffer mode {} specified'.format(
                        self.userbuffer_mode))
        else:
            if self.buffertypes is "All":
                for runtime in flavors:
                    flavors[runtime].extend(
                        self._product.RUNTIME_BUFFER_MODES[runtime])
            else:
                if not all(
                        [x in self._product.BUFFER_MODES for x in self.buffertypes]):
                    raise ConfigError(
                        'Wrong buffer mode specified in config file')
                for runtime in flavors:
                    flavors[runtime].extend(self.buffertypes)
        runtimes = []
        runtimes.extend(self.runtimes)
        # TODO figure out a better way to decide whether GPU_s is supported
        if self._product.RUNTIME_GPU in runtimes and \
                any([x in self._artifacts_config[self._product.CONFIG_ARTIFACTS_KEY] for x in self._product.CONFIG_ARTIFACTS_COMPILER_KEY_TARGET_ANDROID_OSPACE]):

            for runtime in flavors:
                flavors[runtime] = [x for x in flavors[runtime]
                                    if x in self._product.RUNTIME_BUFFER_MODES[runtime]]
        return [(i, (lambda:j, lambda:"")[j == "float"]())
                for i in runtimes for j in flavors[i]]

    @property
    def runtime_flavors(self):
        rf = self.return_valid_run_flavors()
        return list(['_'.join(filter(''.__ne__, x)) for x in rf])

    @property
    def host_artifacts(self):

        _host_artifacts = {}
        for _compiler, _artifacts in self._artifacts_config[self._product.CONFIG_ARTIFACTS_KEY].items(
        ):
            if _compiler == self._product.CONFIG_ARTIFACTS_COMPILER_KEY_HOST or \
            _compiler == self._product.CONFIG_ARTIFACTS_COMPILER_KEY_HOST_WIN:
                for _artifact_path in _artifacts:
                    if not os.path.isabs(_artifact_path):
                        _artifact_path = self.__default_artifact_path(
                            _compiler, _artifact_path, self._product)
                    if os.path.exists(_artifact_path):
                        _base_name = os.path.basename(_artifact_path)
                        if _base_name == self._product.DIAGVIEW_EXE or \
                        _base_name == self._product.DIAGVIEW_EXE_WIN:
                            _host_artifacts[self._product.DIAGVIEW_EXE] = _artifact_path

        _host_artifacts[self._product.MODEL_INFO_EXE] = None
        return _host_artifacts

    @property
    def artifacts(self):
        arch_map = {
            'aarch64-android': 'arm64-v8a',
            'aarch64-windows-msvc': 'dummy',
            'aarch64-qnx': 'dummy',
            'aarch64-oe-linux-gcc11.2': 'dummy',
            'aarch64-oe-linux-gcc9.3': 'dummy',
            'aarch64-oe-linux-gcc8.2': 'dummy',
            'aarch64-ubuntu-gcc9.4': 'dummy',
            'aarch64-ubuntu-gcc7.5': 'dummy'
        }
        _tmp = {}
        backend_config_done = False
        for _compiler, _artifacts in self._artifacts_config[self._product.CONFIG_ARTIFACTS_KEY].items(
        ):
            _tmp[_compiler] = []

            if _compiler == self._product.CONFIG_ARTIFACTS_COMPILER_KEY_HOST and self._htp_serialized:
                model = self.dnn_model.model
                backend_path = ''
                output_dir = self.dnn_model.host_dir
                exe = ''
                for _artifact_path in _artifacts:
                    if not os.path.isabs(_artifact_path):
                        _artifact_path = self.__default_artifact_path(_compiler, _artifact_path, self._product)
                    if os.path.exists(_artifact_path):
                        _base_name = os.path.basename(_artifact_path)
                        if _base_name == 'libQnnHtp.so':
                            backend_path = _artifact_path
                        if "qnn-context-binary-generator" in _base_name:
                            exe = _artifact_path

                model_cmd = [exe, '--binary_file', 'qnngraph.serialized', '--model', model, '--backend', backend_path,
                             '--output_dir', '.']

                if self.backend_config:
                    with open(self.backend_config, 'r') as conf:
                        data_conf = json.load(conf)
                        if "backend_extensions" in data_conf.keys():
                            if data_conf['backend_extensions']['config_file_path']:
                                data_conf['backend_extensions']['config_file_path'] = os.path.abspath(data_conf['backend_extensions']['config_file_path'])
                            with open(data_conf['backend_extensions']['config_file_path'], 'r') as be_conf:
                                be_data_conf = json.load(be_conf)
                                if "custom" in be_data_conf:
                                    be_data_conf.pop("custom", None)
                                    tmp_be_cfg_path = os.path.join(os.path.abspath(self.dnn_model.host_dir),'tmp_be_config.json')
                                    with open(tmp_be_cfg_path, 'w') as tmp_conf:
                                        json.dump(be_data_conf, tmp_conf, indent=4)
                                    data_conf['backend_extensions']['config_file_path'] = tmp_be_cfg_path
                            data_conf['backend_extensions']['shared_library_path'] = \
                                self.__default_artifact_path(_compiler, data_conf['backend_extensions']['shared_library_path'], self._product)
                            with open(os.path.join(os.path.abspath(self.dnn_model.host_dir),'tmp_config.json'), 'w') as conf:
                                json.dump(data_conf, conf, indent=4)
                    model_cmd += ['--config_file', os.path.join(os.path.abspath(self.dnn_model.host_dir),'tmp_config.json')]

                logger.info("The model generation command: %s" %(model_cmd))
                check_output(model_cmd, cwd=output_dir).decode()
                _tmp[_compiler].append([os.path.join(output_dir, 'qnngraph.serialized.bin'),
                                        self.dnn_model.device_rootdir])
                continue

            _dev_bin_dir = '/'.join([
                self.device_path,
                self._product.ARTIFACT_DIR,
                _compiler,
                "bin"])
            _dev_lib_dir = '/'.join([
                self.device_path,
                self._product.ARTIFACT_DIR,
                _compiler,
                "lib"])
            _dev_model_dir = '/'.join([
                self.device_path,self.dnn_model.name])
            for _artifact_path in _artifacts:
                if "cdsp0" in _artifact_path:
                    _artifact_path = _artifact_path.replace('cdsp0:', '')
                    _tmp[_compiler].append([_artifact_path, "/mnt/etc/images/cdsp0"])
                    continue
                if "cdsp1" in _artifact_path:
                    _artifact_path = _artifact_path.replace('cdsp1:', '')
                    _tmp[_compiler].append([_artifact_path, "/mnt/etc/images/cdsp1"])
                    continue

                if not os.path.isabs(_artifact_path):
                    _artifact_path = self.__default_artifact_path(
                        _compiler, _artifact_path, self._product)

                if os.path.exists(_artifact_path):
                    _base_name = os.path.basename(_artifact_path)
                    if _base_name.endswith(
                            ".so") or _base_name.endswith(
                            ".dll") or _base_name.startswith("lib"):
                        if "hexagon" in _compiler and ("DSP" in self.runtimes or "DSP_FP16" in self.runtimes):
                            if self._dsp_type is None:
                                logger.error("pass --dsp_type option")
                                return
                            else:
                                device_type = "hexagon-" + str(self._dsp_type)
                                if _compiler == device_type:
                                    if self.device_os_type == self._product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                                        _dev_lib_dir = _dev_lib_dir.replace(_compiler, self.device_os_type)
                                    else:
                                        _dev_lib_dir = _dev_lib_dir.replace(_compiler, 'dsp')
                                    _tmp[_compiler].append([_artifact_path, _dev_lib_dir])
                        _tmp[_compiler].append([_artifact_path, _dev_lib_dir])
                    else:
                        _tmp[_compiler].append([_artifact_path, _dev_bin_dir])

                if _compiler in arch_map:
                    if self.backend_config and not backend_config_done:
                        with open(self.backend_config, 'r') as conf:
                            data_conf = json.load(conf)
                            if "backend_extensions" in data_conf.keys():
                                if data_conf['backend_extensions']['config_file_path']:
                                    conf_file = data_conf['backend_extensions']['config_file_path']
                                    _tmp[_compiler].append([os.path.abspath(conf_file), _dev_lib_dir])
                                    new_config_file_path = ''
                                    if self.device_os_type == self._product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
                                        _tmp[_compiler].append([os.path.abspath(conf_file), _dev_model_dir])
                                        new_config_file_path = os.path.basename(data_conf['backend_extensions']['config_file_path'])
                                        data_conf['backend_extensions']['shared_library_path'] = \
                                        data_conf['backend_extensions']['shared_library_path'].replace('lib', '').replace('.so', '.dll')
                                    else:
                                        new_config_file_path = '/'.join([_dev_lib_dir,
                                        os.path.basename(data_conf['backend_extensions']['config_file_path'])])
                                    data_conf['backend_extensions']['config_file_path'] = new_config_file_path
                                with open(os.path.join(self.dnn_model.host_dir,os.path.basename(self.backend_config)), 'w') as conf:
                                    json.dump(data_conf, conf, indent=4)
                                _tmp[_compiler].append([os.path.join(self.dnn_model.host_dir,os.path.basename(self.backend_config)), _dev_lib_dir])
                                backend_config_done = True
                    if "android" in _compiler:
                        if 'ANDROID_NDK_ROOT' in os.environ:
                            _artifact_path = os.path.join(
                               os.environ['ANDROID_NDK_ROOT'],
                               'sources', 'cxx-stl', 'llvm-libc++', 'libs',
                                arch_map[_compiler],
                                'libc++_shared.so')
                            _tmp[_compiler].append([_artifact_path, _dev_lib_dir])
                        else:
                            logger.error("ANDROID_NDK_ROOT is not set. Exiting")
                            sys.exit(Error.ERRNUM_NOBENCHMARKRAN_ERROR)
            if len(_tmp[_compiler]) == 0:
                del _tmp[_compiler]
        return _tmp

    @property
    def device_artifacts_bin(self):
        raise ConfigError('Deprecated call')

    @property
    def device_artifacts_lib(self):
        raise ConfigError('Deprecated call')

    def get_device_artifacts_bin(self, runtime):
        return '/'.join([self.__device_artifacts_helper(runtime), 'bin'])

    def get_device_artifacts_lib(self, runtime):
        return '/'.join([self.__device_artifacts_helper(runtime), 'lib'])

    def get_exe_name(self):
        return self._product.BATCHRUN_EXE

    def __device_artifacts_helper(self, runtime):
        '''
        Given a runtime (CPU, DSP, GPU, etc), return the artifacts folder

        Note that the logic is as simple as:
            1. There should be at least one artifacts folder.
            2. If you are asking for GPU_S, there needs to be 2
            3. GPU_S will return the one ending with "_s", all others will get the first one
        '''
        if self.device_os_type and self.device_os_type == self._product.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64:
            anchor = self._product.RUNTIME_LIB_WIN
        else:
            anchor = self._product.RUNTIME_LIB
        for _compiler, _artifacts in self._artifacts_config[self._product.CONFIG_ARTIFACTS_KEY].items(
        ):
            if anchor in _artifacts:
                return '/'.join([
                    self.device_path, self._product.ARTIFACT_DIR, _compiler])
        raise ConfigError('Unable to find device artifacts')

    def __str__(self):
        input_type = ('  %s:%s\n' % (self._product.CONFIG_MODEL_KEY + ":" + self._product.CONFIG_MODEL_DATA_SUBKEY,
                                     self._cfg_from_json[self._product.CONFIG_MODEL_KEY][self._product.CONFIG_MODEL_DATA_SUBKEY]))
        return ("\n--CONFIG--\n" +
                ('  %s:%s \n' % (self._product.CONFIG_NAME_KEY, self.name)) +
                ('  %s:%s \n' % (self._product.CONFIG_HOST_ROOTPATH_KEY, self.host_rootpath)) +
                ('  %s:%s \n' % (self._product.CONFIG_HOST_RESULTSDIR_KEY, self.host_resultspath)) +
                ('  %s:%s \n' % (self._product.CONFIG_DEVICES_KEY, self.devices)) +
                ('  %s:%s\n' % (self._product.CONFIG_DEVICE_PATH_KEY, self.device_path)) +
                ('  %s:%s\n' % (self._product.CONFIG_HOST_NAME_KEY, self.hostname)) +
                ('  %s:%s\n' % (self._product.CONFIG_RUNS_KEY, self.iterations)) +
                ('  %s:%s\n' % (self._product.CONFIG_MODEL_KEY + ":" + self._product.CONFIG_MODEL_NAME_SUBKEY, self._dnnmodel.name)) +
                ('  %s:%s\n' % (self._product.CONFIG_MODEL_KEY + ":" + self._product.CONFIG_MODEL_SUBKEY, self._dnnmodel.model)) +
                input_type +
                ('  %s:%s\n' % (self._product.CONFIG_RUNTIMES_KEY, self.runtime_flavors)) +
                ('  %s:%s\n' % (self._product.CONFIG_ARCHITECTURES_KEY, self.architectures)) +
                ('  %s:%s\n' % (self._product.CONFIG_COMPILER_KEY, self.compiler)) +
                ('  %s:%s\n' % (self._product.CONFIG_STL_LIBRARY_KEY, self.stl_library)) +
                ('  %s:%s\n' % (self._product.CONFIG_MEASUREMENTS_KEY, self.measurements)) +
                ('  %s:%s\n' % (self._product.CONFIG_PERF_PROFILE_KEY, self.perfprofile)) +
                ('  %s:%s\n' % (self._product.CONFIG_PROFILING_LEVEL_KEY, self.profilinglevel)) +
                "--END CONFIG--\n")


class ConfigFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def make_config(cfgfile, outputbasedir, devicelist, hostname,
                    deviceostype, userbuffer_mode, backend_config,
                    perfprofile, profilinglevel, iterations, runtimes, product,
                    dsp_type, htp_serialized, arm_prepare, use_signed_skel,cache, use_shared_buffer,
                    discard_output, test_duration):
        _config_from_json = load_json(cfgfile)
        if hostname is None:
            try:
                hostname = _config_from_json[product.CONFIG_HOST_NAME_KEY]
            except KeyError:
                pass
        if runtimes is not None:
            _config_from_json[product.CONFIG_RUNTIMES_KEY] = runtimes
        if not devicelist:
            devicelist = _config_from_json['Devices']

        if 'MODELZOO' in os.environ:
            model_path = os.path.join(
                os.environ['MODELZOO'], _config_from_json[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_SUBKEY])
        else:
            model_path = _config_from_json[product.CONFIG_MODEL_KEY][product.CONFIG_MODEL_SUBKEY]
        if not os.path.exists(model_path):
            logger.error(model_path + " doesn't exist")
            return
        else:
            return Config(cfgfile, _config_from_json, outputbasedir, devicelist, hostname, deviceostype,
                          userbuffer_mode, backend_config, perfprofile, iterations,
                          profilinglevel, product, dsp_type, htp_serialized, arm_prepare, use_signed_skel,
                          cache, use_shared_buffer, discard_output, test_duration)
