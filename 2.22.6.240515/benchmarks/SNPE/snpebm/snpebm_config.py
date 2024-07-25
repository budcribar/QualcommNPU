#==============================================================================
#
#  Copyright (c) 2016-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from __future__ import absolute_import
import json
import os
import shutil
from .snpebm_jsonkeys import *
from .snpebm_config_restrictions import *
import datetime
from subprocess import check_output
import numpy as np
from common_utils  import protocol as Adb
from common_utils.exceptions import ConfigError, AdbShellCmdFailedException
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
    def __init__(self, config, host_artifacts):
        self._name = config[CONFIG_MODEL_KEY][CONFIG_MODEL_NAME_SUBKEY]
        self._dlc = DnnModel.__default_path(config[CONFIG_MODEL_KEY][CONFIG_MODEL_DLC_SUBKEY])
        if CONFIG_MODEL_UDO_SUBKEY in config[CONFIG_MODEL_KEY].keys():
            self._udo = self.__default_path(config[CONFIG_MODEL_KEY][CONFIG_MODEL_UDO_SUBKEY])
        else:
            self._udo = None
        self._dev_root_dir = os.path.join(config[CONFIG_DEVICE_PATH_KEY], self._name)
        self._host_artifacts = host_artifacts
        subkeys = []
        for sub_key in config[CONFIG_MODEL_KEY]:
            subkeys.append(sub_key)
        if "InputList" in subkeys:
            self._input_list_name = os.path.basename(config[CONFIG_MODEL_KEY][CONFIG_MODEL_INPUTLIST_SUBKEY])
            self._artifacts = []
            self._artifacts.append([DnnModel.__default_path(config[CONFIG_MODEL_KEY][CONFIG_MODEL_INPUTLIST_SUBKEY]),
                                    self._dev_root_dir])
            for data in config[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]:
                _abs_data_path = DnnModel.__default_path(data)
                self._artifacts.append([_abs_data_path, os.path.join(self._dev_root_dir, os.path.basename(data))])
        elif "RandomInput" in subkeys:
            # Get the input dimensions and generate random data
            input_layer_dim = self.input_layers
            num_inputs = config[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]
            if num_inputs < 1:
                raise ConfigError('Input can not be less than 1')

            random_inputs_dir = os.path.join(config[CONFIG_HOST_ROOTPATH_KEY], 'random_inputs')
            _abs_random_inputs_path = os.path.abspath(random_inputs_dir)
            if os.path.isdir(_abs_random_inputs_path):
                shutil.rmtree(_abs_random_inputs_path)
            os.makedirs(_abs_random_inputs_path)
            self._input_list_name = "random_raw_list.txt"
            input_file_path = os.path.join(_abs_random_inputs_path, self._input_list_name)
            input_file = open(input_file_path, 'w')

            rand_idx = 0
            for i in range (1, num_inputs+1):
                for inp in range(len(input_layer_dim)):
                    rand_idx += 1
                    random_data_dim_str = input_layer_dim[inp].split(":")[-1]
                    random_data_dim = random_data_dim_str.split(",")
                    random_data_dim = list(map(int, random_data_dim))

                    raw_filepath = os.path.join(_abs_random_inputs_path, 'random_input_' + str(rand_idx) + '.raw')
                    rand_raw = np.random.uniform(-1.0, +1.0, random_data_dim).astype(np.float32)
                    with open(raw_filepath, 'wb') as fid:
                        fid.write(rand_raw)
                    raw_rel_path = os.path.join(os.path.basename(random_inputs_dir), os.path.basename(raw_filepath))
                    input_file.write(raw_rel_path + " ")
                input_file.write("\n")
            input_file.close()
            self._artifacts = []
            self._artifacts.append([input_file_path, self._dev_root_dir])
            self._artifacts.append([_abs_random_inputs_path, os.path.join(self._dev_root_dir, os.path.basename(_abs_random_inputs_path))])

        self._artifacts.append([self._dlc, self._dev_root_dir])

    @staticmethod
    def __default_path(artifact_path):
        _abs_path = artifact_path
        if '.dlc' in _abs_path or "udo" in artifact_path.lower():
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
    def dlc(self):
        return self._dlc

    @property
    def udo(self):
        return self._udo

    @property
    def device_rootdir(self):
        return self._dev_root_dir

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def input_layers(self):
        dlc_cmd = [self._host_artifacts[SNPE_DLC_INFO_EXE], '-i', self._dlc]
        try:
            dlc_info_output = check_output(dlc_cmd).decode()
        except Exception as de:
            logger.warning("Failed to parse {0}".format(self._dlc))
            logger.warning(repr(de))
            return []

        inputs = []
        found = False
        count = 0
        for line in dlc_info_output.split('\n'):
            if "Input Name" in line:
                found = True
            elif found and "--------" in line:
                count = count + 1
                if count >= 2:
                    break;
            elif found:
                split_line = line.replace(" ", "").split("|")
                layer_name = split_line[1]
                input_dimensions = split_line[2]
                inputs.append("{}:{}".format(layer_name, input_dimensions))
        return inputs


class Config(object):
    def __init__(self, cfg_file, cfg_from_json, outputbasedir, devicelist, hostname, deviceostype, userbuffer_mode, perfprofile, profilinglevel, enable_init_caching):
        config_prefix = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
        self._cfg_from_json = cfg_from_json
        self.pt = [s for s in os.listdir(os.path.join(os.environ['SNPE_ROOT'], "lib")) if "hexagon" in s]
        if deviceostype and deviceostype not in CONFIG_VALID_DEVICEOSTYPES:
            raise ConfigError('Device OS Type not valid.  Only specify one of %s'%CONFIG_VALID_DEVICEOSTYPES)
        elif deviceostype == CONFIG_DEVICEOSTYPES_LE64_GCC53:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC53
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_LE64_GCC53_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_ANDROID_AARCH64:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_ANDROID
            self._stl_library = STL_LIBCXX_SHARED
            if (len(self.pt)>0):
                self._artifacts_config = SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS.replace(SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS[SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS.rfind("-"):],"")
                self._compiler = ""
            else:
                self._artifacts_config = SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS
                self._compiler = COMPILER_CLANG60
        elif deviceostype == CONFIG_DEVICEOSTYPES_LE_OE_GCC82:
            self._architectures = [ARCH_ARM]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC82
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_LE_OE_GCC82_HF_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_LE64_OE_GCC82:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC82
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_LE64_OE_GCC82_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_QNX64_GCC54:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_QNX
            self._compiler = COMPILER_GCC54
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_QNX64_GCC54_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_QNX64_GCC83:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_QNX
            self._compiler = COMPILER_GCC83
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_QNX64_GCC83_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_GCC75:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC75
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_AARCH64_UBUNTU_GCC75_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_AARCH64_OE_LINUX_GCC93:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC93
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_AARCH64_OE_LINUX_GCC93_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_AARCH64_OE_LINUX_GCC112:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC112
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_AARCH64_OE_LINUX_GCC112_ARTIFACTS
        elif deviceostype == CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_GCC94:
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_LINUX
            self._compiler = COMPILER_GCC94
            self._stl_library = None
            self._artifacts_config = SNPE_BENCH_AARCH64_UBUNTU_GCC94_ARTIFACTS
        elif deviceostype:
            # presumed to be Android aarch64
            self._architectures = [ARCH_AARCH64]
            self._platform_os = PLATFORM_OS_ANDROID
            self._stl_library = STL_LIBCXX_SHARED
            if (len(self.pt)>0):
                self._artifacts_config = SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS.replace(SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS[SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS.rfind("-"):],"")
                self._compiler = ""
            else:
                self._artifacts_config = SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS
                self._compiler = COMPILER_CLANG60
        self.__override_cfgfile__(outputbasedir, devicelist, hostname)
        self.__quick_verify__()
        self._dnnmodel = DnnModel(self._cfg_from_json, self.host_artifacts)

        self.userbuffer_mode = userbuffer_mode
        self.enable_init_caching = enable_init_caching
        try:
            self._hostname = self._cfg_from_json[CONFIG_HOST_NAME_KEY]
        except KeyError:
            self._cfg_from_json[CONFIG_HOST_NAME_KEY] = 'localhost'

        try:
            self.cpu_fallback = self._cfg_from_json[CONFIG_CPU_FALLBACK_KEY]
            if not isinstance(self.cpu_fallback, bool):
                raise ConfigError(CONFIG_CPU_FALLBACK_KEY + " key in " + cfg_file + " expects boolean type. Found " + str(type(self.cpu_fallback)))

        except KeyError:
            self.cpu_fallback = False

        try:
            self.platform_option = self._cfg_from_json[CONFIG_PLATFORM_OPTIONS_KEY]
            if not isinstance(self.platform_option, list):
                raise ConfigError(CONFIG_PLATFORM_OPTIONS_KEY + " key in " + cfg_file + " expects list type. Found " + str(type(self.platform_option)))

        except KeyError:
            self.platform_option = []

        try:
            self.ubfloat_output = self._cfg_from_json[CONFIG_USERBUFFER_FLOAT_OUTPUT_KEY]
            if not isinstance(self.ubfloat_output, bool):
                raise ConfigError(CONFIG_USERBUFFER_FLOAT_OUTPUT_KEY + " key in " + cfg_file +
                                  " expects boolean type. Found " + str(type(self.ubfloat_output)))
        except KeyError:
            self.ubfloat_output = False

        try:
            if not self._cfg_from_json[CONFIG_PERF_PROFILE_KEY] is 'null':
                self.perfprofile = self._cfg_from_json[CONFIG_PERF_PROFILE_KEY]
        except KeyError as ke:
            self.perfprofile = perfprofile

        try:
            if not self._cfg_from_json[CONFIG_PROFILING_LEVEL_KEY] is 'null':
                self.profilinglevel = self._cfg_from_json[CONFIG_PROFILING_LEVEL_KEY]
        except KeyError as ke:
            self.profilinglevel = profilinglevel

        try:
            self.buffertypes = self._cfg_from_json[CONFIG_BUFFERTYPES_KEY]
        except KeyError:
            self.buffertypes = "All"

        # Required to set it to default 'high_performance' in case
        # perf_profile is not provided as runtime argument or in the config file
        if self.perfprofile == '' and perfprofile == '':
            self.perfprofile = 'high_performance'
        # Required to give preference to runtime argument in case
        # perf_profile is provided both as runtime argument and in the config file
        elif perfprofile != '':
            self.perfprofile = perfprofile

        if self.profilinglevel == '' and profilinglevel == '':
            self.profilinglevel = 'basic'
        elif profilinglevel != '':
            self.profilinglevel = profilinglevel


    def __override_cfgfile__(self, outputbasedir, devicelist, hostname):
        # Override output base dir if the host paths are relative paths
        # after override, it will become an absolute path
        if outputbasedir is not None:
            logger.info('Overriding output base dir to %s, instead of %s' % (outputbasedir, os.getcwd()))
            for _key in [CONFIG_HOST_ROOTPATH_KEY, CONFIG_HOST_RESULTSDIR_KEY]:
                _value = self._cfg_from_json[_key]
                if not os.path.isabs(_value):
                    _abs_path = os.path.abspath(outputbasedir+'/'+_value)
                    self._cfg_from_json[_key] = _abs_path
        # Override device id if one's supplied
        if devicelist is not None:
            _prevDevices = self._cfg_from_json.get(CONFIG_DEVICES_KEY, None)
            self._cfg_from_json[CONFIG_DEVICES_KEY] = devicelist
            logger.info('Overriding device id to %s, instead of %s from config file' % (self._cfg_from_json[CONFIG_DEVICES_KEY], _prevDevices))
        # Override Host Name is one's supplied
        if hostname is not None:
            self._cfg_from_json[CONFIG_HOST_NAME_KEY] = hostname
            logger.info('Overriding host name to %s' % (self._cfg_from_json[CONFIG_HOST_NAME_KEY]))

    def __quick_verify__(self):
        try:
            # Check that we have at least all top level keys
            # note that this will give an exception on any key that
            # isn't present
            # NOTE: We do not validate the values provided for that key
            for _key in CONFIG_JSON_ROOTKEYS:
                # check optional keys
                if not _key is CONFIG_PERF_PROFILE_KEY and not _key is CONFIG_CPU_FALLBACK_KEY and not _key is CONFIG_HOST_NAME_KEY and \
                        not _key is CONFIG_BUFFERTYPES_KEY and not _key is CONFIG_PROFILING_LEVEL_KEY and not _key is CONFIG_USERBUFFER_FLOAT_OUTPUT_KEY and \
                        not _key is CONFIG_PLATFORM_OPTIONS_KEY:
                    if self._cfg_from_json[_key] is 'null':
                        raise ConfigError("Missing value for " + _key)
            # Check for no foreign top level keys
            for _key in self._cfg_from_json.keys():
                if _key not in CONFIG_JSON_ROOTKEYS:
                    raise ConfigError("Found invalid top level key: " + _key)
        except KeyError as ke:
            raise ConfigError('Missing key in config file: ' + repr(ke))
        # We have all top level keys.  Check some of the sub keys
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            subkeys = CONFIG_JSON_MODEL_COMMON_SUBKEYS + CONFIG_JSON_MODEL_RANDOM_INPUT_SUBKEYS
            for _key in subkeys:
                if not _key in self._cfg_from_json[CONFIG_MODEL_KEY]:
                    raise ConfigError("No " + CONFIG_MODEL_KEY + ":" + _key + " found")
        else:
            subkeys = CONFIG_JSON_MODEL_COMMON_SUBKEYS + CONFIG_JSON_MODEL_DEFINED_INPUT_SUBKEYS
            for _key in subkeys:
                if not _key in self._cfg_from_json[CONFIG_MODEL_KEY]:
                    raise ConfigError("No " + CONFIG_MODEL_KEY + ":" + _key + " found")

        # All relative paths are relative to the current directory
        for _key in [CONFIG_HOST_ROOTPATH_KEY, CONFIG_HOST_RESULTSDIR_KEY]:
            _value = self._cfg_from_json[_key]
            if not os.path.isabs(_value):
                _abs_path = os.path.abspath(_value)
                if os.path.isfile(_abs_path):
                    raise ConfigError(_key + " is not a directory")
                self._cfg_from_json[_key] = os.path.abspath(_value)

        # at the momemnt, despite devices takes in a list, benchmark does not work correctly when multiple devices are specified
        device_list = self._cfg_from_json.get(CONFIG_DEVICES_KEY, None)
        if len(device_list) == 0 or not device_list[0]:
            raise ConfigError('Benchmark does not have any device specified')
        elif len(device_list) != 1:
            raise ConfigError('Benchmark does not yet support more than 1 device')

        # Measurements allowed are "timing" and "mem"
        if  0 == len(self._cfg_from_json.get(CONFIG_MEASUREMENTS_KEY, None)):
            raise ConfigError('Benchmark does not specify what to measure')
        else:
            for measurement in self._cfg_from_json.get(CONFIG_MEASUREMENTS_KEY, None):
                if measurement not in [MEASURE_TIMING, MEASURE_MEM]:
                    raise ConfigError('"%s" is unknown measurement'%measurement)


    @staticmethod
    def __default_artifact_path(compiler):
        artifacts = {}
        # first check for ZDL_ROOT, because if you have both, contributor role takes precedence
        if ZDL_ROOT not in os.environ:
            if SNPE_SDK_ROOT not in os.environ:
                raise ConfigError("Environment variables 'SNPE_ROOT' and 'ZDL_ROOT' are not defined, absolute path is needed for " + artifact + " in snpebm_artifacts.json")
            _sdk_root = os.environ[SNPE_SDK_ROOT]
            if os.path.exists(os.path.join(_sdk_root, "lib", compiler)):
                artifacts['lib'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_sdk_root, "lib", compiler)), x),os.listdir(os.path.join(_sdk_root, "lib", compiler))))
            if os.path.exists(os.path.join(_sdk_root, "bin", compiler)):
                artifacts['bin'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_sdk_root, "bin", compiler)), x),os.listdir(os.path.join(_sdk_root, "bin", compiler))))
        else:
            _zdl_root = os.environ[ZDL_ROOT]
            if os.path.exists(os.path.join(_zdl_root, compiler, "lib")):
                artifacts['lib'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_zdl_root, compiler, "lib")), x),os.listdir(os.path.join(_zdl_root, compiler, "lib"))))
            if os.path.exists(os.path.join(_zdl_root, compiler, "bin")):
                artifacts['bin'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_zdl_root, compiler, "bin")), x),os.listdir(os.path.join(_zdl_root, compiler, "bin"))))
        return artifacts

    def measurement_types_are_valid(self):
        for item in self.measurements:
            if item not in CONFIG_VALID_MEASURMENTS:
                return False
        return True


    @property
    def csvrows(self):
        _csvrows = []
        _csvrows.append([CONFIG_NAME_KEY] + [self.name])
        _csvrows.append([CONFIG_HOST_ROOTPATH_KEY] + [self.host_rootpath])
        _csvrows.append([CONFIG_HOST_RESULTSDIR_KEY] + [self.host_resultspath])
        _csvrows.append([CONFIG_DEVICE_PATH_KEY] + [self.device_path])
        _csvrows.append([CONFIG_DEVICES_KEY] + [','.join(self.devices)])
        _csvrows.append([CONFIG_HOST_NAME_KEY] + [self.hostname])
        _csvrows.append([CONFIG_RUNS_KEY] + [self.iterations])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_NAME_SUBKEY] + [self.dnn_model.name])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DLC_SUBKEY] + [self.dnn_model.dlc])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_UDO_SUBKEY] + [self.dnn_model.udo])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTLIST_SUBKEY] + [self.dnn_model.input_list_name])
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_RANDOMINPUT_SUBKEY] + [self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]])
        else:
            for data in self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]:
                _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DATA_SUBKEY] + [data])
        _csvrows.append([CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTS] + [','.join(self.dnn_model.input_layers)])
        _csvrows.append([CONFIG_RUNTIMES_KEY] + [','.join(self.runtime_flavors)])
        _csvrows.append([CONFIG_ARCHITECTURES_KEY] + [','.join(self.architectures)])
        _csvrows.append([CONFIG_COMPILER_KEY] + [self.compiler])
        _csvrows.append([CONFIG_STL_LIBRARY_KEY] + [self.stl_library])
        _csvrows.append([CONFIG_MEASUREMENTS_KEY] + [','.join(self.measurements)])
        _csvrows.append([CONFIG_PERF_PROFILE_KEY] + [self.perfprofile])
        _csvrows.append([CONFIG_PROFILING_LEVEL_KEY] + [self.profilinglevel])
        _csvrows.append(['Date'] + [datetime.datetime.now()])

        return _csvrows

    @property
    def jsonrows(self):
        _jsonrows = {}
        _jsonrows.update({CONFIG_NAME_KEY: self.name})
        _jsonrows.update({CONFIG_HOST_ROOTPATH_KEY: self.host_rootpath})
        _jsonrows.update({CONFIG_HOST_RESULTSDIR_KEY: self.host_resultspath})
        _jsonrows.update({CONFIG_DEVICE_PATH_KEY: self.device_path})
        _jsonrows.update({CONFIG_DEVICES_KEY: ','.join(self.devices)})
        _jsonrows.update({CONFIG_HOST_NAME_KEY: self.hostname})
        _jsonrows.update({CONFIG_RUNS_KEY: self.iterations})
        _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_NAME_SUBKEY: self.dnn_model.name})
        _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DLC_SUBKEY: self.dnn_model.dlc})
        _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_UDO_SUBKEY: self.dnn_model.udo})
        _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTLIST_SUBKEY: self.dnn_model.input_list_name})
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_RANDOMINPUT_SUBKEY: self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]})
        else:
            for data in self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]:
                _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DATA_SUBKEY: data})
        _jsonrows.update({CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTS: ','.join(self.dnn_model.input_layers)})
        _jsonrows.update({CONFIG_RUNTIMES_KEY:  ','.join(self.runtime_flavors)})
        _jsonrows.update({CONFIG_ARCHITECTURES_KEY: ','.join(self.architectures)})
        _jsonrows.update({CONFIG_COMPILER_KEY: self.compiler})
        _jsonrows.update({CONFIG_STL_LIBRARY_KEY: self.stl_library})
        _jsonrows.update({CONFIG_MEASUREMENTS_KEY: ','.join(self.measurements)})
        _jsonrows.update({CONFIG_PERF_PROFILE_KEY: self.perfprofile})
        _jsonrows.update({CONFIG_PROFILING_LEVEL_KEY : self.profilinglevel})
        _jsonrows.update({'Date': datetime.datetime.now().strftime("%Y-%m-%d")})

        return _jsonrows

    @property
    def name(self):
        return self._cfg_from_json[CONFIG_NAME_KEY]

    @property
    def host_rootpath(self):
        return self._cfg_from_json[CONFIG_HOST_ROOTPATH_KEY]

    @property
    def host_resultspath(self):
        return self._cfg_from_json[CONFIG_HOST_RESULTSDIR_KEY]

    @property
    def devices(self):
        return self._cfg_from_json[CONFIG_DEVICES_KEY]

    @property
    def hostname(self):
        return self._cfg_from_json[CONFIG_HOST_NAME_KEY]

    @property
    def device_path(self):
        return self._cfg_from_json[CONFIG_DEVICE_PATH_KEY]

    @property
    def iterations(self):
        return self._cfg_from_json[CONFIG_RUNS_KEY]

    @property
    def dnn_model(self):
        return self._dnnmodel

    @property
    def runtimes(self):
        return self._cfg_from_json[CONFIG_RUNTIMES_KEY]

    @property
    def userbuffer_mode(self):
        return self.userbuffer_mode

    def userbuffer_mode(self, value):
        self.userbuffer_mode = value

    @property
    def perfprofile(self):
        return self._cfg_from_json[CONFIG_PERF_PROFILE_KEY]

    def perfprofile(self, value):
        self.perfprofile = value

    @property
    def profilinglevel(self):
        return self._cfg_from_json[CONFIG_PROFILING_LEVEL_KEY]
    def profilinglevel(self, value):
        self.profilinglevel = value

    @property
    def cpu_fallback(self):
        return self._cfg_from_json[CONFIG_CPU_FALLBACK_KEY]

    def cpu_fallback(self, value):
        self.cpu_fallback = value

    @property
    def ubfloat_output(self):
        return self._cfg_from_json[CONFIG_USERBUFFER_FLOAT_OUTPUT_KEY]

    def ubfloat_output(self, value):
        self.ubfloat_output = value

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
        return self._cfg_from_json[CONFIG_MEASUREMENTS_KEY]

    '''
       Support to add optional field "BufferTypes" : ["float","ub_float","ub_tf8"]
       when BufferTypes is not present in json, it runs with defaut behaviour i.e., adds all possible runtimes.
       "BufferTypes" : when given, runs for all given buffer types.
    '''
    def return_valid_run_flavors(self):
        # '' corresponds to no userbuffer mode
        flavors = {i: [] for i in RUNTIMES}
        # give higher precedence to command line arg
        if self.userbuffer_mode != '':
            if self.userbuffer_mode in BUFFER_MODES:
                for runtime in flavors:
                    flavors[runtime].append(self.userbuffer_mode)
            else:
                raise ConfigError('Wrong userbuffer mode {} specified'.format(self.userbuffer_mode))
        else:
            if self.buffertypes is "All":
                for runtime in flavors:
                    flavors[runtime].extend(RUNTIME_BUFFER_MODES[runtime])
            else:
                if not all([x in BUFFER_MODES for x in self.buffertypes]):
                    raise ConfigError('Wrong buffer mode specified in config file')
                for runtime in flavors:
                    flavors[runtime].extend(self.buffertypes)
        runtimes = []
        runtimes.extend(self.runtimes)
        return [(i, (lambda:j, lambda:"")[j == "float"]()) for i in runtimes for j in flavors[i]]

    @property
    def runtime_flavors(self):
        rf = self.return_valid_run_flavors()
        return list(['_'.join(filter(''.__ne__, x)) for x in rf])

    @property
    def host_artifacts(self):
        _host_artifacts = {}
        _compiler = CONFIG_ARTIFACTS_COMPILER_KEY_HOST
        _artifacts =  self.__default_artifact_path(_compiler)
        for _artifact_path in _artifacts['bin']:
            if os.path.exists(_artifact_path):
                _base_name = os.path.basename(_artifact_path)
                if _base_name == SNPE_DLC_INFO_EXE:
                    _host_artifacts[SNPE_DLC_INFO_EXE] = _artifact_path
                elif _base_name == SNPE_DIAGVIEW_EXE:
                    _host_artifacts[SNPE_DIAGVIEW_EXE] = _artifact_path
        return _host_artifacts

    @property
    def artifacts(self):
        if (len(self.pt)>0):
            _udo_map = {
                "aarch64-android": "arm64-v8a",
                "dsp": "dsp/signed"
            }
        else:
            _udo_map = {
                "aarch64-android-clang6.0": "arm64-v8a",
                "dsp": "dsp/signed"
            }
        _compiler = self._artifacts_config
        _artifacts = self.__default_artifact_path(_compiler)
        if (len(self.pt)>0):
            for ptl in range(0,len(self.pt)):
                if 'dsp' in _artifacts.keys():
                    for p in (self.__default_artifact_path(self.pt[ptl]+'/unsigned'))['lib']:
                        _artifacts['dsp'].append(p)
                else:
                    _artifacts['dsp'] = (self.__default_artifact_path(self.pt[ptl]+'/unsigned'))['lib']
        else:
            _artifacts['dsp'] = (self.__default_artifact_path('dsp'))['lib']
        _tmp = {_compiler:[]}
        _dev_bin_dir = os.path.join(self.device_path, ARTIFACT_DIR, _compiler, "bin")
        _dev_lib_dir = os.path.join(self.device_path, ARTIFACT_DIR, _compiler, "lib")
        for key, _artifact_paths in _artifacts.items():
            for _artifact_path in _artifact_paths:
                if os.path.exists(_artifact_path):
                    _base_name = os.path.basename(_artifact_path)
                    if _base_name.endswith(".so") or _base_name.startswith("lib"):
                        _tmp[_compiler].append([_artifact_path, _dev_lib_dir])
                    elif _base_name =='snpe-net-run':
                        _tmp[_compiler].append([_artifact_path, _dev_bin_dir])
        if self._dnnmodel.udo:
            if  _compiler in _udo_map.keys():
                _udo_lib_name = _udo_map[_compiler]
            else:
                _udo_lib_name = _compiler
            _tmp[_compiler].append([os.path.join(self._dnnmodel.udo,
                                                 "libs",
                                                 _udo_lib_name),
                                    _dev_lib_dir])
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
        return os.path.join(self.__device_artifacts_helper(runtime), 'bin')

    def get_device_artifacts_lib(self, runtime):
        return os.path.join(self.__device_artifacts_helper(runtime), 'lib')

    def get_exe_name(self, runtime):
        return SNPE_BATCHRUN_EXE

    def __device_artifacts_helper(self, runtime):
        '''
        Given a runtime (CPU, DSP, GPU, etc), return the artifacts folder

        Note that the logic is as simple as:
            1. There should be at least one artifacts folder.
            2. If you are asking for GPU_S, there needs to be 2
            3. GPU_S will return the one ending with "_s", all others will get the first one
        '''
        anchor = SNPE_RUNTIME_LIB
        _artifacts = self.__default_artifact_path(self._artifacts_config)
        for _artifact in _artifacts['lib']:
            if _artifact.endswith(anchor):
                return os.path.join(self.device_path, ARTIFACT_DIR, self._artifacts_config)
        raise ConfigError('Unable to find device artifacts')

    def __str__(self):
        if 'RandomInput' in self._cfg_from_json[CONFIG_MODEL_KEY]:
            input_type = ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_RANDOMINPUT_SUBKEY, self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_RANDOMINPUT_SUBKEY]))
        else:
            input_type = ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DATA_SUBKEY, self._cfg_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DATA_SUBKEY]))
        return ("\n--CONFIG--\n" +
                ('  %s:%s \n' % (CONFIG_NAME_KEY, self.name)) +
                ('  %s:%s \n' % (CONFIG_HOST_ROOTPATH_KEY, self.host_rootpath)) +
                ('  %s:%s \n' % (CONFIG_HOST_RESULTSDIR_KEY, self.host_resultspath)) +
                ('  %s:%s \n' % (CONFIG_DEVICES_KEY, self.devices)) +
                ('  %s:%s\n' % (CONFIG_DEVICE_PATH_KEY, self.device_path)) +
                ('  %s:%s\n' % (CONFIG_HOST_NAME_KEY, self.hostname)) +
                ('  %s:%s\n' % (CONFIG_RUNS_KEY, self.iterations)) +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_NAME_SUBKEY , self._dnnmodel.name)) +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_DLC_SUBKEY , self._dnnmodel.dlc)) +
                input_type +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_UDO_SUBKEY, self._dnnmodel.udo)) +
                ('  %s:%s\n' % (CONFIG_MODEL_KEY + ":" + CONFIG_MODEL_INPUTS, self._dnnmodel.input_layers)) +
                ('  %s:%s\n' % (CONFIG_RUNTIMES_KEY, self.runtime_flavors)) +
                ('  %s:%s\n' % (CONFIG_ARCHITECTURES_KEY, self.architectures)) +
                ('  %s:%s\n' % (CONFIG_COMPILER_KEY, self.compiler)) +
                ('  %s:%s\n' % (CONFIG_STL_LIBRARY_KEY, self.stl_library)) +
                ('  %s:%s\n' % (CONFIG_MEASUREMENTS_KEY, self.measurements)) +
                ('  %s:%s\n' % (CONFIG_PERF_PROFILE_KEY, self.perfprofile)) +
                ('  %s:%s\n' % (CONFIG_PROFILING_LEVEL_KEY, self.profilinglevel)) +
                ('  %s:%s\n' % (CONFIG_CPU_FALLBACK_KEY, self.cpu_fallback)) +
                ('  %s:%s\n' % (CONFIG_PLATFORM_OPTIONS_KEY, self.platform_option)) +
                ('  %s:%s\n' % (CONFIG_USERBUFFER_FLOAT_OUTPUT_KEY, self.ubfloat_output)) +
                ('  %s:%s\n' % (CONFIG_BUFFERTYPES_KEY, self.buffertypes)) +
                "--END CONFIG--\n")


class ConfigFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def make_config(cfgfile, outputbasedir, devicelist, hostname, usedevicesconnected, deviceostype, userbuffer_mode, perfprofile, profilinglevel, enable_init_caching):
        _config_from_json = load_json(cfgfile)
        if hostname is None:
            try:
                hostname = _config_from_json[CONFIG_HOST_NAME_KEY]
            except KeyError:
                pass
        if usedevicesconnected:
            ret, devicelist, err = Adb('adb', '', hostname=hostname).get_devices()
            if ret != 0:
                logger.error('Failed to get device list')
                raise AdbShellCmdFailedException(err)
        elif not devicelist:
            devicelist = _config_from_json['Devices']

        if 'MODELZOO' in os.environ:
            dlc_path = os.path.join(os.environ['MODELZOO'], _config_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DLC_SUBKEY])
            if CONFIG_MODEL_UDO_SUBKEY in _config_from_json[CONFIG_MODEL_KEY]:
                udo_path = os.path.join(os.environ['MODELZOO'], _config_from_json[
                                            CONFIG_MODEL_KEY][CONFIG_MODEL_UDO_SUBKEY])
            else:
                udo_path = None
        else:
            dlc_path = _config_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_DLC_SUBKEY]
            if CONFIG_MODEL_UDO_SUBKEY in _config_from_json[CONFIG_MODEL_KEY]:
                udo_path = _config_from_json[CONFIG_MODEL_KEY][CONFIG_MODEL_UDO_SUBKEY]
            else:
                udo_path = None

        if not os.path.exists(dlc_path):
            logger.error(dlc_path + " doesn't exist")
            return
        elif udo_path and not os.path.exists(udo_path):
            logger.error(udo_path + " doesn't exist")
            return
        else:
            return Config(cfgfile, _config_from_json, outputbasedir, devicelist, hostname, deviceostype, userbuffer_mode, perfprofile, profilinglevel, enable_init_caching)
