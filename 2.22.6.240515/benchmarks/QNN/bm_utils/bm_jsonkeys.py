# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class BmJsonKeys:
    def __init__(self):

        self.CONFIG_NAME_KEY = "Name"
        self.CONFIG_HOST_ROOTPATH_KEY = "HostRootPath"
        self.CONFIG_HOST_RESULTSDIR_KEY = "HostResultsDir"
        self.CONFIG_DEVICE_PATH_KEY = "DevicePath"
        self.CONFIG_DEVICES_KEY = "Devices"
        self.CONFIG_HOST_NAME_KEY = "HostName"
        self.CONFIG_RUNS_KEY = "Runs"
        self.CONFIG_MODEL_KEY = "Model"
        self.CONFIG_MODEL_NAME_SUBKEY = "Name"
        self.CONFIG_MODEL_DATA_SUBKEY = "Data"
        self.CONFIG_MODEL_INPUTLIST_SUBKEY = "InputList"
        self.CONFIG_RUNTIMES_KEY = "Backends"
        self.CONFIG_ARCHITECTURES_KEY = "Architectures"
        self.CONFIG_COMPILER_KEY = "Compiler"
        self.CONFIG_STL_LIBRARY_KEY = "C++ Standard Library"
        self.CONFIG_MEASUREMENTS_KEY = "Measurements"
        self.CONFIG_PERF_PROFILE_KEY = "PerfProfile"
        self.CONFIG_BUFFERTYPES_KEY = "BufferTypes"
        self.CONFIG_PROFILING_LEVEL_KEY = "ProfilingLevel"
        self.CONFIG_GRAPH_PREPARE_TYPE = "Graph Prepare"
        self.CONFIG_USE_SHARED_BUFFER = "SharedBuffer"
        self.CONFIG_ARTIFACTS_KEY = "Artifacts"

        self.CONFIG_JSON_ROOTKEYS = [self.CONFIG_NAME_KEY, self.CONFIG_HOST_ROOTPATH_KEY,
                                     self.CONFIG_HOST_RESULTSDIR_KEY, self.CONFIG_DEVICE_PATH_KEY,
                                     self.CONFIG_DEVICES_KEY, self.CONFIG_HOST_NAME_KEY, self.CONFIG_RUNS_KEY,
                                     self.CONFIG_MODEL_KEY, self.CONFIG_RUNTIMES_KEY,
                                     self.CONFIG_MEASUREMENTS_KEY, self.CONFIG_PERF_PROFILE_KEY,
                                     self.CONFIG_USE_SHARED_BUFFER,
                                     self.CONFIG_BUFFERTYPES_KEY, self.CONFIG_PROFILING_LEVEL_KEY]
        self.CONFIG_JSON_MODEL_COMMON_SUBKEYS = [self.CONFIG_MODEL_NAME_SUBKEY]
        self.CONFIG_JSON_MODEL_DEFINED_INPUT_SUBKEYS = [self.CONFIG_MODEL_INPUTLIST_SUBKEY,
                                                        self.CONFIG_MODEL_DATA_SUBKEY]
        self.CONFIG_MODEL_SUBKEY = "qnn_model"
        self.CONFIG_ARTIFACTS_COMPILER_KEY_HOST = "x86_64-linux-clang"
        self.CONFIG_ARTIFACTS_COMPILER_KEY_HOST_WIN = "x86_64-windows-msvc"
        self.CONFIG_ARTIFACTS_COMPILER_KEY_TARGET_ANDROID_OSPACE = [
            "aarch64-android"
        ]
        self.CONFIG_JSON_MODEL_COMMON_SUBKEYS.append(self.CONFIG_MODEL_SUBKEY)


