#==============================================================================
#
#  Copyright (c) 2016-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

SNPE_SDK_ROOT = 'SNPE_ROOT'
# for running as a SNPE contributor
ZDL_ROOT = 'ZDL_ROOT'
DIAGVIEW_OPTION = 'DIAGVIEW_OPTION'
SNPE_BENCH_NAME = 'snpe_bench'
SNPE_BENCH_ROOT = 'SNPE_BENCH_ROOT'
SNPE_BENCH_LOG_FORMAT = '%(asctime)s - %(levelname)s - {}: %(message)s'

SNPE_BATCHRUN_EXE = 'snpe-net-run'
SNPE_RUNTIME_LIB = 'libSNPE.so'

SNPE_BENCH_SCRIPT = 'snpe-bench_cmds.sh'
SNPE_DLC_INFO_EXE = "snpe-dlc-info"
SNPE_DIAGVIEW_EXE = "snpe-diagview"
DEVICE_TYPE_ARM_ANDROID = 'arm_android'
DEVICE_TYPE_ARM_LINUX = 'arm_linux'
DEVICE_TYPE_ARM_QNX= 'arm_qnx'
DEVICE_ID_X86 = 'localhost'
ARTIFACT_DIR = "artifacts"
SNPE_BENCH_DIAG_OUTPUT_FILE = "SNPEDiag_0.log"
SNPE_BENCH_DIAG_REMOVE = "SNPEDiag*"
SNPE_BENCH_OUTPUT_DIR_DATETIME_FMT = "%4d-%02d-%02d_%02d:%02d:%02d"
SNPE_BENCH_ANDROID_AARCH64_ARTIFACTS = "aarch64-android-clang8.0"
SNPE_BENCH_LE64_GCC53_ARTIFACTS = "aarch64-linux-gcc5.3"
SNPE_BENCH_LE_OE_GCC82_HF_ARTIFACTS = "arm-oe-linux-gcc8.2hf"
SNPE_BENCH_LE64_OE_GCC82_ARTIFACTS = "aarch64-oe-linux-gcc8.2"
SNPE_BENCH_QNX64_GCC54_ARTIFACTS = "aarch64-qnx-gcc5.4"
SNPE_BENCH_QNX64_GCC83_ARTIFACTS = "aarch64-qnx-gcc8.3"
SNPE_BENCH_AARCH64_UBUNTU_GCC75_ARTIFACTS = "aarch64-ubuntu-gcc7.5"
SNPE_BENCH_AARCH64_OE_LINUX_GCC93_ARTIFACTS = "aarch64-oe-linux-gcc9.3"
SNPE_BENCH_AARCH64_OE_LINUX_GCC112_ARTIFACTS = "aarch64-oe-linux-gcc11.2"
SNPE_BENCH_AARCH64_UBUNTU_GCC94_ARTIFACTS = "aarch64-ubuntu-gcc9.4"

CONFIG_DEVICEOSTYPES_ANDROID_AARCH64 = 'android-aarch64'
CONFIG_DEVICEOSTYPES_LE64_GCC53 = 'le64_gcc5.3'
CONFIG_DEVICEOSTYPES_LE_OE_GCC82 = 'le_oe_gcc8.2'
CONFIG_DEVICEOSTYPES_LE64_OE_GCC82 = 'le64_oe_gcc8.2'
CONFIG_DEVICEOSTYPES_QNX64_GCC54 = 'qnx64_gcc5.4'
CONFIG_DEVICEOSTYPES_QNX64_GCC83 = 'qnx64_gcc8.3'
CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_GCC75 = 'ubuntu64_gcc75'
CONFIG_DEVICEOSTYPES_AARCH64_OE_LINUX_GCC93 = 'le64_oe_gcc9.3'
CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_GCC94 = 'ubuntu64_gcc94'
CONFIG_DEVICEOSTYPES_AARCH64_OE_LINUX_GCC112 = 'le64_oe_gcc11.2'

# some values in the JSON fields. used in JSON and in directory creation.
RUNTIMES = {
    'AIP': ' --use_aip',
    'AIP_ACT16': ' --use_aip',
    'CPU': '',
    'DSP': ' --use_dsp',
    'DSP_ACT16': ' --use_dsp',
    'DSP_FP16': ' --use_dsp',
    'GPU': ' --use_gpu',
    'GPU_FP16': ' --use_gpu --gpu_mode float16'
}

RUNTIME_CPU = 'CPU'
RUNTIME_GPU = 'GPU'

BUFFER_MODES = {
    'float': '',
    'ub_float': ' --userbuffer_float',
    'ub_tf8': ' --userbuffer_tfN 8',
    'ub_tf16': ' --userbuffer_tfN 16',
    'ub_auto': ' --userbuffer_auto'
}

RUNTIME_BUFFER_MODES = {
    'CPU': ['float', 'ub_float', 'ub_auto'],
    'DSP': ['float', 'ub_float', 'ub_tf8', 'ub_auto'],
    'DSP_ACT16': ['float', 'ub_float', 'ub_tf16', 'ub_auto'],
    'DSP_FP16': ['float', 'ub_float', 'ub_auto'],
    'GPU': ['float', 'ub_float', 'ub_auto'],
    'GPU_FP16': ['float', 'ub_float', 'ub_auto'],
    'AIP': ['float', 'ub_float','ub_tf8', 'ub_auto'],
    'AIP_ACT16': ['float', 'ub_float', 'ub_tf16', 'ub_auto']
}

ARCH_AARCH64 = "aarch64"
ARCH_ARM = "arm"
ARCH_DSP = "dsp"
ARCH_X86 = "x86"

PLATFORM_OS_LINUX = "linux"
PLATFORM_OS_ANDROID = "android"
PLATFORM_OS_QNX = "qnx"

COMPILER_GCC82 = "gcc8.2"
COMPILER_GCC75 = "gcc7.5"
COMPILER_GCC83 = "gcc8.3"
COMPILER_GCC54 = "gcc5.4"
COMPILER_GCC53 = "gcc5.3"
COMPILER_GCC93 = "gcc9.3"
COMPILER_GCC94 = "gcc9.4"
COMPILER_GCC112 = "gcc11.2"
COMPILER_CLANG60 = "clang6.0"
STL_LIBCXX_SHARED = "libc++_shared.so"

MEASURE_SNPE_VERSION = 'snpe_version'
MEASURE_TIMING = "timing"
MEASURE_MEM = "mem"

PROFILING_LEVEL_BASIC = "basic"
PROFILING_LEVEL_MODERATE = "moderate"
PROFILING_LEVEL_DETAILED = "detailed"
PROFILING_LEVEL_BACKEND_LINTING = "linting"
PROFILING_LEVEL_OFF = "off"

PERF_PROFILE_BALANCED = "balanced"
PERF_PROFILE_DEFAULT = "default"
PERF_PROFILE_POWER_SAVER = "power_saver"
PERF_PROFILE_HIGH_PERFORMANCE = "high_performance"
PERF_PROFILE_SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"
PROFILE_MODE_ON = 1

LATEST_RESULTS_LINK_NAME = "latest_results"

MEM_LOG_FILE_NAME = "MemLog.txt"
SNPE_DIAG_PARSE_OUT_FILE = "diaglog_out.txt"
SNPE_DIAG_JSON_FILE = "diaglog"

ERRNUM_CONFIG_ERROR = 1
ERRNUM_PARSEARGS_ERROR = 3
ERRNUM_GENERALEXCEPTION_ERROR = 4
ERRNUM_ADBSHELLCMDEXCEPTION_ERROR = 5
ERRNUM_MD5CHECKSUM_FILE_NOT_FOUND_ON_DEVICE = 14
ERRNUM_MD5CHECKSUM_CHECKSUM_MISMATCH = 15
ERRNUM_MD5CHECKSUM_UNKNOWN_ERROR = 16
ERRNUM_NOBENCHMARKRAN_ERROR = 17

ENABLE_CACHE_SUPPORTED_RUNTIMES = ("DSP", "AIP")

