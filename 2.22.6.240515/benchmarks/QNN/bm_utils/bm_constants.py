# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class BmConstants:

    CONFIG_DEVICEOSTYPES_ANDROID_AARCH64 = 'aarch64-android'
    CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64 = 'aarch64-windows-msvc'
    CONFIG_DEVICEOSTYPES_QNX_AARCH64 = 'aarch64-qnx'
    CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC112 = 'aarch64-oe-linux-gcc11.2'
    CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC93 = 'aarch64-oe-linux-gcc9.3'
    CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC82 = 'aarch64-oe-linux-gcc8.2'
    CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_OE_GCC94 = 'aarch64-ubuntu-gcc9.4'
    CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_OE_GCC75 = 'aarch64-ubuntu-gcc7.5'
    MEASURE_TIMING = "timing"
    MEASURE_MEM = "mem"

    def __init__(self):

        self.DIAGVIEW_OPTION = 'DIAGVIEW_OPTION'
        self.BENCH_LOG_FORMAT = '%(asctime)s - %(levelname)s - {}: %(message)s'
        self.DEVICE_TYPE_ARM_ANDROID = 'arm_android'
        self.DEVICE_TYPE_WINDOWS = 'windows'
        self.DEVICE_TYPE_ARM_QNX = 'arm_qnx'
        self.ARTIFACT_DIR = "artifacts"
        self.PRODUCT_BM = "bm_utils"

        self.RUNTIME_CPU = 'CPU'
        self.RUNTIME_GPU = 'GPU'

        self.ARCH_AARCH64 = "aarch64"
        self.ARCH_ARM = "arm"
        self.ARCH_DSP = "dsp"
        self.ARCH_X86 = "x86"

        self.PLATFORM_OS_ANDROID = "android"
        self.PLATFORM_OS_WINDOWS = "windows"
        self.PLATFORM_OS_QNX = "qnx"
        self.COMPILER_CLANG90 = "clang9.0"
        self.COMPILER_MSVC = "msvc"
        self.STL_LIBCXX_SHARED = "libc++_shared.so"

        self.PROFILING_LEVEL_BASIC = "baic"
        self.PROFILING_LEVEL_DETAILED = "detailed"

        self.LATEST_RESULTS_LINK_NAME = "latest_results"
        self.MEM_LOG_FILE_NAME = "MemLog.txt"

        self.BENCH_OUTPUT_DIR_DATETIME_FMT = "%4d-%02d-%02d_%02d_%02d_%02d"
        self.ARCH_VENDOR_LIB_PATH = {
            "arm": "/vendor/lib/",
            "aarch64": "/vendor/lib64/"
            }
        self.HOST_TMP_DIR = "tmp_work"
        self.SDK_ROOT = 'QNN_SDK_ROOT'
        self.BENCH_NAME = 'qnn_bench'
        self.BM_ARTIFACTS = "qnnbm_artifacts.json"
        self.BENCH_ROOT = 'QNN_BENCH_ROOT'
        self.BATCHRUN_EXE = 'qnn-net-run'
        self.CHACHE_EXE = 'qnn-context-binary-generator'
        self.RUNTIME_LIB = 'libQnnCpu.so'
        self.RUNTIME_LIB_WIN = 'QnnCpu.dll'
        self.BENCH_SCRIPT = 'qnn-bench_cmds.sh'
        self.BENCH_SCRIPT_WIN = 'qnn-bench_cmds.bat'
        self.DIAGVIEW_SCRIPT = 'qnn-diagview.sh'
        self.DIAGVIEW_SCRIPT_WIN = 'qnn-diagview.bat'
        self.MODEL_INFO_EXE = "qnn-model-info"
        self.DIAGVIEW_EXE = "qnn-profile-viewer"
        self.DIAGVIEW_EXE_WIN = "qnn-profile-viewer.exe"
        self.DIAGVIEW_LIB = "libQnnChrometraceProfilingReader.so"
        self.BENCH_DIAG_CSV_FILE = "diaglog.csv"
        self.BENCH_DIAG_JSON_FILE = "diaglog.json"
        self.BENCH_DIAG_OUTPUT_FILE = "qnn-profiling-data_0.log"
        self.MEASURE_PRODUCT_VERSION = 'qnn_version'
        self.BENCH_DIAG_REMOVE = "qnn-profiling-data*"
        self.SDK_VERSION_HEADER = "QNN SDK version:"
        self.PARSE_OUT_FILE = "diaglog_out.txt"
        self.run_cmd_options = "{0} --model {1} --input_list {2} --output_dir {3}"
        self.cache_cmd_options = "{0} --model {1} --binary_file {2} --output_dir {3}"
        self.serialized_run_cmd_options = "{0} --retrieve_context {1} --input_list {2} " \
                                          "--output_dir {3} "
        self.diagview_cmd = "{0} --input_log {1} --output {2} > {3}"
        self.RUNTIMES = {
            'CPU': ' --backend libQnnCpu.so',
            'GPU': ' --backend libQnnGpu.so',
            'GPU_FP16': ' --backend libQnnGpu.so',
            'HTP_v68': ' --backend libQnnHtp.so',
            'HTP_v69': ' --backend libQnnHtp.so',
            'HTP_v69-plus': ' --backend libQnnHtp.so',
            'HTP_v73': ' --backend libQnnHtp.so',
            'HTP_v75': ' --backend libQnnHtp.so',
            'HTP_v79': ' --backend libQnnHtp.so',
            'DSP_v66': ' --backend libQnnDsp.so',
            'DSP_v65': ' --backend libQnnDsp.so',
            'DSP': ' --backend libQnnHtp.so',
            'DSP_FP16': ' --backend libQnnHtp.so',
            'HTA' : ' --backend libQnnHta.so'
        }
        self.RUNTIMES_WIN = {
            'CPU': ' --backend QnnCpu.dll',
            'GPU': ' --backend QnnGpu.dll',
            'GPU_FP16': ' --backend QnnGpu.dll',
            'HTP_v68': ' --backend QnnHtp.dll',
            'HTP_v69': ' --backend QnnHtp.dll',
            'HTP_v69-plus': ' --backend QnnHtp.dll',
            'HTP_v73': ' --backend QnnHtp.dll',
            'HTP_v75': ' --backend QnnHtp.dll',
            'HTP_v79': ' --backend QnnHtp.dll',
            'DSP_v66': ' --backend QnnDsp.dll',
            'DSP_v65': ' --backend QnnDsp.dll',
            'DSP': ' --backend QnnHtp.dll',
            'DSP_FP16': ' --backend QnnHtp.dll',
            'HTA' : ' --backend QnnHta.dll'
        }
        self.BUFFER_MODES = {
            'float': ''
        }
        self.RUNTIME_BUFFER_MODES = {
            'CPU': ['float'],
            'GPU': ['float'],
            'GPU_FP16' : ['float'],
            'DSP': ['float'],
            'DSP_FP16': ['float'],
            'HTP_v68': ['float'],
            'HTP_v69': ['float'],
            'HTP_v69-plus': ['float'],
            'HTP_v73': ['float'],
            'HTP_v75': ['float'],
            'HTP_v79': ['float'],
            'DSP_v65': ['float'],
            'DSP_v66': ['float'],
            'HTA': ['float'],
        }

        self.BENCH_ANDROID_AARCH64_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-android.json"
        self.BENCH_WINDOWS_AARCH64_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-windows-msvc.json"
        self.BENCH_QNX_AARCH64_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-qnx.json"
        self.BENCH_AARCH64_LINUX_OE_GCC112_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-oe-linux-gcc11.2.json"
        self.BENCH_AARCH64_LINUX_OE_GCC93_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-oe-linux-gcc9.3.json"
        self.BENCH_AARCH64_LINUX_OE_GCC82_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-oe-linux-gcc8.2.json"
        self.BENCH_AARCH64_UBUNTU_OE_GCC94_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-ubuntu-gcc9.4.json"
        self.BENCH_AARCH64_UBUNTU_OE_GCC75_ARTIFACTS_JSON = "qnnbm_artifacts_aarch64-ubuntu-gcc7.5.json"

        self.MAJOR_STATISTICS_1 = {
            "Init Stats": self.PROFILING_LEVEL_BASIC,
            "Compose Graph Stats": self.PROFILING_LEVEL_BASIC,
            "Finalize Stats": self.PROFILING_LEVEL_BASIC,
            "De-Init Stats": self.PROFILING_LEVEL_BASIC,
            "Total Inference Time": self.PROFILING_LEVEL_BASIC
        }

