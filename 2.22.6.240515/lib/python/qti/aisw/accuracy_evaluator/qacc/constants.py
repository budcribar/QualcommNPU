##############################################################################
#
# Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################


class Constants:
    # Plugin constants.
    PLUG_INFO_TYPE_MEM = 'mem'
    PLUG_INFO_TYPE_PATH = 'path'
    PLUG_INFO_TYPE_DIR = 'dir'

    # Inference Engine constants
    INFER_ENGINE_QNN = "qnn"
    INFER_ENGINE_AIC = 'aic'
    INFER_ENGINE_ONNXRT = 'onnxrt'
    INFER_ENGINE_TFRT = 'tensorflow'
    INFER_ENGINE_TORCHSCRIPTRT = 'torchscript'
    INFER_ENGINE_PYTORCH_MODULE = 'pytorch-module'
    INFER_ENGINE_TFRT_SESSION = 'tensorflow-session'

    BACKEND_CPU = "cpu"
    BACKEND_GPU = "gpu"
    BACKEND_AIC = "aic"
    BACKEND_HTP = "htp"
    BACKEND_HTP_MCP = "htp_mcp"
    BACKEND_DSPV69 = "dspv69"
    BACKEND_DSPV73 = "dspv73"
    BACKEND_DSPV75 = "dspv75"

    PRECISION_FP16 = 'fp16'
    PRECISION_FP32 = 'fp32'
    PRECISION_INT8 = 'int8'
    PRECISION_QUANT = "quant"

    # IO info keys
    IO_TYPE = 'type'
    IO_DTYPE = 'dtype'
    IO_FORMAT = 'format'

    # Datatypes
    DTYPE_FLOAT16 = 'float16'
    DTYPE_FLOAT32 = 'float32'
    DTYPE_FLOAT64 = 'float64'
    DTYPE_INT8 = 'int8'
    DTYPE_INT16 = 'int16'
    DTYPE_INT32 = 'int32'
    DTYPE_INT64 = 'int64'

    # Formats
    FMT_NPY = 'np'
    FMT_CV2 = 'cv2'
    FMT_PIL = 'pil'

    # Plugin status
    STATUS_SUCCESS = 0
    STATUS_ERROR = 1
    STATUS_REMOVE = 2

    # Calibration type
    CALIBRATION_TYPE_INDEX = 'index'
    CALIBRATION_TYPE_RAW = 'raw'
    CALIBRATION_TYPE_DATASET = 'dataset'

    # Pipeline stage names
    STAGE_PREPROC_CALIB = 'calibration'
    STAGE_PREPROC = 'preproc'
    STAGE_POSTPROC = 'postproc'
    STAGE_COMPILE = 'compiled'
    STAGE_INFER = 'infer'
    STAGE_METRIC = 'metric'

    # File type
    BINARY_PATH = 'binary'

    # output file names
    PROCESSED_OUTFILE = 'processed-outputs.txt'
    QNN_PROCESSED_OUTFILE = 'qnn-processed-outputs.txt'
    INFER_OUTFILE = 'infer-outlist.txt'
    PROFILE_YAML = 'profile.yaml'
    RESULTS_TABLE_CSV = 'metrics-info.csv'
    PROFILING_TABLE_CSV = 'profiling-info.csv'
    INPUT_LIST_FILE = 'processed-inputlist.txt'
    CALIB_FILE = 'processed-calibration.txt'
    INFER_RESULTS_FILE = 'runlog_inf.txt'

    # output directory names
    DATASET_DIR = 'dataset'

    # qacc inference schema runstatus
    SCHEMA_INFER_SUCCESS = 0
    SCHEMA_INFER_FAIL = 1
    SCHEMA_POSTPROC_SUCCESS = 2
    SCHEMA_POSTPROC_FAIL = 3
    SCHEMA_METRIC_SUCCESS = 4
    SCHEMA_METRIC_FAIL = 5

    def get_inference_schema_status(code):
        if code == Constants.SCHEMA_INFER_FAIL:
            return 'Inference failed'
        elif code == Constants.SCHEMA_POSTPROC_FAIL:
            return 'PostProcess failed'
        elif code == Constants.SCHEMA_METRIC_FAIL:
            return 'Metric Failed'
        else:
            return 'Success'

    # search space delimiter
    SEARCH_SPACE_DELIMITER = '|'
    RANGE_BASED_DELIMITER = '-'
    RANGE_BASED_SWEEP_PREFIX = 'range=('

    # cleanup options
    CLEANUP_AT_END = 'end'
    CLEANUP_INTERMEDIATE = 'intermediate'
    INFER_SKIP_CLEANUP = '/temp/'

    # config info
    MODEL_INFO_BATCH_SIZE = 'batchsize'

    # pipeline pipeline_cache keys
    PIPELINE_BATCH_SIZE = 'config.info.batchsize'
    PIPELINE_WORK_DIR = 'qacc.work_dir'
    PIPELINE_MAX_INPUTS = 'qacc.dataset.max_inputs'
    PIPELINE_MAX_CALIB = 'qacc.dataset.max_calib'

    # preproc
    PIPELINE_PREPROC_DIR = 'qacc.preproc_dir'
    PIPELINE_PREPROC_FILE = 'qacc.preproc_file'
    # calib
    PIPELINE_CALIB_DIR = 'qacc.calib_dir'
    PIPELINE_CALIB_FILE = 'qacc.calib_file'
    PIPELINE_PREPROC_IS_CALIB = 'qacc.preproc_is_calib'
    # postproc
    PIPELINE_POSTPROC_DIR = 'qacc.postproc_dir'  # contains nested structure
    PIPELINE_POSTPROC_FILE = 'qacc.postproc_file'  # contains nested structure
    # infer
    PIPELINE_INFER_DIR = 'qacc.infer_dir'  # contains nested structure
    PIPELINE_INFER_FILE = 'qacc.infer_file'  # contains nested structure
    PIPELINE_INFER_INPUT_INFO = 'qacc.infer_input_info'
    PIPELINE_INFER_OUTPUT_INFO = 'qacc.infer_output_info'
    PIPELINE_NETWORK_BIN_DIR = 'qacc.network_bin_dir'  # contains nested structure
    PIPELINE_NETWORK_DESC = 'qacc.network_desc'  # contains nested structure
    PIPELINE_PROGRAM_QPC = 'qacc.program_qpc'  # contains nested structure

    # internal pipeline cache keys
    INTERNAL_CALIB_TIME = 'qacc.calib_time'
    INTERNAL_PREPROC_TIME = 'qacc.preproc_time'
    INTERNAL_QUANTIZATION_TIME = 'qacc.quantization_time'  # contains nested structure
    INTERNAL_COMPILATION_TIME = 'qacc.compilation_time'  # contains nested structure
    INTERNAL_INFER_TIME = 'qacc.infer_time'  # contains nested structure
    INTERNAL_POSTPROC_TIME = 'qacc.postproc_time'  # contains nested structure
    INTERNAL_METRIC_TIME = 'qacc.metric_time'  # contains nested structure
    INTERNAL_EXEC_BATCH_SIZE = 'qacc.exec_batch_size'

    # file naming convention
    NETWORK_DESC_FILE = 'networkdesc.bin'
    PROGRAM_QPC_FILE = 'programqpc.bin'

    # options for get orig file paths API
    LAST_BATCH_TRUNCATE = 1
    LAST_BATCH_REPEAT_LAST_RECORD = 2
    LAST_BATCH_NO_CHANGE = 3

    # dataset filter plugin keys
    DATASET_FILTER_PLUGIN_NAME = 'filter_dataset'
    DATASET_FILTER_PLUGIN_PARAM_RANDOM = 'random'
    DATASET_FILTER_PLUGIN_PARAM_MAX_INPUTS = 'max_inputs'
    DATASET_FILTER_PLUGIN_PARAM_MAX_CALIB = 'max_calib'

    # Data reader related
    DATAREADER_SENTINEL_OBJ = 'END'

    # model configurator artifacts
    MODEL_CONFIGURATOR_RESULT_FILE = 'results.csv'
    MODEL_CONFIGURATOR_DIR = 'model_configurator'
    MODEL_SETTING_FILE = 'model_settings.yaml'

    # Infernce Toolkit Stages: Placeholder values
    STAGE_EVALUATOR = 'accuracy_evaluator'
    STAGE_MODEL_CONFIGURATOR = 'model_configurator'
    STAGE_FILTER_INFERENCE_SCHEMAS = 'filter_inference_schemas'

    # QNN related cache keys
    QNN_SDK_DIR = "qnn_sdk_dir"

    # QNN executables
    MODEL_LIB_GENERATOR = "qnn-model-lib-generator"
    CONTEXT_BINARY_GENERATOR = "qnn-context-binary-generator"
    NET_RUN = "qnn-net-run"

    # QNN binaries
    AIC_BACKEND = "libQnnAic.so"
    AIC_NETRUN_EXTENSION = "libQnnAicNetRunExtensions.so"
    HTP_BACKEND = "libQnnHtp.so"
    HTP_NETRUN_EXTENSION = "libQnnHtpNetRunExtensions.so"
    HTP_MCP_BACKEND = "libQnnHtpMcp.so"
    HTP_MCP_NETRUN_EXTENSION = "libQnnHtpMcpNetRunExtensions.so"
    CPU_BACKEND = "libQnnCpu.so"
    GPU_BACKEND = "libQnnGpu.so"
    DEFAULT_ADB_PATH = "/opt/bin/adb"
    DEFAULT_MODEL_ZOO_PATH = "/home/model_zoo"
    DEFAULT_MCP_ELF_PATH = "lib/hexagon-v68/unsigned/libQnnHtpMcpV68.elf"
    QNN_DLC_MODEL_SO = "libQnnModelDlc.so"

    # QNN intermediate files
    MODEL_IR_FILE = "model"
    MODEL_IR_FOLDER = "qnn_ir"
    MODEL_BINARIES_FOLDER = "model_binaries"
    CONTEXT_BINARY_FILE = "context_binary"
    CONTEXT_BACKEND_EXTENSION_CONFIG = "context_backend_extensions.json"
    NETRUN_BACKEND_EXTENSION_CONFIG = "netrun_backend_extensions.json"
    CONTEXT_CONFIG = "context_config.json"
    NETRUN_CONFIG = "netrun_config.json"

    # Backend libraries
    DSP_BACKEND_LIBRARIES = [
        "lib/aarch64-android/libQnnHtp.so", "lib/aarch64-android/libQnnHtpPrepare.so",
        "lib/aarch64-android/libQnnHtpNetRunExtensions.so"
    ]
    CPU_BACKEND_LIB = "lib/aarch64-android/libQnnCpu.so"
    GPU_BACKEND_LIB = "lib/aarch64-android/libQnnGpu.so"

    # Supported converter parameter
    SUPPORTED_CONVERTER_PARAMS = [
        "quantization_overrides", "keep_quant_nodes", "input_list", "param_quantizer",
        "act_quantizer", "algorithms", "bias_bitwidth", "bias_bw", "act_bitwidth", "act_bw",
        "weights_bitwidth", "weight_bw", "float_bias_bitwidth", "float_bias_bw",
        "use_per_row_quantization", "use_per_channel_quantization", "use_native_input_files",
        "use_native_dtype", "use_native_output_files", "disable_relu_squashing",
        "restrict_quantization_steps", "ignore_encodings", "op_package_lib",
        "converter_op_package_lib", "package_name", "op_package_config", "input_type",
        "input_dtype", "input_encoding", "input_layout", "custom_io", "preserve_io",
        "disable_batchnorm_folding", "keep_disconnected_nodes", "float_bitwidth", "float_bw",
        "overwrite_model_prefix", "debug", "arch_checker", "exclude_named_tensors", "dump_relay",
        "show_unconsumed_nodes", "saved_model_tag", "saved_model_signature_key",
        "dump_custom_io_config_template", "no_simplification", "define_symbol", "input_dim",
        "input_network", "out_node", "copyright_file", "dry_run", "batch", "output_path",
        "extra_args", "float_fallback", "pack_4_bit_weights", "param_quantizer_calibration",
        "act_quantizer_calibration", "act_quantizer_schema", "param_quantizer_schema",
        "percentile_calibration_value"
    ]
    SUPPORTED_CONTEXT_BINARY_PARAMS = [
        "model_prefix", "op_packages", "profiling_level", "enable_intermediate_outputs",
        "set_output_tensors", "extra_args"
    ]
    SUPPORTED_NETRUN_PARAMS = [
        "model_prefix", "op_packages", "profiling_level", "use_native_input_files",
        "use_native_output_files", "native_input_tensor_names", "perf_profile", "num_inferences",
        "keep_num_outputs", "max_input_cache_tensor_sets", "max_input_cache_size_mb", "extra_args"
    ]
    # Supported compiler parameters based on backend extension
    SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS = {
        "aic": [
            "compiler_compilation_target", "compiler_hardware_version", "compiler_num_of_cores",
            "compiler_do_host_preproc", "compiler_stat_level", "compiler_stats_batch_size",
            "compiler_printDDRStats", "compiler_printPerfMetrics", "compiler_perfWarnings",
            "compiler_PMU_events", "compiler_PMU_recipe_opt", "compiler_buffer_dealloc_delay",
            "compiler_genCRC", "compiler_crc_stride", "compiler_enable_depth_first",
            "compiler_cluster_sizes", "compiler_max_out_channel_split",
            "compiler_overlap_split_factor", "compiler_compilationOutputDir",
            "compiler_depth_first_mem", "compiler_VTCM_working_set_limit_ratio",
            "compiler_userDMAProducerDMAEnabled", "compiler_size_split_granularity",
            "compiler_do_DDR_to_multicast", "compiler_enableDebug", "compiler_combine_inputs",
            "compiler_combine_outputs", "compiler_directApi", "compiler_compileThreads",
            "compiler_force_VTCM_spill", "compiler_convert_to_FP16"
        ],
        "htp": [
            "vtcm_mb", "fp16_relaxed_precision", "O", "dlbc", "hvx_threads", "soc_id", "soc_model",
            "dsp_arch", "pd_session", "profiling_level", "weight_sharing_enabled"
        ],
        "htp_mcp": [
            "fp16_relaxed_precision", "O", "device_id", "num_cores", "heap_size", "elf_path",
            "mode", "combined_io_dma_enabled"
        ]
    }
    # Supported runtime parameters based on backend extension
    SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS = {
        "aic": [
            "runtime_device_ids", "runtime_num_activations", "runtime_profiling_start_iter",
            "runtime_profiling_num_samples", "runtime_profiling_type", "runtime_profiling_out_dir",
            "runtime_submit_timeout", "runtime_num_retries", "runtime_set_size",
            "runtime_threads_per_queue"
        ],
        "htp": [
            "fp16_relaxed_precision", "rpc_control_latency", "vtcm_mb", "O", "dlbc", "hvx_threads",
            "device_id", "soc_id", "soc_model", "dsp_arch", "pd_session", "profiling_level",
            "use_client_context", "core_id", "perf_profile", "rpc_polling_time", "hmx_timeout_us",
            "max_spill_fill_buffer_for_group", "group_id", "file_read_memory_budget_in_mb"
        ],
        "htp_mcp":
        ["profiling_level", "device_id", "timeout", "retries", "mode", "combined_io_dma_enabled"]
    }
    PIPE_SUPPORTED_CONVERTER_PARAMS = {
        "param_quantizer": "",
        "param_quantizer_calibration": "",
        "param_quantizer_schema": "",
        "act_quantizer": "",
        "act_quantizer_calibration": "",
        "act_quantizer_schema": "",
        "algorithms": "",
        "float_bitwidth": "float",
        "float_bw": "float",
        "bias_bitwidth": "bias",
        "bias_bw": "bias",
        "act_bitwidth": "act",
        "act_bw": "act",
        "float_bias_bitwidth": "floatbias",
        "float_bias_bw": "floatbias",
        "use_per_channel_quantization": "pcq",
        "use_per_row_quantization": "prq"
    }
    SUPPORTED_CONVERTER_EXPORT_FORMATS = ['cpp', 'dlc']
    EXPORT_FORMAT_DLC = 'dlc'
    EXPORT_FORMAT_CPP = 'cpp'
    DEFAULT_CONVERTER_EXPORT_FORMAT = 'cpp'
    CUSTOM_OP_FLAGS = ['op_package_config', 'op_package_lib', 'package_name']
    SIMPLIFIED_CLEANED_MODEL_ONNX = 'cleanmodel_simplified.onnx'
    CLEANED_MODEL_ONNX = 'cleanmodel.onnx'
