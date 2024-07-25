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

import logging
import os
import subprocess
import time
import yaml
import copy
import json
import shutil
import re
import numpy as np
from collections import OrderedDict

import qti.aisw.accuracy_evaluator.common.defaults as df
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.common.infer_engines.executors import LocalExecutor, AdbExecutor
from qti.aisw.accuracy_evaluator.common.infer_engines.converters import (OnnxConverter,
                                                                         TensorflowConverter,
                                                                         TFLiteConverter,
                                                                         PytorchConverter)

defaults = df.Defaults.getInstance()


class QnnInferenceEngine():

    def __init__(self, model, inputlistfile, calibration_file, output_path, multithread=False,
                 input_info=None, output_info=None, gen_out_file=None, backend_extensions=None,
                 netrun_params=None, converter_params=None, contextbin_params=None,
                 binary_path=None, backend="cpu", target_arch="x86_64-linux-clang", qnn_sdk_dir="",
                 device_id=None, converter_export_format=qcc.DEFAULT_CONVERTER_EXPORT_FORMAT):
        self.model_path = model
        self.input_path = inputlistfile
        self.calibration_file = calibration_file
        self.input_info = input_info
        self.output_info = output_info
        self.output_path = output_path
        self.backend_extensions = backend_extensions
        self.netrun_params = netrun_params
        self.converter_params = converter_params
        self.contextbin_params = contextbin_params
        self.gen_out_file = gen_out_file
        self.binary_path = self.output_path + '/temp'
        self.debug_log_path = self.output_path + '/debug_'
        self.backend = backend  #cpu/aic/htp
        self.target_arch = target_arch
        self.engine_path = qnn_sdk_dir
        self.device_id = device_id
        self.converter_export_format = converter_export_format
        self.use_native_output_files = False
        self.compile_arch = "x86_64-linux-clang"  # converter, model-lib-generator and context⁠-binary-generator binaries from x86_64-linux-clang bin will be used irrespective of given target.
        self.model_lib_target = "x86_64-linux-clang"  # Default target for model-lib-generator command

        if target_arch == "aarch64-android" and backend in [qcc.BACKEND_CPU, qcc.BACKEND_GPU]:
            # For Android CPU/GPU backend inference-schemas, aarch64-android model lib is required
            self.model_lib_target = "aarch64-android"

            # Validate if valid NDK paths are set
            android_root = os.environ.get('ANDROID_NDK_ROOT')
            if not (android_root and os.path.exists(android_root)):
                raise ce.ConfigurationException(
                    f"Invalid value found for environment variable $ANDROID_NDK_ROOT={android_root}, please set valid path."
                )

        self.is_adb = (("dspv" in backend or backend in [qcc.BACKEND_CPU, qcc.BACKEND_GPU])
                       and target_arch == "aarch64-android")

        # DLC Path for cpu backend is available only in AIC SDK
        if self.backend == qcc.BACKEND_CPU and self.target_arch == "x86_64-linux-clang" and not (
                os.path.exists(
                    os.path.join(self.engine_path, 'lib', self.target_arch, qcc.AIC_BACKEND))):
            qacc_file_logger.warning(
                "Using CPP based evaluation as Context caching is not supported by current backend."
            )
            self.converter_export_format = qcc.EXPORT_FORMAT_CPP

        #Initialize all the executable paths
        self._setup()

        # Set given AIC device ID with backend extension param
        if backend == qcc.BACKEND_AIC and self.device_id is not None:
            # Expected format of providing device ids for AIC backend extensions: "runtime_device_ids": [0,1]
            self.backend_extensions.update({"runtime_device_ids": [self.device_id]})
        # Create config JSON files for relevant backends
        if self.backend in [
                qcc.BACKEND_AIC, qcc.BACKEND_HTP, qcc.BACKEND_HTP_MCP, qcc.BACKEND_DSPV69,
                qcc.BACKEND_DSPV73, qcc.BACKEND_DSPV75
        ]:
            self._create_config_files()

        # Status for each of inference stage: converter, model_lib_generator,
        # context_binary_generator, net_run
        self.stage_status = OrderedDict([('qnn-converter', True), ('qnn-model-lib-generator', True),
                                         ('qnn-context-binary-generator', True),
                                         ('qnn-net-run', True)])
        # Model lib generator step is not required for DLC Format
        if self.converter_export_format == qcc.EXPORT_FORMAT_DLC:
            del self.stage_status['qnn-model-lib-generator']

    def _create_htp_config(self, params, is_context_backend_extensions=False):
        """Create json config with context-binary and netrun backend extension
        params for HTP.

        The json is expected to have the following format:
        {
            graphs : [
                {
                    graph_names : ["model"],
                    vtcm_mb : 4,
                    fp16_relaxed_precision : 1,
                    O : 2,
                    hvx_threads : 1,
                    dlbc : 0
                }
            ],
            devices : [
                {
                    dsp_arch : v75,
                    device_id : 0,
                    soc_id : 0,
                    soc_model : 0,
                    pd_session : unsigned,
                    use_client_context : true,
                    cores : [
                        {
                            core_id : 0,
                            perf_profile : high_performance,
                            rpc_control_latency : 100,
                            rpc_polling_time : 9999,
                            hmx_timeout_us : 300000
                        }
                    ]
                }
            ],
            context : {
                weight_sharing_enabled : true,
                max_spill_fill_buffer_for_group : 0,
                group_id : 0,
                file_read_memory_budget_in_mb : 0
            }
        }
        """

        expected_params = {
            "graphs":
            ["vtcm_mb", "fp16_relaxed_precision", "graph_names", "O", "dlbc", "hvx_threads"],
            "devices": {
                "contextbin": ["soc_id", "soc_model", "dsp_arch", "pd_session", "profiling_level"],
                "netrun": [
                    "device_id", "soc_id", "soc_model", "dsp_arch", "pd_session", "profiling_level",
                    "use_client_context"
                ]
            },
            "cores": {
                "contextbin": [],
                "netrun": [
                    "core_id", "perf_profile", "rpc_control_latency", "rpc_polling_time",
                    "hmx_timeout_us"
                ]
            },
            "context": {
                "contextbin": ["weight_sharing_enabled"],
                "netrun":
                ["max_spill_fill_buffer_for_group", "group_id", "file_read_memory_budget_in_mb"]
            }
        }
        new_params = {"graphs": [{}], "devices": [{"cores": [{}]}], "context": {}}
        params_type = "contextbin" if is_context_backend_extensions else "netrun"

        # Required param for both context-binary and netrun
        new_params["graphs"][0]["graph_names"] = [self.model_name]

        for key, value in params.items():
            if key in expected_params["graphs"]:
                new_params["graphs"][0][key] = value
            elif key in expected_params["cores"][params_type]:
                new_params["devices"][0]["cores"][0][key] = value
            elif key in expected_params["devices"][params_type]:
                new_params["devices"][0][key] = value
            elif key in expected_params["context"][params_type]:
                new_params["context"][key] = value
            else:
                raise ce.UnsupportedException("Invalid {} parameter - {} for HTP backend".format(
                    params_type, key))
        return new_params

    def _create_htp_mcp_config(self, params, is_context_backend_extensions=False):
        """Create json config with context-binary and netrun backend extension
        params for HTP-MCP.

        The json is expected to have the following format:
        {
            graphs : [
                {
                    graph_name : model,
                    fp16_relaxed_precision : 1,
                    profiling_level : basic,
                    num_cores : 1,
                    O : 0
                }
            ],
            device : {
                device_id : 0
            },
            context : {
                heap_size : 256,
                elf_path : network.elf,
                timeout : 5000,
                retries : 5,
                mode : auto,
                combined_io_dma_enabled : true
            }
        }
        """

        expected_params = {
            "graphs": {
                "contextbin": ["fp16_relaxed_precision", "num_cores", "O"],
                "netrun": ["profiling_level"]
            },
            "device": ["device_id"],
            "context": {
                "contextbin": ["heap_size", "elf_path", "mode", "combined_io_dma_enabled"],
                "netrun": ["timeout", "retries", "mode", "combined_io_dma_enabled"]
            }
        }
        new_params = {"graphs": [{}], "device": {}, "context": {}}
        params_type = "netrun"
        if is_context_backend_extensions:
            params_type = "contextbin"
            new_params["graphs"][0]["graph_name"] = self.model_name

        for key, value in params.items():
            if key in expected_params["graphs"][params_type]:
                new_params["graphs"][0][key] = value
            elif key in expected_params["device"]:
                new_params["device"][key] = value
            elif key in expected_params["context"][params_type]:
                if key == "elf_path":
                    if not os.path.isabs(value):
                        value = os.path.join(self.engine_path, value)
                    assert os.path.exists(value), f"Invalid elf_path {value}, "\
                        "should be either an absolute path or relative to QNN_SDK_ROOT"
                new_params["context"][key] = value
            else:
                raise ce.UnsupportedException(
                    "Invalid {} parameter - {} for HTP MCP backend".format(params_type, key))

        # remove empty params from config
        new_params = {key: val for key, val in new_params.items() if val}
        return new_params

    def _create_config_json(self, params, is_context_backend_extensions=False):
        """Create json config file with context-binary and netrun backend
        extension params."""

        #Add the graph_names to the json.
        if self.backend == qcc.BACKEND_AIC and is_context_backend_extensions:
            params["graph_names"] = [self.model_name]
            # Explicit type cast required for compiler_num_of_cores and other integer
            # TODO: Need to add backend specific JSON creation based on schema
            if 'compiler_num_of_cores' in params:
                params['compiler_num_of_cores'] = int(params['compiler_num_of_cores'])
        qacc_file_logger.debug(f"Setting the backend extension options : {params}")
        if self.backend == qcc.BACKEND_HTP or "dspv" in self.backend:
            params = self._create_htp_config(params, is_context_backend_extensions)
        if self.backend == qcc.BACKEND_HTP_MCP:
            params = self._create_htp_mcp_config(params, is_context_backend_extensions)
        if is_context_backend_extensions:
            out_file = self.context_backend_extensions_json
        else:
            out_file = self.netrun_backend_extensions_json

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

        return out_file

    def _create_top_json(self, net_run_extension_path, config_json,
                         is_context_backend_extensions=False):
        """Create top level json with extension path and config json path."""

        data = {}
        if self.is_adb:
            if is_context_backend_extensions:
                data["backend_extensions"] = {
                    "shared_library_path": os.path.basename(net_run_extension_path),
                    "config_file_path": config_json,
                }
            else:
                data["backend_extensions"] = {
                    "shared_library_path": os.path.basename(net_run_extension_path),
                    "config_file_path": os.path.basename(config_json)
                }
        else:
            data["backend_extensions"] = {
                "shared_library_path": net_run_extension_path,
                "config_file_path": config_json,
            }
        if is_context_backend_extensions:
            out_file = self.context_config_json
        else:
            out_file = self.netrun_config_json

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _create_config_files(self):
        """Create the json files with the context-binary and netrun backend
        extension params and path to the extension binary."""

        context_backend_extensions = {}
        netrun_backend_extensions = {}
        if self.backend_extensions:
            params = copy.deepcopy(self.backend_extensions)
            backend = qcc.BACKEND_HTP if "dspv" in self.backend else self.backend
            for key, val in params.items():
                if key in qcc.SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS[backend]:
                    context_backend_extensions[key] = val
                if key in qcc.SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS[backend]:
                    netrun_backend_extensions[key] = val

        if context_backend_extensions:
            context_backend_extensions_json = self._create_config_json(
                context_backend_extensions, is_context_backend_extensions=True)
            self._create_top_json(self.net_run_extension_path, context_backend_extensions_json,
                                  is_context_backend_extensions=True)

        if netrun_backend_extensions:
            netrun_backend_extensions_json = self._create_config_json(netrun_backend_extensions)
            self._create_top_json(self.net_run_extension_path, netrun_backend_extensions_json)

    def qnn_converter(self):
        """Converts the reference model to QNN IR."""
        model_type = Helper.get_model_type(self.model_path)
        if model_type == ModelType.ONNX:
            converter_cls = OnnxConverter
        elif model_type == ModelType.TENSORFLOW:
            converter_cls = TensorflowConverter
        elif model_type == ModelType.TFLITE:
            converter_cls = TFLiteConverter
        elif model_type == ModelType.TORCHSCRIPT:
            converter_cls = PytorchConverter

        if model_type == ModelType.TORCHSCRIPT:
            converter = converter_cls(sdkPath=self.engine_path, inputNetwork=self.model_path,
                                      outputPath=self.converter_output,
                                      inputList=self.calibration_file,
                                      converterParams=self.converter_params,
                                      input_info=self.input_info,
                                      export_format=self.converter_export_format)
        else:
            converter = converter_cls(sdkPath=self.engine_path, inputNetwork=self.model_path,
                                      outputPath=self.converter_output,
                                      inputList=self.calibration_file,
                                      converterParams=self.converter_params,
                                      export_format=self.converter_export_format)
        try:
            debug_log_file = self.debug_log_path + 'converter.log'
            converter.convert(debug_log_path=debug_log_file)
        except Exception as e:
            self.stage_status['qnn-converter'] = False
            Helper.dump_stage_error_log(debug_log_file)

    def qnn_model_lib_generator(self):
        """Compiles the QNN IR to .so for a specific target architecture."""
        model_lib_gen_command = [
            self.MODEL_LIB_GENERATOR,
            f"-c {self.converter_output}",
            f"-o {self.model_binaries}",
            f"-b {self.converter_bin}",
            f"-l {self.model_so}",
            f"-t {self.model_lib_target}",
        ]
        cmd = ' '.join(model_lib_gen_command)
        qacc_file_logger.info(cmd)
        executor = LocalExecutor()
        debug_log_file = self.debug_log_path + 'model_lib_gen.log'
        status = executor.run(cmd, log_file=debug_log_file)
        self.stage_status['qnn-model-lib-generator'] = not bool(status)
        if status != 0:
            Helper.dump_stage_error_log(debug_log_file)
            qacc_file_logger.error("qnn-model-lib-generator failed to run successfully")
            raise ce.QnnModelLibGeneratorException(
                "model-lib-generator failed to run successfully.")

    def qnn_context_binary_generator(self):
        """Creates the compiled binary for AIC backend."""

        context_bin_command = [
            self.CONTEXT_BINARY_GENERATOR, f"--backend {self.compiler_backend_path}",
            f"--binary_file {self.context_binary}", f"--output_dir {self.output_path}"
        ]

        if self.converter_export_format == qcc.EXPORT_FORMAT_DLC:
            context_bin_command.append(f'--model {self.libQnnModelDlc_path}')
            context_bin_command.append(f'--dlc_path {self.converter_output}')
        else:
            context_bin_command.append(
                f"--model {os.path.join(self.model_binaries, self.compile_arch, self.model_so)}")

        if os.path.exists(self.context_config_json):
            context_bin_command.append(f'--config_file {self.context_config_json}')

        if "extra_args" in self.contextbin_params:
            self._handle_extra_args_io_tensors(["set_output_tensors"])

        contextbin_param_list = Helper.cli_params_to_list(self.contextbin_params)
        context_bin_command.extend(contextbin_param_list)

        cmd = ' '.join(context_bin_command)
        qacc_file_logger.info(cmd)
        executor = LocalExecutor()
        debug_log_file = self.debug_log_path + 'context_bin_gen.log'
        status = executor.run(cmd, log_file=debug_log_file)
        self.stage_status['qnn-context-binary-generator'] = not bool(status)
        if status != 0:
            Helper.dump_stage_error_log(debug_log_file)
            qacc_file_logger.error("qnn-context-binary-generator failed to run successfully")
            raise ce.QnnContextBinaryGeneratorException(
                "context-binary-generator failed to run successfully.")

    def qnn_net_run(self):
        if "extra_args" in self.netrun_params:
            self._handle_extra_args_io_tensors(["native_input_tensor_names", "set_output_tensors"])
            if 'use_native_output_files' in self.netrun_params["extra_args"]:
                self.use_native_output_files = True

        netrun_param_list = Helper.cli_params_to_list(self.netrun_params)
        """Inference on the device."""
        if self.is_adb:
            status = self._run_on_target_device(netrun_param_list)
            if status == -1:
                raise ce.QnnNetRunException("qnn-net-run failed to run successfully on target.")
            return

        net_run_cmd = [
            self.NET_RUN, f"--backend {self.backend_path}", f"--input_list {self.input_path}",
            f"--output_dir {self.output_path}"
        ]

        if os.path.exists(self.netrun_config_json):
            net_run_cmd.append(f"--config_file {self.netrun_config_json}")

        if self.backend == qcc.BACKEND_AIC or self.backend == qcc.BACKEND_HTP_MCP or self.converter_export_format == qcc.EXPORT_FORMAT_DLC:
            net_run_cmd.append(f"--retrieve_context {self.output_path}/{self.context_binary}.bin")
        else:
            net_run_cmd.append(f"--model {self.model_binaries}/{self.target_arch}/{self.model_so}")

        net_run_cmd.extend(netrun_param_list)

        cmd = ' '.join(net_run_cmd)
        qacc_file_logger.info(cmd)
        executor = LocalExecutor()
        debug_log_file = self.debug_log_path + 'netrun.log'
        status = executor.run(cmd, log_file=debug_log_file)
        self.stage_status['qnn-net-run'] = not bool(status)
        if status != 0:
            Helper.dump_stage_error_log(debug_log_file)
            qacc_file_logger.error("qnn-net-run failed to run successfully")
            raise ce.QnnNetRunException("qnn-net-run failed to run successfully.")

    def _handle_extra_args_io_tensors(self, param_list):
        if "extra_args" not in self.netrun_params:
            return
        extra_args_str = self.netrun_params["extra_args"]
        extra_args_list = extra_args_str.split()
        for param in param_list:
            param_index = [i + 1 for i, val in enumerate(extra_args_list) if param in val]
            for inx in param_index:
                extra_args_list[inx] = Helper.sanitize_native_tensor_names(extra_args_list[inx])
        extra_args_str = ' '.join(extra_args_list)
        self.netrun_params["extra_args"] = extra_args_str

    def _run_on_target_device(self, netrun_param_list):
        """Runs the qnn-net-run inference on the target device through adb."""

        ADB_PATH = os.environ.get('ADB_PATH', qcc.DEFAULT_ADB_PATH)
        inputDataPath = os.path.dirname(self.input_path)
        configFilename, settingsFilename = None, None
        if os.path.exists(self.netrun_config_json):
            configFilename = os.path.basename(self.netrun_config_json)
        if os.path.exists(self.netrun_backend_extensions_json):
            settingsFilename = os.path.basename(self.netrun_backend_extensions_json)

        # extract just version number from backend
        dsp_version = None
        model_lib = None
        if 'dspv' in self.backend:
            dsp_version = 'v' + re.findall(r'\d+', self.backend)[0]
        elif self.backend == qcc.BACKEND_CPU or self.backend == qcc.BACKEND_GPU:
            model_lib = os.path.join(self.model_binaries, self.target_arch, self.model_so)

        adbDevice = AdbExecutor(pathToAdbBinary=ADB_PATH, deviceSerialNumber=self.device_id,
                                inputList=self.input_path, inputDataPath=inputDataPath,
                                graphDir=self.model_binaries, sdkDir=self.engine_path,
                                outputDir=self.work_dir, configFilename=configFilename,
                                settingsFilename=settingsFilename, dspVersion=dsp_version,
                                backend=self.backend, model_lib=model_lib,
                                netrun_param_list=netrun_param_list)

        adbDevice.buildQnnNetRunArgs()
        if adbDevice.pushArtifacts() == -1:
            qacc_logger.info(
                'Error pushing artifacts to target device (adb). Please check the console or logs for details.'
            )
            return -1
        else:
            if adbDevice.runModel() == -1:
                qacc_logger.info(
                    'Error running model on target (adb). Please check the console or logs for details.'
                )
                return -1
            else:
                if adbDevice.pullOutput() == -1:
                    qacc_logger.info(
                        'Error pulling output files from target device (adb). Please check the console or logs for details.'
                    )
                    return -1
                else:
                    if adbDevice.cleanup() == -1:
                        qacc_logger.info(
                            'Error cleaning up target device (adb). Please check the console or logs for details.'
                        )
                        return -1

        # Move all the files to one level above in the output directory for comparison
        source_dir = os.path.join(self.work_dir, "output")
        dest_dir = self.work_dir
        all_outputs = os.listdir(source_dir)
        for op_folder in all_outputs:
            shutil.move(os.path.join(source_dir, op_folder), os.path.join(dest_dir, op_folder))

    def _setup(self):
        """This function sets up the working directory and environment to
        execute QNN inferences.

        It should:
        - Setup the QNN execution environment on host x86 device
        """

        # Setting the paths to the executables
        self.MODEL_LIB_GENERATOR = os.path.join(self.engine_path, "bin", self.compile_arch,
                                                qcc.MODEL_LIB_GENERATOR)
        self.CONTEXT_BINARY_GENERATOR = os.path.join(self.engine_path, "bin", self.compile_arch,
                                                     qcc.CONTEXT_BINARY_GENERATOR)
        self.NET_RUN = os.path.join(self.engine_path, "bin", self.target_arch, qcc.NET_RUN)
        # DLC Path requires this so file to be supplied to --model cli argument
        self.libQnnModelDlc_path = os.path.join(self.engine_path, "lib", self.compile_arch,
                                                qcc.QNN_DLC_MODEL_SO)

        work_dir = os.path.join(os.getcwd(), self.output_path)
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir

        self.model_name = qcc.MODEL_IR_FILE
        # Output of Converter stage:
        # converted model output file path: i.e model.cpp or model.dlc
        self.converter_output = os.path.join(self.work_dir, qcc.MODEL_IR_FOLDER,
                                             self.model_name + '.' + self.converter_export_format)
        # converted model.bin file path. Generated only when export format is 'cpp'
        # Only Required by model lib generator
        self.converter_bin = os.path.join(self.work_dir, qcc.MODEL_IR_FOLDER,
                                          self.model_name + ".bin")

        # Output of model lib generator stage:
        self.model_binaries = os.path.join(self.work_dir, qcc.MODEL_IR_FOLDER,
                                           qcc.MODEL_BINARIES_FOLDER)
        self.model_so = "lib" + self.model_name + ".so"
        # Output of context binary generator stage
        self.context_binary = qcc.CONTEXT_BINARY_FILE
        self.context_backend_extensions_json = os.path.join(self.work_dir,
                                                            qcc.CONTEXT_BACKEND_EXTENSION_CONFIG)
        self.netrun_backend_extensions_json = os.path.join(self.work_dir,
                                                           qcc.NETRUN_BACKEND_EXTENSION_CONFIG)
        self.context_config_json = os.path.join(self.work_dir, qcc.CONTEXT_CONFIG)
        self.netrun_config_json = os.path.join(self.work_dir, qcc.NETRUN_CONFIG)

        # setup backend_path
        if self.backend == qcc.BACKEND_AIC:
            backend = qcc.AIC_BACKEND
            netrun_extension = qcc.AIC_NETRUN_EXTENSION
        elif self.backend == qcc.BACKEND_HTP or "dspv" in self.backend:
            backend = qcc.HTP_BACKEND
            netrun_extension = qcc.HTP_NETRUN_EXTENSION
        elif self.backend == qcc.BACKEND_HTP_MCP:
            backend = qcc.HTP_MCP_BACKEND
            netrun_extension = qcc.HTP_MCP_NETRUN_EXTENSION
        elif self.backend == qcc.BACKEND_CPU:
            backend = qcc.CPU_BACKEND
            netrun_extension = ""
        elif self.backend == qcc.BACKEND_GPU:
            backend = qcc.GPU_BACKEND
            netrun_extension = ""

        self.compiler_backend_path = os.path.join(self.engine_path, "lib", self.compile_arch,
                                                  backend)
        self.backend_path = os.path.join(self.engine_path, "lib", self.target_arch, backend)
        self.net_run_extension_path = os.path.join(self.engine_path, "lib", self.target_arch,
                                                   netrun_extension)

    def get_output_names(self):
        output_names = []
        # Add logic to use output_info from config in the same order
        if self.output_info:
            for output_name, output_node_info in self.output_info.items():
                if output_node_info[0] != 'float32' and self.use_native_output_files:
                    output_names.append(f"{output_name}_native")
                else:
                    output_names.append(output_name)
            # handle output name change introduced by tf converter
            model_type = Helper.get_model_type(self.model_path)
            if model_type == ModelType.TENSORFLOW:
                output_names = [
                    f"{output_name}_{idx}" for idx, output_name in enumerate(output_names)
                ]
        else:
            # Determine the output node names based on inference output filenames
            # This is required for minimal mode to work: output info is not mandatory in minimal mode
            for root, dirs, files in os.walk(os.path.join(self.work_dir, "Result_0")):
                for file in files:
                    filename, fileExt = os.path.splitext(file)
                    if fileExt == '.raw':
                        output_names.append(filename)
            output_names.sort()
        return output_names

    def gen_output_file(self):
        # Create the output file if requested.
        qacc_file_logger.debug(f"Generating output file {self.gen_out_file}")
        out_list_file = open(self.gen_out_file, 'w')
        # Output file names
        #assert self.output_info, 'Output names is mandatory'
        output_names = self.get_output_names()

        self.num_inputs = sum(1 for line in open(self.input_path))
        with open(self.gen_out_file, 'w') as F:
            for i in range(self.num_inputs):
                paths = []
                for out_name in output_names:
                    _path = os.path.join(self.output_path, f"Result_{i}/{out_name}.raw")
                    paths.append(_path)
                F.write(','.join(paths) + '\n')

    def convertInputsForBackend(self, isInt64to32=True):
        # Convert inputs to lower precision if not supported in backend
        toConvertInx = []
        if self.backend == qcc.BACKEND_HTP or "dspv" in self.backend:
            # Check which inputs are int64 and store their indices
            if self.input_info:
                inx = 0
                isInt64Inp = False
                for in_name, in_info in self.input_info.items():
                    if in_info[0] == 'int64':
                        toConvertInx.append(inx)
                        isInt64Inp = True
                    inx += 1
                if not isInt64Inp:
                    return
            file_list = [self.input_path]
            if self.calibration_file and (os.path.dirname(self.input_path) != os.path.dirname(
                    self.calibration_file)):
                file_list.append(self.calibration_file)
            # read the inputlist file and calibration file, select the int64 inputs and cast them
            # to int32
            for file_path in file_list:
                with open(file_path, 'r') as F:
                    paths = F.readlines()
                src_dt = np.int64
                dst_dt = np.int32
                if not isInt64to32:
                    src_dt = np.int32
                    dst_dt = np.int64
                for path_per_line in paths:
                    input_paths = path_per_line.split()
                    for i, path in enumerate(input_paths):
                        if i in toConvertInx:
                            input_path = path.split(":=")[1] if ":=" in path else path
                            inputSrc = np.fromfile(input_path, src_dt)
                            inputSrc = inputSrc.flatten()
                            inputDst = inputSrc.astype(dst_dt)
                            inputDst.tofile(input_path)

    def execute(self):
        """
        Executes the QNN workflow in sequence
        TODO: Separate the compile and execution stages.
        """

        self.convertInputsForBackend(isInt64to32=True)
        self.qnn_converter()
        # Model lib generator is required only when export format is 'cpp'
        if self.converter_export_format == qcc.EXPORT_FORMAT_CPP:
            self.qnn_model_lib_generator()
        # Invoke context binary generator only in case the backend is AIC or HTP-MCP
        # or if adb is enabled
        if self.require_qnn_context_binary_generator_step():
            self.qnn_context_binary_generator()
            self.qnn_net_run()
        else:
            self.qnn_net_run()
        self.convertInputsForBackend(isInt64to32=False)

        #Generate the infer output file to compare
        self.gen_output_file()

    @classmethod
    def get_calibration_skip_params(self):
        return ['quantization_overrides']

    def require_qnn_context_binary_generator_step(self):
        """Checks if context binary generator step is required during model
        evaluation.

        Returns:
            bool: Whether context binary generation step is required.
        """
        require_context_binary = False
        # when the backend is AIC or HTP-MCP then we need to generate context binary.
        if self.backend == qcc.BACKEND_AIC or self.backend == qcc.BACKEND_HTP_MCP:
            require_context_binary = True
        # if adb based evaluation is enabled then we need to generate context binary; HTP Device Case
        elif self.is_adb:
            require_context_binary = True
        # if the user sets export_format='dlc' and backend is not cpu with target_arch is x86 (i.e HTP Simulator Case) generate context binary
        elif self.converter_export_format == qcc.EXPORT_FORMAT_DLC:
            if (self.backend not in [qcc.BACKEND_CPU] and self.target_arch == "x86_64-linux-clang"):
                require_context_binary = True
        return require_context_binary
