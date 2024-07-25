# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import json
import logging
from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.inference_engine.executors.nd_executor import Executor
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Framework, Engine, \
    X86_windows_Architectures, Windows_Architectures


@inference_engine_repository.register(cls_type=ComponentType.executor, framework=None,
                                      engine=Engine.QNN, engine_version="0.5.0.11262")
class QNNExecutor(Executor):

    def __init__(self, context):
        super(QNNExecutor, self).__init__(context)
        self.executable = context.executable
        # Windows executable are differentiated by having '.exe' in the name.
        if context.architecture in ['x86_64-windows-msvc', 'wos-remote', 'wos']:
            self.executable = context.windows_executable

        self.model_path_flag = context.arguments["retrieve_context"]     \
        if context.offline_prepare else context.arguments["qnn_model_path"]

        self.input_list_flag = context.arguments["input_list"]
        self.backend_flag = context.arguments["backend"]
        self.op_package_flag = context.arguments["op_package"]
        self.output_dir_flag = context.arguments["output_dir"]
        self.debug_flag = context.arguments["debug"]
        self.use_native_input_files = context.arguments["use_native_input_files"]
        self.use_native_output_files = context.arguments["use_native_output_files"]
        self.profiling_level_flag = context.arguments["profiling_level"]
        self.perf_profile_flag = context.arguments["perf_profile"]
        self.config_file_flag = context.arguments["config_file"]
        self.log_level_flag = context.arguments["log_level"]
        self.version_flag = context.arguments["version"]
        self.output_tensor_flag = context.arguments["output_tensors"]

        self.environment_variables = {}
        self.engine_path = context.engine_path
        self.target_arch = context.architecture
        if self.target_arch == "x86_64-linux-clang":
            self.target_path = ''
        elif self.target_arch == "aarch64-qnx":
            self.target_path = context.target_path["qnx"]
            self.environment_variables = context.environment_variables["qnx"]
        elif self.target_arch == "wos-remote":
            self.target_path = context.target_path["wos"]
        else:  # android backend
            self.target_path = context.target_path["android"]
            self.environment_variables = context.environment_variables["android"]
        self.qnn_model_name = context.model_name if context.model_name is not None else "model"

    def get_execute_environment_variables(self):

        def fill(v):
            return v.format(sdk_tools_root=self.engine_path, target_arch=self.target_arch)

        return {(k, fill(v)) for k, v in self.environment_variables.items()}

    def updateField(self, attr, value):
        if hasattr(self, attr):
            setattr(self, attr, value)
            return True
        else:
            return False

    def build_execute_command(self, model_binary_path, backend_path, input_list_path, op_packages,
                              execution_output_dir, use_native_input_files, use_native_output_files,
                              perf_profile, profiling_level, debug_option_on, log_level,
                              print_version, config_json_file, extra_runtime_args=None,
                              intermediate_output_tensors=None):
        # type: (str, str, str, List[str]) -> str
        """Build execution command using qnn-net-run.

        model_binary_path: Path to QNN model binary
        backend_path: Path to backend (runtime) binary
        input_list_path: Path to .txt file with list of inputs
        op_package_list: List of paths to different op packages to include in execution

        return value: string of overall execution command using qnn-net-run
        """
        # includes required flags or those w/ default vals
        execute_command_list = [
            self.executable, self.model_path_flag, model_binary_path, self.backend_flag,
            backend_path, self.input_list_flag, input_list_path, self.output_dir_flag,
            execution_output_dir, self.perf_profile_flag, perf_profile
        ]

        if len(op_packages) > 0:
            execute_command_list.append(self.op_package_flag)
            execute_command_list.append('\'' + ' '.join(op_packages) + '\'')
        if profiling_level is not None:
            execute_command_list.append(self.profiling_level_flag)
            execute_command_list.append(profiling_level)
        if log_level is not None:
            execute_command_list.append(self.log_level_flag)
            execute_command_list.append(log_level)
        if config_json_file:
            execute_command_list.append(self.config_file_flag)
            execute_command_list.append(config_json_file)
        #'--set_output_tensors' option of qnn-net-run should not be used with '--retrieve_context'
        # and '--debug'options
        if self.model_path_flag != "--retrieve_context" and intermediate_output_tensors:
            execute_command_list.extend(
                [self.output_tensor_flag, self.qnn_model_name + ':' + intermediate_output_tensors])
        else:
            logging.warning("Skipped --set_output_tensors option by executor of inference engine")
        if debug_option_on and not intermediate_output_tensors:
            execute_command_list.append(self.debug_flag)
        if print_version:
            execute_command_list.append(self.version_flag)
        if extra_runtime_args:
            execute_command_list.append(extra_runtime_args)
        if use_native_input_files:
            execute_command_list.append(self.use_native_input_files)
        if use_native_output_files:
            execute_command_list.append(self.use_native_output_files)
        return " ".join(execute_command_list)
