# =============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import json

from lib.inference_engine import inference_engine_repository
from lib.inference_engine.executors.nd_executor import Executor
from lib.utils.nd_constants import ComponentType, Framework, Engine


@inference_engine_repository.register(cls_type=ComponentType.executor,
                                      framework=None,
                                      engine=Engine.QNN,
                                      engine_version="0.5.0.11262")
class QNNExecutor(Executor):
    def __init__(self, context):
        super(QNNExecutor, self).__init__(context)
        self.executable = context.executable
        self.model_path_flag = context.arguments["retrieve_context"]     \
        if context.offline_prepare else context.arguments["qnn_model_path"]

        self.input_list_flag = context.arguments["input_list"]
        self.backend_flag = context.arguments["backend"]
        self.op_package_flag = context.arguments["op_package"]
        self.output_dir_flag = context.arguments["output_dir"]
        self.debug_flag = context.arguments["debug"]
        self.input_data_type_flag = context.arguments["input_data_type"]
        self.output_data_type_flag = context.arguments["output_data_type"]
        self.profiling_level_flag = context.arguments["profiling_level"]
        self.perf_profile_flag = context.arguments["perf_profile"]
        self.config_file_flag = context.arguments["config_file"]
        self.log_level_flag = context.arguments["log_level"]
        self.version_flag = context.arguments["version"]

        self.environment_variables = context.environment_variables

        self.engine_path = context.engine_path
        self.target_arch = context.architecture
        self.target_path = context.target_path

    def get_execute_environment_variables(self):
        def fill(v):
            return v.format(sdk_tools_root=self.engine_path, target_arch=self.target_arch)

        return {(k, fill(v)) for k, v in self.environment_variables.items()}

    #TODO: temporary solution for updating paths, will need to correct this by supplying the same arguments for all modules
    # mainly used to update engine_path
    def updateField(self,attr,value):
        if hasattr(self, attr):
            setattr(self,attr,value)
            return True
        else:
            return False

    def build_execute_command(self, model_binary_path, backend_path, input_list_path, op_packages, execution_output_dir,
                              input_data_type, output_data_type, perf_profile, profiling_level,
                              debug_mode, log_level, print_version, config_json_file):
        # type: (str, str, str, List[str]) -> str
        """
        Build execution command using qnn-net-run

        model_binary_path: Path to QNN model binary
        backend_path: Path to backend (runtime) binary
        input_list_path: Path to .txt file with list of inputs
        op_package_list: List of paths to different op packages to include in execution

        return value: string of overall execution command using qnn-net-run
        """

        # includes required flags or those w/ default vals
        execute_command_list = [self.executable, self.model_path_flag,
                                model_binary_path,
                                self.backend_flag, backend_path,
                                self.input_list_flag, input_list_path, self.output_dir_flag, execution_output_dir,
                                self.input_data_type_flag, input_data_type,
                                self.output_data_type_flag, output_data_type,
                                self.perf_profile_flag, perf_profile]

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
        if debug_mode:
            execute_command_list.append(self.debug_flag)
        if print_version:
            execute_command_list.append(self.version_flag)
        return " ".join(execute_command_list)
