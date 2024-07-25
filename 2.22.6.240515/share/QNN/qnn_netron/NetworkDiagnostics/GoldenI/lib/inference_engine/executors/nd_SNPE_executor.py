# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.inference_engine import inference_engine_repository
from lib.inference_engine.executors.nd_executor import Executor
from lib.utils.nd_constants import ComponentType, Framework, Engine


@inference_engine_repository.register(cls_type=ComponentType.executor,
                                      framework=None,
                                      engine=Engine.SNPE,
                                      engine_version="1.22.2.233")
class SNPEExecutor(Executor):
    def __init__(self, context):
        super(SNPEExecutor, self).__init__(context)
        self.executable = context.executable
        self.container = context.arguments["container"]
        self.input_list = context.arguments["input_list"]
        self.runtime = context.arguments["runtime"][context.runtime]
        self.environment_variables = context.environment_variables

        self.engine_path = context.engine_path
        self.target_arch = context.architecture
        self.target_path = context.target_path

    def get_execute_environment_variables(self):
        def fill(variable):
            return variable.format(target_path=self.target_path, target_arch=self.target_arch)

        return {(k, fill(v)) for k, v in self.environment_variables.items()}

    def build_execute_command(self, container, input_list):
        # type: (str, str) -> str
        execute_command_list = [self.executable, self.container, container, self.input_list, input_list, self.runtime]
        execute_command_str = ' '.join(execute_command_list)
        return execute_command_str
