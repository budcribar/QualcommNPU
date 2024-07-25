# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os

from lib.inference_engine import inference_engine_repository
from lib.inference_engine.executors.nd_executor import Executor
from lib.utils.nd_constants import ComponentType, Framework, Engine


@inference_engine_repository.register(cls_type=ComponentType.executor,
                                      framework=None,
                                      engine=Engine.ANN,
                                      engine_version="1.2")
class ANNExecutor(Executor):
    def __init__(self, context):
        super(ANNExecutor, self).__init__(context)
        self.executable = context.executable
        self.model = context.arguments["model"]
        self.input = context.arguments["input"]
        self.output = context.arguments["output"]
        self.acceleration = context.arguments["acceleration"]
        self.relaxed = context.arguments["relaxed"]
        self.target_path = context.target_path

    def build_execute_command(self, acceleration, relaxed, model, input_data, output_data):
        # type: (str, str, str, list, list) -> str
        path_executable = os.path.join(self.target_path, self.executable)
        execute_command_list = [path_executable, self.acceleration, acceleration, self.relaxed,
                                relaxed, self.model, model, self.input, str(len(input_data))]

        execute_command_list.extend(input_data)
        execute_command_list.append(self.output)
        execute_command_list.append(str(len(output_data)))
        execute_command_list.extend(output_data)

        execute_command_str = ' '.join(execute_command_list)

        return execute_command_str
