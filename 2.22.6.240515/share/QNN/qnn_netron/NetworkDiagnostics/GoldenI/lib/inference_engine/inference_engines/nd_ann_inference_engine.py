# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import shutil
import time
import importlib

import absl.logging
import logging

from subprocess import CalledProcessError
from distutils.version import StrictVersion

from lib.inference_engine import inference_engine_repository
from lib.inference_engine.inference_engines.nd_inference_engine import InferenceEngine
from lib.utils.nd_constants import ComponentType, Engine, Framework, Runtime, Status
from lib.utils.nd_exceptions import InferenceEngineError
from lib.utils.nd_errors import get_message
from lib.utils.nd_logger import setup_logger


@inference_engine_repository.register(cls_type=ComponentType.inference_engine,
                                      framework=None,
                                      engine=Engine.ANN,
                                      engine_version="1.2")
class ANNInferenceEngine(InferenceEngine):

    def __init__(self, context, converter, executor):
        super(ANNInferenceEngine, self).__init__(context, converter, executor)
        # Import these in the __init__ function, so we don't have to have these
        # libraries in other inference engine venvs:
        self.tf = importlib.import_module('tensorflow')
        self.tflite = importlib.import_module('tflite')

        self.host_device = context.host_device
        self.target_device = context.target_device
        self.logger = context.logger
        self.model_inputs = context.model_inputs
        self.model_outputs = context.model_outputs
        self.model_path = context.model_path
        self.wd = context.wd
        self.host_output_dir = context.output_dir
        self.target_path = self.executor.target_path
        self.executables = context.executables
        self.framework = context.framework
        self.framework_version = context.framework_version
        self.executable = self.executor.executable
        self.local_executable_path = None
        self.runtime = context.runtime
        self.controls = context.controls
        self.available_runtimes = self.controls["runtimes"]
        self.available_runtime_names = self.available_runtimes.keys()
        self.acceleration = '1'
        self.relaxed = '0'
        self.engine_type = context.engine_type
        self.nnhal_runtimes = [Runtime.nnhal_cpu, Runtime.nnhal_gpu, Runtime.nnhal_gpu_relaxed,
                               Runtime.nnhal_dsp]
        self.process_tag = self.controls["process_tag"]
        self.local_executable_folder = os.path.join(os.path.dirname(__file__), '..')

        # Fields for creating intermediate tflite files:
        self._interpreter = None
        self._tensor_map = None
        self._tensor_to_op_map = None
        self._model_buffer = None
        self._model = None
        self.intermediate_tensors = None
        self._model_files = None

        # Working directory and file location constants
        self._INTERMEDIATE_MODEL_FILE_PATH = os.path.join(self.wd, 'tflite_intermediate_models')
        self._INTERMEDIATE_MODEL_NAME = 'intermediate_model.tflite'

        # To stop duplicate logging from Tensorflow:
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False

    def _copy_executable(self, rel_path):
        new_exec_path = os.path.join(self.local_executable_folder, os.path.dirname(rel_path),
                                     self.executable)
        shutil.copyfile(os.path.join(self.local_executable_folder, rel_path), new_exec_path)
        return new_exec_path

    def _select_executable(self):

        valid_versions = self.executables[self.framework.value].keys()

        if not self.framework_version:
            latest_version = str(max(valid_versions,
                                     key=lambda x: StrictVersion(x)))
            self.logger.info("No framework version specified, defaulting to version {}"
                             .format(latest_version))
            self.local_executable_path = self._copy_executable(
                self.executables[self.framework.value][latest_version])
        elif self.framework_version in str(valid_versions):
            self.local_executable_path = self._copy_executable(
                self.executables[self.framework.value][self.framework_version])
        else:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_INVALID_EXECUTABLE")
                                       (self.framework_version, list(valid_versions)))

    def _setup(self):
        if not self.framework.value == Framework.tflite.value:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_UNSUPPORTED_FRAMEWORK")
                                       (self.framework.value, Engine.ANN))
        self._select_executable()

        self.logger.info('Pushing binaries to target')
        target_executable_path = os.path.join(self.target_path, self.executable)
        self._push_to_device(self.local_executable_path, target_executable_path)
        chmod_command = 'chmod 777 ' + target_executable_path
        code, _, err = self.target_device.execute(commands=[chmod_command], cwd=self.target_path)
        if code != 0:
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CHMOD_FAILED'))

        self.logger.info('Pushing input data to target')
        for _, _, input in self.model_inputs:
            self._push_to_device(input, os.path.join(self.target_path, os.path.basename(input)))

    def _push_to_device(self, source, target):
        code, _, err = self.target_device.push(source, target)
        if code != 0:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

    def _execute_inference(self):

        inputs = [os.path.basename(data) for _, _, data in self.model_inputs]
        for output in self.intermediate_tensors:
            try:
                output_filename = self._model_files[output]
            except:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_MODEL_FILE_DOES_NOT_EXIST")(output))

            intermediate_model_path = os.path.join(self._INTERMEDIATE_MODEL_FILE_PATH, output_filename)
            self.logger.info('Pushing {} model to target'.format(output))
            self._push_to_device(intermediate_model_path, self.target_path + '/')
            # Rename file, so it is overwritten everytime we push (saves space on device):
            rename_command = 'mv {} {}'.format(os.path.basename(intermediate_model_path), self._INTERMEDIATE_MODEL_NAME)

            try:
                code, _, err = self.target_device.execute(commands=[rename_command],
                                                          cwd=self.target_path)
                if code != 0:
                    raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))
            except CalledProcessError as exc:
                self.logger.error(str(exc))
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))

            self.target_device.make_directory(os.path.join(self.target_path,
                                                           os.path.dirname(output)))
            execute_command = self.executor.build_execute_command(self.acceleration, self.relaxed,
                                                                  self._INTERMEDIATE_MODEL_NAME,
                                                                  inputs, [output])

            log_string = 'Running inference with: ' + \
                         'Inputs: ' + str(inputs) + ' ' + \
                         'Outputs: ' + str(output)
            self.logger.info(log_string)

            try:
                code, _, err = self.target_device.execute(commands=[execute_command],
                                                          cwd=self.target_path)
                if code != 0:
                    raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))
                self.logger.info('Inference executed successfully')
            except CalledProcessError as exc:
                self.logger.error(str(exc))
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))

    def _pull_inference_results(self):

        for index, name in enumerate(self.intermediate_tensors):
            # Use 0 in the raw file name, since we only outputted 1 output tensor
            # from each inference:
            raw_file = name + '0' + '.raw'
            new_raw_file_name = name + '.raw'
            code, _, err = self.target_device.pull(
                os.path.join(self.target_path, raw_file),
                os.path.join(self.host_output_dir, new_raw_file_name))
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED"))

    def _prepare_runtime(self):
        def toggle(runtime, status):
            if status == Status.off:
                self.logger.info("Attempting to disable {}".format(runtime))
                error_msg = "ERROR_INFERENCE_ENGINE_DISABLE_ACCELERATION_FAILED"
                success_msg = "Disabled {}".format(runtime)
            elif status == Status.on:
                self.logger.info("Attempting to enable {}".format(runtime))
                error_msg = "ERROR_INFERENCE_ENGINE_ENABLE_ACCELERATION_FAILED"
                success_msg = "Enabled {}".format(runtime)
            else:
                assert status == Status.on or status == Status.off, ("Invalid status. "
                                                                     "Status must be on/off.")

            command = self.available_runtimes[runtime][status.value]
            code, _, err = self.target_device.execute(commands=[command],
                                                      cwd=self.target_path)

            if code != 0:
                raise InferenceEngineError(get_message(error_msg)(runtime, command))
            self.logger.info(success_msg)

        def kill():
            code, output, _ = self.target_device.execute(commands=[self.controls["find_process"]],
                                                         cwd=self.target_path)

            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_PROCESS_SEARCH_FAILED'))
            if not output:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_PROCESS_NOT_FOUND')(self.process_tag))

            code, _, _ = self.target_device.execute(commands=[self.controls["kill"]
                                                              + output[0]], cwd=self.target_path)

            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_KILL_PROCESS_FAILED')
                    (output[0]))

        for runtime in self.available_runtime_names:
            toggle(runtime, Status.off)

        if self.runtime == Runtime.ann_cpu:
            # Do not enable any NNHAL properties
            pass
        elif self.runtime in self.nnhal_runtimes:
            toggle(self.runtime.value, Status.on)
            if self.runtime == Runtime.nnhal_gpu_relaxed:
                self.relaxed = '1'
        elif self.runtime == Runtime.tflite_cpu:
            self.acceleration = '0'
        else:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_RUNTIME_INVALID")
                                       (self.runtime.value, self.engine_type.value))

        kill()
        time.sleep(5)

    def _generate_intermediate_tflite_files(self):
        self.logger.info('Generating intermediate TFLite Model Files')
        self._load_model(self.model_path)
        input_tensor_names = [name[0] for name in self.model_inputs]
        self.intermediate_tensors = []
        self._model_files = {}  # dictionary of tensor name to filename
        tensor_pairs = self._get_intermediate_tensors(input_tensor_names, self.model_outputs)
        for _, output_tensor_names in tensor_pairs:
            for output_name in output_tensor_names:
                self.intermediate_tensors.append(output_name)
                output_index = self._tensor_map[output_name]
                new_model_buffer = self._buffer_change_output_tensor_to(output_index)
                # Save model_buffer to working directory:
                output_filename = os.path.join(self._INTERMEDIATE_MODEL_FILE_PATH, output_name + '.tflite')
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                with open(output_filename, 'wb') as f:
                    f.write(new_model_buffer)
                # After model file is created, store name in dictionary:
                self._model_files[output_name] = output_filename
    def run(self):
        self._generate_intermediate_tflite_files()
        self._setup()
        self._prepare_runtime()
        self._execute_inference()
        self._pull_inference_results()

    def get_graph_structure(self):
        pass

    def _load_model(self, model_path):
        self._interpreter = self.tf.lite.Interpreter(model_path=model_path)
        self._model_buffer = open(model_path, "rb").read()
        self._model = self.tflite.Model.GetRootAsModel(bytearray(self._model_buffer), 0)
        self._interpreter.allocate_tensors()

    def _buffer_change_output_tensor_to(self, new_tensor_i):
        """
        Reads model_buffer as a proper flatbuffer file and gets the offset programatically.
        Set subgraph 0's output(s) to new_tensor_i.
        """

        # Custom added function (OutputsOffset) to return the file offset to this vector :
        try:
            output_tensor_index_offset = self._model.Subgraphs(0).OutputsOffset(0)
        except:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_CUSTOM_FUNCTION_NOT_ADDED"))
        # Flatbuffer scalars are stored in little-endian.
        new_tensor_i_bytes = bytes([
          new_tensor_i & 0x000000FF, \
          (new_tensor_i & 0x0000FF00) >> 8, \
          (new_tensor_i & 0x00FF0000) >> 16, \
          (new_tensor_i & 0xFF000000) >> 24 \
        ])
        # Replace the 4 bytes corresponding to the first output tensor index
        return self._model_buffer[:output_tensor_index_offset] + \
            new_tensor_i_bytes + \
            self._model_buffer[output_tensor_index_offset + 4:]

    def _get_intermediate_tensors(self, input_tensors, output_tensors):
        # type: (List[str], List[str]) -> List[Tuple[List[str]]]

        # tensor_details = self._interpreter.get_tensor_details()
        output_found = [False for i in range(len(output_tensors))]

        input_details = self._interpreter.get_input_details()
        model_input_names = [tensor['name'] for tensor in input_details]

        # Build up tensor_map (dictionary of tensor names to indices):
        self._tensor_map = {}
        self._tensor_to_op_map = {}
        tensor_list = self._interpreter.get_tensor_details()
        for tensor in tensor_list:
            self._tensor_map[tensor['name']] = tensor['index']
            self._tensor_to_op_map[tensor['index']] = -1

        # All input_tensors must be part of the model's input details!:
        for name in input_tensors:
            if (name not in model_input_names):
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_UNSUPPORTED_INPUT_TENSOR")
                                                      (input_tensors, model_input_names))
        # All output tensors must be a valid tensor:
        for name in output_tensors:
            if (name not in self._tensor_map):
                raise InferenceEngineError(get_message
                                           ("ERROR_INFERENCE_ENGINE_UNSUPPORTED_OUTPUT_TENSOR")
                                           (name))

        # Let us see what inputs are not provided in input_tensors, but are in the model's
        # input details:
        # We may not need these inputs depending on the model structure and output tensors given:
        neglected_inputs = []
        for name in model_input_names:
            if (name not in input_tensors):
                neglected_inputs.append(name)

        # Build up tensor_to_op_map (dictionary of tensor names to lists of ops):
        # In tensor_to_op_map, the keys are the tensor index
        # Each value in the dict is the op that outputs the corresponding tensor key
        for op in range(self._model.Subgraphs(0).OperatorsLength()):
            operator = self._model.Subgraphs(0).Operators(op)
            for k in range(operator.OutputsLength()):
                self._tensor_to_op_map[operator.Outputs(k)] = op

        # After this, if the tensor_to_op_map value is still -1, then the tensor is an input tensor

        op_list = self._operator_list(input_tensors, output_tensors)
        tensor_pairs = []
        for i in op_list:
            inputs = []
            outputs = []
            operator = self._model.Subgraphs(0).Operators(i)
            for j in range(operator.InputsLength()):
                name = tensor_list[operator.Inputs(j)]['name']
                inputs.append(name)
                # If a neglected input is encountered, the user didn't give enough inputs:
                if (name in neglected_inputs):
                    raise InferenceEngineError(get_message
                                               ("ERROR_INFERENCE_ENGINE_UNSUPPORTED_INPUT_TENSOR")
                                               (input_tensors, model_input_names))
            for k in range(operator.OutputsLength()):
                name = tensor_list[operator.Outputs(k)]['name']
                outputs.append(name)
                if name in output_tensors:
                    output_found[output_tensors.index(name)] = True
            tensor_pairs.append((inputs, outputs))

        if (not all(output_found)):  # If we didn't find some outputs, something went wrong
            for i in range(len(output_found)):
                if (not output_found[i]):
                    raise InferenceEngineError(get_message
                                               ("ERROR_INFERENCE_ENGINE_UNSUPPORTED_OUTPUT_TENSOR")
                                               (output_tensors[i]))

        return tensor_pairs

    def _operator_list(self, input_tensors, output_tensors):
        # type: (List[str], List[str]) -> List[int]
        """
        Calls DFS on model starting from each operation that outputs an element from
        output_tensors.
        Returns a list of operator indices that were encountered in this process

        input_tensors: Inputted list of input tensor names
        output_tensors: Inputted list of output tensor names

        op_list: Outputted list of operator indices that were encountered in DFS
        """

        op_list = []
        output_indices = [self._tensor_map[name] for name in output_tensors]
        # Start from output node:
        for i, output_idx in enumerate(output_indices):
            output_op = self._tensor_to_op_map[output_idx]
            if (output_op == -1):
                # This tensor is not outputted from any ops, meaning it is an input, constant, etc.
                # As such, it should not be labelled as an output tensor!:
                raise InferenceEngineError(get_message
                                           ("ERROR_INFERENCE_ENGINE_UNSUPPORTED_OUTPUT_TENSOR")
                                           (output_tensors[i]))
            if (output_op not in op_list):
                self._dfs_operator_list_stack(output_op, op_list)

        return op_list

    def _dfs_operator_list_stack(self, op_idx, op_list):
        # type: (int, List[int]) -> List[int]
        """
        Performs DFS on the model starting from operator index op_idx, using
        stack based implementation.

        op_idx: Inputted index where DFS starts from
        op_list: Inputted list of operator indices visited so far. Is appended to in this function,
        and since Lists are mutable, changes in it are seen by the calling function.
        """

        stack = []
        stack.append(op_idx)
        while (len(stack) != 0):
            curr_idx = stack.pop()
            if (curr_idx not in op_list):
                op_list.append(curr_idx)
                # Find adjacent Nodes:
                # Adjacent nodes (ops) are ones where the outputs of the adjacent op
                # include the input of the current op:
                operator = self._model.Subgraphs(0).Operators(curr_idx)
                for i in range(operator.InputsLength()):
                    # Use tensor_to_op_map to see which ops have the current input
                    # as an output:
                    adj_op = self._tensor_to_op_map[operator.Inputs(i)]
                    if (adj_op != -1):
                        stack.append(adj_op)
