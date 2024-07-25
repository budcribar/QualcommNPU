# =============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import absl.logging
import zipfile
import subprocess
import os
import json
from collections import OrderedDict
import builtins #builtins library was included to mock builtins.open in the unittests.
from packaging import version

from lib.inference_engine import inference_engine_repository
from lib.inference_engine.inference_engines.nd_inference_engine import InferenceEngine
from lib.utils.nd_constants import ComponentType, Engine
from lib.utils.nd_exceptions import InferenceEngineError, ProfilingError, DependencyError
from lib.utils.nd_errors import get_message
from lib.utils.nd_path_utility import get_absolute_path


@inference_engine_repository.register(cls_type=ComponentType.inference_engine,
                                      framework=None,
                                      engine=Engine.QNN,
                                      engine_version="1.1.0.22262")
class QNNInferenceEngine(InferenceEngine):
    def __init__(self, context, converter, executor):
        super().__init__(context, converter, executor)

        # Instantiate Class Fields from context:
        # Fields from context
        self.engine_type = context.engine
        self.engine_version = context.engine_version
        self.stage=context.stage
        if context.engine_path.endswith(".zip"):
            self.engine_zip_path = context.engine_path
            self.engine_path=None
        else:
            self.engine_path=context.engine_path
            self.engine_zip_path=None
        self.host_device = context.host_device
        self.target_device = context.target_device
        #input_tensor accepted here should be a list of lists of len of 3 [tensor, dim, data]
        self.model_inputs = context.input_tensor
        self.model_outputs = context.output_tensor
        self.model_path = context.model_path
        self.host_output_dir = context.output_dir
        self.target_arch = context.architecture
        self.target_path = self.executor.target_path
        self.runtime = context.runtime
        self.executable_location = context.executable_location
        if self.target_arch == "x86_64-linux-clang":
            self.backend_path = context.backend_locations["x86"][self.runtime]
            self.op_packages = context.op_packages["x86"][self.runtime]
        else:  # android backend
            self.backend_path = context.backend_locations["android"][self.runtime]
            self.op_packages = context.op_packages["android"][self.runtime]
        self.interface_module = context.op_packages["interface"]
        self.ndk_path = context.ndk_path
        self.clang_path = context.clang_path.format(android_ndk_path=self.ndk_path)
        self.sdk_tools_root = context.sdk_tools_root
        self.env_variables = context.environment_variables
        self.logger = context.logger
        #quantization
        self.input_list_txt = context.input_list
        self.quantization_overrides = context.quantization_overrides
        self.param_quantizer = context.param_quantizer
        self.act_quantizer = context.act_quantizer
        self.weights_bitwidth = context.weights_bitwidth
        self.bias_bitwidth = context.bias_bitwidth
        self.act_bitwidth = context.act_bitwidth
        self.algorithms = context.algorithms
        self.ignore_encodings = context.ignore_encodings
        self.use_per_channel_quantization = context.per_channel_quantization

        self.input_data_type = context.input_data_type
        self.output_data_type = context.output_data_type

        self.binaries_dir = os.path.join(self.host_output_dir, context.binaries_dir) if context.binaries_dir is not None else None
        self.qnn_model_cpp = context.qnn_model_cpp_path
        self.qnn_model_bin = context.qnn_model_bin_path
        self.qnn_model_binary = context.qnn_model_binary_path
        self.qnn_model_net_json = context.qnn_model_net_json

        # Lib Generator
        self.qnn_model_name = context.model_name if context.model_name is not None else "qnn_model"
        self.lib_target = context.lib_target
        self.lib_name = context.lib_name if context.lib_name is not None else 'qnn_model'
        self.context_binary_generator_config = context.context_binary_generator_config
        self.offline_prepare = context.offline_prepare
        # Lib Generator commands:
        self.lib_generator_executable = context.lib_generator["executable"]
        self.lib_generator_args = context.lib_generator["arguments"]

        # Profiler
        self.profiler_executable = context.profiler["executable"]
        self.profiler_path = context.profiler["executable_path"]
        self.profiler_args = context.profiler["arguments"]

        # libcpp_dependency
        self.libcpp_dependency = context.libcpp_dependency

        # qnn-net-run parameters
        self.profiling_level = context.profiling_level
        self.perf_profile = context.perf_profile
        self.print_version = context.print_version
        self.debug_mode = context.debug_mode
        self.log_level = context.log_level
        self.netrun_config_file = context.qnn_netrun_config_file
        self.be_ext_shared_library = context.backend_extension_shared_library_path

        # To stop duplicate logging from Tensorflow:
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False

        # Other private parameters

        #both _original_input_paths and _full_path_input_paths could contain tensor specific
        #inputs like input:=datapath.raw or just normal data path like path/to/data1.raw
        #_full_path_input_paths are basically the same as _original_input_paths except all the path inside are absolute path
        #paths in _original_input_paths could be absolute or relative. We should use _full_path_input_paths to refer to data path
        #from the input_list for most of the time
        self._original_input_paths = None

        #This is used to store the comments from the input_list.txt used to identify output node for SNPE, not used for QNN but keeping for compatibility purpose.
        self._input_list_comments = ""
        self._full_path_input_paths = None
        self._target_output_dir = None
        self._host_env= {}
        self._target_model_path = None
        self._target_backend = None


    # -------------------------------------------------- HELPER FUNCTIONS ----------------------------------------------------
    def _setup(self):
        """
        This function sets up the working directory and environment to execute QNN inferences
        It should:
        - Unzip the QNN SDK into the working directory
        - Setup the QNN execution environment on host x86 device
        """
        # Unzip SDK:
        self._validate_engine_path()

        #Update Executor engine_path to be the unzipped path if originally provided with .zip path:
        if(not self.executor.updateField('engine_path', self.engine_path)):
            self.logger.error("failed to update executor engine_path")

        #setup backend_path
        self.backend_path=[source.format(engine_path=self.engine_path,target_arch=self.target_arch) for source in self.backend_path]

        #setup executable_location
        self.executable_location=self.executable_location.format(engine_path=self.engine_path,target_arch=self.target_arch)

        #moved from init incase engine_path not setup first
        for pkg in self.op_packages:
            if not os.path.exists(pkg.format(engine_path=self.engine_path, target_arch=self.target_arch)):
                self.op_packages.remove(pkg)

        #setting up the profiler_path
        self.profiler_path=self.profiler_path.format(engine_path=self.engine_path,target_arch=self.target_arch)

        # get original input list paths
        #_original_input_paths stores all rel input paths in form of list of lists;
        # ie. if a input list has 2 batch and each batch require 3 inputs then the _original_input_paths would look like:
        # [[batch1_input1,batch1_input2,batch1_input3],[batch2_input1,batch2_input2,batch2_input3]]
        with open(self.input_list_txt, "r") as input_list:
            self._original_input_paths = []
            for line in input_list.readlines():
                if line.startswith("#"):
                    self._input_list_comments = line
                else:
                    self._original_input_paths.append(line.strip(' \n').split(' '))

        #set _full_path_input_paths:
        self._set_input_list()

        self._set_host_environment()

        # starting with source framework
        if self.stage == 'source':
            self._execute_conversion()
            self._create_model_binaries()
        # starting with .cpp and .bin
        elif self.stage == 'converted':
            self._create_model_binaries()
        self._push_required_files()

    def _validate_engine_path(self):
        """
        This helper function unzips engine_zip and sets the engine_path to the correct path.
        """
        if not self.engine_path and self.engine_zip_path:
            #Zipfile is breaking the symlink while extracting. So using subprocess for extracting
            try:
                subprocess.run(['unzip', '-q', self.engine_zip_path, '-d', self.host_output_dir], stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print("ERROR: Extracting SDK with the following error: ", err.returncode)
            with zipfile.ZipFile(self.engine_zip_path, 'r') as f:
                filelists=f.namelist()
                for file in filelists:
                    os.chmod(os.path.join(self.host_output_dir, file), 0o755)
            if './' in  filelists[0]:
                self.engine_path = os.path.join(self.host_output_dir, os.path.dirname(filelists[1]))
            else:
                self.engine_path = os.path.join(self.host_output_dir, os.path.dirname(filelists[0]))
        elif not os.path.isdir(self.engine_path):
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_PATH_INVALID")
                                       (self.engine_path))

    def _set_host_environment(self):
        """
        This helper function sets up the QNN execution environment on host x86 device.
        """
        # Get file paths:
        self.sdk_tools_root = self.sdk_tools_root.format(engine_path=self.engine_path)

        for var in self.env_variables:
            self.env_variables[var] = (self.env_variables[var]).format(sdk_tools_root=self.sdk_tools_root)

        # set environment:
        for var in self.env_variables:
            self._host_env[var] = self.env_variables[var] + os.pathsep + '$' + var
        # Add   path to PATH:
        self._host_env['ANDROID_NDK_ROOT'] = self.ndk_path
        self._host_env['QNN_SDK_ROOT'] = self.engine_path
        self._host_env['PATH'] = self.ndk_path + os.pathsep + self.clang_path + os.pathsep + self._host_env['PATH']

    @staticmethod
    def build_be_ext_config_json(output_file_path, config_file_path, shared_library_path):
        """
        utility method to help building the backend extension config json file by providng the
        .conf file. If .conf file not provided, returning None, else return output_file_path
        """
        if not config_file_path:
            return None

        config={
            "backend_extensions": {
                "shared_library_path": shared_library_path,
                "config_file_path": config_file_path
            }
        }
        with open(output_file_path, "w") as file:
            json.dump(config,file,indent=4)
        return output_file_path

    def _get_binaries_to_push(self):
        """
        Get QNN binaries used to convert and run a model.

        :return: List of binary paths
        """
        self.backend_path.append(self.executable_location)
        self.backend_path.append(self.profiler_path)
        for pkg in self.op_packages:
            self.backend_path.append(pkg)
        return self.backend_path

    def _push_input_list(self):
        """
        Create an input list on the host device. and push to target device,
        if the target device is x86, it should not call this function
        """
        if self.model_inputs is not None:
            # Set input list file name
            #using the tensor_name as the input_list_name
            layers = ['-'.join(tensor_name.split('/')) for tensor_name, _, _ in self.model_inputs]
            input_list_name = '_'.join(layers) + '.txt'

            #device_input_list_host_path is the inputlist to be used on device but needs to store a copy on the host (x86), this is because on the file path
            #within the device_intput_list should be device path based ie. /data/local/tmp/, hence this device_input_list can not be used on host.
            device_input_list_host_path = os.path.join(self.host_output_dir, input_list_name)
            self.target_input_list_path = os.path.join(self.target_path, input_list_name)

            string_to_write = ' '.join([tensor_name + ":=" + data_path.split("/")[-1] for tensor_name, dims, data_path in self.model_inputs])
            string_to_write += '\n'
            with open(device_input_list_host_path, 'w+') as f:
                f.write(self._input_list_comments)
                f.write(string_to_write)
            code, _, err = self.target_device.push(device_input_list_host_path, self.target_input_list_path)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
        else:
            # create a new input list with on-device paths to input data
            if ":=" not in self._full_path_input_paths[0][0]:
                on_device_input_paths = [[os.path.join(self.target_path, os.path.basename(rel_path)) for rel_path in line] for line in self._full_path_input_paths]
            else:
                on_device_input_paths = [[rel_path.split(":=")[0]+":="+os.path.join(self.target_path, os.path.basename(rel_path.split(":=")[1])) for rel_path in line] for line in self._full_path_input_paths]
            device_input_list_host_path = os.path.join(self.host_output_dir, 'device_input_list.txt')
            self.target_input_list_path = os.path.join(self.target_path, os.path.basename(device_input_list_host_path))
            with open(device_input_list_host_path, 'w') as d:
                d.write(self._input_list_comments)
                d.write(('\n'.join([' '.join(per_batch) for per_batch in on_device_input_paths])) + '\n')
            code, _, err = self.target_device.push(device_input_list_host_path, self.target_input_list_path)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

    def _push_config_file(self):
        """
        Create configFile and push backend Ext Files to proper path. Only called for On Device runtime not for x86.
        """
        if self.netrun_config_file:
            self.logger.info('Pushing config file to target')
            config_json_file=os.path.join(self.host_output_dir, "be_ext_config_file.json")
            netrun_config_file_target_path=os.path.join(self.target_path,os.path.basename(self.netrun_config_file))
            self.build_be_ext_config_json(config_json_file, netrun_config_file_target_path, self.be_ext_shared_library)

            code, _, err = self.target_device.push(self.netrun_config_file, netrun_config_file_target_path)
            code, _, err = self.target_device.push(config_json_file, self.target_path)
            code, _, err = self.target_device.push(os.path.join(os.path.dirname(self.backend_path[0]), self.be_ext_shared_library), self.target_path)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
            return os.path.basename(config_json_file)
        else:
            return None

    def _set_input_list(self):
        """
        This function prepends input list's current directory to each of the
        relative paths in input list, resulting in an input list with absolute paths.
        :param curr_dir: input list's current directory
        """
        curr_dir = os.path.dirname(self.input_list_txt)

        # this here basically means for each item in each line of _original_input_paths, make it an absolute path
        self._full_path_input_paths = [[get_absolute_path(rel_path, checkExist=True, pathPrepend=curr_dir) if ":=" not in rel_path \
            else rel_path.split(":=")[0]+":="+ get_absolute_path(rel_path.split(":=")[1], checkExist=True, pathPrepend=curr_dir) for rel_path in per_batch] \
            for per_batch in self._original_input_paths]
        #create a new input_list_file in the output_dir and use that
        self.input_list_txt=os.path.join(self.host_output_dir,os.path.basename(self.input_list_txt))
        with open(self.input_list_txt, "w") as input_list:
            input_list.write(self._input_list_comments)
            input_list.write('\n'.join([' '.join(per_batch) for per_batch in self._full_path_input_paths]))

    def _push_required_files(self):
        """
        This function sends the required QNN files to device, including:
        - model binary
        - runtime library binaries
        - input data
        """

        # if target device is x86, no need to push
        if self.target_device.device == "x86":
            self._target_backend = self.backend_path[0]
            self._target_model_path = self.qnn_model_binary
            self.target_input_list_path = self.input_list_txt
            self._target_output_dir = os.path.join(self.host_output_dir, "output")
            self.target_path = self.host_output_dir
            self.target_config_json_file = self.build_be_ext_config_json(os.path.join(self.host_output_dir,"be_ext_config_file.json"),\
                self.netrun_config_file, self.be_ext_shared_library)
            return

        try:
            # Push binaries to target device
            binary_paths = self._get_binaries_to_push()
            self.logger.info('Pushing binaries')

            for source in binary_paths:
                code, _, err = self.target_device.push(source, self.target_path)
                if code != 0:
                    raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))
            self._target_backend = self.backend_path[0].split("/")[-1]

            # Push model to target device
            self.logger.info('Pushing model to target')
            code, _, err = self.target_device.push(self.qnn_model_binary, self.target_path)
            if self.libcpp_dependency and not self.offline_prepare:
                libcpp_file = os.path.join(os.path.dirname(self.qnn_model_binary), "libc++_shared.so")
                if not os.path.exists(libcpp_file):
                    raise DependencyError(get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))
                code, _, err = self.target_device.push(
                    libcpp_file, self.target_path)
                if code != 0:
                    raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE"))
            self._target_model_path = os.path.join(self.target_path, self.qnn_model_binary.split("/")[-1])

            # Push input data to target device
            # input data are pushed to a device folder named input_data
            self.logger.info('Pushing input data to target')
            if self.model_inputs is not None:
                for _, _, data_path in self.model_inputs:
                    device_model_input_path = os.path.join(self.target_path, os.path.basename(data_path))
                    code, _, err = self.target_device.push(data_path, device_model_input_path)
                    if code != 0:
                        raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
            else:
                for perbatch in self._full_path_input_paths:
                    for full_path in perbatch:
                        if ":=" in full_path:
                            full_path=full_path.split(":=")[1]
                        code, _, err = self.target_device.push(full_path, self.target_path)
                        if code != 0:
                            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

            self.logger.info('Pushing input list to target')
            self._push_input_list()

            # Push config file on target device if provided
            self.target_config_json_file = self._push_config_file()
            self._target_output_dir = os.path.join(self.target_path, 'output')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE'))

    def _execute_conversion(self):
        """
        This function calls on the proper Converter class and creates the proper
        model binaries from the conversion tools
        """
        # set paths of the the to-be generated .ccp and .bin files
        self.qnn_model_cpp = os.path.join(self.host_output_dir, self.qnn_model_name + '.cpp')
        self.qnn_model_bin = os.path.join(self.host_output_dir, self.qnn_model_name + '.bin')

        # since including input list as a conversion parameter triggers quantization,
        # this sets input list to None for cpu and gpu because cpu and gpu don't support quantized models
        if self.runtime == 'cpu' or self.runtime == 'gpu':
            input_list = None
        else:
            input_list = self.input_list_txt

        convert_command = self.converter.build_convert_command(self.model_path,
                                                               self.model_inputs,
                                                               self.model_outputs,
                                                               self.qnn_model_cpp,
                                                               input_list,
                                                               self.quantization_overrides,
                                                               self.param_quantizer,
                                                               self.act_quantizer,
                                                               self.weights_bitwidth,
                                                               self.bias_bitwidth,
                                                               self.act_bitwidth,
                                                               self.algorithms,
                                                               self.ignore_encodings,
                                                               self.use_per_channel_quantization)
        try:
            code, _, err = self.host_device.execute(commands=[convert_command],
                                                    cwd=self.engine_path,
                                                    env=self._host_env)
            if code != 0:
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))
            self.logger.info('Model converted successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))

    def _model_lib_generate(self, lib_name_default, target_arch):
        # if lib_name is not specified, sets default lib_name to model_name
        if self.lib_name is None:
            self.lib_name = self.qnn_model_name
        lib_gen_command = [self.lib_generator_executable,
                           self.lib_generator_args["model_cpp"],
                           self.qnn_model_cpp,
                           self.lib_generator_args["output_path"],
                           self.binaries_dir,
                           self.lib_generator_args["lib_name"],
                           self.lib_name + '.so',
                           self.lib_generator_args["lib_target"],
                           self.lib_target]

        if self.qnn_model_bin is not None and os.path.exists(self.qnn_model_bin):
            lib_gen_command.extend([self.lib_generator_args["model_bin"], self.qnn_model_bin])
        else:
            self.logger.warning('No Model BIN found for Model at {}. This is ok if model does not have any static tensors.'.format(self.qnn_model_bin))
        lib_gen_command_str = ' '.join(lib_gen_command)
        code, out, err = self.host_device.execute(commands=[lib_gen_command_str],
                                                cwd=self.engine_path,
                                                env=self._host_env)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED')(target_arch, err_msg))
        # stores path to model.so
        self.qnn_model_binary = os.path.join(self.binaries_dir, self.target_arch, lib_name_default)

    def _create_model_binaries(self):
        """
        This function calls the qnn-model-lib-generator tool to create the model binaries from the
        .cpp and .bin files
        """
        try:
            lib_name_default = 'lib' + self.lib_name + '.so'
            # generate qnn model lib
            self._model_lib_generate(lib_name_default, self.target_arch)
            # generate qnn serialized bin
            if self.offline_prepare:
                if self.target_arch != "x86_64-linux-clang":
                    self._model_lib_generate(lib_name_default, "x86_64-linux-clang")
                context_binary_generate_command = [self.context_binary_generator_config["executable"],
                            self.context_binary_generator_config["arguments"]["model_path"],
                            os.path.join(self.binaries_dir, "x86_64-linux-clang", lib_name_default),
                            self.context_binary_generator_config["arguments"]["backend"],
                            self.context_binary_generator_config["backend_location"].format(engine_path=self.engine_path),
                            self.context_binary_generator_config["arguments"]["binary_file"],
                            self.lib_name,
                            self.context_binary_generator_config["arguments"]["output_dir"],
                            self.binaries_dir]
                context_binary_gen_command_str = ' '.join(context_binary_generate_command)
                code, out, err = self.host_device.execute(commands=[context_binary_gen_command_str],
                                                        cwd=self.engine_path,
                                                        env=self._host_env)
                if code != 0:
                    err_msg = str(err) if err else str(out)
                    raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')
                                               (err_msg))
                self.qnn_model_binary=os.path.join(self.binaries_dir, self.lib_name + ".bin")
            self.logger.info('Model binaries generated successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED')(self.target_arch,
                                                                                                  str(exc)))

    def _set_device_environment(self):
        """
        This helper function sets up the QNN execution environment on the target device
        """
        if self.target_device.device != 'x86':
            self.device_env = {}
            for library_path_name, path in self.executor.get_execute_environment_variables():
                self.device_env[library_path_name] = path
        else:
            self.device_env = self._host_env

    def _execute_inference(self):
        """
        This function calls on the Executor class and executes the model inference
        on device
        """
        self._set_device_environment()
        execute_command = self.executor.build_execute_command(self._target_model_path,
                                                              self._target_backend,
                                                              self.target_input_list_path,
                                                              self.op_packages,
                                                              self._target_output_dir,
                                                              self.input_data_type,
                                                              self.output_data_type,
                                                              self.perf_profile,
                                                              self.profiling_level,
                                                              self.debug_mode,
                                                              self.log_level,
                                                              self.print_version,
                                                              self.target_config_json_file)

        log_string = 'Running inference with command: ' + str(execute_command)
        self.logger.info(log_string)

        try:
            code, out, err = self.target_device.execute(commands=[execute_command], cwd=self.target_path,
                                                      env=self.device_env)
            if code != 0:
                err_msg = str(err) if err else str(out)
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED')(err_msg))
            self.logger.info('Inference executed successfully')

            if self.profiling_level is not None:
                self._parse_profiling_data()
            if self.target_device.device != "x86":
                self._pull_results()

        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED')(str(exc)))

    def _parse_profiling_data(self):
        """
        This function parses profiling data generated from qnn-net-run and saves the parsed
        performance metrics to a csv file
        """
        profiler_log_file = os.path.join(self._target_output_dir, "qnn-profiling-data.log")
        profiler_output = os.path.join(self._target_output_dir, "profiling.csv")

        prof_viewer_command = [self.profiler_executable,
                               self.profiler_args["input_log"],
                               profiler_log_file,
                               self.profiler_args["output_csv"],
                               profiler_output
                               ]

        prof_viewer_command_str = ' '.join(prof_viewer_command)

        try:
            code, _, err = self.target_device.execute(commands=[prof_viewer_command_str], env=self.device_env)
            if code != 0:
                raise ProfilingError(get_message('ERROR_PROFILER_DATA_EXTRACTION_FAILED'))
            self.logger.info('Profiling data extracted successfully')

        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise ProfilingError(get_message('ERROR_PROFILER_DATA_EXTRACTION_FAILED'))

    def _load_graph(self):
        """
        This function loads the net json which is generated by qnn converter
        """
        if self.qnn_model_net_json is None:
            self.qnn_model_net_json = os.path.join(self.host_output_dir, self.qnn_model_name + '_net.json')

        with open(self.qnn_model_net_json) as graph_file:
            self.graph_data = json.load(graph_file)

    def _extract_graph_structure(self):
        """
        This function extract qnn graph structure from the net json
        """

        def construct_encodings(tensors, output_tensor_names):
            """
            The encodings will be written into graph structure json which then be retrieved for
            usage in ScaledDiff verifier in verification module.
            """
            encs = {}
            switcher = {
                         "0x416": 16,
                         "0x316": 16,
                         "0x308": 8,
                         "0x408": 8
                        }
            for o_tensor_name in output_tensor_names:
                bw = switcher.get(hex(tensors[o_tensor_name]["data_type"]), 8)
                scale = tensors[o_tensor_name]["quant_params"]["scale_offset"]["scale"]
                offset = tensors[o_tensor_name]["quant_params"]["scale_offset"]["offset"]
                encs[o_tensor_name] = {
                                        "min":    scale*offset,
                                        "max":    scale*offset+scale*(2**bw - 1),
                                        "scale":  scale,
                                        "offset": offset,
                                        "bw":     bw
                                      }
            return encs

        tensors = self.graph_data["graph"]["tensors"]
        nodes = self.graph_data["graph"]["nodes"]

        graph_list_structure = OrderedDict()

        dim_field = "dims"
        # version 1.x uses max_dims as the field name while in 2.x and above, it is changed to dims
        if (self.engine_version is not None and version.parse(self.engine_version) < version.Version("2.0")):
            dim_field = "max_dims"
        for tensor_name in tensors:
            if tensors[tensor_name]['type'] == 0:
                input_tensors = {tensor_name : tensors[tensor_name][dim_field]}
                output_tensors = {tensor_name : tensors[tensor_name][dim_field]}
                encodings = construct_encodings(tensors, [tensor_name])
                graph_list_structure[tensor_name] = ["data", input_tensors, output_tensors, encodings]

        for node_name, node in nodes.items():
            input_tensors = {input_tensor: tensors[input_tensor][dim_field]
                             for input_tensor in node["input_names"]}
            output_tensors = {output_tensor: tensors[output_tensor][dim_field]
                              for output_tensor in node["output_names"]}
            encodings = construct_encodings(tensors, list(output_tensors.keys()))
            node_data = [node["type"], input_tensors, output_tensors, encodings]
            graph_list_structure[node_name] = node_data
        return graph_list_structure

    def _pull_results(self):
        """
        This function pulls the results from device and clears the on-device results directory.
        """
        code, out, err = self.target_device.pull(os.path.join(self.target_path, "output"), self.host_output_dir)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED")(err_msg))
        self.logger.info('Pull device results successfully')

        code, out, err = self.target_device.remove(target_path=self.target_path)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_REMOVE_RESULTS_FAILED')(err_msg))
        self.logger.info('Removed inference results from device successfully')



    # ------------------------------------------------ ABSTRACT FUNCTIONS --------------------------------------------------

    def run(self):
        self._setup()
        self._execute_inference()

    def get_graph_structure(self):
        self._load_graph()
        return self._extract_graph_structure()
