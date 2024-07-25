# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging
import os
import re
import subprocess
import zipfile
from collections import OrderedDict

import absl.logging
import yaml
from packaging import version

from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.inference_engine.inference_engines.nd_inference_engine import InferenceEngine
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, Runtime, X86_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError, ProfilingError, DependencyError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path


@inference_engine_repository.register(cls_type=ComponentType.inference_engine, framework=None,
                                      engine=Engine.QNN, engine_version="1.1.0.22262")
class QNNInferenceEngine(InferenceEngine):

    def __init__(self, context, converter, executor):
        super().__init__(context, converter, executor)

        # Instantiate Class Fields from context:
        # Fields from context
        self.engine_type = context.engine
        self.engine_version = context.engine_version
        self.stage = context.stage
        if context.engine_path.endswith(".zip"):
            self.engine_zip_path = context.engine_path
            self.engine_path = None
        else:
            self.engine_path = context.engine_path
            self.engine_zip_path = None
        self.host_device = context.host_device
        self.target_device = context.target_device
        #input_tensor accepted here should be a list of lists of len of 3 [tensor, dim, data]
        self.model_inputs = context.input_tensor
        self.model_outputs = context.output_tensor
        self.intermediate_outputs = context.add_layer_outputs
        self.add_layer_types = context.add_layer_types
        self.model_path = context.model_path
        self.host_output_dir = context.output_dir
        self.target_arch = context.architecture
        self.runtime = context.runtime
        self.remote_server = context.remote_server
        self.remote_username = context.remote_username
        self.remote_password = context.remote_password
        self.executable_location = context.executable_location
        # Setup backend_path and op_packages depending on target_arch
        if self.target_arch == "x86_64-linux-clang":
            self.backend_path = context.backend_locations["x86"][self.runtime]
            self.op_packages = context.op_packages["x86"][self.runtime]
            self.target_path = self.host_output_dir
        elif self.target_arch == "x86_64-windows-msvc":
            self.backend_path = context.backend_locations["x86_64_windows_msvc"][self.runtime]
            self.op_packages = context.op_packages["x86_64_windows_msvc"][self.runtime]
            self.target_path = self.host_output_dir
        elif self.target_arch == "wos":
            self.backend_path = context.backend_locations["wos"][self.runtime]
            self.op_packages = context.op_packages["wos"][self.runtime]
            self.target_path = self.host_output_dir
        elif self.target_arch == "aarch64-qnx":
            self.backend_path = context.backend_locations["qnx"][self.runtime]
            self.op_packages = context.op_packages["qnx"][self.runtime]
            self.target_path = context.target_path["qnx"]
        elif self.target_arch == "wos-remote":
            self.backend_path = context.backend_locations["wos"][self.runtime]
            self.op_packages = context.op_packages["wos"][self.runtime]
            self.target_path = context.target_path["wos"].format(username=self.remote_username)
        else:  # android backend
            self.backend_path = context.backend_locations["android"][self.runtime]
            self.op_packages = context.op_packages["android"][self.runtime]
            self.target_path = context.target_path["android"]

        self.interface_module = context.op_packages["interface"]
        self.compiler_config_json = context.compiler_config
        self.sdk_tools_root = context.sdk_tools_root
        # Get environment variables depending on the host device architecture.
        if self.host_device.device == "x86_64-windows-msvc":
            self.env_variables = context.x86_64_windows_msvc_environment_variables
        elif self.host_device.device == "wos":
            self.env_variables = context.wos_environment_variables
        else:
            self.env_variables = context.environment_variables
        self.logger = context.logger
        #quantization
        self.precision = context.precision
        self.input_list_txt = context.input_list
        self.quantization_overrides = context.quantization_overrides
        self.param_quantizer = context.param_quantizer
        self.act_quantizer = context.act_quantizer
        self.act_quantizer_calibration = context.act_quantizer_calibration
        self.param_quantizer_calibration = context.param_quantizer_calibration
        self.act_quantizer_schema = context.act_quantizer_schema
        self.param_quantizer_schema = context.param_quantizer_schema
        self.percentile_calibration_value = context.percentile_calibration_value
        self.weights_bitwidth = context.weights_bitwidth
        self.bias_bitwidth = context.bias_bitwidth
        self.act_bitwidth = context.act_bitwidth
        self.float_bias_bitwidth = context.float_bias_bitwidth
        self.restrict_quantization_steps = context.restrict_quantization_steps
        self.algorithms = context.algorithms
        self.ignore_encodings = context.ignore_encodings
        self.use_per_channel_quantization = context.per_channel_quantization
        self.extra_converter_args = context.extra_converter_args
        self.extra_contextbin_args = context.extra_contextbin_args
        self.extra_runtime_args = context.extra_runtime_args

        self.use_native_input_files = context.use_native_input_files
        self.use_native_output_files = context.use_native_output_files

        self.binaries_dir = os.path.join(
            self.host_output_dir,
            context.binaries_dir) if context.binaries_dir is not None else None
        self.qnn_model_cpp = context.qnn_model_cpp_path
        self.qnn_model_bin = context.qnn_model_bin_path
        self.qnn_model_binary = context.qnn_model_binary_path
        self.qnn_model_net_json = context.qnn_model_net_json

        # Lib Generator
        self.qnn_model_name = context.model_name if context.model_name is not None else "qnn_model"
        if self.target_arch in ["wos-remote", "wos"]:
            self.lib_name = context.lib_name if context.lib_name is not None else 'qnngraph.serialized'
        else:
            self.lib_name = context.lib_name if context.lib_name is not None else 'qnn_model'
        self.lib_target = context.lib_target
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
        if (self.target_arch in ['wos-remote', 'x86_64-windows-msvc', 'wos']):
            self.htp_be_ext_shared_library_path = context.htp_backend_extension_shared_library_path[
                "windows"]
            self.aic_be_ext_shared_library_path = context.aic_backend_extension_shared_library_path[
                "windows"]
        else:
            self.htp_be_ext_shared_library_path = context.htp_backend_extension_shared_library_path[
                "linux"]
            self.aic_be_ext_shared_library_path = context.aic_backend_extension_shared_library_path[
                "linux"]

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
        self._host_env = {}
        self._target_model_path = None
        self._target_backend = None
        self.context_config_json = None

    # -------------------------------------- HELPER FUNCTIONS --------------------------------------
    def _setup(self):
        """This function sets up the working directory and environment to
        execute QNN inferences.

        It should:
        - Unzip the QNN SDK into the working directory
        - Setup the QNN execution environment on host x86 device
        """
        # Unzip SDK:
        self._validate_engine_path()

        # TODO: Fix the target arch name to arm64x-windows once libs and bins are shipped in this arch
        arch = 'aarch64-windows-msvc' if self.target_arch in ['wos-remote', 'wos'] else self.target_arch

        # For wos-remote backend_path will be formatted in a dedicated function
        if self.target_arch != 'wos-remote':
            # setup backend_path
            self.backend_path = [
                source.format(engine_path=self.engine_path, target_arch=arch)
                for source in self.backend_path
            ]

        self.htp_be_ext_shared_library_path = self.htp_be_ext_shared_library_path.format(
            engine_path=self.engine_path, target_arch=arch)
        self.aic_be_ext_shared_library_path = self.aic_be_ext_shared_library_path.format(
            engine_path=self.engine_path, target_arch=arch)

        # validates the given runtime with sdk version
        self._validate_runtime()

        #Update Executor engine_path to be the unzipped path if originally provided with .zip path:
        if (not self.executor.updateField('engine_path', self.engine_path)):
            self.logger.error("failed to update executor engine_path")

        #setup executable_location
        self.executable_location = self.executable_location.format(engine_path=self.engine_path,
                                                                   target_arch=arch)

        #moved from init incase engine_path not setup first
        for pkg in self.op_packages:
            if not os.path.exists(pkg.format(engine_path=self.engine_path, target_arch=arch)):
                self.op_packages.remove(pkg)

        #setting up the profiler_path
        self.profiler_path = self.profiler_path.format(engine_path=self.engine_path,
                                                       target_arch=arch)

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
                    #This assumes per batch input is separated by either comma or space
                    self._original_input_paths.append(re.split(' ,|, |,| ', line.strip(' \n')))

        #set _full_path_input_paths:
        self._set_input_list()

        self._set_host_environment()

        if self.compiler_config_json:
            self.context_config_json = f"{self.host_output_dir}/context_config.json"

            if self.runtime == Runtime.aic.value:
                shared_lib_path = self.aic_be_ext_shared_library_path
            elif self.runtime == Runtime.htp.value or "dsp" in self.runtime:
                # For context-bin-generator, we will be using x86_64-linux-clang htp extension regardless of the given architecture
                shared_lib_path = self.htp_be_ext_shared_library_path.replace(
                    arch, "x86_64-linux-clang")
            else:
                self.logger.error(
                    "--compiler_config is supported only for aic and htp/dsp runtimes.")
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_BACKEND_CONFIG_PARSING_FAILED"))

            self.build_be_ext_config_json(self.context_config_json, self.compiler_config_json,
                                          shared_lib_path)

        if self.target_arch == 'wos-remote':
            return

        # starting with source framework
        if self.stage == 'source':
            self._execute_conversion()
            self._create_model_binaries()
        # starting with .cpp and .bin
        elif self.stage == 'converted':
            self._create_model_binaries()
        self._push_required_files()

    def _validate_engine_path(self):
        """This helper function unzips engine_zip and sets the engine_path to
        the correct path."""
        if not self.engine_path and self.engine_zip_path:
            #Zipfile is breaking the symlink while extracting. So using subprocess for extracting
            try:
                subprocess.run(['unzip', '-q', self.engine_zip_path, '-d', self.host_output_dir],
                               stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print("ERROR: Extracting SDK with the following error: ", err.returncode)
            with zipfile.ZipFile(self.engine_zip_path, 'r') as f:
                filelists = f.namelist()
                for file in filelists:
                    os.chmod(os.path.join(self.host_output_dir, file), 0o755)
            if './' in filelists[0]:
                self.engine_path = os.path.join(self.host_output_dir, os.path.dirname(filelists[1]))
            else:
                self.engine_path = os.path.join(self.host_output_dir, os.path.dirname(filelists[0]))
        elif not os.path.isdir(self.engine_path):
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_PATH_INVALID")(self.engine_path))

    def _set_host_environment(self):
        """This helper function sets up the QNN execution environment on host
        x86 device."""
        # Get file paths:
        self.sdk_tools_root = self.sdk_tools_root.format(engine_path=self.engine_path)

        for var in self.env_variables:
            self.env_variables[var] = (self.env_variables[var]).format(
                sdk_tools_root=self.sdk_tools_root)

        # set environment variables depending on host device architecture
        if self.host_device.device in ["x86_64-windows-msvc", "wos"]:
            for var in self.env_variables:
                self._host_env[var] = self.env_variables[var] + os.pathsep
        else:
            for var in self.env_variables:
                self._host_env[var] = self.env_variables[var] + os.pathsep + '$' + var
        # Add   path to PATH:
        self._host_env['QNN_SDK_ROOT'] = self.engine_path

    @staticmethod
    def build_be_ext_config_json(output_file_path, config_file_path, shared_library_path):
        """Utility method to help building the backend extension config json
        file by providing the .json file.

        :param output_file_path: path to backend extension file.
        :param config_file_path: path to backend specific compile/execute parameters .json
        :param shared_library_path: Backend extensions shared library path
        :return: If .json file not provided, returning None, else return
                 output_file_path
        """
        if not config_file_path:
            return None

        config = {
            "backend_extensions": {
                "shared_library_path": shared_library_path,
                "config_file_path": config_file_path
            }
        }
        with open(output_file_path, "w") as file:
            json.dump(config, file, indent=4)
        return output_file_path

    def _get_binaries_to_push(self):
        """Get QNN binaries used to convert and run a model.

        :return: List of binary paths
        """
        self.backend_path.append(self.executable_location)
        self.backend_path.append(self.profiler_path)
        for pkg in self.op_packages:
            self.backend_path.append(pkg)
        return self.backend_path

    def _push_input_list(self):
        """Create an input list on the host device.

        and push to target device, if the target device is x86, it
        should not call this function
        """
        if not self.input_list_txt:

            # Set input list file name
            # using the tensor_name as the input_list_name
            layers = ['-'.join(tensor_name.split('/')) for tensor_name, _, _ in self.model_inputs]
            input_list_name = '_'.join(layers) + '.txt'

            #device_input_list_host_path is the inputlist to be used on device but needs to store a copy on the host (x86), this is because on the file path
            #within the device_intput_list should be device path based ie. /data/local/tmp/, hence this device_input_list can not be used on host.
            device_input_list_host_path = os.path.join(self.host_output_dir, input_list_name)
            self.target_input_list_path = os.path.join(self.target_path, input_list_name)

            string_to_write = ' '.join([
                tensor_name + ":=" + data_path.split("/")[-1]
                for tensor_name, dims, data_path in self.model_inputs
            ])
            string_to_write += '\n'
            with open(device_input_list_host_path, 'w+') as f:
                f.write(self._input_list_comments)
                f.write(string_to_write)
            code, _, err = self.target_device.push(device_input_list_host_path,
                                                   self.target_input_list_path)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
        else:
            # create a new input list with on-device paths to input data
            if ":=" not in self._full_path_input_paths[0][0]:
                on_device_input_paths = [[
                    os.path.join(self.target_path, os.path.basename(rel_path)) for rel_path in line
                ] for line in self._full_path_input_paths]
            else:
                on_device_input_paths = [[
                    rel_path.split(":=")[0] + ":=" +
                    os.path.join(self.target_path, os.path.basename(rel_path.split(":=")[1]))
                    for rel_path in line
                ] for line in self._full_path_input_paths]
            device_input_list_host_path = os.path.join(self.host_output_dir,
                                                       'device_input_list.txt')
            self.target_input_list_path = os.path.join(
                self.target_path, os.path.basename(device_input_list_host_path))
            with open(device_input_list_host_path, 'w') as d:
                d.write(self._input_list_comments)
                d.write(('\n'.join([' '.join(per_batch)
                                    for per_batch in on_device_input_paths])) + '\n')
            code, _, err = self.target_device.push(device_input_list_host_path,
                                                   self.target_input_list_path)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

    def _push_config_file(self):
        """Create configFile and push backend Ext Files to proper path.

        Only called for On Device runtime not for x86.
        """
        if self.netrun_config_file:
            self.logger.info('Pushing config file to target')
            config_json_file = os.path.join(self.host_output_dir, "be_ext_config_file.json")
            netrun_config_file_target_path = os.path.join(self.target_path,
                                                          os.path.basename(self.netrun_config_file))
            be_ext_shared_library_target_path = os.path.join(
                self.target_path, os.path.basename(self.htp_be_ext_shared_library_path))

            if self.target_arch == 'wos-remote':
                netrun_config_file_target_path_for_target = "..\\..\\" + str(
                    os.path.basename(self.netrun_config_file))
                be_ext_shared_library_target_path_config_for_target = "..\\..\\" + str(
                    os.path.basename(self.htp_be_ext_shared_library_path))
            else:
                netrun_config_file_target_path_for_target = netrun_config_file_target_path
                be_ext_shared_library_target_path_config_for_target = be_ext_shared_library_target_path

            self.build_be_ext_config_json(config_json_file,
                                          netrun_config_file_target_path_for_target,
                                          be_ext_shared_library_target_path_config_for_target)

            code, _, err = self.target_device.push(self.netrun_config_file,
                                                   netrun_config_file_target_path)
            code, _, err = self.target_device.push(
                config_json_file, os.path.join(self.target_path,
                                               os.path.basename(config_json_file)))
            code, _, err = self.target_device.push(self.htp_be_ext_shared_library_path,
                                                   be_ext_shared_library_target_path)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
            return os.path.basename(config_json_file)
        else:
            return None

    def _prepend_input_tensor_to_list(self):
        """
        This funtion prepends input sample supplied via input_tensor to input_list file
        """
        if self._full_path_input_paths is None or self.model_inputs is None:
            return

        if ":=" not in self._full_path_input_paths[0][0]:
            input_sample_path = [
                get_absolute_path(data_path) for tensor_name, _, data_path in self.model_inputs
            ]
        else:
            input_sample_path = [
                tensor_name + ":=" + get_absolute_path(data_path)
                for tensor_name, _, data_path in self.model_inputs
            ]

        if set(input_sample_path) != set(self._full_path_input_paths[0]):
            self._full_path_input_paths.insert(0, input_sample_path)

    def _set_input_list(self):
        """This function prepends input list's current directory to each of the
        relative paths in input list, resulting in an input list with absolute
        paths.

        :param curr_dir: input list's current directory
        """
        curr_dir = os.path.dirname(self.input_list_txt)

        # this here basically means for each item in each line of _original_input_paths, make it an absolute path
        self._full_path_input_paths = [[get_absolute_path(rel_path, checkExist=True, pathPrepend=curr_dir) if ":=" not in rel_path \
            else rel_path.split(":=")[0]+":="+ get_absolute_path(rel_path.split(":=")[1], checkExist=True, pathPrepend=curr_dir) for rel_path in per_batch] \
            for per_batch in self._original_input_paths]
        # Prepend raw file supplied with --input_tensor to input_list, so that output corresponding to this sample get
        # dumped first.
        # TODO:Remove this when AI studio dependency on debugger is removed.
        self._prepend_input_tensor_to_list()
        # create a new input_list_file in the output_dir and use that
        self.input_list_txt = os.path.join(self.host_output_dir,
                                           os.path.basename(self.input_list_txt))
        with open(self.input_list_txt, "w") as input_list:
            input_list.write(self._input_list_comments)
            input_list.write('\n'.join(
                [' '.join(per_batch) for per_batch in self._full_path_input_paths]))

    def _push_required_files(self):
        """This function sends the required QNN files to device, including:

        - model binary
        - runtime library binaries
        - input data
        """
        # if target device is x86, no need to push
        if self.target_device.device in ["x86", "x86_64-windows-msvc", "wos"]:
            self._target_backend = self.backend_path[0]
            self._target_model_path = self.qnn_model_binary
            self.target_input_list_path = self.input_list_txt
            self._target_output_dir = os.path.join(self.host_output_dir, "output")
            # In case of x86, construct full path to backend extension library file.
            # In case of android, construct relative path to backend extension library file.
            if self.runtime == Runtime.aic.value:
                self.target_config_json_file = self.build_be_ext_config_json(
                    os.path.join(self.host_output_dir, "be_ext_config_file.json"),
                    self.netrun_config_file, self.aic_be_ext_shared_library_path)
            else:
                self.target_config_json_file = self.build_be_ext_config_json(
                    os.path.join(self.host_output_dir, "be_ext_config_file.json"),
                    self.netrun_config_file, self.htp_be_ext_shared_library_path)
            return

        try:
            # Push binaries to target device
            binary_paths = self._get_binaries_to_push()
            self.logger.info('Pushing binaries')

            for source in binary_paths:
                code, _, err = self.target_device.push(
                    source, os.path.join(self.target_path, os.path.basename(source)))
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_PUSH_BINARIES_FAILED_DEVICE"))
            self._target_backend = self.backend_path[0].split("/")[-1]

            # Push model to target device
            self.logger.info('Pushing model to target')
            code, _, err = self.target_device.push(
                self.qnn_model_binary,
                os.path.join(self.target_path, os.path.basename(self.qnn_model_binary)))
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_PUSH_MODEL_FAILED_DEVICE"))
            if self.libcpp_dependency and not self.offline_prepare:
                libcpp_file = os.path.join(os.path.dirname(self.qnn_model_binary),
                                           "libc++_shared.so")
                if not os.path.exists(libcpp_file):
                    raise DependencyError(
                        get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))
                code, _, err = self.target_device.push(
                    libcpp_file, os.path.join(self.target_path, os.path.basename(libcpp_file)))
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE"))
            self._target_model_path = os.path.join(self.target_path,
                                                   self.qnn_model_binary.split("/")[-1])

            self._push_input_samples()
            self._push_input_list()

            # Push config file on target device if provided
            self.target_config_json_file = self._push_config_file()
            self._target_output_dir = os.path.join(self.target_path, 'output')
            # Push custom_op files to target device
            self._push_custom_op_packages()

        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE'))

    def _push_custom_op_packages(self):
        if self.extra_runtime_args and '--op_packages' in self.extra_runtime_args:
            op_packages_entries = self.extra_runtime_args.split('--op_packages')[-1].strip().split(
                ' ')[0].split(',')
            self.extra_runtime_args = self.extra_runtime_args.split('--op_packages')[0] + \
                ' '.join(self.extra_runtime_args.split('--op_packages')[-1].strip().split(' ')[1:])
            self.extra_runtime_args = self.extra_runtime_args.strip()

            op_package_path = []
            target_op_package_path = []
            interface_provider = []
            target = []
            for i, op_package_entry in enumerate(op_packages_entries):
                op_package_list = op_package_entry.split(':')
                op_package_path.append(op_package_list[0].strip())
                target_op_package_path.append(
                    os.path.join(self.target_path, os.path.basename(op_package_path[i])))
                #include push
                code, _, err = self.target_device.push(op_package_path[i],
                                                       target_op_package_path[i])
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                interface_provider.append(op_package_list[1].strip() if len(op_package_list) ==
                                          2 else '')
                target.append(op_package_list[2].strip() if len(op_package_list) == 3 else '')
            self.extra_runtime_args += ' --op_packages ' + ','.join([
                target_op_package_path[i] + ':' + interface_provider[i] + ':' + target[i]
                for i in range(len(target_op_package_path))
            ])

            if self.extra_contextbin_args and '--op_packages' in self.extra_contextbin_args:
                self.extra_contextbin_args = self.extra_contextbin_args.split('--op_packages')[0] + \
                ' '.join(self.extra_contextbin_args.split('--op_packages')[-1].strip().split(' ')[1:])
                self.extra_contextbin_args = self.extra_contextbin_args.strip()
                self.extra_contextbin_args += ' --op_packages ' + ','.join([
                    target_op_package_path[i] + ':' + interface_provider[i] + ':' + target[i]
                    for i in range(len(target_op_package_path))
                ])

    def _execute_conversion(self):
        """This function calls on the proper Converter class and creates the
        proper model binaries from the conversion tools."""
        # set paths of the to-be generated .ccp and .bin files
        self.qnn_model_cpp = os.path.join(self.host_output_dir, self.qnn_model_name + '.cpp')
        self.qnn_model_bin = os.path.join(self.host_output_dir, self.qnn_model_name + '.bin')

        # since including input list as a conversion parameter triggers quantization,
        # this sets input list to None for cpu and gpu because cpu and gpu don't support quantized models

        if self.runtime in ['cpu', 'gpu']:
            input_list = None
        elif self.runtime == 'aic':
            if self.precision in ['fp16']:
                input_list = None
            else:
                input_list = self.input_list_txt
        else:
            input_list = self.input_list_txt

        convert_command = self.converter.build_convert_command(
            model_path=self.model_path, input_tensors=self.model_inputs,
            output_tensors=self.model_outputs, output_path=self.qnn_model_cpp,
            input_list_txt=input_list, quantization_overrides=self.quantization_overrides,
            param_quantizer=self.param_quantizer, act_quantizer=self.act_quantizer,
            weight_bw=self.weights_bitwidth, bias_bw=self.bias_bitwidth, act_bw=self.act_bitwidth,
            float_bias_bw=self.float_bias_bitwidth,
            restrict_quantization_steps=self.restrict_quantization_steps,
            algorithms=self.algorithms, ignore_encodings=self.ignore_encodings,
            per_channel_quantization=self.use_per_channel_quantization,
            act_quantizer_calibration=self.act_quantizer_calibration,
            param_quantizer_calibration=self.param_quantizer_calibration,
            act_quantizer_schema=self.act_quantizer_schema,
            param_quantizer_schema=self.param_quantizer_schema,
            percentile_calibration_value=self.percentile_calibration_value,
            extra_converter_args=self.extra_converter_args)

        try:
            self.logger.debug('Model converter command : {}'.format(convert_command))
            code, _, err = self.host_device.execute(commands=[convert_command],
                                                    cwd=self.engine_path, env=self._host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))
            self.logger.info(get_progress_message("PROGRESS_INFERENCE_ENGINE_CONVERSION_FINISHED"))
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))

    def _model_lib_generate(self, lib_name_default, target_arch):
        # if lib_name is not specified, sets default lib_name to model_name
        if self.lib_name is None:
            self.lib_name = self.qnn_model_name

        # For windows, lib_name should not have .so
        if (target_arch in ["x86_64-windows-msvc", "wos"]):
            lib_name_param = self.lib_name
        else:
            lib_name_param = self.lib_name + '.so'

        # model lib generator command expect 'windows-x86_64' as the lib target for x86_64_windows_msvc
        if "x86_64-windows-msvc" in self.lib_target:
            lib_target_param = 'windows-x86_64'
        elif "wos" in self.lib_target:
            lib_target_param = 'windows-aarch64'
        else:
            lib_target_param = self.lib_target

        lib_gen_command = [
            self.lib_generator_executable, self.lib_generator_args["model_cpp"], self.qnn_model_cpp,
            self.lib_generator_args["output_path"], self.binaries_dir,
            self.lib_generator_args["lib_name"], lib_name_param,
            self.lib_generator_args["lib_target"], lib_target_param
        ]

        if self.qnn_model_bin is not None and os.path.exists(self.qnn_model_bin):
            lib_gen_command.extend([self.lib_generator_args["model_bin"], self.qnn_model_bin])
        else:
            self.logger.warning(
                'No Model BIN found for Model at {}. This is ok if model does not have any static tensors.'
                .format(self.qnn_model_bin))
        lib_gen_command_str = ' '.join(lib_gen_command)
        self.logger.debug('Model libgenerate command : {}'.format(lib_gen_command_str))
        code, out, err = self.host_device.execute(commands=[lib_gen_command_str],
                                                  cwd=self.engine_path, env=self._host_env)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED')(target_arch, err_msg))
        # stores path to model.so or .dll, in case of windows x86_64, wos output file is produced
        # in directory named as "x64", "ARM64" respectively

        if target_arch == "x86_64-windows-msvc":
            target_arch_param = "x64"
        elif target_arch == "wos":
            target_arch_param = "ARM64"
        else:
            target_arch_param = target_arch
        self.qnn_model_binary = os.path.join(self.binaries_dir, target_arch_param, lib_name_default)

    def _create_model_binaries(self):
        """This function calls the qnn-model-lib-generator tool to create the
        model binaries from the .cpp and .bin files."""
        try:
            # In case of windows library name should end with .dll otherwise
            # it should end with .so for android and linux
            if (self.target_arch in ["x86_64-windows-msvc", "wos"]):
                lib_name_default = self.lib_name + '.dll'
            else:
                lib_name_default = 'lib' + self.lib_name + '.so'
            # generate qnn model lib
            self._model_lib_generate(lib_name_default, self.target_arch)
            # generate qnn serialized bin
            if self.offline_prepare:
                if self.runtime in ["cpu", "gpu"]:
                    self.logger.error("offline_prepare can not be set for CPU and GPU backend")
                    raise InferenceEngineError(
                        get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')(
                            "offline_prepare can not be set for CPU and GPU backend"))
                if (self.target_arch not in  ["x86_64-linux-clang", "x86_64-windows-msvc", "wos"]):
                    self._model_lib_generate(lib_name_default, "x86_64-linux-clang")

                if self.target_arch == "x86_64-windows-msvc":
                    model_path_param = os.path.join(self.binaries_dir, "x64", lib_name_default)
                elif self.target_arch == "wos":
                    model_path_param = os.path.join(self.binaries_dir, "ARM64", lib_name_default)
                else:
                    model_path_param = os.path.join(self.binaries_dir, "x86_64-linux-clang",
                                                    lib_name_default)

                b_end = "aic_backend_location" if self.runtime == 'aic' else "backend_location"
                context_binary_generate_command = [
                    self.context_binary_generator_config["executable"],
                    self.context_binary_generator_config["arguments"]["model_path"],
                    model_path_param, self.context_binary_generator_config["arguments"]["backend"],
                    self.context_binary_generator_config[b_end].format(
                        engine_path=self.engine_path),
                    self.context_binary_generator_config["arguments"]["binary_file"], self.lib_name,
                    self.context_binary_generator_config["arguments"]["output_dir"],
                    self.binaries_dir
                ]
                if self.profiling_level:
                    context_binary_generate_command.extend([
                        self.context_binary_generator_config["arguments"]["profiling_level"],
                        self.profiling_level
                    ])
                # if both debug_mode and intermediate_outputs are enabled , intermediate_outputs takes precedence
                # and dumps only specified intermediate outputs
                if self.intermediate_outputs:
                    context_binary_generate_command.extend([
                        self.context_binary_generator_config["arguments"]["output_tensors"],
                        self.qnn_model_name + ':' + self.intermediate_outputs
                    ])
                if self.debug_mode and not self.intermediate_outputs:
                    context_binary_generate_command.append(
                        self.context_binary_generator_config["arguments"]
                        ["enable_intermediate_outputs"])
                if self.precision in ['fp16', 'int8'] and self.compiler_config_json:
                    context_binary_generate_command += [
                        self.context_binary_generator_config["arguments"]["config_file"],
                        self.context_config_json
                    ]
                if self.extra_contextbin_args:
                    context_binary_generate_command.append(self.extra_contextbin_args)

                context_binary_gen_command_str = ' '.join(context_binary_generate_command)
                self.logger.debug(
                    'context bin generator command : {}'.format(context_binary_gen_command_str))
                code, out, err = self.host_device.execute(commands=[context_binary_gen_command_str],
                                                          cwd=self.engine_path, env=self._host_env)
                if code != 0:
                    err_msg = str(err) if err else str(out)
                    raise InferenceEngineError(
                        get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')(
                            err_msg))
                self.qnn_model_binary = os.path.join(self.binaries_dir, self.lib_name + ".bin")
            self.logger.info(get_progress_message("PROGRESS_INFERENCE_ENGINE_MODEL_BINARIES"))
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED')(self.target_arch,
                                                                           str(exc)))

    def _set_device_environment(self):
        """This helper function sets up the QNN execution environment on the
        target device."""
        if (self.target_device.device not in ['x86', 'x86_64-windows-msvc', 'wos']):
            self.device_env = {}
            for library_path_name, path in self.executor.get_execute_environment_variables():
                self.device_env[library_path_name] = path
        else:
            # For x86 and x86_64_windows_msvc, WOS, target and host device is currently same
            self.device_env = self._host_env

    def _set_remote_environment(self):
        """It sets up the QNN execution environment on the
        remote target device."""

        self._remote_env = {}
        for k, v in self.remote_env_variables.items():
            self._remote_env[k] = v

    def _push_input_samples(self):
        # Push input data to target device
        # input data are pushed to a device folder named input_data
        self.logger.info('Pushing input data to target')
        if not self.input_list_txt:
            for _, _, data_path in self.model_inputs:
                device_model_input_path = os.path.join(self.target_path,
                                                       os.path.basename(data_path))
                code, _, err = self.target_device.push(data_path, device_model_input_path)
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
        else:
            for perbatch in self._full_path_input_paths:
                for full_path in perbatch:
                    if ":=" in full_path:
                        full_path = full_path.split(":=")[1]
                    device_model_input_path = os.path.join(self.target_path,
                                                           os.path.basename(full_path))
                    code, _, err = self.target_device.push(full_path, device_model_input_path)
                    if code != 0:
                        raise InferenceEngineError(
                            get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

    def _execute_inference(self):
        """This function calls on the Executor class and executes the model
        inference on device."""
        self._set_device_environment()

        # if offline_prepare and debug mode both on, don't need to pass in debug option to execute command builder as
        # it's handled inside context binary generator
        debug_option_on = self.debug_mode
        if debug_option_on and self.offline_prepare: debug_option_on = False

        execute_command = self.executor.build_execute_command(
            self._target_model_path, self._target_backend, self.target_input_list_path,
            self.op_packages, self._target_output_dir, self.use_native_input_files,
            self.use_native_output_files, self.perf_profile, self.profiling_level, debug_option_on,
            self.log_level, self.print_version, self.target_config_json_file,
            self.extra_runtime_args, self.intermediate_outputs)

        if self.target_arch == "aarch64-qnx":
            execute_command = './' + execute_command

        log_string = 'Using inference command: ' + str(execute_command)
        self.logger.debug(log_string)

        try:
            self.logger.info(
                get_progress_message('PROGRESS_INFERENCE_ENGINE_GENERATE_OUTPUTS')(
                    self._target_output_dir))
            if self.target_arch == 'wos-remote':
                code, out, err = self.target_device.execute(
                    commands=[execute_command], cwd=os.path.join(self.target_path, 'bin',
                                                                 'aarch64-windows-msvc'),
                    env=os.path.join(self.target_path, 'bin'))
            else:
                code, out, err = self.target_device.execute(commands=[execute_command],
                                                            cwd=self.target_path,
                                                            env=self.device_env)

            if code != 0:
                err_msg = str(err) if err else str(out)
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED')(err_msg))
            self.logger.info(
                get_progress_message('PROGRESS_INFERENCE_ENGINE_GENERATED_INTERMEDIATE_TENSORS')(
                    self.engine_type))

            if self.profiling_level is not None:
                self._parse_profiling_data()
            if self.target_device.device not in ["x86", "x86_64-windows-msvc", "wos"]:
                self._pull_results()

        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED')(str(exc)))

    def _parse_profiling_data(self):
        """This function parses profiling data generated from qnn-net-run and
        saves the parsed performance metrics to a csv file."""
        profiler_log_file = os.path.join(self._target_output_dir, "qnn-profiling-data.log")
        profiler_output = os.path.join(self._target_output_dir, "profiling.csv")

        if self.target_arch == "wos-remote":
            self.profiler_executable = './' + self.profiler_executable + '.exe'
        elif self.target_arch == "aarch64-qnx":
            self.profiler_executable = './' + self.profiler_executable

        prof_viewer_command = [
            self.profiler_executable, self.profiler_args["input_log"], profiler_log_file,
            self.profiler_args["output_csv"], profiler_output,
            self.profiler_args["extract_opaque_objects"]
        ]

        prof_viewer_command_str = ' '.join(prof_viewer_command)

        try:
            if self.target_arch == 'wos-remote':
                code, _, _ = self.target_device.execute(
                    commands=[prof_viewer_command_str],
                    cwd=os.path.join(self.target_path, 'bin', 'aarch64-windows-msvc'),
                    env=os.path.join(self.target_path, 'bin'))
            else:
                code, _, _ = self.target_device.execute(commands=[prof_viewer_command_str],
                                                        env=self.device_env)
            if code != 0:
                raise ProfilingError(get_message('ERROR_PROFILER_DATA_EXTRACTION_FAILED'))
            self.logger.info('Profiling data extracted successfully')

        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise ProfilingError(get_message('ERROR_PROFILER_DATA_EXTRACTION_FAILED'))

    def _load_graph(self):
        """This function loads the net json which is generated by qnn
        converter."""
        if self.qnn_model_net_json is None:
            self.qnn_model_net_json = os.path.join(self.host_output_dir,
                                                   self.qnn_model_name + '_net.json')

        with open(self.qnn_model_net_json) as graph_file:
            self.graph_data = json.load(graph_file)

    def _extract_graph_structure(self):
        """This function extract qnn graph structure from the net json."""

        def construct_encodings(tensors, output_tensor_names):
            """The encodings will be written into graph structure json which
            then be retrieved for usage in ScaledDiff verifier in verification
            module."""
            encs = {}
            switcher = {"0x416": 16, "0x316": 16, "0x308": 8, "0x408": 8}
            for o_tensor_name in output_tensor_names:
                bw = switcher.get(hex(tensors[o_tensor_name]["data_type"]), 8)
                scale = tensors[o_tensor_name]["quant_params"]["scale_offset"]["scale"]
                offset = tensors[o_tensor_name]["quant_params"]["scale_offset"]["offset"]
                encs[o_tensor_name] = {
                    "min": scale * offset,
                    "max": scale * offset + scale * (2**bw - 1),
                    "scale": scale,
                    "offset": offset,
                    "bw": bw
                }
            return encs

        tensors = self.graph_data["graph"]["tensors"]
        nodes = self.graph_data["graph"]["nodes"]

        graph_list_structure = OrderedDict()

        dim_field = "dims"
        # version 1.x uses max_dims as the field name while in 2.x and above, it is changed to dims
        if (self.engine_version is not None
                and version.parse(self.engine_version) < version.Version("2.0")):
            dim_field = "max_dims"
        for tensor_name in tensors:
            if tensors[tensor_name]['type'] == 0:
                input_tensors = {tensor_name: tensors[tensor_name][dim_field]}
                output_tensors = {tensor_name: tensors[tensor_name][dim_field]}
                encodings = construct_encodings(tensors, [tensor_name])
                graph_list_structure[tensor_name] = [
                    "data", input_tensors, output_tensors, encodings
                ]

        for node_name, node in nodes.items():
            input_tensors = {
                input_tensor: tensors[input_tensor][dim_field]
                for input_tensor in node["input_names"]
            }
            output_tensors = {
                output_tensor: tensors[output_tensor][dim_field]
                for output_tensor in node["output_names"]
            }
            encodings = construct_encodings(tensors, list(output_tensors.keys()))
            node_data = [node["type"], input_tensors, output_tensors, encodings]
            graph_list_structure[node_name] = node_data
        return graph_list_structure

    def _pull_results(self):
        """This function pulls the results from device and clears the on-device
        results directory."""
        code, out, err = self.target_device.pull(os.path.join(self.target_path, "output"),
                                                 self.host_output_dir)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED")(err_msg))
        self.logger.debug('Pull device results successfully')

        if self.target_arch == "wos-remote" and self.target_device.is_path(
                os.path.join(self.target_path, "bin/aarch64-windows-msvc/model_schematic.bin")):
            code, out, err = self.target_device.pull(
                os.path.join(self.target_path, "bin/aarch64-windows-msvc/model_schematic.bin"),
                self.host_output_dir)
            if code != 0:
                err_msg = str(err) if err else str(out)
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_PULL_FILE_FAILED")(err_msg))
            self.logger.debug('Pull schematic binary file successful')

        code, out, err = self.target_device.remove(target_path=self.target_path)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_REMOVE_RESULTS_FAILED')(err_msg))
        self.logger.debug('Removed inference results from device successfully')

    def get_engine_version(self):
        """This function returns the engine version if not available."""

        # If engine version is already provided by user, return it
        if self.engine_version:
            return self.engine_version

        # Define the path to the sdk.yaml file, which should be present in sdk
        sdk_yaml_file_path = os.path.join(self.engine_path, "sdk.yaml")
        # Define the regular expression pattern for version number
        regex_ver = r'\d+(\.\d+)*'

        # If user does not provide version number,
        # Check if the sdk.yaml file exists,
        # Get the version field from the yaml file.
        if os.path.isfile(sdk_yaml_file_path):
            with open(sdk_yaml_file_path, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                    version_field = data.get('version')
                    if version_field:
                        # Use regex to find the version number
                        version_number = re.search(regex_ver, version_field)
                        if version_number:
                            return version_number.group().strip()
                except yaml.YAMLError as exc:
                    return str(exc)

        # If sdk.yaml file is not able to provide version number,
        # Get the last directory from the engine path,
        # Extract version number from sdk directory name using regex.
        engine_dir_name = os.path.basename(os.path.normpath(self.engine_path))
        # Used regex to find the version number
        version_number = re.search(regex_ver, engine_dir_name)
        if version_number:
            return version_number.group()

        # Log a warning if the version number can't be found and return empty string
        self.logger.warning('Cannot find engine version')
        return ""

    def _validate_runtime(self):
        """This function validates the sdk version requirement for aic
        runtime."""
        ver = version.parse(self.get_engine_version())
        if self.runtime == 'aic' and ver < version.parse('1.12.0'):
            raise InferenceEngineError('AIC runtime is not supported on qnn sdk version < 1.12.0')

    def _create_remote_directory(self, base_path, dir_path):
        remote_shared_path = os.path.join(base_path, dir_path)
        dir_creator_cmd = f'mkdir -p {dir_path}'
        if not self.target_device.is_path(remote_shared_path):
            code, out, err = self.target_device.execute(commands=[dir_creator_cmd], cwd=base_path)
            if code != 0:
                err_msg = str(err) if err else str(out)
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_CREATE_DIR_FAILED')(err_msg))

    def _push_to_wos(self):
        """
        It pushes required files to remote target device
        """
        arch = "aarch64-windows-msvc"
        # Make remote working directory
        code, _, err = self.target_device.make_directory(self.target_path)
        if code != 0:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_TARGET_DIR_CREATION_FAILED")(err))

        # Push bin and header folders
        folders_to_push = [
            f"bin/{arch}", "include/QNN", "share/QNN/converter", "share/QNN/converter/jni",
            "share/QNN/converter/jni/windows"
        ]
        for push_folder_name in folders_to_push:
            self._create_remote_directory(self.target_path, push_folder_name)
            local_dir = os.path.join(self.engine_path, push_folder_name)
            for generator_file in os.listdir(local_dir):
                src_path = os.path.join(local_dir, generator_file)
                dst_path = os.path.join(self.target_path, push_folder_name,
                                        os.path.basename(generator_file))
                code, _, err = self.target_device.push(src_path, dst_path)
                if code != 0:
                    raise InferenceEngineError(
                        get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

        # Push lib folder
        push_folder_name = f"lib/{arch}"
        self._create_remote_directory(self.target_path, push_folder_name)
        local_dir = os.path.join(self.engine_path, push_folder_name)
        for lib_file in self.backend_path:
            lib_file = lib_file.format(engine_path=self.engine_path, target_arch=arch)
            dst_path = os.path.join(self.target_path, push_folder_name)
            code, _, err = self.target_device.push(
                lib_file, os.path.join(dst_path, os.path.basename(lib_file)))
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

        # Push remaining files
        env_setter_path = os.path.join(self.engine_path, 'bin', 'envsetup.ps1')
        dependency_checker_path = os.path.join(self.engine_path, 'bin',
                                               'check-windows-dependency.ps1')
        target_files = [env_setter_path, dependency_checker_path, self.qnn_model_cpp]
        if self.qnn_model_bin is not None and os.path.exists(self.qnn_model_bin):
            target_files.append(self.qnn_model_bin)

        for i, t_file in enumerate(target_files):
            # copy envsetup.ps1 and check-windows-dependency.ps1 to bin folder
            if i < 2:
                dst_path = os.path.join(self.target_path, 'bin', os.path.basename(t_file))
            # copy model .cpp and .bin files to root folder
            else:
                dst_path = os.path.join(self.target_path, os.path.basename(t_file))
            code, _, err = self.target_device.push(t_file, dst_path)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))

        # copy input sample(s) to target
        self._push_input_samples()
        self._push_input_list()

        # Push config file on target device if provided
        if self.compiler_config_json or self.netrun_config_file:
            self.target_config_json_file = os.path.join(self.target_path, self._push_config_file())
        else:
            self.target_config_json_file = None

        # Push custom_op files to target device
        self._push_custom_op_packages()

    def _execute_wos_remote_inference(self):
        """This function calls on the Executor class and executes the model
        inference on device."""

        self._push_to_wos()

        arch = "aarch64-windows-msvc"

        # Run model lib generator
        qnn_model_cpp_target = os.path.join(self.target_path, os.path.basename(self.qnn_model_cpp))
        qnn_model_bin_target = None
        if self.qnn_model_bin is not None and os.path.exists(self.qnn_model_bin):
            qnn_model_bin_target = os.path.join(self.target_path,
                                                os.path.basename(self.qnn_model_bin))
        model_lib_path = os.path.join(self.target_path, 'model_lib')
        lib_gen_command = [
            self.lib_generator_executable, self.lib_generator_args["model_cpp"],
            qnn_model_cpp_target, self.lib_generator_args["output_path"], model_lib_path
        ]

        if qnn_model_bin_target is not None and self.target_device.is_path(qnn_model_bin_target):
            lib_gen_command.extend([self.lib_generator_args["model_bin"], qnn_model_bin_target])
        else:
            self.logger.warning(
                'No Model BIN found for Model at {}. This is ok if model does not have any static tensors.'
                .format(self.qnn_model_bin))
        lib_gen_command_str = 'python ' + ' '.join(lib_gen_command)
        self.logger.debug('Model libgenerate command : {}'.format(lib_gen_command_str))
        code, out, err = self.target_device.execute(commands=[lib_gen_command_str],
                                                    cwd=os.path.join(self.target_path, 'bin', arch),
                                                    env=os.path.join(self.target_path, 'bin'))
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED')(err_msg))
        model_lib_path = os.path.join(model_lib_path, 'ARM64')
        self._target_model_path = os.path.join(
            model_lib_path,
            os.path.basename(qnn_model_cpp_target).split('.')[0] + '.dll')

        # Run context binary generator
        if self.offline_prepare:
            if self.runtime in ["cpu", "gpu"]:
                self.logger.error("offline_prepare can not be set for CPU and GPU backend")
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')(
                        "offline_prepare can not be set for CPU and GPU backend"))

            context_binary_generate_command = [
                self.context_binary_generator_config["executable"] + '.exe',
                self.context_binary_generator_config["arguments"]["model_path"],
                self._target_model_path,
                self.context_binary_generator_config["arguments"]["backend"],
                self.context_binary_generator_config["remote_backend_location"].format(
                    engine_path=self.target_path, target_arch=arch),
                self.context_binary_generator_config["arguments"]["binary_file"], self.lib_name,
                self.context_binary_generator_config["arguments"]["output_dir"], model_lib_path
            ]
            if self.profiling_level:
                context_binary_generate_command.extend([
                    self.context_binary_generator_config["arguments"]["profiling_level"],
                    self.profiling_level
                ])
            # if both debug_mode and intermediate_outputs are enabled , intermediate_outputs takes precedence
            # and dumps only specified intermediate outputs
            if self.intermediate_outputs:
                context_binary_generate_command.extend([
                    self.context_binary_generator_config["arguments"]["output_tensors"],
                    self.qnn_model_name + ':' + self.intermediate_outputs
                ])
            if self.debug_mode and not self.intermediate_outputs:
                context_binary_generate_command.append(
                    self.context_binary_generator_config["arguments"]
                    ["enable_intermediate_outputs"])
            if self.precision in ['fp16', 'int8'] and self.compiler_config_json:
                context_binary_generate_command += [
                    self.context_binary_generator_config["arguments"]["config_file"],
                    self.target_config_json_file
                ]
            if self.extra_contextbin_args:
                context_binary_generate_command.append(self.extra_contextbin_args)

            context_binary_gen_command_str = ' '.join(context_binary_generate_command)
            self.logger.debug(
                'context bin generator command : {}'.format(context_binary_gen_command_str))
            code, out, err = self.target_device.execute(
                commands=[context_binary_gen_command_str],
                cwd=os.path.join(self.target_path, 'bin',
                                 arch), env=os.path.join(self.target_path, 'bin'))
            if code != 0:
                err_msg = str(err) if err else str(out)
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED')(err_msg))

            self._target_model_path = os.path.join(model_lib_path, self.lib_name + '.bin')

        # run executor
        self._target_backend = self.backend_path[0].format(engine_path=self.target_path,
                                                           target_arch=arch)
        self._target_output_dir = os.path.join(self.target_path, 'output')
        self._execute_inference()

    # ------------------------------------------------ ABSTRACT FUNCTIONS ---------------------------------------------

    def run(self):
        self._setup()
        if self.target_arch == 'wos-remote':
            if self.stage == 'source':
                self._execute_conversion()
            self._execute_wos_remote_inference()
        else:
            self._execute_inference()

        if self.target_arch == 'wos-remote' or self.target_arch == 'aarch64-qnx':
            self.target_device.close()

    def get_graph_structure(self):
        self._load_graph()
        return self._extract_graph_structure()
