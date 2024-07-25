# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import os
from typing import Dict
import zipfile
import importlib
import subprocess
import builtins #needed for test mocking
from collections import OrderedDict

from lib.inference_engine import inference_engine_repository
from lib.inference_engine.inference_engines.nd_inference_engine import InferenceEngine
from lib.utils.nd_constants import ComponentType, Engine
from lib.utils.nd_exceptions import InferenceEngineError
from lib.utils.nd_errors import get_message
from lib.utils.nd_path_utility import get_absolute_path


@inference_engine_repository.register(cls_type=ComponentType.inference_engine,
                                      framework=None,
                                      engine=Engine.SNPE,
                                      engine_version="1.22.2.233")
class SNPEInferenceEngine(InferenceEngine):
    class DLCData:
        def __init__(self, output_layers, host_dlc_path, host_model_inputs, target_model_inputs):
            self.output_layers = output_layers
            # dict of input tensor names to input data locations on host device
            self.host_model_inputs = host_model_inputs
            # dict of input tensor names to input data locations on target device
            self.target_model_inputs = target_model_inputs

            # note that the dlc paths are the host and target base dlc in coarse grained mode
            self.host_dlc_path = host_dlc_path
            self.target_dlc_path = None

            self.host_input_list_path = None
            self.target_input_list_path = None

    def __init__(self, context, converter, executor):
        super().__init__(context, converter, executor)
        # Fields from context
        self.engine_type = context.engine
        self.stage = context.stage
        self.fine_grained_mode = context.fine_grain_mode
        self.engine_path = context.engine_path
        self.host_device = context.host_device
        self.target_device = context.target_device
        self.model_inputs = context.input_tensor
        self.model_outputs = context.output_tensor
        self.model_path = context.model_path
        self.snpe_lib_python = context.snpe_lib_python
        self.snpe_dlc_utils_package = context.snpe_dlc_utils_package
        self.host_output_dir = context.output_dir
        self.binary_paths = context.binary_paths
        self.target_arch = self.executor.target_arch
        self.target_path = self.executor.target_path
        self.static_model = context.static_model
        self.runtime = context.runtime
        self.input_list = context.input_list
        self.weights_bitwidth = context.weights_bitwidth
        self.bias_bitwidth = context.bias_bitwidth
        self.act_bitwidth = context.act_bitwidth
        self.no_weight_quantization = context.no_weight_quantization
        self.use_symmetric_quantize_weights = context.use_symmetric_quantize_weights
        self.use_adjusted_weights_quantizer = context.use_adjusted_weights_quantizer
        self.use_enhanced_quantizer = context.use_enhanced_quantizer
        self.override_params  = context.override_params
        self.offline_prepare = context.offline_prepare
        self.htp_socs = context.htp_socs
        self.debug_mode = context.debug_mode
        self.snpe_quantizer_config = context.snpe_quantizer_config

        if self.engine_path.endswith('.zip'):
            self.engine_zip_path=self.engine_path
            with zipfile.ZipFile(self.engine_zip_path, 'r') as zip_file:
                zip_file.extractall(self.host_output_dir)
                for file in zip_file.namelist():
                    os.chmod(os.path.join(self.host_output_dir, file), 0o755)
            if './' in  zip_file.namelist()[0]:
                self.engine_path = os.path.join(self.host_output_dir, os.path.dirname(zip_file.namelist()[1]))
            else:
                self.engine_path = os.path.join(self.host_output_dir, os.path.dirname(zip_file.namelist()[0]))
        elif os.path.isdir(self.engine_path):
            self.engine_zip_path=None
        else:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_PATH_INVALID")
                                       (self.engine_path))

        # Working directory and file location constants
        self._INPUT_LIST_DIR = os.path.join(self.host_output_dir, 'input_list_files')

        self._CONVERTER_LOCATION = context.converter_location.format(engine_path=self.engine_path)
        self._BASE_DLC_PATH = os.path.join(self.host_output_dir, 'base.dlc')

        # base dlc and associated info
        self.base_dlc_data = None
        self.base_dlc_info = None
        # list of sub dlcs and their associated information
        self.sub_dlc_data_list = []

        self.logger = context.logger

        #both _original_input_paths and _full_path_input_paths could contain tensor specific
        #inputs like input:=datapath.raw or just normal data path like path/to/data1.raw
        #_full_path_input_paths are basically the same as _original_input_paths except all the path inside are absolute path
        #paths in _original_input_paths could be absolute or relative. We should use _full_path_input_paths to refer to data path
        #from the input_list for most of the time
        self._original_input_paths = None
        self._full_path_input_paths = None

# -------------------------------------------------- COMMON ------------------------------------------------------------
    def _common_setup(self):
        """
        Set the base dlc, and create directories for input lists on host and target.
        """
        if self.target_device.device == "x86":
            self._TARGET_INPUT_LIST_DIR = self._INPUT_LIST_DIR
            self._TARGET_EXCUTE_DIR = self.host_output_dir
        else:
            self._TARGET_INPUT_LIST_DIR = os.path.join(self.executor.target_path, 'data')
            self._TARGET_EXCUTE_DIR = self.executor.target_path

        self._set_original_input_paths()
        self._set_input_list()

        self._set_host_environment()
        # Create the base dlc and info
        self._set_base_dlc()

        # Create directory for input list files on host
        os.makedirs(self._INPUT_LIST_DIR, exist_ok=True)
        # Create directory for input list files on target
        if not os.path.exists(self._TARGET_INPUT_LIST_DIR):
            code, _, err = self.target_device.make_directory(self._TARGET_INPUT_LIST_DIR)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_MKDIR_FAILED")(err))

    def _set_original_input_paths(self):
        """
        This function should be called before any modification to the original input_list
        """
        if self.input_list is not None:
        # get original input list paths
        #_original_input_paths stores all rel input paths in form of list of lists;
        # ie. if a input list has 2 batch and each batch require 3 inputs then the _original_input_paths would look like:
        # [[batch1_input1,batch1_input2,batch1_input3],[batch2_input1,batch2_input2,batch2_input3]]
            with open(self.input_list, "r") as input_list:
                self._original_input_paths = [line.strip(' \n').split(' ') for line in input_list.readlines()]
            if self._original_input_paths==[]:self.input_list=None

    def _set_host_environment(self):
        self.host_env = {}
        self.host_env["PATH"] = self._CONVERTER_LOCATION + os.pathsep + '$PATH'
        self.host_env['PYTHONPATH'] = os.path.join(self.engine_path, self.snpe_lib_python) + os.pathsep + '$PYTHONPATH'

        print(self.host_env)
        sys.path.insert(0, os.path.join(self.engine_path, self.snpe_lib_python))
        sys.path.insert(0, os.path.join(self.engine_path, self.snpe_dlc_utils_package))
        self.snpe_dlc = importlib.import_module('snpe_dlc_utils')


    def _set_base_dlc(self):
        """
        Convert the entire model to set the base dlc and extract model info.
        """
        base_model_inputs=None
        if self.stage == 'source':
            # Execute the conversion
            conversion_inputs = {input_name: dim for input_name, dim, data in self.model_inputs}
            self._execute_conversion(self._BASE_DLC_PATH, conversion_inputs, self.model_outputs)
            base_model_inputs = {input_name: data for input_name, dim, data in self.model_inputs}
        elif self.stage == 'converted' or self.stage == 'compiled':
            self._BASE_DLC_PATH = self.static_model
        else:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_SNPE_STAGE_ERROR"))

        # quantize the model if it's float dlc ie. from stage source or converted.
        if (self.stage == 'source' or self.stage == 'converted') and "dsp" in self.runtime:
            self._BASE_DLC_PATH = self._execute_dlc_quantization(self._BASE_DLC_PATH)
        # Set base dlc data object
        self.base_dlc_data = self.DLCData(output_layers=self.model_outputs,
                                          host_dlc_path=self._BASE_DLC_PATH,
                                          host_model_inputs=base_model_inputs,
                                          target_model_inputs=None)
        # Extract info from initial dlc
        m = self.snpe_dlc.ModelInfo()
        self.base_dlc_info = m.extract_model_info(self.base_dlc_data.host_dlc_path)
        # with open("./base_dlc_info_dump", "w") as basedlc:
        #     basedlc.writelines(self.base_dlc_info)


    def _set_input_list(self):
        """
        This function prepends input list's current directory to each of the
        relative paths in input list, resulting in an input list with absolute paths.
        :param curr_dir: input list's current directory
        """

        if not self.input_list: return

        curr_dir = os.path.dirname(self.input_list)

        #this here basically means for each item in each line of _original_input_paths, make it an absolute path
        self._full_path_input_paths = [[get_absolute_path(rel_path, checkExist=True, pathPrepend=curr_dir) if ":=" not in rel_path \
            else rel_path.split(":=")[0]+":="+ get_absolute_path(rel_path.split(":=")[1], checkExist=True, pathPrepend=curr_dir) for rel_path in per_batch] \
            for per_batch in self._original_input_paths]
        #create a new input_list_file in the output_dir and use that
        self.input_list=os.path.join(self.host_output_dir,os.path.basename(self.input_list))
        with open(self.input_list, "w") as input_list:
            input_list.write('\n'.join([' '.join(per_batch) for per_batch in self._full_path_input_paths]))


# -------------------------------------------------- COMMON HELPERS ----------------------------------------------------

    @staticmethod
    def _write_input_list(dlc_data, input_list_dir):
        """
        Create an input list on the host device.

        :param dlc_data: DLC data object
        :param input_list_dir: Directory to place input list files
        """

        # Set input list file name
        layers = ['-'.join(layer_name.split('/')) for layer_name in dlc_data.output_layers]
        layer_file_name = '&'.join(layers) + '.txt'
        file_path = os.path.join(input_list_dir, layer_file_name)
        dlc_data.host_input_list_path = file_path
        print("####file_path####:{}".format(file_path))
        string_to_write = "#"
        for output in dlc_data.output_layers:
            string_to_write += output + ' '
        string_to_write += '\n'

        for batch in dlc_data.target_model_inputs:
            for items in batch:
                print("item in write_input_list: {}".format(items))
                string_to_write += items + ' '
            string_to_write += '\n'
        with open(dlc_data.host_input_list_path, 'w+') as f:
            print("####stringtowrite#####: {}".format(string_to_write))
            f.write(string_to_write)


    def _execute_dlc_quantization(self, dlc_path):
        """
        Execute DLC quantization.
        :param dlc_path: Path to the converter dlc result
        :param inputs: Input list of raw files
        :param outputs: Output names of dlc modle after quntize
        :weights_bitwidth: the bitwidth to use when quantizing the weights
        :act_bitwidth: the bitwidth to use when quantizing the act
        :bias_bitwidth: the bitwidth to use when quantizing the bias
        """
        snpe_quantize_outputs = dlc_path.split('.', 2)[0] + "_quantized.dlc"
        try:
            convert_command = [self.snpe_quantizer_config["executable"], self.snpe_quantizer_config["arguments"]["dlc_path"], dlc_path]
            if self.input_list:
                convert_command += [self.snpe_quantizer_config["arguments"]["input_list"], self.input_list]
            else:
                raise InferenceEngineError("snpe dlc quantization should be input the inputlist, but you miss it!")

            convert_command += [self.snpe_quantizer_config["arguments"]["act_bitwidth"] + "=" + str(self.act_bitwidth)]
            convert_command += [self.snpe_quantizer_config["arguments"]["bias_bitwidth"] + "=" + str(self.bias_bitwidth)]
            convert_command += [self.snpe_quantizer_config["arguments"]["output_path"], snpe_quantize_outputs]

            if not self.no_weight_quantization:
                if self.use_symmetric_quantize_weights:
                    convert_command += [self.snpe_quantizer_config["arguments"]["use_symmetric_quantize_weights"]]
                else:
                    convert_command += [self.snpe_quantizer_config["arguments"]["weights_bitwidth"] + "=" + str(self.weights_bitwidth)]
            else:
                convert_command += [self.snpe_quantizer_config["arguments"]["no_weight_quantization"]]

            if self.use_adjusted_weights_quantizer:
                convert_command += [self.snpe_quantizer_config["arguments"]["use_adjusted_weights_quantizer"]]
            if self.use_enhanced_quantizer:
                convert_command += [self.snpe_quantizer_config["arguments"]["use_enhanced_quantizer"]]
            if self.override_params:
                convert_command += [self.snpe_quantizer_config["arguments"]["override_params"]]
            if self.offline_prepare:
                convert_command += [self.snpe_quantizer_config["arguments"]["enable_htp"]]
            if self.htp_socs:
                convert_command += [self.snpe_quantizer_config["arguments"]["htp_socs"] + "=" + self.htp_socs]

            convert_command += self.snpe_quantizer_config["arguments"]["flags"]
            convert_command_str = ' '.join(convert_command)

            log_string = 'Running DLC quantize with: ' + \
                        'Inputs: ' + str(dlc_path) + ' ' + \
                        'Outputs: ' + str(snpe_quantize_outputs)
            self.logger.info(log_string)
            code, _, err = self.host_device.execute(commands=[convert_command_str],
                                                    cwd=self.engine_path,
                                                    env=self.host_env)
            if code != 0:
                raise InferenceEngineError(get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))
            self.logger.info('DLC model quantized successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))

        if os.path.isfile(snpe_quantize_outputs):
            return snpe_quantize_outputs


    def _execute_conversion(self, dlc_path, inputs, outputs):
        """
        Convert a model into a dlc.

        :param dlc_path: Path to save the new dlc
        :param inputs: Input names and dimensions to the model
        :param outputs: Output names of the model
        """

        convert_command = self.converter.build_convert_command(self.model_path,
                                                               inputs,
                                                               outputs,
                                                               dlc_path)

        log_string = 'Starting conversion with: ' + \
                     'Inputs: ' + str(list(inputs.keys())) + ' ' + \
                     'Outputs: ' + str(outputs)
        self.logger.info(log_string)

        print(self.host_env)

        try:
            code, _, err = self.host_device.execute(commands=[convert_command], env=self.host_env)
            if code != 0:
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))

            self.logger.info('Model converted successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))

    def _execute_inference(self, dlc_data):
        """
        Run inference.

        :param dlc_data: DLC data object
        """
        if self.target_device.device == "x86":
            self.host_env['LD_LIBRARY_PATH'] = os.path.join(self.engine_path,"lib", self.target_arch)
        else:
            for library_path_name, path in self.executor.get_execute_environment_variables():
                self.host_env[library_path_name] = path

        execute_command = self.executor.build_execute_command(dlc_data.target_dlc_path,
                                                              dlc_data.target_input_list_path)
        log_string = 'Running inference with: ' + \
                     'Inputs: ' + str(dlc_data.target_model_inputs) + ' ' + \
                     'Outputs: ' + str(dlc_data.output_layers)
        self.logger.info(log_string)

        try:
            code, _, err = self.target_device.execute(commands=[execute_command], cwd=self._TARGET_EXCUTE_DIR, env=self.host_env)
            if code != 0:
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))
            self.logger.info('Inference executed successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))

    def _pull_inference_results(self):
        """
        Pull inference results from target device to host.
        """
        code, _, err = self.target_device.pull(os.path.join(self._TARGET_EXCUTE_DIR, "output"), self.host_output_dir)
        if code != 0:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED"))

        code, out, err = self.target_device.remove(target_path=self._TARGET_EXCUTE_DIR)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_REMOVE_RESULTS_FAILED')(err_msg))
        self.logger.info('Removed inference results from device successfully')

    def get_binaries_to_push(self):
        """
        Get SNPE binaries used to convert and run a DLC.

        :return: List of binary paths
        """

        def fill(path):
            src, target = path
            return src.format(engine_path=self.engine_path, target_arch=self.target_arch), \
                   target.format(target_path=self.target_path, target_arch=self.target_arch)

        return [fill(binary_path) for binary_path in self.binary_paths]

# -------------------------------------------------- COARSE GRAINED ----------------------------------------------------
    def _push_libs(self):
        """
        Push binaries and base dlc to target device.

        """
        # Push binaries to target device
        binary_paths = self.get_binaries_to_push()
        self.logger.info('Pushing binaries')
        for source, target in binary_paths:
            code, _, err = self.target_device.push(os.path.join(source, '.'), target)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))

        # Push base dlc to target device
        try:
            self.logger.info('Pushing base dlc to target')
            code, _, err = self.target_device.push(self.base_dlc_data.host_dlc_path, self.executor.target_path)
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE"))
            self.base_dlc_data.target_dlc_path = os.path.join(self.executor.target_path,
                                                              os.path.basename(self.base_dlc_data.host_dlc_path))
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE'))

    def _push_input_data(self,target_device):
        """
        Push user specified input data to target device.
        x86 target does not push, just need to set the correct path
        """
        on_x86=target_device=='x86'
        # Push input data to target device
        self.logger.info('Pushing input data to target')
        if self.model_inputs is not None:
            per_batch_inputs=[]
            for input_name, dim, data_path in self.model_inputs:

                if on_x86:
                    per_batch_inputs.append(input_name+":="+data_path)
                else:
                    device_model_input_path = os.path.join(self.executor.target_path, os.path.basename(data_path))
                    code, _, err = self.target_device.push(data_path, device_model_input_path)
                    if code != 0:
                        raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                    per_batch_inputs.append(input_name+":="+device_model_input_path)
            target_model_inputs=[per_batch_inputs]
        elif self.input_list:
            target_model_inputs=[]
            # get from input_list.
            #assumes the path is already full correct
            for batch in self._full_path_input_paths:
                print("####Frank_first_line####"+str(batch))
                per_batch_inputs=[]
                for inputs in batch:
                    if on_x86:
                        device_model_input_path=inputs
                    else:
                        if ":=" in inputs:
                            splited_inputs=inputs.split(":=")
                            tensor = splited_inputs[0]
                            data_path=splited_inputs[1]
                        else:
                            data_path=inputs

                        device_model_input_path=os.path.join(self.executor.target_path, os.path.basename(data_path))
                        code, _, err = self.target_device.push(data_path, device_model_input_path)
                        if code != 0:
                            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                        #Update the path after push to become tensor specific path
                        if ":=" in inputs:
                            device_model_input_path=tensor+":="+os.path.join(self.executor.target_path, os.path.basename(data_path))

                    per_batch_inputs.append(device_model_input_path)
                target_model_inputs.append(per_batch_inputs)
        print(target_model_inputs)
        return target_model_inputs


    def _coarse_grained_run(self):
        """
        Convert and run in coarse grained mode.
        Creates an input list for every layer in the base dlc that uses, each using user specified model inputs
        for the input and the layer output for the output, then pushes the file to target, runs inference,
        and pulls the result.
        """
        # if target device is x86, no need to push
        if self.target_device.device == "x86":
            self.base_dlc_data.target_dlc_path = self.base_dlc_data.host_dlc_path
        else:
            # Push binaries and base dlc
            self._push_libs()
        target_model_inputs=self._push_input_data(self.target_device.device)
            # Target model inputs are same for every inference in coarse grained mode

        # Set inputs to the model on the target device
        self.base_dlc_data.target_model_inputs = target_model_inputs
        # Create dlc data objects and add to the engine's list of dlc data
        def dlcdata_inference(layer_row):
            # always use the base dlc and base inputs in coarse grained mode
            dlc_data = self.DLCData(output_layers=[layer_row.name],
                                    host_dlc_path=self.base_dlc_data.host_dlc_path,
                                    host_model_inputs=self.base_dlc_data.host_model_inputs,
                                    target_model_inputs=self.base_dlc_data.target_model_inputs)
            dlc_data.target_dlc_path = self.base_dlc_data.target_dlc_path
            self.sub_dlc_data_list.append(dlc_data)

            # Create input list
            SNPEInferenceEngine._write_input_list(dlc_data, self._INPUT_LIST_DIR)

            # Push input list to target and set dlc_data's target input list
            self.logger.info('Pushing input list to target: ' + dlc_data.host_input_list_path)
            input_list_file_dest = os.path.join(self._TARGET_INPUT_LIST_DIR,
                                                os.path.basename(dlc_data.host_input_list_path))
            dlc_data.target_input_list_path = input_list_file_dest
            if not self.target_device.device == "x86":
                try:
                    code, _, err = self.target_device.push(dlc_data.host_input_list_path, input_list_file_dest)
                    if code != 0:
                        raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                    dlc_data.target_input_list_path = input_list_file_dest
                except subprocess.CalledProcessError as exc:
                    self.logger.error('Error pushing input list to target. Continuing with inference on next layer.')
                    self.logger.error(str(exc))
                    return

            # Execute inference
            try:
                self._execute_inference(dlc_data)
            except InferenceEngineError as exc:
                self.logger.error('Error executing inference. Continuing with inference on next layer.')
                self.logger.error(str(exc))
                return

        if not self.offline_prepare and self.debug_mode:
            firstRow = True
            for layer_row in self.base_dlc_info:
                # skip first row
                if firstRow:
                    firstRow = False
                    continue
                dlcdata_inference(layer_row)
        else:
            dlcdata_inference(self.base_dlc_info[-1])

        if not self.target_device.device == "x86":
            # Pull results
            self._pull_inference_results()
# -------------------------------------------------- FINE GRAINED ------------------------------------------------------

    def _fine_grained_run(self):
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_FINE_GRAINED_NOT_SUPPORTED'))

# ----------------------------------------------------------------------------------------------------------------------

    def run(self):
        self._common_setup()
        if self.fine_grained_mode:
            self._fine_grained_run()
        else:
            self._coarse_grained_run()


    def get_graph_structure(self):
        if self.base_dlc_info is None:
            self.logger.info('Converting model to obtain graph structure.')
            self._set_base_dlc()
        graph_list_structure = [(layer.name, [layer.type, dict(zip(layer.input_names, layer.input_dims)),
                                  dict(zip(layer.output_names, layer.output_dims_list))])
                                for layer in self.base_dlc_info]
        return OrderedDict(graph_list_structure)
