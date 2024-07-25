# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
import shutil
import signal
import sys
import traceback
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from qti.aisw.accuracy_debugger.lib.inference_engine.nd_execute_inference_engine import exec_inference_engine
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import show_progress
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError, SnoopingError
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, transpose_to_nhwc
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import AxisFormat
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import permute_tensor_data_axis_order

# used to mark layerwise snooping started.
snoop_stop = False
pgq_data = {}


class LayerStatus():
    LAYER_STATUS_SUCCESS = ''
    LAYER_STATUS_CON_ERROR = 'err_con'
    LAYER_STATUS_LIB_ERROR = 'err_lib'
    LAYER_STATUS_CNTX_ERROR = 'err_cntx'
    LAYER_STATUS_EXEC_ERROR = 'err_exec'
    LAYER_STATUS_PARTITION_ERR = 'err_part'
    LAYER_STATUS_SKIPPED = 'skip'
    LAYER_STATUS_PARTITION = ' part'
    LAYER_STATUS_COMPARE_ERROR = 'err_compare'


def signal_handler(sig, frame):
    global snoop_stop
    logging.info('Stopping snooping on user request.')
    if snoop_stop:
        logging.info('Waiting for current layer to complete.')
        snoop_stop = True
    else:
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


def files_to_compare(framework_path, inference_path, cur_layer_out_name, d_type, logger,
                     out_folder=None):
    """This method returns the file paths to compare.

    Args:
        cur_layer_out_name  : output name of layer
        framework_path      : path to the reference framework results
        inference_path      : path to the qnn inference results
        d_type              : datatype of the layer
        out_folder          : name of output folder of QNN
    Returns:
        inf_path : path of output file from qnn platform
        rt_path  : path of output file from reference platform
    """
    rt_raw = None
    inf_raw = None
    sanitized_cur_layer_out_name = santize_node_name(cur_layer_out_name)
    folder_name = out_folder if out_folder else sanitized_cur_layer_out_name
    rt_path = os.path.join(framework_path, sanitized_cur_layer_out_name + '.raw')
    if os.path.exists(rt_path):
        rt_raw = np.fromfile(rt_path, dtype=d_type)
    inf_path = os.path.join(inference_path, 'inference_engine', folder_name, 'output/Result_0',
                            sanitized_cur_layer_out_name + '.raw')

    if os.path.exists(inf_path):
        inf_raw = np.fromfile(inf_path, dtype=d_type)
    logger.debug('compare files inf_path : {} \t rt_path : {}'.format(inf_path, rt_path))

    return inf_raw, rt_raw


class LayerwiseSnooping:
    """Class that runs layer wise snooping."""

    def __init__(self, args, logger, c_logger=None):
        self.args = args
        self.logger = logger
        self.c_logger = c_logger
        self.input_list_file = args.input_list
        from qti.aisw.accuracy_debugger.lib.wrapper.nd_tool_setup import ToolConfig
        self.envToolConfig = ToolConfig()
        self.model = args.model_path
        self.engine_path = args.engine_path
        self.engine = args.engine
        self.framework = args.framework
        self.framework_version = None
        self.runtime = args.runtime
        self.framework_results = args.golden_output_reference_directory
        self.work_dir = args.working_dir
        self.output_dir = args.output_dir
        self.model_traverser = None
        self.model_handler = None
        self.target_device = args.target_device
        self.host_device = args.host_device
        self.architecture = args.architecture
        self.precision = args.precision
        self.compiler_config = args.compiler_config
        self.profile_info = None
        self.extra_converter_args = args.extra_converter_args
        self.extra_runtime_args = args.extra_runtime_args
        self.remote_server = args.remote_server
        self.remote_username = args.remote_username
        self.remote_password = args.remote_password
        self.act_quantizer = args.act_quantizer
        self.param_quantizer = args.param_quantizer
        self.bias_bitwidth = args.bias_bitwidth
        self.weights_bitwidth = args.weights_bitwidth
        self.act_bitwidth = args.act_bitwidth
        self.quantization_overrides = args.quantization_overrides
        self.algorithms = args.algorithms
        self.ignore_encodings = args.ignore_encodings
        self.per_channel_quantization = args.per_channel_quantization
        self.add_layer_outputs = args.add_layer_outputs
        self.add_layer_types = args.add_layer_types
        self.use_native_input_files = args.use_native_input_files
        self.use_native_output_files = args.use_native_output_files
        self.symbol_data_map = self.define_symbols()
        self.is_transpose_needed_dict = {}

    def define_symbols(self):
        """Populate symbol_data_map with mappings of onnx model symbols with
        corresponding data which is provided by user in extra_converter_args.
        Symbols are defined in the extra_converter_args as following examples:

        'define_symbol batch_size = 1'
        'define_symbol seq_len 128'

        """
        symbol_data_map = {}
        if self.extra_converter_args:
            converter_args = self.extra_converter_args.split(';')
            for arg in converter_args:
                arg = " ".join(arg.split())
                split_arg = re.split(' |=', arg)
                if split_arg[0] == "define_symbol":
                    if len(split_arg) == 3:
                        symbol, data = split_arg[1], split_arg[2]
                        symbol_data_map[symbol] = data
                    else:
                        raise ConfigError('Symbols are not defined correctly in extra_converter_args.')

        return symbol_data_map

    def get_input_tensors(self, list_file, model=None):
        input_tensors = []
        with open(list_file, 'r') as f:
            input_paths = f.readline().rstrip().split(',')

        # get input_layers of extracted model by creating a new handler for it
        from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
        framework_args = Namespace(None, framework=self.framework, version=self.framework_version,
                                   model_path=model, output_dir=self.output_dir, engine=self.engine)
        model_handler = FrameworkRunner(self.logger, framework_args)
        model_handler.load_framework()
        input_layers = model_handler.framework_instance.get_input_layers()

        for i, item in enumerate(input_layers):
            if i >= len(input_paths):
                break
            dim_str = str(item[2])
            for symbol, data in self.symbol_data_map.items():
                dim_str = dim_str.replace(symbol, data)
            dim_str = dim_str.replace(' ', '')[1:-1]
            if ":=" in input_paths[i]:
                input_paths[i] = input_paths[i].split(":=")[1]
            input_data = (item[0], dim_str, input_paths[i])
            input_tensors.append(input_data)
        return input_tensors

    def update_list_file(self, graph_inputs):
        """Create a new input list file (temp-list.txt) based on the given
        input names."""
        updated_input_list = []
        handleInputNames = False
        # check is needed for caffe
        if isinstance(graph_inputs, dict):
            input_map = graph_inputs.copy()
            handleInputNames = True
            graph_inputs = list(graph_inputs.values())
            os.makedirs(self.work_dir + '/temp_inp/', exist_ok=True)

        for ip in graph_inputs:
            if ip in self.original_input_names_raw_map:
                updated_input_list.append(self.original_input_names_raw_map[ip].strip())
            else:
                s_ip = santize_node_name(ip)
                inp_path = os.path.join(self.framework_results, s_ip + '.raw')
                inp_shape = self.profile_info[s_ip][1]
                inp_dtype = self.profile_info[s_ip][0]
                inp_raw = np.fromfile(inp_path, dtype=inp_dtype)
                if ip in self.is_transpose_needed_dict:
                    inp_raw = transpose_to_nhwc(inp_raw, inp_shape)
                inp_path = os.path.join(self.framework_results, s_ip + '_nhwc.raw')
                inp_raw.tofile(inp_path)

                if handleInputNames:
                    # move req input files to temp folder
                    dst_path = self.work_dir + '/temp_inp/' + list(input_map.keys())[list(
                        input_map.values()).index(ip)] + '.raw'
                    try:
                        shutil.copy(inp_path, dst_path)
                        self.logger.debug('copied file {} to {}'.format(inp_path, dst_path))
                        inp_path = dst_path
                    except:
                        inp_path = self.work_dir + '/temp_inp/' + list(input_map.keys())[list(
                            input_map.values()).index(ip)] + '.raw'
                updated_input_list.append(inp_path)

        # creating new input-list-file for extracted model
        if len(updated_input_list) > 0:
            with open(self.output_dir + '/temp-list.txt', "w") as f:
                f.write(','.join(updated_input_list))

        list_file = self.output_dir + '/temp-list.txt'
        return list_file

    def initiate_model_extraction(self, model, start_layer=None, end_layer=None, set_model=True):
        """
        This method partitions the model at start layer output till end layer and generates
        updated input list file
        Args:
            model : path to the model which needs to be partitioned
        Returns:
            status          : True if success
            model           : path to partitioned model
            list_file       : input list file for partitioned model
            new_g_inputs    : list of new inputs of partitioned model
        """
        s_utility = su.getInstance()
        self.model_handler = s_utility.getFrameworkInstance()

        # populate original_input_names_raw_map needed for end layer extraction.
        if set_model:
            start_layer = s_utility.getStartLayer()
            end_layer = s_utility.getEndLayer()
        valid_layers = [item[1] for item in self.model_traverser._layerlist]
        # check if valid layers are provided as start/end layers
        if start_layer and start_layer not in valid_layers:
            raise ConfigError('{} is not present in {}. Please provide valid start_layer'.format(
                start_layer, model))
        if end_layer and end_layer not in valid_layers:
            raise ConfigError('{} is not present in {}. Please provide valid end_layer'.format(
                end_layer, model))

        list_file = self.input_list_file
        original_input_names = self.model_traverser.framework_instance.get_input_layers(
            names_only=True)

        with open(list_file, 'r') as F:
            file_names = F.readline().strip().split(',')
            self.original_input_names_raw_map = dict(zip(original_input_names, file_names))
        (ret_status, model,
         new_g_inputs) = self.model_handler.extract_sub_graph(start_layer, end_layer,
                                                              self.output_dir)

        if not ret_status:
            return False, None, None, None
        # create input list file for partitioned model
        list_file = self.update_list_file(new_g_inputs)

        return True, model, list_file, new_g_inputs

    def handle_qnn_run_failure(self, std_out, cur_layer_out_name, layer_status_map, conv_fail_nodes,
                               lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes):
        """
        This method handles the compilation and execution failures of qnn run
        Args:
            std_out             : output of qnn inference engine
            cur_layer_out_name  : output name of layer
            layer_status_map    : dict that gives status of each layer
            conv_fail_nodes     : list of qnn converter fail layers
            lib_fail_nodes      : list of qnn lib-generator fail layers
            cntx_fail_nodes     : list of qnn context binary generator fail layers
            exec_fail_nodes     : list of qnn net-run fail layers
        Returns:
            conv_fail_nodes     : updated list of qnn converter fail layers
            lib_fail_nodes      : updated list of qnn lib-generator fail layers
            cntx_fail_nodes     : updated list of qnn context binary generator fail layers
            exec_fail_nodes     : updated list of qnn net-run fail layers
            layer_status_map    : updated dict that gives status of each layer
        """
        s_cur_layer_out_name = santize_node_name(cur_layer_out_name)
        if 'ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED' in std_out:
            # handles qnn_converter failure
            conv_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CON_ERROR

        elif 'ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED' in std_out:
            # handles qnn_lib_generator failure
            lib_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_LIB_ERROR

        elif 'ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED' in std_out:
            # handles qnn_context_bin_gen failure
            cntx_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CNTX_ERROR

        elif 'ERROR_INFERENCE_ENGINE_INFERENCE_FAILED' in std_out:
            # handles qnn_net_run failure
            exec_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_EXEC_ERROR
        return conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map

    def partition_initial_model(self, model):
        s_utility = su.getInstance(self.args)

        if s_utility.getStartLayer() or s_utility.getEndLayer():
            self.set_profile_info(self.logger, self.args, model)
            status, model, list_file, _ = self.initiate_model_extraction(model)
            if status is False:
                return status, None, None
            # keep a copy of extracted model as there is chance of replacement due to partitions
            if os.path.exists(os.path.join(self.work_dir, 'cleaned')) and os.path.isdir(
                    os.path.join(self.work_dir, 'cleaned')):
                if os.path.exists(model):
                    model_dir = os.path.dirname(model)
                    if not os.path.exists(
                            os.path.join(
                                self.output_dir, 'transformed' +
                                self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)):
                        os.makedirs(
                            os.path.join(
                                self.output_dir, 'transformed' +
                                self.model_traverser.framework_instance.FRAMEWORK_SUFFIX))
                    for path in os.listdir(model_dir):
                        shutil.copy(
                            os.path.join(model_dir, path),
                            os.path.join('cleaned', 'cleanmodel' + os.path.splitext(path)[1]))
            else:
                if os.path.exists(model):
                    shutil.copy(
                        model,
                        os.path.join(
                            self.output_dir, 'cleanmodel' +
                            self.model_traverser.framework_instance.FRAMEWORK_SUFFIX))
            model = os.path.join(
                self.output_dir,
                'cleanmodel' + self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)
        else:
            list_file = self.input_list_file

        return True, model, list_file

    def set_profile_info(self, logger, args, model, list_file=None):
        """
        Create and set profile info of model.
        """
        s_utility = su.getInstance(self.args)
        self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args, model)
        self.model_traverser = s_utility.setModelTraverserInstance(self.logger, self.args, model,
                                                                   self.add_layer_outputs,
                                                                   self.add_layer_types)

        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)
        original_input_names = self.model_handler.framework_instance.get_input_layers(
            names_only=True)

        if not list_file:
            list_file = self.input_list_file

        with open(list_file, 'r') as F:
            file_names = F.readline().strip().split(',')
            self.original_input_names_raw_map = dict(zip(original_input_names, file_names))

        # get profile info like tensor dimensions, dtype, min, max and median values
        profile_path = os.path.join(self.framework_results, 'profile_info.json')
        temp_profile_path = os.path.join(self.work_dir, 'temp', 'profile_info.json')
        if not os.path.exists(profile_path) and not os.path.exists(temp_profile_path):
            inputs = self.model_handler.framework_instance.get_input_layers()
            for idx, ip in enumerate(inputs):
                input_dim_str = ','.join(str(d) for d in ip[2])
                inputs[idx] = (ip[0], input_dim_str,
                               self.original_input_names_raw_map[inputs[idx][0]], ip[1])
            self.model_handler.generate_intermediate_outputs(os.path.join(self.work_dir,
                                                                          'temp'), input=inputs,
                                                             output=original_output_names)
            profile_path = os.path.join(self.work_dir, 'temp', 'profile_info.json')
        profile_info = read_json(profile_path)
        self.profile_info = profile_info

    def run(self, model=None, profile_info=None):
        """This method contains the sequence of debugger for
        LayerwiseSnooping."""
        model = self.model if not model else model
        ret_status = True
        s_utility = su.getInstance(self.args)

        global snoop_stop
        snoop_stop = False

        layer_status_map = {}
        layer_perc_map = {}
        layer_compare_info_map = {}
        layer_type_map = {}
        layer_shape_map = {}
        layer_dtype_map = {}
        layer_profile_map = {}
        conv_fail_nodes = []
        lib_fail_nodes = []
        cntx_fail_nodes = []
        exec_fail_nodes = []
        extract_fail_nodes = []
        compare_skip_nodes = []
        overall_comp_list = []
        comparators_list = s_utility.getComparator()
        layer_output_comp_map = {}
        layerwise_report_path = self.output_dir + '/layerwise.csv'

        # partition the model from user supplied -start-from-layer-output
        # the input list file is updated accordingly.
        status, model, list_file = self.partition_initial_model(model)
        self.input_list_file = list_file
        if status is False:
            return status

        self.set_profile_info(self.logger, self.args, model)
        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)

        profile_info = self.profile_info
        total_layers = self.model_traverser.get_layer_count()
        skip_count = 0

        tensor_mapping = self.initial_run(model)
        # get list of nodes from qnn inference tensor mapping
        valid_nodes = list(read_json(tensor_mapping).values())
        self.logger.info('Started layerwise snooping')

        # Snooping loop
        count = 0
        prev_layer_out_name = None
        while (True):
            skip_compare = False
            # Get next layer
            (status, layer_name, cur_layer_out_name,
             layer_type) = self.model_traverser.get_next_layer()
            if status == 1 or snoop_stop:
                # Reached end.
                break
            # check if cur_layer_out_name is in qnn valid nodes
            if not cur_layer_out_name:
                continue
            s_cur_layer_out_name = santize_node_name(cur_layer_out_name)
            if s_cur_layer_out_name not in valid_nodes:
                continue

            if prev_layer_out_name is not None:
                # Populate layer details with default and known values.
                layer_perc_map[s_cur_layer_out_name] = '-'
                layer_compare_info_map[s_cur_layer_out_name] = '-'

                if cur_layer_out_name in original_output_names:
                    layer_type_map[s_cur_layer_out_name] = '(*)' + layer_type
                else:
                    layer_type_map[s_cur_layer_out_name] = layer_type

                if profile_info and s_cur_layer_out_name in profile_info:
                    layer_profile_map[s_cur_layer_out_name] = profile_info[s_cur_layer_out_name][2:]
                    layer_shape_map[s_cur_layer_out_name] = profile_info[s_cur_layer_out_name][1]
                    layer_dtype_map[s_cur_layer_out_name] = profile_info[s_cur_layer_out_name][0]
                else:
                    layer_profile_map[s_cur_layer_out_name] = '-'
                    layer_shape_map[s_cur_layer_out_name] = '-'
                    layer_dtype_map[s_cur_layer_out_name] = '-'

                layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_SUCCESS

                if status == 2:  # Skipped layer
                    count += 1
                    skip_count += 1
                    prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(
                        skip_count)
                    show_progress(total_layers, count, prog_info)
                    layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_SKIPPED
                    continue

                # Check if snooping needs to be stopped
                if snoop_stop:
                    break

                count += 1
                # Show progress
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)

                self.logger.debug('Debugging layer ' + layer_name)

                # Extract sub model. Continue to next layer if extraction fails.
                try:
                    ret_status, extracted_model_path, list_file, new_g_inputs = \
                        self.initiate_model_extraction(model, prev_layer_out_name, cur_layer_out_name, set_model = False)
                    if ret_status:
                        # update status as partition success if no error
                        layer_status_map[s_cur_layer_out_name] += LayerStatus.LAYER_STATUS_PARTITION
                except Exception as e:
                    ret_status = False
                    traceback.print_exc()
                    self.logger.error('Extraction error {}'.format(e))
                if not ret_status:
                    extract_fail_nodes.append(cur_layer_out_name)
                    self.logger.error('Extraction failed at node {}'.format(cur_layer_out_name))
                    if cur_layer_out_name in layer_status_map:
                        layer_status_map[
                            s_cur_layer_out_name] += ',' + LayerStatus.LAYER_STATUS_PARTITION_ERR
                    else:
                        layer_status_map[
                            s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_PARTITION_ERR
                    continue

                # Use this extracted model for debugging.
                temp_model = extracted_model_path
                # self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args, temp_model)

                output_tensors = None

                # Execute model on QNN
                ret_inference_engine, std_out = self.execute_on_qnn(temp_model, list_file,
                                                                    output_tensors,
                                                                    [cur_layer_out_name],
                                                                    s_cur_layer_out_name)

                if ret_inference_engine != 0:
                    self.logger.warning(
                        f'inference-engine failed for the layer -> {s_cur_layer_out_name}, skipping verification'
                    )
                    skip_compare = True
                    percent_match = 0.0
                    conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map = self.handle_qnn_run_failure(
                        std_out, cur_layer_out_name, layer_status_map, conv_fail_nodes,
                        lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes)

                # Compare current layer outputs
                if not skip_compare:
                    d_type = [layer_dtype_map[s_cur_layer_out_name]]
                    inf_raw, rt_raw = files_to_compare(self.framework_results, self.output_dir,
                                                       cur_layer_out_name, d_type[0], self.logger)
                    sanitized_cur_layer_out_name = santize_node_name(cur_layer_out_name)
                    qnn_model_json_path = os.path.join(self.output_dir, 'inference_engine',
                                            sanitized_cur_layer_out_name, 'model_net.json')
                    qnn_model_tensors_json = None
                    if qnn_model_json_path is not None:
                        qnn_model_json = read_json(qnn_model_json_path)
                        qnn_model_tensors_json = qnn_model_json['graph']['tensors']

                    if qnn_model_tensors_json is not None and sanitized_cur_layer_out_name in qnn_model_tensors_json:
                            tensor_dict = qnn_model_tensors_json[sanitized_cur_layer_out_name]
                            src_axis_format = tensor_dict['src_axis_format']
                            axis_format = tensor_dict['axis_format']
                            tensor_dims = tensor_dict['dims']
                            rt_raw, is_permuted = permute_tensor_data_axis_order(
                                src_axis_format, axis_format, tensor_dims, rt_raw)
                            self.is_transpose_needed_dict[cur_layer_out_name] = is_permuted

                    info_origin = {}
                    if (inf_raw is not None) and (rt_raw is not None):
                        percent_match = {}
                        if cur_layer_out_name in layer_output_comp_map:
                            comp_list = layer_output_comp_map[s_cur_layer_out_name]
                        else:
                            comp_list = comparators_list.copy()

                        for idx, comp in enumerate(comp_list):
                            try:
                                match_info = '-'
                                _, percent = comp.verify(layer_type, None, [rt_raw], [inf_raw],
                                                         False)
                            except Exception:
                                percent, match_info = 0.0, ''
                                compare_skip_nodes.append(cur_layer_out_name)
                                self.logger.debug(
                                    'Skipping comparision for node : {}, and marking 0.0% match'.
                                    format(cur_layer_out_name))
                                layer_status_map[
                                    cur_layer_out_name] = LayerStatus.LAYER_STATUS_COMPARE_ERROR
                            # store percentage match for each user supplied comparator
                            comp_name = comp.V_NAME
                            percent_match[comp_name] = round(percent, 4)
                            if match_info:
                                info_origin[comp_name] = comp_name + ": " + match_info
                            # maintain a list of over all comparators used in snooping
                            if comp_name not in overall_comp_list:
                                overall_comp_list.append(comp_name)
                    else:
                        percent_match = 0.0

                    self.logger.info('Debug Layer {}, output {} match percent {}'.format(
                        layer_name, cur_layer_out_name, percent_match))

                    layer_perc_map[s_cur_layer_out_name] = percent_match
                    layer_compare_info_map[s_cur_layer_out_name] = "\n".join(
                        list(info_origin.values()))

            prev_layer_out_name = cur_layer_out_name

            #  Exit if end layer is provided
            if s_utility.getEndLayer() == cur_layer_out_name:
                skip_count += (total_layers - count)
                count = total_layers
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)
                break

        print("============== Layerwise Debug Results ==============")
        pd.set_option('display.max_rows', None, 'display.max_colwidth', 30, 'expand_frame_repr',
                      False)

        # to split the layer_perc_map into multiple dicts comparator wise
        overall_comp_list.sort()
        perc_compwise_map = {}
        for idx, elem in enumerate(overall_comp_list):
            _updated = {}
            for k, v in layer_perc_map.items():
                try:
                    if overall_comp_list[idx] in v:
                        _updated[k] = v[overall_comp_list[idx]]
                    else:
                        _updated[k] = '-'
                except:
                    _updated[k] = '-'
            perc_compwise_map[elem] = _updated

        #Check if info column is populated for all the keys:
        for op in layer_perc_map.keys():
            if op not in layer_compare_info_map or layer_compare_info_map[op] == '':
                layer_compare_info_map[op] = '-'

        perc_compwise_list = [perc_compwise_map[elem] for elem in overall_comp_list]
        results_dicts = ([layer_status_map, layer_type_map, layer_shape_map, layer_profile_map] +
                         perc_compwise_list + [layer_compare_info_map])
        results_dict = {}
        for k in layer_perc_map.keys():
            results_dict[k] = tuple(d[k] for d in results_dicts)
        if len(results_dict) == 0:
            logging.info('No layers has been debugged.')
            return ret_status

        df = pd.DataFrame.from_dict(results_dict, orient='index')
        labels = ['Status', 'Layer Type', 'Shape', 'Activations (Min,Max,Median)'
                  ] + overall_comp_list + ['Info']
        df.columns = labels
        df.index.name = 'O/P Name'
        print('\n' + str(df))
        df.to_csv(layerwise_report_path)
        print('Results saved at {}'.format(layerwise_report_path))
        print("\n============== Error details ==============")
        print(
            'Converter Failures at nodes : {} \nLibgenerator Failures at nodes : {} \nContext Binary Genrator Failures at nodes : {} \nExtraction Failures at nodes : {} \nExecution '
            'Failures at nodes : {} \nComparition Failures at nodes : {}'.format(
                str(conv_fail_nodes), str(lib_fail_nodes), str(cntx_fail_nodes),
                str(extract_fail_nodes), str(exec_fail_nodes), str(compare_skip_nodes)))

        # Layer Snooping completed.
        self.logger.debug('Layerwise snooping completed successfully')
        return ret_status

    def execute_on_qnn(self, model=None, list_file=None, output_tensors=None,
                       orig_model_outputs=None, out_dir=None, capture_intermediate_outputs=False):
        """This method executes the given model on qnn platform.

        Args:
            model                           : path of the model
            list_file                       : file containing input paths to model
            output_tensors                  : output node names of model
            out_dir                         : output folder name inside work directory
            capture_intermediate_outputs    : boolean flag to save intermediate outputs of model
        Returns:
            ret_status                      : status of qnn execution
            std_out                         : console output of qnn inference engine
        """
        model = model if model else self.model
        list_file = list_file if list_file else self.input_list_file
        input_tensors = self.get_input_tensors(list_file, model)
        extra_converter_list = []
        extra_netrun_list = []
        args = {
            'framework':
            '{} {}'.format(self.framework,
                           (self.framework_version if self.framework_version else '')),
            'engine_path':
            self.engine_path,
            'runtime':
            self.runtime,
            'working_dir':
            os.path.join(self.output_dir),
            'output_dirname':
            out_dir,
            'input_list':
            list_file,
            'deviceId':
            '0',
            'host_device':
            self.host_device,
            'target_device':
            self.target_device,
            'model_path':
            model,
            'model_inputs':
            ''.join([
                ' --input_tensor {} {} {}'.format(item[0], item[1], item[2])
                for item in input_tensors
            ]),
            'model_outputs':
            ''.join([' --output_tensor {}'.format(name) for name in orig_model_outputs]),
            'target_architecture':
            self.architecture,
            'precision':
            self.precision,
            'extra_converter_args':
            self.extra_converter_args,
            'extra_runtime_args':
            self.extra_runtime_args,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else ''),
            'act_quantizer':
            self.act_quantizer,
            'param_quantizer':
            self.param_quantizer,
            'bias_bitwidth':
            self.bias_bitwidth,
            'weights_bitwidth':
            self.weights_bitwidth,
            'act_bitwidth':
            self.act_bitwidth,
            'quantization_overrides':
            self.quantization_overrides,
            'algorithms':
            self.algorithms,
            'ignore_encodings':
            self.ignore_encodings,
            'per_channel_quantization':
            self.per_channel_quantization,
            'remote_server':
            self.remote_server,
            'remote_username':
            self.remote_username,
            'remote_password':
            self.remote_password,
        }

        inference_args = (' --framework {args[framework]}'
                          ' --engine_path {args[engine_path]}'
                          ' --runtime {args[runtime]}'
                          ' --working_dir {args[working_dir]}'
                          ' --output_dirname {args[output_dirname]}'
                          ' --input_list {args[input_list]}'
                          ' --deviceId {args[deviceId]}'
                          ' --host_device {args[host_device]}'
                          ' --model_path {args[model_path]}'
                          ' --architecture {args[target_architecture]}'
                          ' --precision {args[precision]}'
                          ' --remote_server {args[remote_server]}'
                          ' --remote_username {args[remote_username]}'
                          '{args[model_inputs]}'
                          '{args[model_outputs]}'
                          '{args[verbose]}').format(args=args)

        quantization_args = (' --act_quantizer {args[act_quantizer]}'
                             ' --param_quantizer {args[param_quantizer]}'
                             ' --bias_bitwidth {args[bias_bitwidth]}'
                             ' --weights_bitwidth {args[weights_bitwidth]}'
                             ' --act_bitwidth {args[act_bitwidth]}').format(args=args)

        if self.remote_password:
            inference_args += ' --remote_password ' + self.remote_password

        if self.use_native_input_files:
            inference_args += ' --use_native_input_files'
        if self.use_native_output_files:
            inference_args += ' --use_native_output_files'

        if self.runtime in [
                'aic', 'dsp', 'dspv65', 'dspv66', 'dspv68', 'dspv69', 'dspv73', 'dspv75'
        ]:
            inference_args += ' --offline_prepare'

        if not capture_intermediate_outputs:
            # Used only for cumulative layerwise because it adds output node at current layer also
            if output_tensors is not None:
                args['add_layer_outputs'] = ','.join(['{}'.format(name) for name in output_tensors])
                inference_args += ' --add_layer_outputs {args[add_layer_outputs]}'.format(args=args)
            inference_args += ' --debug_mode_off'

        if self.precision in ['int8', 'fp16'] and self.compiler_config:
            inference_args += ' --compiler_config ' + self.compiler_config

        if self.quantization_overrides:
            quantization_args += ' --quantization_overrides ' + self.quantization_overrides
        if self.algorithms: quantization_args += ' --algorithms ' + self.algorithms
        if self.ignore_encodings:
            quantization_args += ' --ignore_encodings {args[ignore_encodings]}'.format(args=args)
        if self.per_channel_quantization:
            quantization_args += ' --per_channel_quantization {args[per_channel_quantization]}'.format(
                args=args)
        if self.precision == 'int8': inference_args += quantization_args
        if self.extra_converter_args:
            extra_converter_list = ['--extra_converter_args', self.extra_converter_args]
        if self.extra_runtime_args:
            extra_netrun_list = ['--extra_runtime_args', self.extra_runtime_args]

        # Execute model on QNN
        self.logger.info("Running exec_inference_engine with parameters: {}".format(
            inference_args + ' ' + ' '.join(extra_converter_list + extra_netrun_list)))
        try:
            #TODO: Need to use Python API call once available
            exec_inference_engine(self.engine, inference_args.split() + extra_converter_list + extra_netrun_list, self.logger)
        except Exception as e:
            return 1, str(e)
        return 0, ''

    def initial_run(self, model):
        """This method checks mismatch in original outputs of model with
        reference framwork outputs.

        Also does tensormapping to map qnn and reference tensor names
        """
        self.logger.debug('Initial run to check final output mismatch is started')
        self.logger.info('Started initial run to compare original outputs of model')
        s_utility = su.getInstance()
        percent_match_origin = OrderedDict()
        overall_comp_list = []
        comparators_list = s_utility.getComparator()
        layer_output_comp_map = {}
        final_outputs_mismatched = False
        output_tensors = self.model_handler.framework_instance.get_output_layers(names_only=True)
        ret_inference_engine, std_out = self.execute_on_qnn(model, self.input_list_file,
                                                            output_tensors, output_tensors,
                                                            out_dir='initial_run')

        if ret_inference_engine != 0:
            skip_compare = True
            match, percent_match = False, 0.0
            raise SnoopingError(std_out)

        temp_model_outputs = self.model_handler.framework_instance.get_output_layers()
        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)

        # comparision of qnn and reference outputs.
        for elem in temp_model_outputs:
            out_name = elem[0]
            out_dtype = elem[1]
            out_optype = elem[2]
            org_inf_raw, org_rt_raw = files_to_compare(self.framework_results, self.output_dir,
                                                       out_name, out_dtype, self.logger,
                                                       out_folder='initial_run')

            if out_name in layer_output_comp_map:
                comp_list = layer_output_comp_map[out_name]
            else:
                comp_list = comparators_list.copy()
            for _, comp in enumerate(comp_list):
                try:
                    match_origin, percent_origin = comp.verify(out_optype, None, [org_rt_raw],
                                                               [org_inf_raw], False)
                except Exception as e:
                    match_origin, percent_origin, _ = False, 0.0, ''
                    self.logger.info('Skipping comparision for node : {}, and marking 0.0% '
                                     'match'.format(out_name))
                # store percentage match for each user supplied comparator
                comp_name = comp.V_NAME
                percent_match_origin[comp_name] = round(percent_origin, 4)

                if comp_name not in overall_comp_list:
                    overall_comp_list.append(comp_name)

            if out_name in original_output_names:
                if not match_origin:
                    final_outputs_mismatched = True

        if final_outputs_mismatched:

            ret_inference_engine = self.execute_on_qnn(model, self.input_list_file, output_tensors,
                                                       output_tensors, out_dir='mapping_run',
                                                       capture_intermediate_outputs=True)
            get_mapping_arg = Namespace(None, framework=self.framework,
                                        version=self.framework_version, model_path=model,
                                        output_dir=self.work_dir, engine=self.engine,
                                        golden_dir_for_mapping=self.framework_results)
            self.logger.info('Creating tensor mapping as final outputs mismatched')
            tensor_mapping = TensorMapping(get_mapping_arg, self.logger)
            self.logger.debug('Completed initial run to check final output mismatch')
            return tensor_mapping

        else:
            self.logger.info('No mismatches seen in final outputs of model. Stopping debugger')
            exit(1)
