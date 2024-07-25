# =============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np
import os
import pandas as pd
import shutil
import signal
import sys
from collections import OrderedDict
import traceback
import logging

from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su
from qti.aisw.accuracy_debugger.lib.snooping.nd_layerwise_snooper import LayerStatus, LayerwiseSnooping, files_to_compare
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import show_progress
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, transpose_to_nhwc
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_execute_inference_engine import exec_inference_engine
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name

# used to mark layerwise snooping started.
snoop_stop = False
pgq_data = {}


def signal_handler(sig, frame):
    global snoop_stop
    logging.info('Stopping snooping on user request.')
    if snoop_stop:
        logging.info('Waiting for current layer to complete.')
        snoop_stop = True
    else:
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


class CumulativeLayerwiseSnooping(LayerwiseSnooping):
    """Class that runs cumulative layerwise snooping."""

    def __init__(self, args, logger, c_logger=None):
        super(CumulativeLayerwiseSnooping, self).__init__(args, logger, c_logger)

    def add_output_node_to_current_layer(self, model, cur_layer_out_name, original_output_names,
                                         out_count):
        """
        This method returns the intermediate outputs to be added to the model
        Args:
            model                   : path to model
            cur_layer_out_name      : layer at which intermediate output to be added
            original_output_names   : list containing original outputs
            out_count               : count of intermediate outputs added
        Returns:
            status                  : execution status of method
            out_count               : updated count of intermediate outputs added
            transformed_model_path  : modified model path
            output_names            : names of output nodes to the model
        """
        s_utility = su.getInstance()

        curr_model_output_names = \
            self.model_handler.framework_instance.get_output_layers(names_only=True)

        out_count += 1

        transformed_model_path = os.path.join(
            self.output_dir,
            'transformed' + self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)
        try:
            if os.path.exists(model):
                shutil.copy(model, transformed_model_path)
        except:
            self.logger.info('Skipped making a copy of model')

        output_names = curr_model_output_names if cur_layer_out_name in curr_model_output_names else curr_model_output_names + [
            cur_layer_out_name
        ]
        return True, out_count, transformed_model_path, output_names, curr_model_output_names

    def run(self, model=None, profile_info=None):
        """This method contains the sequence of debugger for
        CumulativeLayerwiseSnooping."""
        model = self.model if not model else model
        ret_status = True
        s_utility = su.getInstance(self.args)
        self.model_traverser = s_utility.setModelTraverserInstance(self.logger, self.args)
        self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args)
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
        layer_orig_perc_map = {}
        comparators_list = s_utility.getComparator()
        layer_output_comp_map = {}
        layerwise_report_path = self.output_dir + '/cumulative_layerwise.csv'

        # partition the model from user supplied -start-from-layer-output
        # the input list file is updated accordingly.
        status, model, list_file = self.partition_initial_model(model)
        self.input_list_file = list_file
        if status is False:
            return status
        # np_fp16_list, np_file = s_utility.getNodePrecisionFP16()
        self.set_profile_info(self.logger, self.args, model)

        total_layers = self.model_traverser.get_layer_count()
        skip_count = 0

        original_outputs = self.model_handler.framework_instance.get_output_layers()
        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)
        profile_info = self.profile_info
        tensor_mapping = self.initial_run(model)
        # get list of nodes from qnn inference tensor mapping
        valid_nodes = list(read_json(tensor_mapping).values())
        temp_model = model
        self.logger.info('Started cumulative layerwise snooping')

        # Snooping loop
        count = 0
        out_count = 0
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

            # Populate layer details with default and known values.
            layer_perc_map[s_cur_layer_out_name] = '-'
            layer_orig_perc_map[s_cur_layer_out_name] = '-'
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
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)
                layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_SKIPPED
                continue

            # Check if snooping needs to be stopped
            if snoop_stop:
                break

            # add intermediate outputs to model
            status, out_count, temp_model, output_names, orig_model_ouputs = self.add_output_node_to_current_layer(
                temp_model, cur_layer_out_name, original_output_names, out_count)

            # Show progress
            count += 1
            if out_count == 0:
                skip_count += 1
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)
                continue
            else:
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)

            self.logger.debug('Debugging layer ' + layer_name)

            run_model = temp_model

            output_tensors = output_names

            # Execute model on QNN
            ret_inference_engine, std_out = self.execute_on_qnn(run_model, list_file,
                                                                output_tensors, orig_model_ouputs,
                                                                s_cur_layer_out_name)

            if ret_inference_engine != 0:
                self.logger.warning(
                    f'inference-engine failed for the layer -> {s_cur_layer_out_name}, skipping verification'
                )
                skip_compare = True
                match, percent_match = False, 0.0
                conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map = self.handle_qnn_run_failure(
                    std_out, cur_layer_out_name, layer_status_map, conv_fail_nodes, lib_fail_nodes,
                    cntx_fail_nodes, exec_fail_nodes)

            # Compare current layer outputs
            if not skip_compare:
                d_type = [layer_dtype_map[s_cur_layer_out_name]]
                inf_raw, rt_raw = files_to_compare(self.framework_results, self.output_dir,
                                                   cur_layer_out_name, d_type[0], self.logger)
                shape = layer_shape_map[s_cur_layer_out_name]
                if self.framework == Framework.onnx.value:
                    rt_raw = transpose_to_nhwc(rt_raw, shape)
                info_origin = {}
                if (inf_raw is not None) and (rt_raw is not None):
                    percent_match = {}
                    if s_cur_layer_out_name in layer_output_comp_map:
                        comp_list = layer_output_comp_map[s_cur_layer_out_name]
                    else:
                        comp_list = comparators_list.copy()

                    for idx, comp in enumerate(comp_list):
                        try:
                            match = True
                            match_info = '-'
                            match, percent = comp.verify(layer_type, None, [rt_raw], [inf_raw],
                                                         False)

                        except Exception as e:
                            match, percent, match_info = False, 0.0, ''
                            compare_skip_nodes.append(cur_layer_out_name)
                            self.logger.debug(
                                'Skipping comparision for node : {}, and marking 0.0% match'.format(
                                    cur_layer_out_name))
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
                    match, percent_match = False, 0.0

                # comparision qnn and reference forms of original outputs
                original_outnode_match = {}
                for elem in original_outputs:
                    original_output_name = elem[0]
                    d_type = elem[1]
                    olayer_type = elem[2]
                    org_inf_raw, org_rt_raw = files_to_compare(self.framework_results,
                                                               self.output_dir,
                                                               original_output_name, d_type,
                                                               self.logger, s_cur_layer_out_name)
                    s_original_output_name = santize_node_name(original_output_name)
                    shape = profile_info[s_original_output_name][1]
                    if self.framework == Framework.onnx.value:
                        org_rt_raw = transpose_to_nhwc(org_rt_raw, shape)
                    percent_match_origin = {}
                    if s_original_output_name in layer_output_comp_map:
                        comp_list = layer_output_comp_map[s_original_output_name]
                    else:
                        comp_list = comparators_list.copy()
                    for idx, comp in enumerate(comp_list):
                        match_origin = True
                        try:
                            match_origin, percent_origin = comp.verify(
                                olayer_type, None, [org_rt_raw], [org_inf_raw], False)

                        except Exception as e:
                            match_origin, percent_origin, _ = False, 0.0, ''
                            self.logger.debug(
                                'Skipping comparision for node : {}, and marking 0.0% '
                                'match'.format(s_original_output_name))
                        # store percentage match for each user supplied comparator
                        comp_name = comp.V_NAME
                        percent_match_origin[comp_name] = round(percent_origin, 4)
                        # maintain a list of over all comparators used in snooping
                        if comp_name not in overall_comp_list:
                            overall_comp_list.append(comp_name)
                    original_outnode_match[s_original_output_name] = percent_match_origin

                self.logger.info(
                    'Debug Layer {}, output {} match percent {}, original outputs {}'.format(
                        layer_name, s_cur_layer_out_name, percent_match,
                        str(original_outnode_match)))

                layer_perc_map[s_cur_layer_out_name] = percent_match
                layer_compare_info_map[s_cur_layer_out_name] = "\n".join(list(info_origin.values()))
                layer_orig_perc_map[s_cur_layer_out_name] = original_outnode_match

            if match:
                # to avoid adding multiple intermediate outputs
                # needed during top-partion where extracted model gets replaced
                temp_model = model

            else:
                # Extract model if mismatch
                if cur_layer_out_name in original_output_names:
                    continue

                # Extract sub model. Continue to next layer if extraction fails.
                while (True):
                    try:
                        ret_status, extracted_model_path, _, new_g_inputs = \
                            self.initiate_model_extraction(temp_model, cur_layer_out_name,set_model = False)
                        if ret_status:
                            # update status as partition success if no error
                            layer_status_map[
                                s_cur_layer_out_name] += LayerStatus.LAYER_STATUS_PARTITION
                    except Exception as e:
                        ret_status = False
                        traceback.print_exc()
                        self.logger.error('Extraction error {}'.format(e))
                    if not ret_status:
                        extract_fail_nodes.append(cur_layer_out_name)
                        self.logger.error('Extraction failed at node {}'.format(cur_layer_out_name))
                        if s_cur_layer_out_name in layer_status_map:
                            layer_status_map[
                                s_cur_layer_out_name] += ',' + LayerStatus.LAYER_STATUS_PARTITION_ERR
                        else:
                            layer_status_map[
                                s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_PARTITION_ERR

                        # Fetch next layer for extraction
                        while (True):
                            (status, layer_name, s_cur_layer_out_name,
                             layer_type) = self.model_traverser.get_next_layer()
                            if cur_layer_out_name in original_output_names:
                                continue
                            elif status == 2:  # Skipped layer
                                count += 1
                                skip_count += 1
                                prog_info = str(count) + '/' + str(
                                    total_layers) + ', skipped:' + str(skip_count)
                                show_progress(total_layers, count, prog_info)
                                continue
                            else:
                                break

                        if status == 1:
                            snoop_stop = True
                            ret_status = True
                            # Reached end.
                            break

                    else:
                        break

                # Reached end layer
                if snoop_stop:
                    break
                list_file = self.update_list_file(new_g_inputs)

                # Use this extracted model for debugging.
                temp_model = extracted_model_path
                model = extracted_model_path
                self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args,
                                                                    temp_model)

            #  Exit if end layer is provided
            if s_utility.getEndLayer() == cur_layer_out_name:
                skip_count += (total_layers - count)
                count = total_layers
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)
                break

        print("============== Cumulative Layerwise Debug Results ==============")
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
                         perc_compwise_list + [layer_orig_perc_map] + [layer_compare_info_map])
        results_dict = {}
        for k in layer_perc_map.keys():
            results_dict[k] = tuple(d[k] for d in results_dicts)
        if len(results_dict) == 0:
            logging.info('No layers has been debugged.')
            return ret_status

        df = pd.DataFrame.from_dict(results_dict, orient='index')
        labels = ['Status', 'Layer Type', 'Shape', 'Activations (Min,Max,Median)'
                  ] + overall_comp_list + ['Orig Outputs', 'Info']
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
        self.logger.debug('Cumulative Layerwise snooping completed successfully')
        return ret_status
