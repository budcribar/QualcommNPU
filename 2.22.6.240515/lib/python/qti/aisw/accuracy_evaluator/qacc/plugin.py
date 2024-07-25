##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import builtins
from collections import OrderedDict
import copy
from functools import partial
import glob
import importlib
import inspect
import os
import shutil
import sys
from joblib import Parallel, delayed
from pathlib import Path

import numpy as np
import re
from typing import Any, DefaultDict, List, Optional, Union
from abc import abstractmethod
import pickle

batched_inputs_idx_parser = re.compile("-(\d+)_(\d+)-")
single_inputs_idx_parser = re.compile("-(\d+)-\d+")
#TODO: Update with Current Ref platform infer out
ref_single_outputs_idx_parser = re.compile("\w+_(\d+)\.")

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.configuration
import qti.aisw.accuracy_evaluator.qacc.reader as rd
import qti.aisw.accuracy_evaluator.qacc.writer as wr
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc._data_transport import PipelineItem
# Max number of input images.
# TODO - replace with BATCH_SIZE when batch mode pipeline is implemented.


def pl_print(*args, **kwargs):
    # TODO - change to debug
    qacc_file_logger.info(*args)


def _parse_range(index_str):
    if len(index_str) == 0:
        return []
    nums = index_str.split("-")
    assert len(nums) <= 2, 'Invalid range in calibration file '
    start = int(nums[0])
    end = int(nums[-1]) + 1
    return range(start, end)


# Dump all prints to qacc log file.
builtins.print = pl_print


class Plugin:
    """QACC Plugin class."""

    def __init__(self, plugin_config):
        self._plugin_config = plugin_config


class PluginManager:
    """QACC Plugin Manager class This class is responsible for running a set of
    plugins during preprocessing, postprocessing and metrics calculation."""
    registered_plugins = {}
    registered_metric_plugins = {}
    registered_dataset_plugins = {}

    def __init__(self, dataset=None, orig_dataset=None):
        # This dataset object is needed for inputlist file during preprocessing and
        # annotation file during post processing
        # dataset is None in case of dataset plugins as dataset object is
        # instantiated post dataset transformation execution.
        self.dataset = dataset
        self.orig_dataset = orig_dataset
        self.reader = rd.Reader()
        self.writer = wr.Writer()

    @classmethod
    def findAvailablePlugins(cls, use_memory_plugins=False):
        qacc_file_logger.info('Searching for plugins.')
        # find all plugin classes in the default plugin location and the given custom path, if it exists.
        if use_memory_plugins:
            plugin_path = str(Path(__file__).resolve().parents[1] / 'plugins/memory_plugins')
        else:
            plugin_path = str(Path(__file__).resolve().parents[1] / 'plugins/filebased_plugins')
        all_plugin_paths = [plugin_path]
        custom_plugin_path = os.getenv('CUSTOM_PLUGIN_PATH')
        if custom_plugin_path:
            if os.path.exists(custom_plugin_path):
                all_plugin_paths.append(custom_plugin_path)
            else:
                qacc_file_logger.warning(
                    f"Custom plugin path {custom_plugin_path} does not exist. No custom plugins would be loaded"
                )

        for path in all_plugin_paths:
            abs_path = os.path.abspath(path)
            sys.path.append(abs_path)
            # add to sys path
            dirs = []
            for dir in os.walk(abs_path):
                if dir[0] not in dirs:
                    sys.path.append(dir[0])
                    dirs.append(dir[0])

            # search plugins recursively
            paths = glob.glob(path + '/**/' + '*.py', recursive=True)
            files = [file.rsplit('/', 1)[-1] for file in paths]
            for idx, file in enumerate(files):
                if file.endswith(".py"):
                    if file.startswith('__'):
                        continue
                    file = os.path.splitext(file)[0]
                    _plugin = importlib.import_module(file)
                    classes = inspect.getmembers(_plugin, predicate=inspect.isclass)
                    for cl in classes:
                        if cl[1].__module__ == file:
                            class_hier = inspect.getmro(cl[1])
                            for class_h in class_hier:
                                if class_h.__name__ == 'qacc_plugin':
                                    PluginManager.registered_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered plugin :' + cl[0])
                                    break
                                elif class_h.__name__ == 'qacc_metric':
                                    PluginManager.registered_metric_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered metric plugin :' + cl[0])
                                    break
                                elif class_h.__name__ == 'qacc_dataset':
                                    PluginManager.registered_dataset_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered dataset plugin :' + cl[0])
                                    break
                                elif class_h.__name__ == 'qacc_memory_plugin':
                                    PluginManager.registered_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered plugin :' + cl[0])
                                    break
                                elif class_h.__name__ == 'qacc_memory_metric':
                                    PluginManager.registered_metric_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered metric plugin :' + cl[0])
                                    break

    @classmethod
    def get_memory_plugin_objects(cls, plugin_list, extra_params={}, dataset=None):
        """Get Plugin objects from the plugin list.

        Extra params contains parameters exposed by the infrastructure for the plugin developers to use
        Note: dataset is passed only for metric plugins. Processor plugins do not require access to dataset
        return a list of plugin objects
        Args:
            plugin_list: List containing Plugin configurations to be created.
            extra_params: Parameters supplied from infrastructure which might be used by plugin during processing information.
            dataset: Dataset object containing information on the source data to be processed and annotation information.
        Returns:
            status: True if success otherwise False
            plugin_objs: List containing the plugin objects
        """
        status = True
        plugin_objs = []
        if plugin_list:
            for plugin in plugin_list:
                plugin_params = OrderedDict()  # Prepare a updated list of parameters
                if plugin._params:
                    plugin_params.update(plugin._params)  # Collect parameters configured

                # Add additional extra_params kwargs and move it to the end (kwargs like behavior)
                plugin_params.update({'extra_params': extra_params})
                plugin_params.update({'_name': plugin._name})
                plugin_params.move_to_end('extra_params')
                try:
                    # Modify author implementation of processor plugin to add extra_params
                    if plugin._name in PluginManager.registered_plugins:
                        plugin_class = PluginManager.registered_plugins[plugin._name]
                        updated_plugin_class = type(plugin._name,
                                                    (plugin_class, qacc_memory_plugin),
                                                    plugin_params)
                        if plugin._params:
                            plugin_objs.append(updated_plugin_class(**plugin._params))
                        else:
                            plugin_objs.append(updated_plugin_class())
                    # Modify author implementation of metric plugin to add extra_params
                    elif plugin._name in PluginManager.registered_metric_plugins:
                        if dataset is None:
                            logger.error(
                                f'Error while creating metric plugin: {plugin._name}. dataset not supplied'
                            )
                        plugin_class = PluginManager.registered_metric_plugins[plugin._name]
                        updated_plugin_class = type(plugin._name,
                                                    (plugin_class, qacc_memory_metric),
                                                    plugin_params)
                        if plugin._params and dataset:
                            plugin_objs.append(
                                updated_plugin_class(dataset=dataset, **plugin._params))
                        else:
                            plugin_objs.append(updated_plugin_class(dataset=dataset))
                    else:
                        status = False
                        raise RuntimeError(f"Plugin not registered: {plugin._name}")
                except Exception as e:
                    status = False
                    raise RuntimeError(f"Failed to create plugin {plugin._name}. Reason :{e}")

        return status, plugin_objs

    @classmethod
    def execute_memory_preprocessors(cls, plugin_objs, pipeline_items, input_info, output_dir,
                                     batchsize=1):
        """Executes all the plugins in the entire plugin_objs chain to
        preprocess calibration/input dataset.

        Note: pipeline_items should contain batchsize number of items to get batched preprocessed output.
        Args:
            plugin_objs: a set of plugins to be performed in a sequence
            pipeline_items: List of pipeline_item containing the data to be processed.
            input_info: Input node name, shape and dtype information from the config.
            output_dir: used as the target path to store intermediate outputs
            batchsize: Number of samples to be fed as a batch. Default is 1
        Returns:
            status: 0 if success otherwise 1
            processed_file_path: path to the file list containing paths of preprocessed input raw files.
        """
        ret_status = 0
        processed_file_path = None
        try:
            for idx, pipeline_item in enumerate(pipeline_items):
                # Plugins always act on a single input i.e. batchsize=1
                for plugin_obj in plugin_objs:
                    pipeline_item = plugin_obj(pipeline_item)
                pipeline_items[idx] = pipeline_item
        except Exception as e:
            qacc_logger.error(f'Failed to execute Plugin {plugin_obj._name}. Reason: {e}')
            return 1, None
        else:
            # Fuse the inputs based on batch dim available as part of input_info into single fused input
            pipeline_item = pack_inputs(pipeline_items, input_info)

        # Use the serialization logic present in the last plugin configured to dump the data to disk
        if not ret_status:
            ret_status, processed_file_path = plugin_objs[-1].to_disk(pipeline_item,
                                                                      output_dir=output_dir)
        # Return the  processed_file_path  to write the final processed-outlist file.
        return ret_status, processed_file_path

    def execute_dataset_transformations(self, dataset_config, out_dir):
        plugin_list = dataset_config._transformations._plugin_config_list

        # create dataset directory
        if len(plugin_list) > 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        # execute plugins
        for pl_idx, plugin in enumerate(plugin_list):
            status, dataset_config = self.execute_dataset_plugin(plugin, dataset_config, out_dir)
            if status:
                qacc_file_logger.error('Error while executing dataset transformation')
        return dataset_config

    def execute_metrics(self, metrics_plugin_config, output_dir, results_str_list, results_dict):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for pl_cfg in metrics_plugin_config:
            plugin = Plugin(pl_cfg)
            m_in = MetricInputInfo(plugin._plugin_config,
                                   result_file=self.dataset.get_input_list_file(),
                                   orig_inp_file=self.orig_dataset.get_orig_input_list_file(),
                                   gt_file=self.orig_dataset.get_dataset_annotation_file(),
                                   out_dir=output_dir)

            # (self, plugin_config, paths, result_file, gt_file, out_dir):
            m_out = MetricResult(out_dir=output_dir)
            self.execute_metric_plugin(plugin, m_in, m_out)
            if m_out.status != 0:
                qacc_file_logger.error('Plugin ({}) failed to execute!!'.format(
                    plugin._plugin_config._name))
                return 1

            results_str_list.append(m_out.result_str)
            results_dict.update(m_out.result)
        return 0

    def get_chained_transformations(self, transformations):
        """
        Returns a list of thread safe transformation list
        eg:
            plugin1 (mem) -> plugin2 (path) -> plugin3 (dir) -> plugin4 (mem)
            should return a transformation list
            chained_transformations = [transformation1, transformation2, transformation3]
            transformation1: [Plugin1, Plugin2]
            transformation2: [Plugin3]
            transformation3: [Plugin4]
        Returns:
            chained_transformations: list of transformations
        """
        chained_transformations = []

        # flag checked before creating a new transformations instance
        is_new_chain = True

        for idx, plugin in enumerate(transformations.plugin_list):
            # create new transformation chain if this is start of
            # the transformations chain
            if is_new_chain is True:
                transformations = Transformations(max_input_len=self.dataset.get_total_entries())
                is_new_chain = False

            # input_info is a tuple (type, dtype, format)
            # check if the type is dir
            if plugin._plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR:
                # check if any earlier transformation chain is not empty
                if not transformations.is_empty():
                    # save old transformation
                    chained_transformations.append(copy.deepcopy(transformations))

                # add new dir based transformation
                # is_dir fields is marked as true to distinguish this transformation
                # as a directory based transformation
                chained_transformations.append(
                    copy.deepcopy(
                        Transformations(plugin_config_list=[plugin._plugin_config], is_dir=True,
                                        max_input_len=self.dataset.get_total_entries())))

                # delete existing transformations as it is already saved
                del transformations

                # create a new transformation in next iteration
                is_new_chain = True
            else:
                transformations.add_plugin(plugin)

        # add last iterated transformations if not empty
        # checking in locals to confirm var transformations
        # was not deleted previously
        if ('transformations' in locals()) and \
                (not transformations.is_empty()):
            # save old transformation
            chained_transformations.append(copy.deepcopy(transformations))

        return chained_transformations

    def execute_transformations(self, transformations, output_dir, batch_offset=0,
                                squash_results=False, input_names=None):
        """Executes all the plugins in the entire transformation chain The
        execute method calls chain transformation method To split the entire
        transformation into smaller transformation chains which can run in
        parallel.

        Args:
            transformations: a set of plugins to be performed in a sequence
            output_dir: used as the target path to store intermediate outputs
        """
        ret_status = 0
        # create output path if not existing.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get thread safe changed transformations
        chained_transformations = self.get_chained_transformations(transformations)

        last_transform_outputs = None

        # iterate the chained transformations and execute them
        for tr_idx, transformations in enumerate(chained_transformations):
            # Execute the transformations based on the is_dir flag of the transformations
            # is_dir flag true shows that the transformations contains single directory based plugin
            # is_dir flag false shows the transformations contains non directory based plugins
            # which can be executed in parallel

            if transformations.is_dir is True:

                # dir_input is in a form of nested paths
                # e.g. [[path1.1], [path2.1]] or
                # [[path1.1, path1.2, path1.3], [path2.1, path2.2, path2.3] … and so on]
                if tr_idx == 0:
                    # If this is the first transformation, the inputs come from dataset.
                    dir_input_paths = [input_path for input_path in self.dataset.input_generator()]
                else:
                    # if not the first transformation, use the output paths from last
                    # transformation.
                    dir_input_paths = last_transform_outputs

                ret_status = self.execute_dir_plugin(transformations, dir_input_paths, output_dir)

                # save the last transformation output paths
                last_transform_outputs = transformations.output_path_list

            else:
                # create multiple threads to execute the plugins in parallel
                # The number of threads is dependents on the available cores in the machine.
                # the n_jobs=-1 tries to create maximum number of threads based on available cores
                # TODO set batch_offset for batch mode

                if last_transform_outputs:
                    # Transformation followed by directory plugin in pipeline.
                    generator = last_transform_outputs
                else:
                    # First transformation in the pipeline stage. Use dataset.
                    generator = self.dataset.input_generator()

                Parallel(n_jobs=-1, verbose=0, prefer="threads") \
                    (delayed(self.execute_transformation_in_parallel)(transformations,
                                                                      input_path,
                                                                      output_dir,
                                                                      input_index + batch_offset)
                     # input_path is a path or a set of paths
                     # e.g. [path1] or [path1, path2, ... pathN]
                     for input_index, input_path in enumerate(generator))

                if transformations.run_status:
                    qacc_file_logger.error('Plugin Transformation failed to run.')
                    ret_status = 1
                    break

                # save the last transformation output paths
                last_transform_outputs = transformations.output_path_list

        # remove any None entries from list
        transformations.output_path_list = [i for i in transformations.output_path_list if i]
        # All transformations have executed. Now create the output list file.
        with open(os.path.join(output_dir, 'processed-outputs.txt'), 'w') as fl:
            for inps in transformations.output_path_list:
                if squash_results:
                    # Append the contents of each file in outputs list file.
                    # Only enabled after postprocessing for merging text file contents of outputs.
                    assert len(inps) == 1, 'Multiple input paths not supported for squashing. ' \
                                           'Disable squashing using squash_results=False in ' \
                                           'postprocessing'
                    with open(inps[0], mode='r') as f:
                        fl.write(f.read())

                else:
                    for i, inp in enumerate(inps):
                        if inp is None:
                            qacc_file_logger.error('Null input found at index {} while creating'
                                                   ' processed-outputs.txt.\n Record {}'.format(
                                                       i, inps))
                            raise RuntimeError('Some inputs were not processed!')
                        if i:
                            fl.write(',' + inp)
                        else:
                            fl.write(inp)
                    fl.write('\n')

        #Write the processed inputs in a different format.
        if input_names is not None:
            with open(os.path.join(output_dir, 'qnn-processed-outputs.txt'), 'w') as fl:
                for inps in transformations.output_path_list:
                    for i, inp in enumerate(inps):
                        if inp is None:
                            qacc_file_logger.error('Null input found at index {} while creating'
                                                   ' qnn-processed-outputs.txt.\n Record {}'.format(
                                                       i, inps))
                            raise RuntimeError('Some inputs were not processed!')
                        if i:
                            fl.write(f" {input_names[i]}:={inp}")
                        else:
                            fl.write(f"{input_names[i]}:={inp}")
                    fl.write('\n')

        return ret_status

    def execute_transformation_in_parallel(self, transformations, input_path_list, output_dir,
                                           input_index):
        """Executes a chain of plugins.

        This method runs in a joblib thread.
        """
        # maintain one output info for number of inputs per record
        idx_plugin_inp_info = [
            PluginInputInfo(path=path, out_dir=output_dir) for path in input_path_list
        ]
        idx_plugin_out_info = [PluginOutputInfo() for _ in range(len(input_path_list))]

        # get original file names for automatically creating output file names by plugins.
        fnames = []
        for input_name in input_path_list:
            _, fname = os.path.split(input_name)
            fnames.append(os.path.splitext(fname)[0])

        for plugin in transformations.plugin_list:

            # execute function based plugin

            # find indexes of input record which this plugin will be passed.
            # default - all the inputs in the input record.
            if plugin._plugin_config._indexes:
                indexes = plugin._plugin_config._indexes
            else:
                indexes = list(range(len(idx_plugin_inp_info)))

            # evaluate inp/out info for mem/path plugin for each index.
            # Note : Only the required indexes are updated. Older objects are preserved.
            for index in indexes:
                # creating plugin input info to be passed to execute plugin
                idx_plugin_inp_info[index] = self.get_plugin_input_info(
                    plugin, idx_plugin_inp_info[index], idx_plugin_out_info[index], input_index)

                idx_plugin_out_info[index] = self.get_plugin_output_info(
                    plugin, idx_plugin_out_info[index], fnames[index], output_dir)
            self.execute_function_plugin(plugin, idx_plugin_inp_info, idx_plugin_out_info)

            deleted_slots = []
            for index in indexes:
                if idx_plugin_out_info[index].status == qcc.STATUS_ERROR:
                    qacc_file_logger.error('Plugin ({}) failed to execute!!'.format(
                        plugin._plugin_config._name))
                    transformations.run_status = 1
                    return
                elif idx_plugin_out_info[index].status == qcc.STATUS_REMOVE:
                    deleted_slots.append(index)

            # If current plugin has disabled some slots, remove them from the chain.
            indexes_to_delete = sorted(deleted_slots, reverse=True)

            for index in indexes_to_delete:
                del idx_plugin_inp_info[index]
                del idx_plugin_out_info[index]
                del fnames[index]

        # Persists in memory data at end of transformation.
        out_paths = transformations.output_path_list[input_index]
        if out_paths is None:
            out_paths = [None] * len(idx_plugin_out_info)
            transformations.output_path_list[input_index] = out_paths

        for idx, pout in enumerate(idx_plugin_out_info):
            if pout.plugin_config:
                if pout.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM:
                    self.writer.write(output_path=pout.get_output_path(), mem_obj=pout.mem_obj,
                                      dtype=pout.get_output_dtype(),
                                      write_format=pout.get_output_format())
                out_paths[idx] = pout.get_output_path()
            else:
                # This input index has not been processed by any of the plugins.
                # Copy the input as it is to the output dir.
                qacc_file_logger.info('Copied unprocessed input {} to output dir {}'.format(
                    input_path_list[idx], output_dir))
                shutil.copy(input_path_list[idx], output_dir)
                out_paths[idx] = input_path_list[idx]

    def get_plugin_output_info(self, plugin, cur_pout, fname, output_dir):

        # point to the new plugin
        cur_pout.plugin_config = plugin._plugin_config
        cur_pout.outdir = output_dir
        cur_pout.status = 1  # Plugins must set this to 0 to mark successful execution.
        cur_pout.fname = fname  # used to generate the same name output file.

        return cur_pout

    def get_plugin_input_info(self, plugin, cur_pinp, last_pout, input_index):

        input_info = plugin._plugin_config._input_info
        cur_pinp.plugin_config = plugin._plugin_config

        last_output_info = None
        if last_pout.plugin_config:
            # This indicates that it is not the first plugin in the chain.
            last_output_info = last_pout.plugin_config._output_info

        if last_output_info is None:
            # first plugin in chain
            if input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM:
                cur_pinp.mem_obj = self.reader.read(input_path=cur_pinp.path,
                                                    dtype=input_info[qcc.IO_DTYPE],
                                                    format=input_info[qcc.IO_FORMAT])
        else:
            # compare prev output format with current input format.
            if last_output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM \
                    and input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH:

                # check formats.
                assert last_output_info[qcc.IO_FORMAT] == input_info[qcc.IO_FORMAT], \
                    'Mem Plugin {} out format must match Path plugin {} input format for the' \
                    'same index'.format(last_pout.plugin_config._name, plugin._plugin_config._name)

                self.writer.write(output_path=last_pout.get_output_path(),
                                  mem_obj=last_pout.mem_obj, dtype=last_output_info[qcc.IO_DTYPE],
                                  write_format=last_output_info[qcc.IO_FORMAT])
                cur_pinp.path = last_pout.get_output_path()

            elif last_output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH and \
                    input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM:

                # check formats.
                assert last_output_info[qcc.IO_FORMAT] == input_info[qcc.IO_FORMAT], \
                    'Path Plugin {} out format must match Mem plugin {} input format for the' \
                    'same index'.format(last_pout.plugin_config._name, plugin._plugin_config._name)

                cur_pinp.mem_obj = self.reader.read(input_path=last_pout.get_output_path(),
                                                    dtype=input_info[qcc.IO_DTYPE],
                                                    format=input_info[qcc.IO_FORMAT])

            elif last_output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH and \
                    input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH:
                cur_pinp.path = last_pout.get_output_path()

            else:
                cur_pinp.mem_obj = last_pout.mem_obj

        # orig dataset input.
        if self.orig_dataset:
            cur_pinp.orig_dataset = self.orig_dataset
            cur_pinp.input_index = input_index

        return cur_pinp

    def execute_dataset_plugin(self, plugin, dataset_config, out_dir):

        def set_dataset_info(dataset_config, dataset_pl_out_info):

            def set_updated(original, updated):
                return updated if updated else original

            dataset_config._inputlist_file = set_updated(dataset_config._inputlist_file,
                                                         dataset_pl_out_info.inputlist_file)
            dataset_config._annotation_file = set_updated(dataset_config._annotation_file,
                                                          dataset_pl_out_info.annotation_file)
            dataset_config._calibration_file = set_updated(dataset_config._calibration_file,
                                                           dataset_pl_out_info.calibration_file)
            dataset_config._calibration_type = set_updated(dataset_config._calibration_type,
                                                           dataset_pl_out_info.calibration_type)

            # check if inputlist path is modified in dataset plugin, if yes, set base path to out_dir
            if dataset_pl_out_info.inputlist_path_modified:
                dataset_config._inputlist_path = dataset_pl_out_info.out_dir
            # check if calibration path is modified in dataset plugin, if yes, set base path to out_dir
            if dataset_pl_out_info.calibration_path_modified:
                dataset_config._calibration_path = dataset_pl_out_info.out_dir

            return dataset_config

        dataset_pl_in_info = DatasetPluginInputInfo(plugin=plugin, dataset_config=dataset_config)
        dataset_pl_out_info = DatasetPluginOutputInfo(out_dir=out_dir)

        # execute plugin
        plugin_name = plugin._name
        if plugin_name not in PluginManager.registered_dataset_plugins:
            raise ce.ConfigurationException('Invalid dataset plugin name {}.'.format(plugin_name))
        plugin_class = PluginManager.registered_dataset_plugins[plugin_name]
        try:
            plugin_class().execute(dataset_pl_in_info, dataset_pl_out_info)
            status = 0
        except Exception as e:
            qacc_logger.error(
                'Plugin: {} failed to execute. See qacc.log for more details.'.format(plugin_name))
            qacc_file_logger.exception('Exception: {}'.format(e))
            status = 1

        # read output info and update dataset config
        if not dataset_pl_out_info.status:
            dataset_config = set_dataset_info(dataset_config, dataset_pl_out_info)
            dataset_config._update_max_inputs()

            if plugin_name == qcc.DATASET_FILTER_PLUGIN_NAME:
                # set the max_inputs and max_calib in pipeline cache
                pipeline_cache = qti.aisw.accuracy_evaluator.qacc.configuration.PipelineCache.getInstance(
                )
                pipeline_cache.set_val(
                    qcc.PIPELINE_MAX_INPUTS,
                    dataset_pl_in_info.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_MAX_INPUTS, -1))
                pipeline_cache.set_val(
                    qcc.PIPELINE_MAX_CALIB,
                    dataset_pl_in_info.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_MAX_CALIB, -1))

        return status, dataset_config

    def execute_dir_plugin(self, transformations, dir_input_paths, output_dir):
        # Execute Dir plugin
        plugin = transformations.plugin_list[0]
        input_info = plugin._plugin_config._input_info
        assert input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR, \
            'Internal error: Expecting dir based plugin'

        # remove any None entries from list
        dir_input_paths = [i for i in dir_input_paths if i]

        # orig dataset
        orig_input_paths = None
        if self.orig_dataset:
            orig_input_paths = self.orig_dataset.get_all_records()

        plugin_input_info = PluginInputInfo(plugin_config=plugin._plugin_config, mem_obj=None,
                                            path=dir_input_paths, orig_dataset=self.orig_dataset,
                                            out_dir=output_dir)
        plugin_output_info = PluginOutputInfo(plugin_config=plugin._plugin_config, status=1,
                                              out_dir=output_dir)

        self.execute_function_plugin(plugin, [plugin_input_info], [plugin_output_info])
        if plugin_output_info.status != 0:
            qacc_file_logger.error('Plugin ({}) failed to execute!!'.format(
                plugin._plugin_config._name))
            return 1

        transformations.output_path_list = plugin_output_info.updated_path
        return 0

    def execute_function_plugin(self, plugin, plugin_input_info_list, plugin_output_info_list):
        plugin_name = plugin._plugin_config._name
        if plugin_name not in PluginManager.registered_plugins:
            raise ce.ConfigurationException('Invalid plugin name {}.'.format(plugin_name))
        plugin_class = PluginManager.registered_plugins[plugin_name]
        try:
            plugin_class().execute(plugin_input_info_list, plugin_output_info_list)
        except Exception as e:
            qacc_logger.error(
                'Plugin: {} failed to execute. See qacc.log for more details.'.format(plugin_name))
            qacc_file_logger.exception('Exception: {}'.format(e))
            plugin_output_info_list[0].status = qcc.STATUS_ERROR

    def execute_metric_plugin(self, plugin, metric_input_info, metric_result):
        plugin_name = plugin._plugin_config._name
        if plugin_name not in PluginManager.registered_metric_plugins:
            raise ce.ConfigurationException('Invalid metric plugin name {}.'.format(plugin_name))
        plugin_class = PluginManager.registered_metric_plugins[plugin_name]
        try:
            plugin_class().execute(metric_input_info, metric_result)
        except Exception as e:
            qacc_logger.error(
                'Plugin: {} failed to execute. See qacc.log for more details.'.format(plugin_name))
            qacc_file_logger.exception('Exception: {}'.format(e))
            metric_result.status = 1


class Transformations:
    """QACC transformation class."""

    def __init__(self, max_input_len, plugin_config_list=None, is_dir=False):
        """
        Args:
            plugin_config_list: list of plugin configs
            is_dir:
                - used to represent if the transformation has dir based plugin
                - it is used in get_chained_transformations method in PluginManager class
                to distinguish between a single directory and other plugin transformations
        """
        if plugin_config_list is None:
            self.plugin_list = []
        else:
            self.plugin_list = self.get_plugin_list_from_plugin_config_list(plugin_config_list)
        self.is_dir = is_dir
        self.run_status = 0
        # path of output files at the end of this transformation.
        # format [[p11,p12],[p21,p22]..], considering two inputs per input record
        self.output_path_list = [None] * max_input_len

    def add_plugin(self, plugin):
        """Adds plugins to the list of transformations."""
        self.plugin_list.append(plugin)

    def is_empty(self):
        return len(self.plugin_list) == 0

    def get_plugin_list_from_plugin_config_list(self, plugin_config_list):
        """Returns a list of plugin objects from the list of plugin config
        objects.

        Args:
            plugin_config_list: list of plugin configs

        Returns:
            plugin_list: list of plugin objects
        """
        plugin_list = []
        for plugin_config in plugin_config_list:
            plugin_list.append(Plugin(plugin_config))
        return plugin_list


class PluginInputInfo:
    """QACC Plugin Input Info class."""

    def __init__(self, plugin_config=None, mem_obj=None, path=None, orig_dataset=None,
                 out_dir=None):
        # Stores the entire plugin configuration
        self.plugin_config = plugin_config

        # In memory object for plugins that work on in memory objects
        # This is useful when multiple plugins are chained together and the output of one plugin
        # is fed as memory object to the next plugin
        self.mem_obj = mem_obj

        # For path based plugin. Also set in case of mem plugins (1D list)
        # For dir based plugins , its a list of list of input record lists. (2D list)
        self.path = path

        # points to original dataset having list of lists of paths for all the inputs.
        # Used mostly by postprocessors which needs original input records.
        self.orig_dataset = orig_dataset

        # input index of the data
        self.input_index = None

        # inference_schema key for pipeline pipeline_cache
        inference_schema = [s for s in out_dir.split('/') if s.startswith('schema')]
        self._inference_schema_name = inference_schema[0] if len(inference_schema) != 0 else None

        # used to store any user object to be transferred to plugins
        # in the chain.
        self.user_obj = None

    def set_user_obj(self, obj):
        self.user_obj = obj

    def get_user_obj(self):
        return self.user_obj

    def get_input_index(self):
        if not self.is_directory_input():
            return self.input_index
        else:
            qacc_file_logger.error('get_input_index() is not supported for directory plugin')
            return None

    def is_memory_input(self):
        return self.plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM

    def is_directory_input(self):
        return self.plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR

    def is_path_input(self):
        return self.plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH

    def get_input_dtype(self):
        return self.plugin_config._input_info[qcc.IO_DTYPE]

    def get_input(self):
        if self.is_memory_input():
            return self.mem_obj
        else:
            return self.path

    def get_orig_path_list(self, num_elements=1, last_batch=qcc.LAST_BATCH_NO_CHANGE):
        """Returns the original input record(s) (as a list of paths) linked to
        the current inference output(s) being processed by the post processor
        based on the num_elements. The last batch is handled using the
        last_batch parameter. Currently last_batch option supports 3 options if
        size of last batch is not equal to num_elements. The options are
        available in qacc constants.

        Example usage from plugins:
            import qacc.constants as qcc
            orig_path_list = pin.get_orig_path_list(num_elements, qcc.LAST_BATCH_TRUNCATE)

        Args:
            num_elements: number of elements to be returned per batch
            last_batch: options to handle the last batch when size of last batch is not equal to
            num_elements
                LAST_BATCH_TRUNCATE: Truncates last batch and returns empty nested list for the
                last batch
                LAST_BATCH_REPEAT_LAST_RECORD: Returns the last batch with last record repeated
                to match the size of num_elements
                LAST_BATCH_NO_CHANGE: Returns the last batch without repeating the last record
        """
        if self.orig_dataset:
            if self.is_memory_input() or self.is_path_input():
                orig_input_paths = self.orig_dataset.get_record(
                    idx=self.input_index, num_records=num_elements if num_elements else 1,
                    last_batch=last_batch)
            elif self.is_directory_input():
                orig_input_paths = self.orig_dataset.get_all_records(group=num_elements,
                                                                     last_batch=last_batch)
            else:
                qacc_file_logger.error('Invalid plugin input info type: {}'.format(
                    self.plugin_config._input_info[qcc.IO_TYPE]))
            return orig_input_paths
        else:
            return None

    def get_env_tag(self):
        return self.plugin_config._env

    def get_param(self, param, default=None):
        if self.plugin_config._params is None:
            return default
        if param in self.plugin_config._params:
            val = self.plugin_config._params[param]
            if isinstance(val, str):
                val = val.strip()
            return val
        return default

    def __str__(self):
        return 'Mem: {}. Plugin Info: {}'.format(self.is_memory_input(), self.plugin_config)

    def get_inference_schema_name(self):
        return self._inference_schema_name.split(
            '_')[1] if self._inference_schema_name is not None else None

    # API to access pipeline_cache
    def read_pipeline_cache_val(self, key):
        pipeline_cache = qti.aisw.accuracy_evaluator.qacc.configuration.PipelineCache.getInstance()
        return pipeline_cache.get_val(key, self._inference_schema_name)


class PluginOutputInfo:
    """QACC Plugin Output Info class."""

    def __init__(self, plugin_config=None, mem_obj=None, status=1, out_dir=None, fname=None):
        # output path for the path and directory based plugins
        self.outdir = out_dir
        self.plugin_config = plugin_config
        self.mem_obj = mem_obj
        self.fname = fname
        self.status = status
        self.extn = '.raw'

        # Path plugins may call setPathOutput() to change the path.
        # Otherwise it may call getOutputPath() to get the desired path.

        # Dir plugins must need to call setDirOutputs() after the processing even if they
        # dont change any file names.

        # This is used only by plugins which update the file paths.
        self.updated_path = None

    def set_status(self, status):
        self.status = status

    def set_mem_output(self, memObj):
        assert (self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM)
        self.mem_obj = memObj

    def set_path_output(self, out_path):
        assert (self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH)
        self.updated_path = out_path

    def set_dir_outputs(self, out_path_list):
        assert (self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR)
        self.updated_path = out_path_list

    def is_memory_output(self):
        return self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM

    def set_output_extn(self, extn):
        self.extn = extn

    def get_output_path(self):
        if self.updated_path:
            return self.updated_path
        return os.path.join(self.outdir, self.fname + self.extn)

    def get_output_dtype(self):
        return self.plugin_config._output_info[qcc.IO_DTYPE]

    def get_output_format(self):
        return self.plugin_config._output_info[qcc.IO_FORMAT]

    def get_out_dir(self):
        return self.outdir


class MetricInputInfo:

    def __init__(self, plugin_config, result_file, orig_inp_file, gt_file, out_dir):
        # Stores the entire plugin configuration
        self.plugin_config = plugin_config

        self.res_file = result_file
        self.orig_inp_file = orig_inp_file
        self.gt_file = gt_file

        # inference_schema key for pipeline pipeline_cache
        inference_schema = [s for s in out_dir.split('/') if s.startswith('schema')]
        self._inference_schema_name = inference_schema[0] if len(inference_schema) != 0 else None

    def get_param(self, param, default=None):
        if self.plugin_config._params is None:
            return default
        if param in self.plugin_config._params:
            return self.plugin_config._params[param]
        return default

    def get_groundtruth(self):
        return self.gt_file

    def get_result_file(self):
        return self.res_file

    def get_orig_inp_file(self):
        return self.orig_inp_file

    def get_inference_schema_name(self):
        return self._inference_schema_name[6:] if self._inference_schema_name is not None else None

    # API to access pipeline_cache
    def read_pipeline_cache_val(self, key):
        pipeline_cache = qti.aisw.accuracy_evaluator.qacc.configuration.PipelineCache.getInstance()
        return pipeline_cache.get_val(key, self._inference_schema_name)


class MetricResult:

    def __init__(self, status=1, out_dir=None):
        self.status = status
        self.result_str = None
        self.result = {}
        self.out_dir = out_dir

    def get_out_dir(self):
        return self.out_dir

    def set_result_str(self, str):
        self.result_str = str

    def set_status(self, status):
        self.status = status

    def set_result(self, result_dict):
        self.result = result_dict
        if self.result_str is None:
            temp_str = ''
            for key, value in self.result.items():
                temp_str += f"{key}: {value}\n"
            self.set_result_str(temp_str)


class DatasetPluginInputInfo:

    def __init__(self, plugin, dataset_config):
        self._plugin_params = plugin._params
        self._dataset_config = dataset_config

    def get_base_path(self):
        return self._dataset_config._path

    def get_inputlist_file(self):
        return self._dataset_config._inputlist_file

    def get_annotation_file(self):
        return self._dataset_config._annotation_file

    def get_calibration_type(self):
        return self._dataset_config._calibration_type

    def get_calibration_file(self):
        return self._dataset_config._calibration_file

    def get_param(self, param, default=None):
        if self._plugin_params is None:
            return default
        if param in self._plugin_params:
            return self._plugin_params[param]
        return default


class DatasetPluginOutputInfo:

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.status = 1  # default status set to failure
        self.inputlist_file = None
        self.annotation_file = None
        self.calibration_type = None
        self.calibration_file = None
        self.inputlist_path_modified = False
        self.calibration_path_modified = False

    def get_out_dir(self):
        return self.out_dir

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def set_inputlist_file(self, inputlist_file):
        self.inputlist_file = inputlist_file

    def set_annotation_file(self, annotation_file):
        self.annotation_file = annotation_file

    def set_calibration_type(self, calibration_type):
        self.calibration_type = calibration_type

    def set_calibration_file(self, calibration_file):
        self.calibration_file = calibration_file

    def set_inputlist_path_modified(self, inputlist_path_modified):
        self.inputlist_path_modified = inputlist_path_modified

    def set_calibration_path_modified(self, calibration_path_modified):
        self.calibration_path_modified = calibration_path_modified


class qacc_plugin:
    """QACC Plugin Interface."""

    def __init__(self):
        pass

    def execute(self, plugin_input_info_list: PluginInputInfo,
                plugin_output_info_list: PluginOutputInfo):
        """Execute method for the plugin class.

        Args:
            plugin_input_info: Takes an list of PluginInputInfo and PluginOutputInfo

        Returns:
        """
        pass


class qacc_metric:

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):
        """Execute method for the metric plugin class.

        Args:
            plugin_input_info: Takes an instance of MetricInputInfo and MetricResult

        Returns:
        """


class qacc_dataset:

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):
        """Execute method for dataset plugin class."""


class qacc_memory_plugin:
    # Note: Plugins are always written to process single input.

    def __init__(self, **kwargs):
        """
        kwargs:  Allows plugin developer to supply arbitrary parameter
        """
        # Needed for automatic attribute setup from config
        self.__dict__.update(kwargs)

    def __call__(self, pipeline_item: PipelineItem, **kwargs):
        # pipeline_items: List containing batchsize number of input samples
        if pipeline_item:
            # For each Sample: Check if it is single input or multi input case.
            data = meta = None
            if not (pipeline_item.data[0], list) == 1:  # if data has only single input
                try:
                    data, meta = self.execute(data=pipeline_item.data, meta=pipeline_item.meta,
                                              **self.__dict__, input_idx=pipeline_item.input_idx,
                                              **kwargs)
                    pipeline_item['data'] = data
                    pipeline_item['meta'] = meta
                    pipeline_item['status'] = True
                except Exception as e:
                    pipeline_item['status'] = False
                    qacc_logger.error(
                        f"Failed to execute plugin : {self.__class__.__name__}. Reason : {e}")
            else:  # Handle case when data has multiple inputs. The processor code needs to use execute logic based on input index
                pipeline_item = self.execute_index(pipeline_item, **kwargs)
        else:
            print(f"Input to the Plugin:{self.__class__.__name__} is empty. input: {pipeline_item}")
        return pipeline_item

    @abstractmethod
    def execute(self, data, meta, input_idx, *args, **kwargs):
        """Execute method for the plugin class.

        Args:
            data  -> [inp_sample_1_node_1,inp_sample_1_node_2]
            meta --> {'meta_key1':val}
            input_idx --> Input identifier (int) (Assigned by infrastructure: 0-N Samples)
            args : Mandatory arguments which can modify the execution logic
            kwargs : optional Keyword based arguments which modifies the execution logic

        Returns: processed data & updated metadata (i.e meta)
        """
        return data, meta

    def execute_index(self, pipeline_item, **kwargs):
        # Default implementation to handle multiple input case in a single pipeline_items i.e multi input evaluation
        # Some postprocessor plugin might need to override this method to act on multiple outputs (if any)
        # i.e input --> preprocessing --> [preproc_out_1, preproc_out_2] --> inference --> [postproc_out_1, postproc_out_2] --> postproc-->metric
        # e.g. if postproc has logic that requires to act upon both postproc_out_1, postproc_out_2, execute_index must be implemented as well.
        meta = pipeline_item.meta
        for processed_idx, processed_data in enumerate(pipeline_item.data):
            # Call execute on each of the input item.
            try:
                data, meta = self.execute(data=input_data, meta=pipeline_item.meta,
                                          input_idx=pipeline_item.input_idx, **self.__dict__,
                                          **kwargs)
                pipeline_item.data[processed_idx] = data
                pipeline_item.meta = meta
                pipeline_item['status'] = True
            except Exception as e:
                pipeline_item['status'] = False
                qacc_logger.error(
                    f"Failed to execute plugin : {self.__class__.__name__}. Reason : {e}")

        return pipeline_item

    def to_disk(self, pipeline_item, output_dir=None):
        # TODO: Might need to update the logic
        work_dir = self.extra_params['work_dir']
        input_names = list(self.extra_params['input_info'].keys())
        output_dir = output_dir if output_dir is not None else os.path.join(
            work_dir, qcc.STAGE_PREPROC)
        # Handling multiple input Scenario:
        file_paths = []
        ret_status = 0  # Return 0 on success
        for inp_node_idx, data in enumerate(pipeline_item['data']):
            if isinstance(data, np.ndarray):
                if len(pipeline_item['input_idx']) > 1:
                    # Create file name using the input index of 1st and last item in the batched list
                    file_name = f"batched-inp-{pipeline_item['input_idx'][0]}_{pipeline_item['input_idx'][-1]}-{inp_node_idx}.raw"
                else:
                    # Create file name using the input index
                    file_name = f"batched-inp-{pipeline_item['input_idx'][0]}-{inp_node_idx}.raw"
                file_path = os.path.join(output_dir, file_name)
                data.tofile(file_path)
                file_paths.append(file_path)
        if len(file_paths) < 1:
            ret_status = 1
        processed_file_path = ','.join(file_paths) + '\n'
        # Return the  processed_file_path  to write the final processed-outlist file.
        return ret_status, processed_file_path

    def from_disk(self, pipeline_item):
        # TODO: Might need to update the logic
        # Note: Below logic doesn't work with random indices.
        # Assumes file naming conventions to read the data from the disk and coupled with the model inputs
        # File name conventions:
        # 1. batchsize>1:  ../batched-inp-{start_idx}_{end_idx}-{input_node_idx}.raw
        # 2. batchsize=1: ../batched-inp-{start_idx}-{input_node_idx}.raw
        # 3. reference_outputs: ../{output_node_name}_{idx}.raw
        # input_node_idx --> Model input node identifier (Applicable for models with multiple input)
        # start_idx, end_idx --> input identifier
        # output_node_name --> Output node name mentioned  in the config
        # Input pipeline_item['data'] contains path to the files on disk
        input_info = list(self.extra_params['input_info'].values())
        for inp_idx, path in enumerate(pipeline_item['data']):
            # load data from disk with dtype mentioned in inference-config section
            pipeline_item['data'][inp_idx] = np.fromfile(path, dtype=input_info[inp_idx][0])
        if len(batched_inputs_idx_parser.findall(path)) > 0:
            start_idx, end_idx = batched_inputs_idx_parser.findall(path)[
                0]  # assume only one pattern is supposed to match
            start_idx, end_idx = int(start_idx), int(end_idx)
            pipeline_item['input_idx'] = list(range(start_idx, end_idx + 1))
            pipeline_item['meta'] = [[] for i in range(start_idx, end_idx + 1)]
        elif len(single_inputs_idx_parser.findall(path)) > 0:
            idx = single_inputs_idx_parser.findall(path)[0]
            pipeline_item['input_idx'] = [int(idx)]
            pipeline_item['meta'] = [[]]
        elif len(ref_single_outputs_idx_parser.findall(path)) > 0:
            idx = ref_single_outputs_idx_parser.findall(path)[0]
            pipeline_item['input_idx'] = [int(idx)]
            pipeline_item['meta'] = [[]]
        return pipeline_item


class qacc_memory_preprocessor(qacc_memory_plugin):
    _type = qcc.STAGE_PREPROC


class qacc_memory_postprocessor(qacc_memory_plugin):
    _type = qcc.STAGE_POSTPROC


class qacc_memory_metric:
    """QACC Memory Metric Plugin Interface."""
    _type = qcc.STAGE_METRIC

    def __init__(self, dataset, **kwargs):
        """Initializer (Optional): Required only if custom control over the
        metric state.

        kwargs:  Allows plugin developer to supply various
        """
        self.dataset = dataset
        self.metric_state = []
        # Needed to make sure parameters supplied during creation are made attributes
        self.__dict__.update(kwargs)

    def __call__(self, pipeline_item=None, **kwargs):
        # Need to handle 2 Scenarios:
        # 1. Setup and Initialization : if we dont pass pipeline_item
        # 2. Calculate step : we pass pipeline_item
        # If pipeline_item given do the actual processing
        try:
            if pipeline_item['data']:
                # Check whether the item inside is a list of list
                if not isinstance(pipeline_item['data'][0], list):
                    result = self.calculate(data=pipeline_item.data, meta=pipeline_item.meta,
                                            input_idx=pipeline_item.input_idx)
                    if result:
                        self.metric_state.append(result)
                else:
                    self.execute_index(pipeline_item)
            else:
                qacc_logger.error(f"Input to the Metric is empty. {pipeline_item}")
        except Exception as e:
            pipeline_item['status'] = False  # fail
            qacc_file_logger.error(
                f"Failed to run calculate in plugin : {self.__class__.__name__}. Reason: {e}")

    def calculate(self, data, meta, input_idx, *args, **kwargs):
        """
        Optional: Describes the logic for computing metric score for single input.
        If user does not override this function, we preserve the entire data state.
        User can optionally prepare the data and meta metric computation
        data  -> [out_sample1_node_1,out_sample1_node_2]
        meta --> {'key1':val}
        input_idx --> Input identifier (int) (Assigned by infrastructure: 0-N Samples)
        args : Mandatory arguments which can modify the execution logic
        kwargs : optional Keyword based arguments which modifies the execution logic
        returns : prepared data & updated metadata (i.e meta)
        """
        return data, meta

    @abstractmethod
    def finalize(self):
        """Compute metric/aggregate metric score computed across input.

        metric_state will contain data required to compute metric score.
        Note: It must return the results as a python dictionary
        """
        pass

    def execute_index(self, pipeline_item, **kwargs):
        # Default implementation to handle multiple output case in a single pipeline_items i.e multiple outputs per single input evaluation
        # Some metric plugin might need to override this method to act on multiple outputs (if any)
        # if metric has logic that requires on both postproc_out_1, postproc_out_2, _execute_index must be implemented as well.
        # pipeline_item['data'] -> [[out_sample1_node_1,out_sample1_node_2],[out_sample2_node_1,out_sample2_node_2]]

        for processed_idx, processed_data in enumerate(pipeline_item['data']):
            # Process individual processed_data using calculate

            result = self.calculate(data=processed_data, meta=pipeline_item.meta[processed_idx],
                                    input_idx=pipeline_item.input_idx[processed_idx])
            if result:
                self.metric_state.append(result)


# _infra_fields = ['data', 'meta', 'input_idx', 'status']
# Add Batching related logic
def pack_inputs(pipeline_items, shape_info):
    """Combine a list of pipeline items into a single pipeline item.

    Uses batch dimension from shape_info to stack the data
    """

    combined_pipeline_item = {k: [] for k in pipeline_items[0].keys()}
    for pipeline_item in pipeline_items:
        for key, value in pipeline_item.items():
            if key not in ['data', 'output_names']:
                combined_pipeline_item[key].append(value)

    # Fuse the data column to be fed to QPC/Session object
    fused_data = []
    for node_idx, node_info in enumerate(shape_info.values()):
        fused_input = []
        for pipeline_item in pipeline_items:
            # Required so that actual values are not updated
            updated_shape = copy.deepcopy(node_info[1])
            if node_info[2] is not None:
                updated_shape[node_info[2]] = 1  # update the batch dimension to 1
            fused_input.append(pipeline_item['data'][node_idx].reshape(updated_shape))
        if node_info[2] is not None:  # if batching dimension is present do fusion
            fused_data.append(np.concatenate(fused_input, axis=node_info[2]))
        else:  # Use the entire data [No fusion of inputs across batch_dimension]
            fused_data.extend(fused_input)
    combined_pipeline_item['data'] = fused_data
    # Add Explicit handling of output_names is present in pipeline_item: onnxrt inference engine
    if 'output_names' in pipeline_item:
        combined_pipeline_item['output_names'] = pipeline_item['output_names']
    return PipelineItem(**combined_pipeline_item)


def unpack_pipeline_items(combined_pipeline_item, output_batch_dims):
    pipeline_items = []
    # Drop padded inputs if any. pick only unique items
    unique_items = len(set(combined_pipeline_item['input_idx']))
    for i in range(unique_items):
        temp_item = {}
        for key, value in combined_pipeline_item.items():
            if key not in ['data', 'output_names']:
                temp_item[key] = value[i]
            elif key == 'output_names' and 'output_names' in combined_pipeline_item:
                temp_item[key] = combined_pipeline_item['output_names']
            elif key == 'data':
                temp_item[key] = []
                for idx, data in enumerate(value):
                    # Skip slicing the batched output arrays that are not configured in model config
                    if idx < len(output_batch_dims):
                        if output_batch_dims[idx] is not None:
                            # if batching dimension is present do unpacking
                            temp_item[key].append(
                                np.array([np.take(data, i, axis=output_batch_dims[idx])]))
                        else:
                            # Use the entire data [No unpacking of inputs across batch_dimension]
                            temp_item[key].append(data)
        pipeline_items.append(PipelineItem(**temp_item))
    return pipeline_items
