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
import copy
import csv
import datetime
import os
import shutil
import sys
import time
from itertools import chain
from joblib import Parallel, delayed
from tabulate import tabulate
import queue
from concurrent.futures import ThreadPoolExecutor, wait

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.configuration as qc
import qti.aisw.accuracy_evaluator.qacc.dataset as ds
import qti.aisw.accuracy_evaluator.qacc.plugin as pl
import qti.aisw.accuracy_evaluator.qacc.inference as infer
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.comparators import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc.reader import DataReader

console = builtins.print

# stage completion status
STAGE_PREPROC_PASS = False
STAGE_INFER_PASS = False

pipeline_cache = qc.PipelineCache.getInstance()


class QACCManager:

    def __init__(self, config_path, work_dir, set_global=None, batchsize=None, model_path=None,
                 use_memory_plugins=False):

        # Stores the runtime info for each inference schema.
        self.inference_schema_run_status = {}
        # available plugins must be loaded before loading configuration.
        self.use_memory_plugins = use_memory_plugins
        pl.PluginManager.findAvailablePlugins(use_memory_plugins=self.use_memory_plugins)

        qacc_logger.info('Loading model config')
        try:
            #Loading the model config from the yaml file.
            self.config = qc.Configuration(config_path=config_path, work_dir=work_dir,
                                           set_global=set_global, batchsize=batchsize,
                                           model_path=model_path,
                                           use_memory_plugins=self.use_memory_plugins)
        except Exception as e:
            qacc_logger.error('qacc failed to load config file. check log for more details.')
            qacc_file_logger.exception(e)
            sys.exit(1)

        self._work_dir = work_dir
        self.input_names = self.config._inference_config._input_names
        self.input_info = self.config._inference_config._input_info
        self.output_info = self.config._inference_config._output_info

    def process_dataset(self, dataset_config):
        qacc_logger.info('Executing dataset plugins')
        out_dir = self.get_output_path(self._work_dir, qcc.DATASET_DIR)
        plugin_manager = pl.PluginManager()
        return plugin_manager.execute_dataset_transformations(dataset_config, out_dir)

    def preprocess(self, dataset, is_calibration=False):
        # Execute preprocessing.
        if is_calibration:
            qacc_logger.info('Executing Preprocessors for calibration inputs')
            out_dir = self.get_output_path(self._work_dir, qcc.STAGE_PREPROC_CALIB)
            pipeline_cache.set_val(qcc.PIPELINE_CALIB_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        else:
            qacc_logger.info('Executing Preprocessors')
            out_dir = self.get_output_path(self._work_dir, qcc.STAGE_PREPROC)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        transformations_config = self.config._preprocessing_config._transformations
        transformations = pl.Transformations(
            plugin_config_list=transformations_config._plugin_config_list,
            max_input_len=dataset.get_total_entries())
        plugin_manager = pl.PluginManager(dataset)

        ret_status = plugin_manager.execute_transformations(transformations=transformations,
                                                            output_dir=out_dir, batch_offset=0,
                                                            input_names=self.input_names)
        return ret_status, self.get_output_path(out_dir, qcc.QNN_PROCESSED_OUTFILE)

    def preprocess_memory(self, dataset, is_calibration=False):
        """Preprocess the configured dataset. All the items are processed at a
        time and a processed outlist file is returned which contains path to
        the preprocessed files.

        Args:
            dataset: Dataset object containing information on the source data to be processed
            is_calibration : Flag to indicate whether the supplied dataset is Calibration set
        Returns:
            status: 0 if success otherwise 1
            processed_file_list_path: path to the file list containing paths of preprocessed input raw files.
        """
        if is_calibration:
            qacc_logger.info('Executing Preprocessors for calibration inputs')
            out_dir = self.get_output_path(self._work_dir, qcc.STAGE_PREPROC_CALIB)
            pipeline_cache.set_val(qcc.PIPELINE_CALIB_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)
        else:
            qacc_logger.info('Executing Preprocessors')
            out_dir = self.get_output_path(self._work_dir, qcc.STAGE_PREPROC)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, out_dir)
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_IS_CALIB, is_calibration)

        if self.config._preprocessing_config:
            # Do Preprocessing only when it is configured within config file.
            transformations_config = self.config._preprocessing_config._transformations
            transformations = pl.Transformations(
                plugin_config_list=transformations_config._plugin_config_list,
                max_input_len=dataset.get_total_entries())

            # Use Memory Plugins to perform Preprocessing and Return a file list
            # Get Plugin Objects:
            if not os.path.exists(out_dir):  # create output path if not existing.
                os.makedirs(out_dir)
            # Get list of plugins to execute and the input item
            extra_params = {
                'is_calibration': is_calibration,
                'work_dir': self._work_dir,
                'input_info': self.input_info
            }
            try:
                status, plugin_objs = pl.PluginManager.get_memory_plugin_objects(
                    transformations_config._plugin_config_list, extra_params=extra_params)
            except Exception as e:
                qacc_logger.error('failed to create plugins. check log for more details.')
                qacc_file_logger.error(e)
                return 1

            # Setup Datareader and preprocessing queue
            preproc_queue = queue.Queue()
            batchsize = self.config._info_config._batchsize
            dr = DataReader(preproc_queue, iter_obj=dataset._ds_generator, batchsize=batchsize)
            futures = dr.start()
            wait(futures)
            if len(plugin_objs) > 0:
                processors = ThreadPoolExecutor(thread_name_prefix=f'Preprocessors', max_workers=1)
                results = []
                for pipeline_items in iter(preproc_queue.get, "END"):
                    if pipeline_items != qcc.DATAREADER_SENTINEL_OBJ:
                        res_futures = processors.submit(
                            pl.PluginManager.execute_memory_preprocessors, plugin_objs=plugin_objs,
                            output_dir=out_dir, pipeline_items=pipeline_items,
                            input_info=self.input_info, batchsize=batchsize)
                        results.append(res_futures)
                wait(results)
                processors.shutdown(wait=True)  # Shutdown all threads after preprocessing
                results = [r.result() for r in results]
                ret_status, processed_file_path = list(zip(*results))
                ret_status = int(any(ret_status))
        else:
            # Pick items from the preprocessed dataset and write it out into processed-outputs.txt
            processed_file_path = [','.join(item) + '\n' for item in dataset.input_generator()]
            ret_status = 0

        preprocessed_out_file = os.path.join(out_dir, qcc.PROCESSED_OUTFILE)
        with open(preprocessed_out_file, 'w') as file:
            file.writelines(processed_file_path)

        preprocessed_out_file_qnn_format = os.path.join(out_dir, qcc.QNN_PROCESSED_OUTFILE)

        #Write the processed inputs in a different format.
        if self.input_names is not None:
            with open(preprocessed_out_file_qnn_format, 'w') as fl:
                for input_paths in processed_file_path:
                    for i, input_path in enumerate(input_paths.split(',')):
                        if input_path is None:
                            qacc_file_logger.error('Null input found at index {} while creating'
                                                   ' qnn-processed-outputs.txt.\n Record {}'.format(
                                                       i, input_path))
                            raise RuntimeError('Some inputs were not processed!')
                        if i:
                            fl.write(f" {self.input_names[i]}:={input_path}")
                        else:
                            fl.write(f"{self.input_names[i]}:={input_path}")
                    # fl.write('\n')
        return ret_status, self.get_output_path(out_dir, qcc.QNN_PROCESSED_OUTFILE)

    def infer(self, model_path, processed_input_file, inference_schema, dataset, device_id,
              inference_schema_name, compile_only=False, load_binary_from_dir=False):

        # Execute Inference.
        qacc_logger.info('({}) Starting inference engine'.format(inference_schema_name))
        dir_name = self.get_output_path(dir=self._work_dir, type=qcc.STAGE_INFER,
                                        inference_schema_name=inference_schema_name)
        infer_ds_path = self.get_output_path(dir=dir_name, type=qcc.INFER_OUTFILE)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_DIR, dir_name, inference_schema_name)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_FILE, infer_ds_path, inference_schema_name)
        binary_path = self.get_output_path(dir=self._work_dir, type=qcc.BINARY_PATH,
                                           inference_schema_name=inference_schema_name)

        # set network binary directory
        network_bin_dir = inference_schema._precompiled_path \
            if inference_schema._precompiled_path is not None else binary_path

        # store values in pipeline cache
        pipeline_cache.set_val(qcc.PIPELINE_NETWORK_BIN_DIR, network_bin_dir, inference_schema_name)
        pipeline_cache.set_val(qcc.PIPELINE_NETWORK_DESC,
                               os.path.join(network_bin_dir, qcc.NETWORK_DESC_FILE),
                               inference_schema_name)
        pipeline_cache.set_val(qcc.PIPELINE_PROGRAM_QPC,
                               os.path.join(network_bin_dir, qcc.PROGRAM_QPC_FILE),
                               inference_schema_name)

        # update exec batchsize in inference schema params
        bs_key = qcc.INTERNAL_EXEC_BATCH_SIZE
        if pipeline_cache.get_val(bs_key) is not None:
            inference_schema._params[qcc.MODEL_INFO_BATCH_SIZE] = pipeline_cache.get_val(bs_key)

        # Set Precompiled path to the corresponding binary path
        qnn_sdk_dir = pipeline_cache.get_val(qcc.QNN_SDK_DIR)
        precompiled_path = binary_path if load_binary_from_dir else None
        infer_mgr = infer.InferenceManager(
            inference_schema, self.config._inference_config, binary_path,
            converter_export_format=self.config._inference_config._converter_export_format)
        if self.config._dataset_config is not None:
            calibration_file = dataset.get_dataset_calibration()
        else:
            #TODO: Handle cli calibration file.
            calibration_file = (qcc.CALIBRATION_TYPE_RAW, processed_input_file)
        err_status, infer_fail_stage, execution_time = infer_mgr.execute(
            model_path=model_path, output_dir=dir_name, input_file=processed_input_file,
            output_file=infer_ds_path, calibration=calibration_file, device_id=device_id,
            precompiled_path=precompiled_path, console_tag=inference_schema_name,
            compile_only=compile_only, qnn_sdk_dir=qnn_sdk_dir)

        return err_status, infer_fail_stage, infer_ds_path, execution_time

    def post_inference(self, inference_schema, dataset, infer_ds_path, inference_schema_name,
                       pipeline_stages, postprocessor_plugin_objs, metric_objs=[]):
        """Performs Postprocessing and metric computation on the entire
        inference outputs.

        Args:
            inference_schema: Schema object to be executed
            dataset: Dataset object containing information on the source data to be processed
            infer_ds_path: Path to the inference output file list.
            inference_schema_name: Name of the inference schema being executed.
            pipeline_stages : List of stages configured for execution
            postprocessor_plugin_objs: List containing postprocessing plugin objects that are to be executed.
            metric_objs: List containing metric plugin objects that are to be executed.

        Returns:
            metric_result: dictionary containing the metrics
        """

        def map_qnn_net_run_output_dtype(dtype, inference_schema):
            # Reference Framework and AIC Case : Stick to Config. User is expected to use preserve io and use native input & output flags appropriately
            if inference_schema._name != qcc.INFER_ENGINE_QNN or inference_schema._backend == qcc.BACKEND_AIC:
                return dtype
            else:
                return 'float32'  # default dtype in qnn-net-run

        def post_infer(pipeline_items, output_dtypes, output_batch_dims, output_info,
                       inference_schema, metric_objs, postprocessor_plugin_objs=[]):
            """Performs Postprocessing and metric computation on a single
            inference output."""
            try:
                for idx, pipeline_item in enumerate(pipeline_items):
                    #TODO: Add logic to read inference outputs based on net runner flags
                    # native output tensor could mean using the native dtypes instead of default float32
                    # native output tensors can be selective for some of the outputs. To Update the inference_schema directly on occurrence
                    # For aic and reference platform use the datatypes used in model config else default to float32
                    # user is expected to use -use_native_output for aic based platforms.
                    pipeline_items[idx]['data'] = [
                        np.fromfile(path, dtype=output_dtypes[out_idx])
                        for out_idx, path in enumerate(pipeline_item['data'])
                    ]  # TODO: qnn-net-run seems to dump always in fp32.
                    pipeline_item = pl.pack_inputs(pipeline_items, output_info)
                pipeline_items = pl.unpack_pipeline_items(pipeline_item, output_batch_dims)
                for pipeline_item in pipeline_items:
                    for postproc_plugin in postprocessor_plugin_objs:
                        pipeline_item = postproc_plugin(pipeline_item)
                    for metric in metric_objs:
                        metric(pipeline_item)
            except Exception as e:
                qacc_logger.error(
                    f"Failed to evaluate input: {pipeline_item['input_idx']}. Reason: {e}")

        # Execute post processing for this inference results.
        # Get inference output dataset
        # Create Postprocessing  & Metric plugin objects based on configuration
        # Create a Data Reader based on File list and populate the Input Item
        # Spawn multiple workers and execute plugins on the data read from memory
        # Create a File list and execute metric on Post processed outputs
        post_infer_queue = queue.Queue()
        batchsize = self.config._info_config._batchsize
        dr = DataReader(post_infer_queue, path_obj=infer_ds_path, batchsize=batchsize,
                        relative_path=False)
        futures = dr.start()
        wait(futures)
        qacc_logger.info('({}) Executing Post Inference processors'.format(inference_schema_name))
        postprocessor_workers = ThreadPoolExecutor(
            thread_name_prefix=f'({inference_schema_name})[PostInference]', max_workers=1)
        output_dtypes = [
            map_qnn_net_run_output_dtype(o[0], inference_schema) for o in self.output_info.values()
        ]
        output_batch_dims = [o[2] for o in self.output_info.values()]
        while True:
            try:
                pipeline_item = post_infer_queue.get()
            except Exception as e:
                qacc_logger.info(e)
                break
            # Sentinel Object to mark end of input queue
            if pipeline_item == 'END' and post_infer_queue.empty():
                break
            else:
                postprocessor_workers.submit(post_infer, pipeline_item, output_dtypes,
                                             output_batch_dims, self.output_info, inference_schema,
                                             metric_objs, postprocessor_plugin_objs)
        postprocessor_workers.shutdown(wait=True)

        metric_result = {}
        for metric in metric_objs:
            qacc_logger.debug(
                f"Invoking Metric.finalize() for {inference_schema_name} --> {metric}")
            metric_result.update(metric.finalize())
        qacc_logger.info(f"Metrics for {inference_schema_name} : {metric_result}")

        return 0, metric_result

    def postprocess(self, idx, dataset, infer_ds_path, inference_schema_name):
        # Execute post processing for this inference results.
        # Get inference output dataset
        if self.config._postprocessing_config:
            qacc_logger.info('({}) Executing Postprocessors'.format(inference_schema_name))
            infer_dataset = ds.DataSet(input_list_file=infer_ds_path)
            squash_results = self.config._postprocessing_config._squash_results
            transformations_config = self.config._postprocessing_config._transformations
            transformations = pl.Transformations(
                plugin_config_list=transformations_config._plugin_config_list,
                max_input_len=dataset.get_total_entries())
            plugin_manager = pl.PluginManager(infer_dataset, orig_dataset=dataset)
            dir_name = self.get_output_path(dir=self._work_dir, type=qcc.STAGE_POSTPROC,
                                            inference_schema_name=inference_schema_name)
            pipeline_cache.set_val(qcc.PIPELINE_POSTPROC_DIR, dir_name, inference_schema_name)
            err_status = plugin_manager.execute_transformations(transformations=transformations,
                                                                output_dir=dir_name, batch_offset=0,
                                                                squash_results=squash_results)
            if err_status:
                return 1, None

            metrics_input_file = self.get_output_path(dir=dir_name, type=qcc.PROCESSED_OUTFILE)
        else:
            metrics_input_file = infer_ds_path
        pipeline_cache.set_val(qcc.PIPELINE_POSTPROC_FILE, metrics_input_file,
                               inference_schema_name)

        return 0, metrics_input_file

    def evaluate_metrics(self, idx, dataset, postproc_file, inference_schema):
        """Evaluate the given metrics on the inferred data."""
        inference_schema_name = inference_schema.get_inference_schema_name()
        if self.config._evaluator_config._metrics_plugin_list:
            qacc_logger.info('({}) Evaluating metrics'.format(inference_schema_name))
            processed_dataset = ds.DataSet(input_list_file=postproc_file)
            plugin_manager = pl.PluginManager(processed_dataset, orig_dataset=dataset)
            metrics_pl_cfg = self.config._evaluator_config._metrics_plugin_list
            metrics_results = []
            metrics_results_dict = {}
            dir_name = self.get_output_path(self._work_dir, qcc.STAGE_METRIC, inference_schema_name)
            err_status = plugin_manager.execute_metrics(metrics_plugin_config=metrics_pl_cfg,
                                                        output_dir=dir_name,
                                                        results_str_list=metrics_results,
                                                        results_dict=metrics_results_dict)
            if err_status:
                self.inference_schema_run_status[inference_schema_name]['metrics'] = {}
                self.inference_schema_run_status[inference_schema_name][
                    'status'] = qcc.SCHEMA_METRIC_FAIL
                return 1
            metrics_info = ''
            for res in metrics_results:
                qacc_logger.info('({}) metric: {}'.format(inference_schema_name,
                                                          res.replace('\n', ' ')))
                if len(metrics_info) > 0:
                    metrics_info += '\n' + res
                else:
                    metrics_info = res
            self.inference_schema_run_status[inference_schema_name][
                'metrics'] = metrics_results_dict
        else:
            self.inference_schema_run_status[inference_schema_name]['metrics'] = {}
            return 0

    def compare_infer_results(self, preproc_file):
        """Compare inference outputs with configured comparator.

        Comparison can be done if there are more than 1 inference
        schemas. User can configure a reference inference schema by
        is_ref=True in inference_schema section in yaml. In absence of
        is_ref, the first defined inference schema is considered as
        reference and the outputs of other inference schemas are
        compared against those of the reference schema.
        """

        def getComparator(config, out_info=None, ref_out_file=None):
            """Return the configured comparators and datatypes for each output
            The order is same as defined in config file."""
            output_comparators = []
            output_comparator_names = []
            output_comparator_dtypes = []
            output_names = []
            if out_info:
                for outname, val in out_info.items():
                    if len(val) > 3:  # idx=2 is now filled with batch_dimension info
                        # output specific comparator
                        cmp = val[3]['type']
                        tol_thresh = val[3]['tol'] if 'tol' in val[3] else 0.001
                        qacc_file_logger.info('Using output specific comparator : ' + cmp)
                    else:
                        cmp = config._evaluator_config._comparator['type']
                        tol_thresh = float(config._evaluator_config._comparator['tol'])

                    if cmp == 'abs':
                        _comparator = TolComparator(tol_thresh)
                    elif cmp == 'avg':
                        _comparator = AvgComparator(tol_thresh)
                    elif cmp == 'rme':
                        _comparator = RMEComparator(tol_thresh)
                    elif cmp == 'l1norm':
                        _comparator = NormComparator(order=1, tol=tol_thresh)
                    elif cmp == 'l2norm':
                        _comparator = NormComparator(order=2, tol=tol_thresh)
                    elif cmp == 'cos':
                        _comparator = CosComparator(tol_thresh)
                    elif cmp == 'std':
                        _comparator = StdComparator(tol_thresh)
                    elif cmp == 'maxerror':
                        _comparator = MaxErrorComparator(tol_thresh)
                    elif cmp == "snr":
                        _comparator = SnrComparator(tol_thresh)
                    elif cmp == "topk":
                        _comparator = TopKComparator(tol_thresh)
                    elif cmp == "pixelbypixel":
                        _comparator = PixelByPixelComparator(tol_thresh)
                    elif cmp == "box":
                        _comparator = BoxComparator(
                            config._evaluator_config._comparator['box_input_json'], tol_thresh)
                    else:
                        qacc_logger.error('Unknown comparator {}. Using default {} instead:'.format(
                            cmp, 'avg'))
                        cmp = 'avg'
                        _comparator = AvgComparator(0.001)

                    output_comparators.append(_comparator)
                    output_comparator_names.append(cmp)
                    output_comparator_dtypes.append(val[0])
                    output_names.append(outname)
            else:
                out_names = self.get_out_names(ref_out_file)
                for outname in out_names:
                    cmp = config._evaluator_config._comparator['type']
                    tol_thresh = float(config._evaluator_config._comparator['tol'])
                    if cmp == 'abs':
                        _comparator = TolComparator(tol_thresh)
                    elif cmp == 'avg':
                        _comparator = AvgComparator(tol_thresh)
                    elif cmp == 'rme':
                        _comparator = RMEComparator(tol_thresh)
                    elif cmp == 'l1norm':
                        _comparator = NormComparator(order=1, tol=tol_thresh)
                    elif cmp == 'l2norm':
                        _comparator = NormComparator(order=2, tol=tol_thresh)
                    elif cmp == 'cos':
                        _comparator = CosComparator(tol_thresh)
                    elif cmp == 'std':
                        _comparator = StdComparator(tol_thresh)
                    elif cmp == 'maxerror':
                        _comparator = MaxErrorComparator(tol_thresh)
                    elif cmp == "snr":
                        _comparator = SnrComparator(tol_thresh)
                    elif cmp == "topk":
                        _comparator = TopKComparator(tol_thresh)
                    elif cmp == "pixelbypixel":
                        _comparator = PixelByPixelComparator(tol_thresh)
                    elif cmp == "box":
                        _comparator = BoxComparator(
                            config._evaluator_config._comparator['box_input_json'], tol_thresh)
                    else:
                        qacc_logger.error('Unknown comparator {}. Using default {} instead:'.format(
                            cmp, 'avg'))
                        cmp = 'avg'
                        _comparator = AvgComparator(0.001)

                    output_comparators.append(_comparator)
                    output_comparator_names.append(cmp)
                    #Default dtype to float32
                    output_comparator_dtypes.append("float32")
                    output_names.append(outname)

            return output_names, output_comparators, output_comparator_dtypes, \
                   output_comparator_names

        inference_schemas = self.config._inference_config._inference_schemas
        if inference_schemas and len(inference_schemas) < 2:
            qacc_logger.info('Not enough inference schemas to compare inference outputs')
            return 0

        ref_inference_schema = self.config.get_ref_inference_schema()

        ref_out_dir = self.get_output_path(self._work_dir, qcc.STAGE_INFER,
                                           ref_inference_schema.get_inference_schema_name())
        ref_out_file = self.get_output_path(ref_out_dir, qcc.INFER_OUTFILE)
        if not os.path.exists(ref_out_file):
            qacc_file_logger.error(
                'reference inference out file {} does not exist'.format(ref_out_file))
            return 1

        outputs_ref = []
        with open(ref_out_file) as ref_file:
            for line in ref_file:
                outputs_ref.append(line.split(','))

        fcomp = FileComparator()
        out_names, comp, comp_dtypes, comp_names = getComparator(self.config,
                                                                 ref_inference_schema._output_info,
                                                                 ref_out_file=ref_out_file)
        qacc_file_logger.info('comparators: {}'.format(comp))
        qacc_file_logger.info('comparator dtypes: {}'.format(comp_dtypes))

        # compare outputs for all inference schemas with reference.
        top = int(self.config._evaluator_config._comparator['fetch_top'])

        qacc_file_logger.info('Comparing inference output files. This may take some time..')
        qacc_file_logger.info('================ Inference output comparisons ====================')
        qacc_file_logger.info('Comparing all files ...')
        for idx, inference_schema in enumerate(inference_schemas):
            if idx == ref_inference_schema._idx:
                continue

            inference_schema_name = inference_schema.get_inference_schema_name()

            try:
                out_file = self.get_output_path(
                    self.get_output_path(self._work_dir, qcc.STAGE_INFER, inference_schema_name),
                    qcc.INFER_OUTFILE)

                if not os.path.exists(out_file):
                    qacc_file_logger.error(
                        'Inference schema infer out file does not exist {}'.format(out_file))
                    continue

                outputs_inference_schema = []
                with open(out_file) as inference_schema_file:
                    for line in inference_schema_file:
                        outputs_inference_schema.append(line.split(','))

                if len(outputs_ref) != len(outputs_inference_schema):
                    qacc_file_logger.error(
                        'Infer output files count for {}:{} does not match for ' +
                        '{}: {}'.format(ref_inference_schema._name + str(ref_inference_schema._idx),
                                        len(outputs_ref), inference_schema._name +
                                        str(idx), len(outputs_inference_schema)))
                    return 1

                # compare each output of each inference schema and reference.
                output_results = {}
                output_results_per_output = {}
                for i, ref_inps in enumerate(outputs_ref):
                    inference_schema_inps = outputs_inference_schema[i]
                    if len(ref_inps) != len(inference_schema_inps):
                        qacc_file_logger.error(
                            'Record {} :Reference number of inputs {} must match '
                            'with inference_schema {} inputs {}'.format(
                                i, len(ref_inps), inference_schema._name + str(idx),
                                len(inference_schema_inps)))
                        return 1

                    if comp[0].name() == "box":
                        match, percent_match, _ = fcomp.compare(
                            [a_path.strip() for a_path in inference_schema_inps],
                            [r_path.strip() for r_path in ref_inps], comp[0], comp_dtypes[0])

                        output_results[i] = round(percent_match, 3)
                    else:
                        out_i_per_match = []
                        for out_i, (a_path,
                                    r_path) in enumerate(zip(inference_schema_inps, ref_inps)):
                            if comp[out_i].name() == "pixelbypixel":
                                save_dir = os.path.dirname(a_path)
                                match, percent_match, _ = fcomp.compare(
                                    a_path.strip(), r_path.strip(), comp[out_i], comp_dtypes[out_i],
                                    save_dir=save_dir)
                            else:
                                match, percent_match, _ = fcomp.compare(
                                    a_path.strip(), r_path.strip(), comp[out_i], comp_dtypes[out_i])
                            out_i_per_match.append(percent_match)
                            if out_i in output_results_per_output:
                                output_results_per_output[out_i].append(percent_match)
                            else:
                                output_results_per_output[out_i] = [percent_match]

                        output_results[i] = round(sum(out_i_per_match) / len(out_i_per_match), 3)

                # sorting by values
                output_results = dict(sorted(output_results.items(), key=lambda item: item[1]))

                mean = round(sum(output_results.values()) / len(output_results), 3)
                qacc_file_logger.info('Avg Match (all outputs) : {} vs {} = {} %'.format(
                    'schema' + str(ref_inference_schema._idx) + '_' + ref_inference_schema._name,
                    inference_schema_name, mean))
                self.inference_schema_run_status[inference_schema_name]['comparator'] = {
                    f'Avg Match (all outputs)': mean
                }
                for out_i, oname in enumerate(out_names):
                    _mean = round(
                        sum(output_results_per_output[out_i]) /
                        len(output_results_per_output[out_i]), 3)
                    self.inference_schema_run_status[inference_schema_name]['comparator'].update(
                        {f'({comp_names[out_i]}) {oname}': _mean})
                    qacc_file_logger.info('\t({}) {} => {} %'.format(comp_names[out_i], oname,
                                                                     _mean))

                matches = sum(float(x) == 100.0 for x in output_results.values())
                qacc_file_logger.info('Complete Matches {} %'.format(
                    round(matches * 100 / len(output_results)), 3))

                qacc_file_logger.info('Top mismatched inputs:')
                qacc_file_logger.info('------------------------------')

                # create 'top' mismatched files reading from preproc file.
                top_indexes = []
                for x in list(output_results)[0:top]:
                    qacc_file_logger.info('Index {} matched {} %  '.format(x, output_results[x]))
                    top_indexes.append(x)

                cache_lines = {}
                with open(preproc_file, 'r') as f1, open(
                        os.path.join(self._work_dir, inference_schema_name + '_mm.txt'), 'w') as f2:
                    for pos, line in enumerate(f1):
                        if pos in top_indexes:
                            cache_lines[pos] = line
                    # write the file inputs in order of top indexes
                    for i in top_indexes:
                        f2.write(cache_lines[i])

                qacc_file_logger.info('')
                qacc_file_logger.info('Top matched inputs:')
                qacc_file_logger.info('-------------------------------')
                for x in list(reversed(list(output_results)))[0:top]:
                    qacc_file_logger.info('Index {} matched {} %  '.format(x, output_results[x]))

            except Exception as e:
                qacc_file_logger.error(e)
                return 1

        return 0

    def _set_test_params(self, max_calib=5):
        self.config._inference_config._max_calib = max_calib

    def run_pipeline(self, work_dir='qacc_temp', inference_schema_name=None,
                     inference_schema_tag=None, cleanup='', onnx_symbol=None, device_id=None,
                     cli_preproc_file=None, cli_infer_file=None, qnn_sdk_dir="", silent=False,
                     backend=None):
        """Executes the E2E pipeline based on the args and model configuration.

        Args:
            Arguments passed from cmd line are supplied to respective variables
        work_dir: path to directory to store the evaluation results and associated artifacts
        inference_schema_name: run only on this inference schema type Allowed values ['qnn','aic','onnxrt',
        'tensorflow','torchscript']
        inference_schema_tag: run only this inference schema tag
        cleanup:'cleanup preprocessing, inference and postprocessing output files.
            cleanup = 'end': deletes the files after all stages are completed.
            cleanup = 'intermediate' : deletes the intermediate inference and postprocessing
            output files. Selecting intermediate option saves space but disables comparator option'
        onnx_symbol: Replace onnx symbols in input/output shapes. Can be passed as list of
        multiple items. Default replaced by 1. e.g __unk_200:1
        device_id: Target Device to be used for accuracy evaluation
        preproc_file: preprocessed output file (if starting at infer stage)
        infer_file: Inference output file (if starting at postproc stage)
        Returns:
            status: 0 if success otherwise 1
        """
        ret_status = 0
        pipeline_stages, pipeline_start, pipeline_end = self.get_pipeline_stages_from_config(
            self.config)
        if len(pipeline_stages):
            qacc_file_logger.info('Configured stages: {}'.format(pipeline_stages))
        else:
            qacc_logger.error('Invalid pipeline start and end stages')
            return 1

        # execute dataset plugins
        if self.config._dataset_config:
            self.config._dataset_config = self.process_dataset(self.config._dataset_config)
            # handle max_inputs and max_calib for backward compatibility
            # override max_calib from dataset plugin to dataset
            if pipeline_cache.get_val(qcc.PIPELINE_MAX_INPUTS):
                self.config._dataset_config._update_max_inputs()  # Update max_inputs post
                # process_dataset()
            # Set max_inputs in pipeline cache with updated values
            pipeline_cache.set_val(qcc.PIPELINE_MAX_INPUTS, self.config._dataset_config._max_inputs)
            # override max_calib from dataset plugin to inference section for backward compatibility
            if pipeline_cache.get_val(qcc.PIPELINE_MAX_CALIB):
                self.config._inference_config._max_calib = pipeline_cache.get_val(
                    qcc.PIPELINE_MAX_CALIB)
            else:
                pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB,
                                       self.config._inference_config._max_calib)
            # create dataset with modified dataset config
            dataset = ds.DataSet(dataset_config=self.config._dataset_config, caching=True)

        preproc_file = None
        infer_file = None
        inference_schema_manager = None
        model_path = None
        inference_schemas = []

        if qcc.STAGE_INFER in pipeline_stages:
            # inference_schema config must be present
            inference_schemas = self.config._inference_config._inference_schemas  # list of inference schemas from config
            # Filter inference schemas using supplied cli inference-schema and inference-schema-tag
            inference_schemas = self.filter_inference_schemas(
                inference_schemas, inference_schema_name=inference_schema_name,
                inference_schema_tag=inference_schema_tag)

            if device_id:
                if backend is not None and "dspv" in backend:
                    device_ids = [device_id]
                elif isinstance(device_id, int):
                    # device_id=0 format
                    device_ids = [device_id]
                elif isinstance(device_id, str):
                    # Assumes '0,1' like string Format. Handle Sting parsing to list conversion
                    device_ids = [int(device) for device in device_id.strip().split(',')]
                else:
                    # Assumes [0,1] like list Format.
                    device_ids = [int(device) for device in device_id]
                # Validate in right range
                if backend is not None and "dspv" in backend:
                    status = True
                else:
                    status = Helper.validate_aic_device_id(device_ids)
                if status:
                    self.config._inference_config._device_ids = device_ids

            # once inference schema(s) is selected perform further actions
            # create inference schema manager
            inference_schema_manager = infer.InferenceSchemaManager(inference_schemas, self.config)

            # search space scan and adding inference schema combination
            inference_schemas, is_calib_req = inference_schema_manager.scan_and_add_inference_schema_permutations(
            )

            # update the config object with all inference schema permutation
            self.config._inference_config._inference_schemas = inference_schemas
            self.config._inference_config._is_calib_req = is_calib_req

            self.config._inference_config.set_inference_schema_names()

            # create schedule for different inference schemas
            inference_schema_manager.create_schedule()

        # get the pipeline_inputs
        if self.config._dataset_config:
            total_inputs = dataset.get_total_entries()

        # clean model if configured.
        if qcc.STAGE_INFER in pipeline_stages:
            # confirm for inference schemas and estimate space
            if self.config._dataset_config:
                self.confirmation_prompt(inference_schemas, self.config, total_inputs, dataset,
                                         inference_schema_manager, cleanup, silent)

            # clean only if the model is not a tf session or pytorch module
            # config._inference_config._model_object is True for tf session and pytorch module. False otherwise
            custom_op_model, quantization_overrides_flag = QACCManager.check_model_for_simplification(
                inference_schemas)
            if not self.config._inference_config._model_object:
                if self.config._inference_config and self.config._inference_config._clean_model:
                    qacc_logger.info('Cleaning up model..')
                    symbols = {}
                    if self.config._inference_config._onnx_define_symbol:
                        sym_from_config = self.config._inference_config._onnx_define_symbol.split(
                            ' ')
                        for sym in sym_from_config:
                            elems = sym.split('=')
                            symbols[elems[0]] = int(elems[1])
                    if onnx_symbol:
                        for sym in onnx_symbol:
                            elems = sym[0].split(':')
                            symbols[elems[0]] = int(elems[1])
                    if custom_op_model:
                        # For custom op model: No clean up to be performed
                        # if user specifies simplify_model=True and provides custom_op_model
                        if self.config._inference_config._simplify_model:
                            self.config._inference_config._simplify_model = False
                            qacc_file_logger.warning(
                                "Can't simplify the model when custom ops is specified, continuing without simplification and cleanup."
                            )
                        model_path = ModelHelper.clean_model_for_qnn(
                            self.config._inference_config._model_path, out_dir=self._work_dir,
                            symbols=symbols, check_model=False, simplify_model=False)

                    # if any of the inference schemas contain "quantization_overrides", Generate both cleaned with/without simplified model
                    elif quantization_overrides_flag:
                        # if user specifies simplify_model=True and provides quantization_overrides
                        if self.config._inference_config._simplify_model:
                            qacc_file_logger.warning(
                                "Can't simplify the model when quantization overrides is specified, continuing only with cleanup."
                            )
                        cleaned_model_path = ModelHelper.clean_model_for_qnn(
                            self.config._inference_config._model_path, out_dir=self._work_dir,
                            symbols=symbols, check_model=self.config._inference_config._check_model,
                            simplify_model=False)
                        self.config._inference_config._cleaned_only_model_path = cleaned_model_path

                    # Always prepare a simplified cleaned model as some inference schemas could still not have quantization_overrides
                    model_path = ModelHelper.clean_model_for_qnn(
                        self.config._inference_config._model_path, out_dir=self._work_dir,
                        symbols=symbols, check_model=self.config._inference_config._check_model,
                        simplify_model=self.config._inference_config._simplify_model)

                else:
                    model_path = self.config._inference_config._model_path
                # check batchsize to be passed to inference engine
                if self.config._inference_config._inference_schemas[0]._input_info:
                    inp_dims = self.config._inference_config._inference_schemas[0]._input_info
                    key_list = list(inp_dims.keys())
                    if len(key_list) == 1:
                        in_node = key_list[0]
                        bs = ModelHelper.get_model_batch_size(model_path, in_node)
                        qacc_file_logger.info(f'Batchsize from Model graph: {bs}')
                        if bs != self.config._info_config._batchsize:
                            # When Model bs != input_bs (supplied) override  input_bs for execution
                            pipeline_cache.set_val(qcc.INTERNAL_EXEC_BATCH_SIZE,
                                                   self.config._info_config._batchsize)
                    else:
                        qacc_file_logger.warning(
                            'Setting batchsize for multiple inputs is currently unsupported')
            else:
                model_path = self.config._inference_config._model_path

        # set values in pipeline pipeline_cache
        pipeline_cache.set_val(qcc.PIPELINE_BATCH_SIZE, self.config._info_config._batchsize)
        pipeline_cache.set_val(qcc.PIPELINE_WORK_DIR, self._work_dir)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_INPUT_INFO,
                               self.config._inference_config._inference_schemas[0]._input_info)
        pipeline_cache.set_val(qcc.PIPELINE_INFER_OUTPUT_INFO,
                               self.config._inference_config._inference_schemas[0]._output_info)
        pipeline_cache.set_val(qcc.QNN_SDK_DIR, qnn_sdk_dir)

        ret_status, preproc_file = self.execute_pipeline_stages(
            pipeline_stages, inference_schema_manager, model_path, cli_work_dir=work_dir,
            pipeline_start=pipeline_start, pipeline_end=pipeline_end,
            cli_preproc_file=cli_preproc_file, cli_infer_file=cli_infer_file, cleanup=cleanup,
            custom_op_model=custom_op_model)

        if ret_status:
            qacc_file_logger.info('Pipeline execution interrupted')

        # do comparison of infer outputs across inference schemas if configured.
        if qcc.STAGE_INFER in pipeline_stages and self.config._evaluator_config._comparator['enabled'] \
                and len(inference_schemas) > 1 and preproc_file is not None and (STAGE_INFER_PASS):
            ret_status = self.compare_infer_results(preproc_file)
            if ret_status:
                qacc_logger.error('qacc comparator failed.')

        # Constant for all inference schemas
        if self.config._dataset_config:
            ds_name = self.config._dataset_config._name
            batch_size = self.config._info_config._batchsize
            max_inputs = pipeline_cache.get_val(qcc.PIPELINE_MAX_INPUTS)
            qacc_file_logger.info(
                f"Using Dataset: {ds_name} Batch Size: {batch_size} Max Inputs : {max_inputs}")
        if pipeline_end != qcc.STAGE_PREPROC:  # Print Summary
            summary = []
            self.results = []  # Used for Api
            # print the results in the same order as config
            for inference_schema_idx, inference_schema in enumerate(inference_schemas):
                entry = []
                inference_schema_name = inference_schema.get_inference_schema_name()
                status_code = self.inference_schema_run_status[inference_schema_name]['status']
                entry.append(inference_schema_name)
                if self.inference_schema_run_status[inference_schema_name]['infer_stage_status']:
                    inference_schema_status_str = f"{qcc.get_inference_schema_status(status_code)} \nin {self.inference_schema_run_status[inference_schema_name]['infer_stage_status']}"
                else:
                    inference_schema_status_str = qcc.get_inference_schema_status(status_code)
                entry.append(inference_schema_status_str)
                entry.append(inference_schema._precision)
                backend_str = ''
                params_str = ''
                converter_params_str = ''
                contextbin_params_str = ''
                netrun_params_str = ''
                if qcc.get_inference_schema_status(status_code) != 'Success':
                    ret_status = 1
                if inference_schema._backend:
                    backend_str = f"{inference_schema._backend}\n{inference_schema._target_arch}"
                entry.append(backend_str)
                if inference_schema._backend_extensions:
                    for k, v in inference_schema._backend_extensions.items():
                        params_str += '{}:{} \n'.format(k, v)
                entry.append(params_str)
                if inference_schema._converter_params:
                    converter_params_str += 'Converter params:\n'
                    for k, v in inference_schema._converter_params.items():
                        converter_params_str += '{}:{} \n'.format(k, v)
                if inference_schema._contextbin_params:
                    contextbin_params_str += '\nContext binary params:\n'
                    for k, v in inference_schema._contextbin_params.items():
                        contextbin_params_str += '{}:{} \n'.format(k, v)
                if inference_schema._netrun_params:
                    netrun_params_str += '\nNetrun params:\n'
                    for k, v in inference_schema._netrun_params.items():
                        netrun_params_str += '{}:{} \n'.format(k, v)
                converter_params_str += contextbin_params_str + netrun_params_str
                entry.append(converter_params_str)
                if 'metrics' in self.inference_schema_run_status[inference_schema_name] and \
                        self.inference_schema_run_status[inference_schema_name]['metrics']:
                    metric_str = ''
                    metrics_dict = self.inference_schema_run_status[inference_schema_name][
                        'metrics']
                    for k, v in metrics_dict.items():
                        metric_str += '{}: {} \n'.format(k, v)
                    entry.append(metric_str)
                else:
                    metrics_dict = {}
                    entry.append('-')
                if 'comparator' in self.inference_schema_run_status[inference_schema_name]:
                    comparator_dict = self.inference_schema_run_status[inference_schema_name][
                        'comparator']
                    comparator_str = ''
                    compare_value = float(list(comparator_dict.values())[0])
                    for k, v in comparator_dict.items():
                        comparator_str += '{}: {} \n'.format(k, v)
                    entry.append(comparator_str)
                    entry.append(compare_value)
                else:
                    comparator_dict = {}
                    entry.append('-')
                    entry.append(float("-inf"))
                summary.append(entry)
                self.results.append([
                    inference_schema._idx, inference_schema._tag, inference_schema_name,
                    qcc.get_inference_schema_status(status_code), inference_schema._precision,
                    inference_schema._converter_params, metrics_dict, comparator_dict
                ])  # appending metric results
            summary.sort(reverse=True, key=lambda x: x[-1])
            summary = [i[:-1] for i in summary]
            qacc_logger.info('Execution Summary:')
            headers = [
                'Inference schema', 'Status', 'Precision', 'Backend', 'Backend extensions',
                'CLI Params', 'Metrics', 'Comparator'
            ]
            console(tabulate(summary, headers=headers))
            result_csv_path = self.get_output_path(self._work_dir, qcc.RESULTS_TABLE_CSV)
            self.write2csv(result_csv_path, summary, header=headers)
            qacc_logger.info(f"\n{tabulate(summary, headers=headers)}")

        # delete output files of all stages.
        if qcc.CLEANUP_AT_END == cleanup:
            self.cleanup_files(self._work_dir, stage='all')
        qacc_file_logger.debug(pipeline_cache._pipeline_cache)

        return ret_status

    def execute_pipeline_stages(self, pipeline_stages, inference_schema_manager, model_path,
                                cli_work_dir, pipeline_start, pipeline_end, cli_preproc_file,
                                cli_infer_file=None, cleanup=None, custom_op_model=False):
        """Execute pipeline stages."""
        # using global stage variables
        global STAGE_PREPROC_PASS
        global STAGE_INFER_PASS

        compile_only = pipeline_end == qcc.STAGE_COMPILE
        load_compiled_binary_from_dir = pipeline_start == qcc.STAGE_COMPILE

        # This is used during inference stage to support reuse_pgq option.
        # The dictionary stores
        # key: group_id and val: path to pgq profile
        # initially has no values and updated once profiles are generated
        group_pgq_dict = {}
        batchsize = self.config._info_config._batchsize
        # capturing start time of calibration
        start_time = time.time()

        # perform preprocessing for calibration inputs
        # this adds support to supply calibration file with inputs files.
        # These inputs can be filenames other than what is mentioned in inputlist.
        # To use this add calibration section in the dataset.yaml as below:
        # calibration:
        #             type: filename
        #             file: calibration_file.txt
        if self.config._dataset_config is not None and self.config._dataset_config._calibration_file \
                and self.config._dataset_config._calibration_type == qcc.CALIBRATION_TYPE_DATASET \
                and (pipeline_end == qcc.STAGE_PREPROC
                     or self.config._inference_config._is_calib_req) and (qcc.STAGE_PREPROC in pipeline_stages):

            # modify the inputlist file to calibration file
            # this is done to execute all the preprocessing plugins
            # using files in calibration file
            calib_dataset_config = copy.deepcopy(self.config._dataset_config)
            calib_dataset_config._inputlist_file = self.config._dataset_config._calibration_file
            calib_dataset_config._max_inputs = self.config._inference_config._max_calib

            # create dataset object with inputlist as calibration file
            calib_dataset = ds.DataSet(dataset_config=calib_dataset_config, caching=True)

            # using batch index 0
            if self.use_memory_plugins:
                err_status, calib_file = self.preprocess_memory(calib_dataset, True)
            else:
                err_status, calib_file = self.preprocess(calib_dataset, True)
            if err_status:
                qacc_file_logger.info('Calibration preprocessing failed')
                return 1
            else:
                # Setting it to RAW as these inputs are already preprocessed
                self.config._dataset_config._calibration_type = qcc.CALIBRATION_TYPE_RAW
                self.config._dataset_config._calibration_file = calib_file

                # updating the max calib
                # This is added as in certain scenarios the number of processed outputs
                # could increase or decrease based on processing technique used like
                # in the case of BERT model.
                self.config._inference_config._max_calib = len(open(calib_file).readlines())
                pipeline_cache.set_val(qcc.PIPELINE_CALIB_FILE, calib_file)
                qacc_file_logger.info(
                    'Calibration preprocessing complete. calibration file: {}'.format(calib_file))
        else:
            # Setting calibration file to None in case of INT8 is not given as calibration is not required
            if not self.config._inference_config._is_calib_req and self.config._dataset_config:
                self.config._dataset_config._calibration_file = None

        # set calibration time
        self.capture_time(qcc.INTERNAL_CALIB_TIME, start_time)
        start_time = None  # reset start time

        # run the pipeline
        # create new dataset object
        dataset = ds.DataSet(dataset_config=self.config._dataset_config, caching=True)

        # Preprocessing
        # capturing start time of preprocessing
        start_time = time.time()
        if (qcc.STAGE_PREPROC in pipeline_stages) and (cli_preproc_file is None):
            if self.use_memory_plugins:
                err_status, preproc_file = self.preprocess_memory(dataset)
            else:
                err_status, preproc_file = self.preprocess(dataset)
            if err_status:
                STAGE_PREPROC_PASS = False
                return 1
            else:
                # calibration_file = dataset.get_dataset_calibration()
                STAGE_PREPROC_PASS = self.validate_pipeline_stage(qcc.STAGE_PREPROC, self._work_dir)
                if not STAGE_PREPROC_PASS:
                    qacc_file_logger.info('{} stage validation failed'.format(qcc.STAGE_PREPROC))
                else:
                    pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)
        else:
            # required for idx based dataset access while
            # post processing
            if self.config._dataset_config:
                qacc_file_logger.info('Loading dataset')
                dataset.load_dataset()

            # if cli preproc not supplied and preprocessing stage is skipped then treat
            # input list as preproc. This is used for supporting scenarios where only
            # preprocessed data is available.
            qacc_logger.info('Loading preprocessed data')
            if cli_preproc_file:
                #Creates new file with absolute paths
                preproc_file = self.update_relative_paths(cli_preproc_file, cli_work_dir)
            elif pipeline_start == qcc.STAGE_COMPILE:
                # When loading from existing compiled output
                dir = self.get_output_path(cli_work_dir, qcc.STAGE_PREPROC)
                preproc_file = self.get_output_path(dir, qcc.QNN_PROCESSED_OUTFILE)
            else:
                # To support AUTO team where generally only preprocessed data is available
                preproc_file = dataset.get_input_list_file()
            STAGE_PREPROC_PASS = True
            pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)

        # set preprocessing time
        self.capture_time(qcc.INTERNAL_PREPROC_TIME, start_time)
        start_time = None  # reset start time
        if qcc.STAGE_INFER in pipeline_stages:

            if self.config._inference_config is None:
                qacc_logger.error('No inference section found in model config.'
                                  'Use -pipeline-start and -pipeline-end flag to skip inference')
                return 1

            # get all the inference schemas
            inference_schemas = self.config._inference_config._inference_schemas

            # run a schedule in distributed manner
            for schedule in inference_schema_manager.get_schedule():

                # get the scheduled inference schemas
                # schedule format: [(inference_schema_idx, device_id), ... , (inference_schema_idx, device_id)]
                # example: [[(0,-1), (1,0), (2,1)], [(3,0), (4,1)]]
                schd_inference_schemas = []

                for schd_inference_schema in schedule:
                    inference_schema_idx = schd_inference_schema[0]
                    inference_schema = inference_schemas[inference_schema_idx]
                    device_id = schd_inference_schema[1]

                    # update load-profile with pgq profile path if available
                    if (inference_schema._reuse_pgq) and (hasattr(inference_schema, '_group_id')) \
                            and (inference_schema._group_id in group_pgq_dict):
                        inference_schema_name = inference_schema.get_inference_schema_name()
                        inference_schema._params['load-profile'] = group_pgq_dict[
                            inference_schema._group_id]
                        qacc_file_logger.debug('({}) loaded pgq profile {} for group id {}'.format(
                            inference_schema_name, pgq_dir, inference_schema._group_id))

                    # store in schd_inference_schemas
                    inference_schema_tuple = (inference_schema_idx, inference_schema, device_id)
                    schd_inference_schemas.append(inference_schema_tuple)

                # run inference sequentially for QNN
                #TODO: Parallelize the inference for QNN backends.
                for inference_schema_idx, inference_schema, device_id in schd_inference_schemas:
                    # Based on the presence of quantization_overrides parameter use the appropriate model
                    # if quantization_overrides present in inference_schema --> Use cleaned model [Not simplified]
                    # if quantization_overrides not present in inference_schema --> Use cleaned + simplified Model path.
                    # Note: model_path would be source model if the model contains custom op, in other cases it will default to
                    # simplified cleaned model. This is applicable only for ONNX Models

                    #  Default case when we use the simplified  + cleaned model
                    _model_path = model_path

                    if Helper.get_model_type(inference_schema._model_path) == ModelType.ONNX:
                        # if the model has custom op or if the user has set simplify_model: False in inference config,
                        # add no_simplification flag to converter params
                        if not self.config._inference_config._simplify_model:
                            inference_schema._converter_params['no_simplification'] = True
                            _model_path = model_path
                            qacc_file_logger.info(
                                f"Adding no_simplification to converter args for {inference_schema.get_inference_schema_name()}"
                                " as model either has custom op or simplify model flag is set to False in inference config"
                            )
                        # if the inference_schema has quantization_overrides, add no_simplification flag to converter params
                        elif 'quantization_overrides' in inference_schema._converter_params.keys():
                            _model_path = self.config._inference_config._cleaned_only_model_path
                            inference_schema._converter_params['no_simplification'] = True
                            qacc_file_logger.info(
                                f"Adding no_simplification to converter args for {inference_schema.get_inference_schema_name()}"
                                "as quantization_overrides is configured")

                    self.run_schedule_in_parallel(
                        preproc_file, dataset, inference_schema_idx, inference_schema, device_id,
                        pipeline_stages, model_path=_model_path, compile_only_flag=compile_only,
                        load_compiled_binary_from_dir_flag=load_compiled_binary_from_dir,
                        cleanup=cleanup, cli_infer_file=cli_infer_file)

                # save pgq profile path
                for inference_schema_idx, inference_schema, device_id in schd_inference_schemas:
                    # update group_pgq_dict with pgq profile path if available
                    if (inference_schema._reuse_pgq) and (hasattr(inference_schema, '_group_id')) \
                            and (inference_schema._group_id not in group_pgq_dict):
                        inference_schema_name = inference_schema.get_inference_schema_name()
                        # path where profile is stored
                        pgq_dir = self.get_output_path(self._work_dir, qcc.STAGE_INFER,
                                                       inference_schema.get_inference_schema_name())
                        pgq_path = os.path.join(pgq_dir, qcc.PROFILE_YAML)
                        if os.path.exists(pgq_path):
                            group_pgq_dict[inference_schema._group_id] = pgq_path
                            qacc_file_logger.debug('({}) updated group id {} with path {}'.format(
                                inference_schema_name, inference_schema._group_id, pgq_path))

            # marking infer stage passed
            STAGE_INFER_PASS = self.validate_pipeline_stage(qcc.STAGE_INFER, self._work_dir)
            if not STAGE_INFER_PASS:
                qacc_file_logger.info('{} stage validation failed'.format(qcc.STAGE_INFER))

        # delete preprocessed outputs
        if STAGE_PREPROC_PASS and (qcc.CLEANUP_INTERMEDIATE == cleanup):
            self.cleanup_files(self._work_dir, qcc.STAGE_PREPROC)

        # terminate pipeline if only preprocessing is configured
        if STAGE_PREPROC_PASS and pipeline_end == qcc.STAGE_PREPROC:
            # squash preproc files
            preproc_dir = self.get_output_path(self._work_dir, qcc.STAGE_PREPROC)
            if os.path.exists(preproc_dir):
                preproc_file = self.get_output_path(preproc_dir, qcc.QNN_PROCESSED_OUTFILE)
                pipeline_cache.set_val(qcc.PIPELINE_PREPROC_DIR, preproc_dir)
                pipeline_cache.set_val(qcc.PIPELINE_PREPROC_FILE, preproc_file)

            if not STAGE_INFER_PASS:
                return 0, preproc_file

        # setting paths and starting metric evaluation
        for inference_schema_idx, inference_schema in enumerate(inference_schemas):
            inference_schema_name = inference_schema.get_inference_schema_name()
            if self.inference_schema_run_status[inference_schema_name]['status'] in [
                    qcc.SCHEMA_POSTPROC_FAIL, qcc.SCHEMA_INFER_FAIL
            ]:
                continue

            # setting postprocessing file
            if qcc.STAGE_POSTPROC in pipeline_stages and self.config._postprocessing_config \
                    and self.inference_schema_run_status[inference_schema_name][
                'status'] == qcc.SCHEMA_POSTPROC_SUCCESS:
                # validate postproc stage
                if not self.validate_pipeline_stage(qcc.STAGE_POSTPROC, self._work_dir):
                    qacc_file_logger.info('{} stage validation failed'.format(qcc.STAGE_POSTPROC))
                else:
                    postproc_file = pipeline_cache.get_val(qcc.PIPELINE_POSTPROC_FILE,
                                                           inference_schema_name)
            else:
                postproc_file = pipeline_cache.get_val(qcc.PIPELINE_INFER_FILE,
                                                       inference_schema_name)

        return 0, preproc_file

    @staticmethod
    def confirmation_prompt(inference_schemas, config, pipeline_batch_inputs, dataset,
                            inference_schema_manager, cleanup, silent):
        """Prompts the user with.

        - number of inference_schemas
        - total space required in Distributed Strategies
        - disabling of comparator
        """

        # disable comparator for intermediate delete
        # calculate based on delete option
        def log_disk_usage(size, msg):
            if size >= 1024:
                qacc_logger.info(msg + '  - {} GB'.format(round(size / 1024, 2)))
            else:
                qacc_logger.info(msg + '  - {} MB'.format(round(size, 2)))

        cleanup_inter = False
        if qcc.CLEANUP_INTERMEDIATE == cleanup:
            qacc_logger.info('Disabling comparator as -cleanup intermediate is selected')
            config._evaluator_config._comparator['enabled'] = False
            cleanup_inter = True

        num_inference_schemas = len(inference_schemas)
        total_req_sizes = QACCManager.get_estimated_req_size(num_inference_schemas, config, dataset,
                                                             cleanup_inter)
        qacc_logger.info('Total inference schemas : {}'.format(num_inference_schemas))
        qacc_logger.info('Total inputs for execution: {} and calibration: {}'.format(
            config._dataset_config._max_inputs, config._inference_config._max_calib))
        preproc_size, calib_size, infer_size = total_req_sizes[0], total_req_sizes[1], \
                                               total_req_sizes[2]
        log_disk_usage(preproc_size + calib_size + infer_size, 'Approximate disk usage')
        if not cleanup_inter and inference_schema_manager.get_schedule() is not None:
            inference_schemas = len(
                inference_schema_manager.get_schedule()[0])  # get len of first schedule
            size = ((infer_size / num_inference_schemas) *
                    inference_schemas) + calib_size + preproc_size
            log_disk_usage(size, 'Approximate disk usage if -cleanup intermediate option is used')

        user_input = input('Do you want to continue execution? (yes/no) :').lower() \
            if not silent else 'y'
        if user_input not in ['yes', 'y']:
            qacc_logger.info('User terminated execution')
            sys.exit(1)

    @staticmethod
    def get_estimated_req_size(num_inference_schemas, config, dataset, cleanup_inter=False):
        """Estimate the required size for Distributed strategy.

        Returns:
            total_req_sizes: [preproc, calib, infer]
        """

        def _parse_range(index_str):
            if len(index_str) == 0:
                return []
            nums = index_str.split("-")
            assert len(nums) <= 2, 'Invalid range in calibration file '
            start = int(nums[0])
            end = int(nums[-1]) + 1
            return range(start, end)

        if not hasattr(config, '_inference_config'):
            return [0, 0]  # inference section not available for calculation

        inputs = dataset.get_total_entries()

        size_dict = {
            'bool': 1,
            'float': 4,
            'float32': 4,
            'float16': 2,
            'float64': 8,
            'int8': 1,
            'int16': 2,
            'int32': 4,
            'int64': 8
        }

        # calculate preproc output size
        inference_schemas = config._inference_config._inference_schemas
        input_dims = inference_schemas[0]._input_info
        qacc_file_logger.debug('input_dims type{} value {}'.format(type(input_dims), input_dims))
        preproc_size = 0
        batch_size = 1
        for in_node, val in input_dims.items():
            qacc_file_logger.debug('val {} for node {}'.format(val, in_node))
            if val[0] not in size_dict:  # datatype
                qacc_file_logger.error('input type {} not supported in input_info '
                                       'in config'.format(val[0]))
            preproc_size_per_out = 1
            for idx, v in enumerate(val[1]):  # tensor shape
                if 0 == idx:
                    batch_size = v
                preproc_size_per_out *= v
            preproc_size += preproc_size_per_out * size_dict[val[0]]
        total_preproc_size = preproc_size * (inputs / batch_size
                                             )  # (inputs/batch_size) --> num of preproc files
        qacc_file_logger.info('preproc size: {} MB'.format(
            round(total_preproc_size / (1024 * 1024)), 3))

        # calculate calibration output size
        calib_size = 0
        if config._dataset_config._calibration_file and config._inference_config._is_calib_req:
            calib_file = config._dataset_config._calibration_file
            if config._dataset_config._calibration_type == qcc.CALIBRATION_TYPE_DATASET \
                    or config._dataset_config._calibration_type == qcc.CALIBRATION_TYPE_RAW:
                calib_inputs = sum(1 for input in open(calib_file))
            else:
                cf = open(calib_file, 'r')
                indexes_str = cf.read().replace('\n', ',').strip()
                indexes = sorted(set(chain.from_iterable(map(_parse_range,
                                                             indexes_str.split(",")))))
                cf.close()
                calib_inputs = len(indexes)

            if -1 != config._inference_config._max_calib:
                calib_inputs = min(calib_inputs, config._inference_config._max_calib)
            else:
                config._inference_config._max_calib = calib_inputs
            calib_size = (calib_inputs / batch_size) * preproc_size
            qacc_file_logger.info('calib_inputs {} preproc_size {} batch_size {}'.format(
                calib_inputs, preproc_size, batch_size))
        else:
            config._inference_config._max_calib = 0
        # Update the Pipeline cache with New Values after pre-processing
        pipeline_cache.set_val(qcc.PIPELINE_MAX_CALIB, config._inference_config._max_calib)
        # calculating infer output size
        inference_schemas = config._inference_config._inference_schemas
        output_dims = inference_schemas[0]._output_info
        qacc_file_logger.debug('output_dims type{} value {}'.format(type(output_dims), output_dims))
        infer_size = 0
        batch_size = 1
        for out_node, val in output_dims.items():
            qacc_file_logger.debug('val {} for node {}'.format(val, out_node))
            if val[0] not in size_dict:
                qacc_file_logger.error('output type {} not supported in outputs_info '
                                       'in config'.format(val[0]))
            infer_size_per_out = 1
            for idx, v in enumerate(val[1]):
                if 0 == idx:
                    batch_size = v
                infer_size_per_out *= v
            infer_size += infer_size_per_out * size_dict[val[0]]

        infer_size = infer_size * num_inference_schemas * (
            inputs / batch_size)  # (inputs/batch_size) --> num of infer files

        MB_divider = (1024 * 1024)
        total_req_sizes = [
            total_preproc_size / MB_divider, calib_size / MB_divider, infer_size / MB_divider
        ]
        return total_req_sizes

    def run_schedule_in_parallel(self, preproc_file, dataset, inference_schema_idx,
                                 inference_schema, device_id, pipeline_stages, model_path,
                                 compile_only_flag=False, load_compiled_binary_from_dir_flag=False,
                                 cleanup='', cli_infer_file=None):
        """Run in parallel."""

        inference_schema_name = inference_schema.get_inference_schema_name()

        if self.use_memory_plugins:
            # Catch failures during to plugin creation at earlier stage before inference
            metric_objs = []
            postprocessor_plugin_objs = []
            extra_params = {
                # 'inference_schema_type': inference_schema._backend, Smart NMS requires
                'work_dir': self._work_dir,
                'inference_schema_name': inference_schema_name,
                'dataset': dataset,
                'output_info': self.output_info,  # required  for smartnms and miou etc.
                'input_info': self.input_info,  #  required for Create batch etc.
            }
            if qcc.STAGE_POSTPROC in pipeline_stages and self.config._postprocessing_config and self.config._postprocessing_config._transformations:
                # Do Postprocessing only when it is configured within config file.
                try:
                    status, postprocessor_plugin_objs = pl.PluginManager.get_memory_plugin_objects(
                        self.config._postprocessing_config._transformations._plugin_config_list,
                        extra_params=extra_params)
                except Exception as e:
                    qacc_logger.error(
                        'Failed to create postprocessor plugins. check log for more details.')
                    qacc_file_logger.error(e)
                    return 1
            # Setup metric plugins if configured
            if qcc.STAGE_METRIC in pipeline_stages and self.config._evaluator_config and self.config._evaluator_config._metrics_plugin_list:
                try:
                    status, metric_objs = pl.PluginManager.get_memory_plugin_objects(
                        self.config._evaluator_config._metrics_plugin_list,
                        extra_params=extra_params, dataset=dataset)
                except Exception as e:
                    qacc_logger.error(
                        f'Failed to create metric plugins for {inference_schema_name}. check log for more details.'
                    )
                    qacc_file_logger.error(e)
                    return 1

        qacc_file_logger.info(
            'Pipeline Execution - Inference schema: {} running on device-id: {}'.format(
                inference_schema_name, device_id if not device_id == -1 else 'Not AIC'))

        err_status, infer_fail_stage, infer_file, execution_time = self.infer(
            model_path, preproc_file, inference_schema, dataset, device_id, inference_schema_name,
            compile_only=compile_only_flag, load_binary_from_dir=load_compiled_binary_from_dir_flag)

        if err_status:
            qacc_logger.error('({}) inference failed'.format(inference_schema_name))
            self.inference_schema_run_status[inference_schema_name] = {
                'status': qcc.SCHEMA_INFER_FAIL
            }
            self.inference_schema_run_status[inference_schema_name][
                'infer_stage_status'] = infer_fail_stage
            # exit the  thread
            return 1
        else:
            self.inference_schema_run_status[inference_schema_name] = {
                'status': qcc.SCHEMA_INFER_SUCCESS
            }
            self.inference_schema_run_status[inference_schema_name][
                'infer_stage_status'] = infer_fail_stage

        # set quantization, compilation and infer time
        pipeline_cache.set_val(qcc.INTERNAL_QUANTIZATION_TIME, execution_time[0],
                               inference_schema_name)
        pipeline_cache.set_val(qcc.INTERNAL_COMPILATION_TIME, execution_time[1],
                               inference_schema_name)
        pipeline_cache.set_val(qcc.INTERNAL_INFER_TIME, execution_time[2], inference_schema_name)

        # Post processing
        # capturing start time of post processing
        start_time = time.time()
        metrics_result = {}
        if self.use_memory_plugins and any(
            [True for stage in [qcc.STAGE_POSTPROC, qcc.STAGE_METRIC] if stage in pipeline_stages]):
            dir_name = self.get_output_path(dir=self._work_dir, type=qcc.STAGE_INFER,
                                            inference_schema_name=inference_schema_name)
            infer_ds_path = self.get_output_path(dir=dir_name, type=qcc.INFER_OUTFILE)
            err_status, metrics_result = self.post_inference(
                inference_schema, dataset, infer_ds_path, inference_schema_name, pipeline_stages,
                postprocessor_plugin_objs=postprocessor_plugin_objs, metric_objs=metric_objs)
            self.inference_schema_run_status[inference_schema_name]['metrics'] = metrics_result
        else:
            if qcc.STAGE_POSTPROC in pipeline_stages:
                if infer_file is None:
                    if cli_infer_file:
                        infer_file = cli_infer_file
                    else:
                        qacc_logger.error('infer-file needed if inference stage is skipped')
                        return 1
                err_status, postproc_file = self.postprocess(inference_schema_idx, dataset,
                                                             infer_file, inference_schema_name)
                if err_status:
                    qacc_logger.error('({}) post processing failed'.format(inference_schema_name))
                    self.inference_schema_run_status[inference_schema_name][
                        'status'] = qcc.SCHEMA_POSTPROC_FAIL
                    return 1
                else:
                    self.inference_schema_run_status[inference_schema_name][
                        'status'] = qcc.SCHEMA_POSTPROC_SUCCESS

                # set post processing time
                self.capture_time(qcc.INTERNAL_POSTPROC_TIME, start_time, inference_schema_name)
                start_time = None  # reset start time

                # delete intermediate inference output files if configured.
                if qcc.CLEANUP_INTERMEDIATE == cleanup and self.config._postprocessing_config:
                    self.cleanup_files(self._work_dir, qcc.STAGE_INFER, inference_schema_name)

            # Metrics
            # capturing start time of infer
            start_time = time.time()
            if qcc.STAGE_METRIC in pipeline_stages:
                ret_status = self.evaluate_metrics(inference_schema_idx, dataset, postproc_file,
                                                   inference_schema)
                if ret_status:
                    qacc_logger.error(
                        '({}) Metrics evaluation failed. See qacc.log for more details.'.format(
                            inference_schema_name))

                # delete postprocessed output files if configured.
                if qcc.CLEANUP_INTERMEDIATE == cleanup:
                    if self.config._postprocessing_config:
                        self.cleanup_files(self._work_dir, qcc.STAGE_POSTPROC,
                                           inference_schema_name)
                    else:
                        self.cleanup_files(self._work_dir, qcc.STAGE_INFER, inference_schema_name)

            # set metric time
            self.capture_time(qcc.INTERNAL_METRIC_TIME, start_time, inference_schema_name)
            start_time = None  # reset start time

    def cleanup_files(self, work_dir, stage, inference_schema_name=None):
        """Cleanup output files generated during various stages of the
        pipeline."""
        # check if cleaning all stages
        cleanup_all = ('all' == stage)

        # cleanup preproc outputs
        if qcc.STAGE_PREPROC == stage or cleanup_all:
            qacc_logger.info('Cleaning up pre-processed outputs')
            shutil.rmtree(self.get_output_path(work_dir, qcc.STAGE_PREPROC), ignore_errors=True)
            shutil.rmtree(self.get_output_path(work_dir, qcc.STAGE_PREPROC_CALIB),
                          ignore_errors=True)

        # cleanup infer outputs
        if qcc.STAGE_INFER == stage or cleanup_all:
            qacc_logger.info('Cleaning up inference outputs')
            dir = self.get_output_path(work_dir, qcc.STAGE_INFER, inference_schema_name)

            infer_files = []
            file_types = defaults.get_value('qacc.file_type.' + qcc.STAGE_INFER)
            file_types = [type.strip() for type in file_types.split(',')]
            for file_type in file_types:
                infer_files.extend(glob.glob(dir + '/**/*.' + file_type, recursive=True))
            for file in infer_files:
                if qcc.INFER_SKIP_CLEANUP in file:
                    continue
                os.remove(file)

        # cleanup postproc outputs
        if qcc.STAGE_POSTPROC == stage or cleanup_all:
            qacc_logger.info('Cleaning up post-processed outputs')
            shutil.rmtree(self.get_output_path(work_dir, qcc.STAGE_POSTPROC, inference_schema_name),
                          ignore_errors=True)

    def validate_pipeline_stage(self, stage, work_dir):
        """Performs validation on the pipeline stage results.

        Returns:
             True: if the results are valid, False otherwise
        """
        exit_execution = False
        # if not enabled only show warning
        if defaults.get_value('qacc.zero_output_check'):
            exit_execution = True

        file_types = defaults.get_value('qacc.file_type.' + stage)
        file_types = [type.strip() for type in file_types.split(',')]

        dir = os.path.join(work_dir, stage)
        if os.path.exists(dir):
            files = []

            # fetch all files based on extension
            for file_type in file_types:
                files.extend(glob.glob(dir + '/**/' + '*.' + file_type, recursive=True))

            # if no files generated mark validation failed
            if 0 == len(files):
                qacc_file_logger.warning('No files found to validate')
                if exit_execution:
                    return False

            # check all files
            for file in files:
                # if file size zero mark validation failed
                if os.path.getsize(file) == 0:
                    qacc_file_logger.warning('File size zero: {}'.format(file))
                    if exit_execution:
                        return False

        # if didn't return False till this point means validation passed
        return True

    def capture_time(self, key, start_time, nested_key=None):
        pipeline_cache.set_val(key, time.time() - start_time, nested_key)

    def copy_pipeline_stage_execution_time(self, inference_schemas, pipeline_stages):

        def get_time_from_dict(key, nested_key=None):
            if pipeline_cache.get_val(key, nested_key) is None:
                return 0
            else:
                return pipeline_cache.get_val(key, nested_key)

        # common execution time
        qacc_file_logger.info('Preprocessing Time Summary:')
        preproc_time = get_time_from_dict(qcc.INTERNAL_CALIB_TIME) + get_time_from_dict(
            qcc.INTERNAL_PREPROC_TIME)
        summary = [['Preprocessing', str(datetime.timedelta(seconds=preproc_time))]]

        table = tabulate(summary, headers=['Preprocessing', 'Time (hh:mm:ss)'])
        console(table)
        qacc_file_logger.info(table)

        if qcc.STAGE_INFER in pipeline_stages:
            qacc_file_logger.info('Inference schema Wise Time Summary (hh:mm:ss):')
            summary = []
            for inference_schema_idx, inference_schema in enumerate(inference_schemas):
                entry = []
                total_time = 0
                inference_schema_name = inference_schema.get_inference_schema_name()
                entry.append(inference_schema_name)
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(
                            qcc.INTERNAL_QUANTIZATION_TIME, inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_COMPILATION_TIME,
                                                                      inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_INFER_TIME,
                                                                      inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_POSTPROC_TIME,
                                                                      inference_schema_name))))
                entry.append(
                    str(
                        datetime.timedelta(seconds=get_time_from_dict(qcc.INTERNAL_METRIC_TIME,
                                                                      inference_schema_name))))
                phases = [
                    qcc.INTERNAL_QUANTIZATION_TIME, qcc.INTERNAL_COMPILATION_TIME,
                    qcc.INTERNAL_INFER_TIME, qcc.INTERNAL_POSTPROC_TIME, qcc.INTERNAL_METRIC_TIME
                ]
                for phase in phases:
                    total_time += get_time_from_dict(phase, inference_schema_name)
                entry.append(str(datetime.timedelta(seconds=total_time)))
                summary.append(entry)
            headers = [
                'Inference schema', 'Quantization', 'Compilation', 'Inference', 'Postprocessing',
                'Metrics', 'Total'
            ]
            table = tabulate(summary, headers=headers)
            profile_csv_path = self.get_output_path(self._work_dir, qcc.PROFILING_TABLE_CSV)
            self.write2csv(profile_csv_path, summary, header=headers)
            console(table)
            qacc_file_logger.info(table)

    def get_output_path(self, dir, type, inference_schema_name=None):
        """Returns the output directory for various stages of the pipeline."""
        # preprocessing or infer file or metric file
        if type in [
                qcc.STAGE_PREPROC, qcc.INFER_OUTFILE, qcc.PROCESSED_OUTFILE,
                qcc.QNN_PROCESSED_OUTFILE, qcc.STAGE_PREPROC_CALIB, qcc.STAGE_INFER,
                qcc.STAGE_POSTPROC, qcc.STAGE_METRIC, qcc.PROFILING_TABLE_CSV,
                qcc.RESULTS_TABLE_CSV, qcc.INFER_RESULTS_FILE, qcc.DATASET_DIR, qcc.INPUT_LIST_FILE,
                qcc.CALIB_FILE
        ] and inference_schema_name is None:
            return os.path.join(dir, type)

        # inference or postprocessing
        elif type in [qcc.STAGE_INFER, qcc.STAGE_POSTPROC, qcc.STAGE_METRIC]:
            return os.path.join(dir, type, inference_schema_name)

        # binary
        elif type == qcc.BINARY_PATH:
            return os.path.join(dir, qcc.STAGE_INFER, inference_schema_name, 'temp')

    def filter_inference_schemas(self, inference_schemas, inference_schema_name=None,
                                 inference_schema_tag=None):
        # select inference schema based on supplied args
        if inference_schema_name:
            inference_schemas = [p for p in inference_schemas if p._name == inference_schema_name]
            if len(inference_schemas) == 0:
                qacc_logger.error('Invalid inference schema name in -inference_schema option')
                sys.exit(1)
        if inference_schema_tag:
            inference_schemas = [
                p for p in inference_schemas
                if p._tag is not None and inference_schema_tag in p._tag
            ]
            if len(inference_schemas) == 0:
                qacc_logger.error('Invalid inference schema tag in -inference_schema_tag option')
                sys.exit(1)
        return inference_schemas

    def write2csv(self, fname, rows, header):
        # check all rows have same length
        assert len(header) == len(rows[0])
        with open(fname, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)

    def get_out_names(self, out_file):
        """Returns the names of the outputs from the out_file."""
        out_names = []
        with open(out_file) as ref_file:
            outputs = ref_file.readline().split(',')
        for op in outputs:
            file_name, _ = os.path.splitext(op.split('/')[-1])
            out_names.append(file_name)

        return out_names

    def update_relative_paths(self, preproc_file, work_dir):
        """Create a new preproc file and modify the relative paths to absolute
        paths."""
        updated_preproc_file = os.path.join(work_dir, "updated_input_list.txt")
        original_list_dir = os.path.dirname(os.path.abspath(preproc_file))
        with open(updated_preproc_file, "w") as write_file, \
             open(preproc_file, "r") as read_file:

            for line in read_file:
                write_file.write(os.path.join(original_list_dir, line))

        return updated_preproc_file

    def get_pipeline_stages_from_config(self, config):
        pipeline_stages = [
            qcc.STAGE_PREPROC, qcc.STAGE_COMPILE, qcc.STAGE_INFER, qcc.STAGE_POSTPROC,
            qcc.STAGE_METRIC
        ]
        pipeline_start = 'infer'
        pipeline_end = 'infer'
        if config._preprocessing_config:
            pipeline_start = qcc.STAGE_PREPROC
            pipeline_end = qcc.STAGE_PREPROC
        if config._inference_config:
            pipeline_end = qcc.STAGE_INFER
        if config._postprocessing_config:
            pipeline_end = qcc.STAGE_POSTPROC
        if config._evaluator_config._metrics_plugin_list:
            pipeline_end = qcc.STAGE_METRIC
        pipeline_stages = pipeline_stages[pipeline_stages.index(pipeline_start):pipeline_stages.
                                          index(pipeline_end) + 1]

        return pipeline_stages, pipeline_start, pipeline_end

    @classmethod
    def check_model_for_simplification(cls, inference_schemas):
        """Skip model simplification when the model contains a custom op."""
        # Assume custom op and quantization_overrides not present
        custom_op_model = False
        quantization_overrides_flag = False

        for inference_schema in inference_schemas:
            converter_parameters = inference_schema._converter_params.keys()
            # Any of the inference schema contains custom op (Ideally all inference
            # schemas should contain custom op field if model contains custom op)
            custom_op_model = custom_op_model or len(
                set(qcc.CUSTOM_OP_FLAGS).intersection(set(converter_parameters))) > 0
            if custom_op_model:
                # can do early exit with single occurrence
                break

        for inference_schema in inference_schemas:
            converter_parameters = inference_schema._converter_params.keys()
            quantization_overrides_flag = quantization_overrides_flag or True \
                    if 'quantization_overrides' in converter_parameters else False
            if quantization_overrides_flag:
                # can do early exit with single occurrence
                break
        return custom_op_model, quantization_overrides_flag
