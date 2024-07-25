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
import copy
import logging
import numpy as np
import os
import json
from itertools import chain

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.configuration as qc
import qti.aisw.accuracy_evaluator.qacc.dataset as ds
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.infer_engines.QnnInferenceEngine import QnnInferenceEngine


class InferenceManager:

    def __init__(self, inference_schema_config, infer_config, binary_path,
                 converter_export_format=None):
        self.inference_schema_config = inference_schema_config
        self.binary_path = binary_path
        self.infer_config = infer_config
        self._converter_export_format = converter_export_format if converter_export_format is not None else qcc.DEFAULT_CONVERTER_EXPORT_FORMAT
        # capture execution time
        # (quantization time, compilation time, inference time)
        self.execution_time = [0, 0, 0]

    def execute(self, model_path, output_dir, input_file, output_file, calibration, device_id,
                precompiled_path, console_tag, compile_only, qnn_sdk_dir=""):
        if self.inference_schema_config._name == qcc.INFER_ENGINE_QNN:
            return self.execute_qnn(model_path, output_dir, input_file, output_file, device_id,
                                    precompiled_path, console_tag, calibration=calibration,
                                    compile_only=compile_only, qnn_sdk_dir=qnn_sdk_dir)
        elif self.inference_schema_config._name == qcc.INFER_ENGINE_ONNXRT:
            return self.execute_onnxrt(model_path, output_dir, input_file, output_file)
        elif self.inference_schema_config._name == qcc.INFER_ENGINE_TFRT:
            return self.execute_tfrt(model_path, output_dir, input_file, output_file)
        elif self.inference_schema_config._name == qcc.INFER_ENGINE_TORCHSCRIPTRT:
            return self.execute_torchscriptrt(model_path, output_dir, input_file, output_file)
        elif self.inference_schema_config._name == qcc.INFER_ENGINE_TFRT_SESSION:
            return self.execute_tfrt_session(model_path, output_dir, input_file, output_file)

        assert ('Invalid inference schema name ' + self.inference_schema_config._name)

    def execute_qnn(self, model_path, output_dir, input_file, output_file, device_id,
                    precompiled_path, console_tag, calibration=None, compile_only=False,
                    qnn_sdk_dir=""):

        backend = self.inference_schema_config._backend
        target_arch = self.inference_schema_config._target_arch

        backend_extensions = self._parse_inference_schema_params(
            self.inference_schema_config._backend_extensions)
        netrun_params = self._parse_inference_schema_params(
            self.inference_schema_config._netrun_params)
        converter_params = self._parse_inference_schema_params(
            self.inference_schema_config._converter_params)
        contextbin_params = self._parse_inference_schema_params(
            self.inference_schema_config._contextbin_params)

        calibration_file = None
        if ("extra_args" in converter_params and "float_fallback" in converter_params["extra_args"]
            ) or ("float_fallback" in converter_params
                  and converter_params["float_fallback"] == "True"):
            pass
        elif calibration and self.inference_schema_config._precision in [
                qcc.PRECISION_QUANT, qcc.PRECISION_INT8
        ]:
            calibration_file = self.parse_generate_calibration(calibration, input_file,
                                                               os.path.dirname(input_file))

        engine = QnnInferenceEngine(
            model=model_path, inputlistfile=input_file, calibration_file=calibration_file,
            output_path=output_dir, input_info=self.inference_schema_config._input_info,
            output_info=self.inference_schema_config._output_info, gen_out_file=output_file,
            backend_extensions=backend_extensions, netrun_params=netrun_params,
            converter_params=converter_params, contextbin_params=contextbin_params, backend=backend,
            target_arch=target_arch, qnn_sdk_dir=qnn_sdk_dir, device_id=device_id,
            converter_export_format=self._converter_export_format)
        if converter_params:
            # dumping all converter params before execution
            outfile = os.path.join(output_dir, 'converter_params_list.json')
            with open(outfile, 'w', encoding='utf-8') as f:
                json.dump(converter_params, f, ensure_ascii=False, indent=4)

        try:
            engine.execute()
            ret_status = True
            qacc_file_logger.info('Inference success on QNN in execution stage.')
        except Exception as e:
            qacc_logger.info(e)
            qacc_file_logger.error('Inference failed on QNN in execution stage.')
            ret_status = False
        finally:
            infer_stages_status = engine.stage_status

        infer_fail_stage = self._get_first_fail_stage(infer_stages_status)
        return not ret_status, infer_fail_stage, [0, 0, 0]

    def execute_onnxrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.OnnxRTEngine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(model=model_path, inputlistfile=input_file,
                                     multithread=self.inference_schema_config._multithreaded,
                                     output_path=output_dir,
                                     input_info=self.inference_schema_config._input_info,
                                     output_info=self.inference_schema_config._output_info,
                                     gen_out_file=output_file,
                                     convert_nchw=self.inference_schema_config._convert_nchw)

        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error('(onnxrt) Inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'onnx-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_tfrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TensorflowRTEngine import TensorflowInferenceEngine
        engine = TensorflowInferenceEngine(model=model_path, inputlistfile=input_file,
                                           multithread=self.inference_schema_config._multithreaded,
                                           output_path=output_dir,
                                           input_info=self.inference_schema_config._input_info,
                                           output_info=self.inference_schema_config._output_info,
                                           gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error('tensorflow runtime inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'tensorflow-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_torchscriptrt(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TorchScriptRTEngine import TorchScriptInferenceEngine
        engine = TorchScriptInferenceEngine(model=model_path, inputlistfile=input_file,
                                            multithread=self.inference_schema_config._multithreaded,
                                            output_path=output_dir,
                                            input_info=self.inference_schema_config._input_info,
                                            output_info=self.inference_schema_config._output_info,
                                            gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error(
                'torchscript runtime inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'torchscript-inference'

        return not status, infer_fail_stage, self.execution_time

    def execute_tfrt_session(self, model_path, output_dir, input_file, output_file):
        from qti.aisw.accuracy_evaluator.common.infer_engines.TensorflowSessionRTEngine import TensorflowSessionInferenceEngine
        engine = TensorflowSessionInferenceEngine(
            model=model_path, inputlistfile=input_file,
            multithread=self.inference_schema_config._multithreaded, output_path=output_dir,
            input_info=self.inference_schema_config._input_info,
            output_info=self.inference_schema_config._output_info, gen_out_file=output_file)
        try:
            status, _, self.execution_time[2], _ = engine.execute()
            infer_fail_stage = None
        except Exception as e:
            qacc_logger.error('tensorflow runtime inference failed. See qacc.log for more details.')
            qacc_file_logger.error('Exception - {}'.format(e))
            status = 0
            infer_fail_stage = 'tensorflow-session-inference'

        return not status, infer_fail_stage, self.execution_time

    def _parse_range(self, index_str):
        if len(index_str) == 0:
            return []
        nums = index_str.split("-")
        assert len(nums) <= 2, 'Invalid range in calibration file '
        start = int(nums[0])
        end = int(nums[-1]) + 1
        return range(start, end)

    def parse_generate_calibration(self, calibration, input_file, output_dir):
        if calibration is None or input_file is None:
            return None
        (calib_type, calib_file) = calibration

        if calib_type == qcc.CALIBRATION_TYPE_RAW:
            return calib_file
        elif calib_type == qcc.CALIBRATION_TYPE_INDEX:
            cf = open(calib_file, 'r')
            indexes_str = cf.read().replace('\n', ',').strip()
            indexes = sorted(
                set(chain.from_iterable(map(self._parse_range, indexes_str.split(",")))))
            cf.close()
            _path = os.path.join(output_dir, 'calibration.txt')
            qacc_file_logger.info('Generating calibration file')
            with open(input_file) as f, open(_path, 'w') as f2:
                for index, line in enumerate(f):
                    if index in indexes:
                        f2.write(line)
            return _path
        else:
            raise RuntimeError('Invalid calibration type {}'.format(calib_type))

    def _parse_inference_schema_params(self, params):
        """Cleans up the unnecessary params."""
        #TODO: Verify if there are any params which needs the False flag.
        param_args = {}
        for k, v in params.items():
            if v is False:
                pass
            else:
                param_args[k] = v

        return param_args

    def _get_first_fail_stage(self, stage_status):
        for stage in stage_status:
            if not stage_status[stage]:
                return stage
        return ""


class InferenceSchemaManager:

    def __init__(self, inference_schemas, config):
        self.inference_schemas = inference_schemas
        self.device_ids = config._inference_config._device_ids
        self.schedule = None

    def scan_and_add_inference_schema_permutations(self):
        """Scans the inference_schema section and finds all the possible
        inference schema permutations. Once the scan is complete, these
        possible inference schema permutations are added to the existing
        inference schema list.

        example:
        Given an inference schema
            inference_schema:
                name: aic
                precision: <value>
                params:
                    param1: input1 | input2 =>
                    param2: input3 | input4 =>
                use_precompiled:

        will create following inference schemas
            inference_schema:
                name: aic
                precision: <value>
                params:
                    param1: input1
                    param2: input3
                use_precompiled:inference_schema:
                name: aic

            inference_schema:
                name: aic
                precision: <value>
                params:
                    param1: input1
                    param2: input4
                use_precompiled:inference_schema:

            inference_schema:
                name: aic
                precision: <value>
                params:
                    param1: input2
                    param2: input3
                use_precompiled:inference_schema:
                name: aic

            inference_schema:
                name: aic
                precision: <value>
                params:
                    param1: input2
                    param2: input4
                use_precompiled:inference_schema:
        """
        # updated inference schemas consisting of original plus newly
        # generated inference schemas
        updated_inference_schemas = []

        # used to tag all the generated inference schemas from
        # one original inference schema with same group id
        group_id = -1

        # use same group_id across same pqq_group
        # key: pqq_group tag and val: group_id
        pgq_group_dict = {}

        # used to perform calibration if int8 inference schema available
        is_calib_req = False

        for inference_schema in self.inference_schemas:

            if (qcc.INFER_ENGINE_QNN != inference_schema._name) and (qcc.INFER_ENGINE_AIC
                                                                     != inference_schema._name):
                qacc_file_logger.debug('scan_and_add: Non QNN inference schema {} added'.format(
                    inference_schema._name))
                updated_inference_schemas.append(inference_schema)
                continue

            # get nested list of values
            param_values = []
            param_keys = []

            for key, val in inference_schema._converter_params.items():
                if isinstance(val, list):
                    # skip keys which have list of values
                    param_values.append(val)
                    param_keys.append(key)
                else:
                    val = str(val)
                    # store list of values
                    vals = [v.strip() for v in val.split(qcc.SEARCH_SPACE_DELIMITER)]
                    if key not in qcc.PIPE_SUPPORTED_CONVERTER_PARAMS and len(vals) > 1:
                        raise ce.ConfigurationException(
                            f"Pipe option not available for converter param {key}")
                    val2remove = []  # Values to Remove
                    for v_idx, v in enumerate(vals):
                        if v.startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and v.endswith(')'):
                            try:
                                start, end, step = v[len(qcc.RANGE_BASED_SWEEP_PREFIX):-1].split(
                                    qcc.RANGE_BASED_DELIMITER)
                                start, end, step = start.strip(), end.strip(), step.strip()
                                val_precision = max([
                                    len(start.split('.')[-1]),
                                    len(end.split('.')[-1]),
                                    len(step.split('.')[-1])
                                ])
                            except:
                                raise ce.ConfigurationException(
                                    f"Check range based parameter syntax in inference_schema params in config "
                                    f"file")
                            _, start = self.get_param_dtype(start, return_val=True)
                            _, end = self.get_param_dtype(end, return_val=True)
                            _, step = self.get_param_dtype(step, return_val=True)
                            range_values = [
                                f'{range_val:0.{val_precision}f}'
                                for range_val in np.arange(start, end, step)
                            ]
                            val2remove.append(v)
                            vals.extend(range_values)
                    for val in val2remove:
                        vals.remove(
                            val)  # Remove the Range based param post expansion of range params

                    param_values.append(list(set(vals)))  # Remove Duplicates if any
                    qacc_file_logger.debug(
                        'Inference schema-{} Added {}:{} values for search space scan'.format(
                            inference_schema._name, key, vals))

                    # store keys
                    param_keys.append(key)

            qacc_file_logger.debug('scan_and_add: Options for keys-{} values-{} added'.format(
                param_keys, param_values))

            # check whether for current inference schema calibration is needed.
            # The key is needed in estimating disk space and performing
            # preprocessing for calibration inputs.
            if not is_calib_req:
                # check only if is_calib_req is False
                # if even inference schema needs calibration then this field will be True
                is_calib_req = (inference_schema._precision == qcc.PRECISION_QUANT
                                and inference_schema._use_precompiled is None)

            if 0 != len(param_values):
                # check if group_id already present
                if inference_schema._pgq_group in pgq_group_dict:
                    group_id = pgq_group_dict[inference_schema._pgq_group]
                else:
                    group_id += 1
                    if inference_schema._pgq_group:
                        pgq_group_dict[inference_schema._pgq_group] = group_id
                self.scan_over_params(param_keys, param_values, inference_schema,
                                      updated_inference_schemas, group_id, True)
                qacc_file_logger.debug(updated_inference_schemas)
            else:
                # add aic inference schema with empty params
                updated_inference_schemas.append(inference_schema)

        for up_inference_schema in updated_inference_schemas:
            qacc_file_logger.info('Inference schema: {} - params: {}'.format(
                up_inference_schema._name, up_inference_schema._converter_params))

        qacc_file_logger.debug('pgq_groups: {}'.format(pgq_group_dict.items()))

        # updating inference schema list
        self.inference_schemas = updated_inference_schemas

        return updated_inference_schemas, is_calib_req

    def scan_over_params(self, param_keys, param_values, inference_schema,
                         updated_inference_schemas, group_id, is_parent, row=0,
                         new_param_values=None):
        """Scan and add inference schemas.

        example format for param_values:
        [[][] ... []]
        [
        [param_val_0 ... param_val_N] => from param_key_0
        [param_val_0 ... param_val_N] => from param_key_1
        .
        .
        .
        [param_val_0 ... param_val_N] => from param_key_N
        ]

        Based on nested param values the function sweeps across all possible combinations
        and adds it as a new AIC inference schema with modified params.
        """

        # new param values
        if new_param_values is None:
            new_param_values = []

        # terminating case
        if row == len(param_values):
            # reached the end so add the inference schema to updated inference schemas
            new_inference_schema = copy.deepcopy(inference_schema)

            # create new param dict
            new_param_dict = dict(zip(param_keys, new_param_values))
            qacc_file_logger.debug(f"New param dict: {new_param_dict}")
            #TODO: Remove all the invalid combinations for the QNN quant params
            # Remove invalid combinations: percentile-calibration-value required only for
            # quantization-calibration == Percentile
            # if 'quantization-calibration' in temp_dict and temp_dict[
            #     'quantization-calibration'] != 'Percentile':
            #     if 'percentile-calibration-value' in temp_dict:
            #         del temp_dict['percentile-calibration-value']
            new_inference_schema._converter_params = new_param_dict

            # mark inference schema with unique group id.
            # This filed is used while reusing pgq profile.
            new_inference_schema._group_id = group_id

            # add new inference schema
            if new_inference_schema not in updated_inference_schemas:
                if is_parent:
                    # add parent inference schema at index 1 (second inference schema)
                    # so that it scheduled before its child inference schemas.
                    # This is done to reuse pgq profile generated by
                    # the parent inference schema. The index 0 is reserved for
                    # reference inference schema eg onnx.
                    updated_inference_schemas.insert(1, copy.deepcopy(new_inference_schema))

                    # add child inference schemas at the end of the list
                    is_parent = False
                else:
                    updated_inference_schemas.append(copy.deepcopy(new_inference_schema))
                qacc_file_logger.debug('Inference schema added: {} - new params: {}'.format(
                    new_inference_schema._name, new_inference_schema._converter_params))

            return is_parent

        for idx, val in enumerate(param_values[row]):

            # check for first element
            if 0 != idx:
                # remove last inserted element
                new_param_values.pop()

            # adding next element
            new_param_values.append(val)

            # call for next params
            is_parent = self.scan_over_params(param_keys, param_values, inference_schema,
                                              updated_inference_schemas, group_id, is_parent,
                                              row + 1, new_param_values)

        # remove last inserted element
        new_param_values.pop()

        # informing previous caller to change it to false
        return is_parent

    def create_schedule(self):
        """Creates a schedule based on distributed inference strategy.

        A schedule has following format:
            [parallel_chuck_1, parallel_chuck_2, ... , parallel_chuck_n]

        Each parallel chunk has following format:
            [(inference_schema_idx, device_id), ... , (inference_schema_idx, device_id)]

        Note: device_id for inference_schemas other than aic is -1

        example:
            case1:
                device_ids = [0,1]
                inference_schemas = [onnx, aic, aic, aic, aic]
                schedule = [[(0,-1), (1,0), (2,1)], [(3,0), (4,1)]]
        """

        self.schedule = []
        slots = len(self.device_ids)
        distributed_inference_schemas = []
        used_slots = 0

        for idx, inference_schema in enumerate(self.inference_schemas):
            if inference_schema._name == qcc.INFER_ENGINE_AIC:

                # if all slots filled
                if used_slots == slots:
                    self.schedule.append(copy.deepcopy(distributed_inference_schemas))
                    distributed_inference_schemas = []
                    used_slots = 0

                distributed_inference_schemas.append((idx, int(self.device_ids[used_slots])))

                # inc used slots
                used_slots += 1

            else:
                # device id for non aic inference schema is -1
                distributed_inference_schemas.append((idx, self.device_ids[0]))

        # copy the last chuck
        self.schedule.append(copy.deepcopy(distributed_inference_schemas))
        qacc_file_logger.info('Distributed schedule: {}'.format(self.schedule))

    def get_schedule(self):
        return self.schedule

    def get_param_dtype(self, param_str, return_val=False):
        """Determine given String is int,float or string Used to in."""
        try:
            val = int(param_str)
            if return_val:
                return int, val
            return int
        except:
            pass
        try:
            val = float(param_str)
            if return_val:
                return float, val
            return float
        except:
            pass
        if return_val:
            return str, val
        return str
