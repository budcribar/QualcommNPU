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
import logging
import numpy as np
import os
import shutil
import sys
import time

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.infer_engine import InferenceEngine
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class OnnxInferenceEngine(InferenceEngine):
    """OnnxInferenceEngine class takes required inputs supplied by user from
    commandline options and calls validate and execute methods.

    TODO: Add an extra dictionary parameter to class which enables to uses extra onnx_runtime
    options
    To use:
    >>> engine = onnxInferenceEngine(model, inputlistfile, output_path, multithread, input_info,
                output_info, gen_out_file, extra_params)
    >>> engine.validate()
    >>> engine.execute()
    """

    def __init__(self, model, inputlistfile, output_path, multithread, input_info, output_info=None,
                 gen_out_file=None, extra_params=None, convert_nchw=False):
        super().__init__(model, inputlistfile, output_path, multithread, input_info, output_info,
                         gen_out_file, extra_params)
        self.convert_nchw = convert_nchw
        onnx = Helper.safe_import_package("onnx")
        onnxruntime = Helper.safe_import_package("onnxruntime", "1.17.1")
        onnxruntime.set_default_logger_severity(3)
        self.validate()

    def execute(self):
        """
        This method runs the given onnx model on onnxruntime returns session and status of execution
        Returns:
            status: execution status
            res   : dictionary containing ort_session
        """
        onnxruntime = Helper.safe_import_package("onnxruntime", "1.17.1")

        # This method runs the given onnx model on onnxruntime and returns session and status of
        # execution
        qacc_file_logger.debug("OnnxInferenceEngine start execution")
        res = {}

        # capture inference time
        inf_time = 0

        do_profile = False
        save_intermediate_outputs = True
        profile_data = None
        if self.extra_params:
            profile_data = {}
            if '-profile' in self.extra_params:
                do_profile = self.extra_params['-profile']
            if '-save-intermediate-outputs' in self.extra_params:
                save_intermediate_outputs = self.extra_params['-save-intermediate-outputs']

        # create onnx runtime session
        ort_session = onnxruntime.InferenceSession(self.model_path)

        # Create a list of input nodes so that it can be used to get node dtype and shapes while
        # running the session
        inp_nodes = [q for q in ort_session.get_inputs()]

        # Create the output file if requested.
        out_list_file = None
        if self.gen_out_file:
            out_list_file = open(self.gen_out_file, 'w')

        start_time = time.time()

        # Create input dictionary for onnx session and run the session for each input.
        with open(self.input_path) as f:
            for iter, line in enumerate(f):
                inputs = {}
                inps = line.strip().split()
                inps = [inp.split(':=')[-1].strip() for inp in inps if inp.strip()]

                for idx, inp in enumerate(inps):
                    if self.input_info is None:
                        # When input shapes and dtypes are not passed by user,inputs dictionary
                        # is formed using inp_nodes list
                        try:
                            inputs[inp_nodes[idx].name] = \
                                np.fromfile(inp,
                                            dtype=Helper.get_np_dtype(
                                                inp_nodes[idx].type)).reshape(inp_nodes[idx].shape)
                        except Exception as e:
                            qacc_logger.error('Unable to extract input info from model.Please try '
                                              'passing input-info ')
                            qacc_file_logger.exception(e)
                            raise ce.InferenceEngineException(
                                "Unable to extract input info from model.Please try "
                                "passing input-info", e)
                    else:
                        if inp_nodes[idx].name not in self.input_info:
                            qacc_file_logger.error(
                                'Input info name not valid for this model. expected: {} '.format(
                                    inp_nodes[idx].name))
                            raise ce.ConfigurationException("Invalid Configuration")

                        inputs[inp_nodes[idx].name] = np.fromfile(inp, dtype=(
                            Helper.get_np_dtype(self.input_info[inp_nodes[idx].name] \
                                                    [0]))).reshape(
                            self.input_info[inp_nodes[idx].name][1])

                # Modify input layout
                if self.convert_nchw:
                    for inp in inputs:
                        inp_shape = inputs[inp].shape
                        inputs[inp] = inputs[inp].reshape(
                            (inp_shape[0], inp_shape[2], inp_shape[3], inp_shape[1])).transpose(
                                (0, 3, 1, 2))

                # Run onnxrt session
                try:
                    outputs = ort_session.run(None, inputs)
                except Exception as e:
                    qacc_file_logger.error('ort_session.run failed')
                    qacc_file_logger.exception(e)
                    raise ce.InferenceEngineException("ort_session.run failed", e)

                # Save output raw files.
                ort_outputs = ort_session.get_outputs()

                # Store the names of the outputs. Use the same for raw file name generation.
                if self.output_info:
                    output_names = []
                    for out_name in list(self.output_info.keys()):
                        output_names.append(Helper.sanitize_node_names(out_name))
                    # reorder onnx rt outputs if needed as per output info names
                    _temp = []
                    _temp_outs = []
                    for name in output_names:
                        _name_found = False
                        for out, node in zip(outputs, ort_outputs):
                            if str(node.name) == str(name):
                                _temp.append(node)
                                _temp_outs.append(out)
                                _name_found = True
                                break
                        if not _name_found:
                            qacc_logger.error('Output name {} in config is incorrect.'.format(name))
                            raise ce.InferenceEngineException(
                                "Invalid model config. Please fix output_info {}".format(name))
                    ort_outputs = _temp
                    outputs = _temp_outs
                else:
                    # set same as ort session
                    output_names = []
                    for node in ort_outputs:
                        output_names.append(node.name)

                if self.output_path:
                    path = self.output_path + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # Check if the output info is configured correctly.
                    if len(output_names) != len(ort_outputs):
                        raise ce.ConfigurationException('The number of outputs in config file'
                                                        '({}) does not match with onnxrt model'
                                                        ' outputs ({})'.format(
                                                            len(output_names), len(ort_outputs)))

                    # Write output files and get profile data.
                    profile_data, _paths = self.save_outputs_and_profile(
                        output_names, outputs, iter, save_intermediate_outputs, do_profile)

                    # generate output text file for each of the inputs.
                    if self.gen_out_file:
                        out_list_file.write(','.join(_paths) + '\n')

        #For generating histogram profile
        input_names = list(inputs.keys())
        input_values = list(inputs.values())
        input_dtypes = [ip.dtype for ip in input_values]
        output_dtypes = [op.dtype for op in outputs]
        output_array_map = list(
            zip(input_names + output_names, input_dtypes + output_dtypes, input_values + outputs))

        if self.gen_out_file:
            out_list_file.close()

        res['ort_session'] = ort_session
        res['profile'] = profile_data

        if do_profile:
            qacc_logger.info('Captured onnx profile')

        inf_time = time.time() - start_time
        qacc_file_logger.debug("OnnxInferenceEngine execution success")
        return True, res, inf_time, output_array_map

    def get_profile(self):
        return self.profile_data

    def validate(self):
        """
        This method checks whether the given model_path ,model,input_path and output_path are
        valid or not
        Returns:
            status: validation status
        """
        onnx = Helper.safe_import_package("onnx", "1.12.0")
        qacc_file_logger.debug("OnnxInferenceEngine validation")
        # check the existence of model path and its authenticity
        if os.path.exists(self.model_path):
            try:
                onnx.checker.check_model(self.model_path)
            except Exception as e:
                qacc_file_logger.warning(
                    f'check_model failed for model : {self.model_path} Reason: {e}')
                # suppressing the exception, as shape info is populated from the onnxrt-session.
                # This did not cause any issue in executing the model on onnxrt.
                # qacc_file_logger.exception(e)
                # raise ce.InferenceEngineException(
                #    'check_model failed for given model : {}'.format(self.model_path))

        else:
            qacc_file_logger.error('Model path : {} does not exist '.format(self.model_path))
            raise ce.InferenceEngineException('Model path : {} does not exist '.format(
                self.model_path))

        # check whether the output path exists and create the path otherwise

        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # check the existence of input path
        if not os.path.exists(self.input_path):
            qacc_logger.error('Input path : {} does not exist '.format(self.input_path))
            raise ce.InferenceEngineException('Input path : {} does not exist '.format(
                self.input_path))
        qacc_file_logger.debug("OnnxInferenceEngine validation success")
        return True
