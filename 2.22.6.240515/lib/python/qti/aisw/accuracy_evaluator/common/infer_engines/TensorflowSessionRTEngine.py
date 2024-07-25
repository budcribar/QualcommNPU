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
import time
import shutil
import sys
from operator import mod
# to avoid printing logs on console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.infer_engine import InferenceEngine
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper


class TensorflowSessionInferenceEngine(InferenceEngine):
    """TensorflowInferenceEngine class takes required inputs supplied by user
    from commandline options and calls validate and execute methods.

    TODO: Add an extra dictionary parameter to class which enables to uses extra tensorflow
    options
    To use:
    >>> engine = TensorflowInferenceEngine(model, inputlistfile, output_path, multithread,
    input_info,
                output_info, gen_out_file, extra_params)
    >>> engine.validate()
    >>> engine.execute()
    """

    def __init__(self, model, inputlistfile, output_path, multithread, input_info, output_info=None,
                 gen_out_file=None, extra_params=None):
        super().__init__(model, inputlistfile, output_path, multithread, input_info, output_info,
                         gen_out_file, extra_params)
        tf = Helper.safe_import_package("tensorflow")
        self.validate()

    def execute(self):
        """
        This method runs the given model on tensorflow
        Returns:
            status: execution status
            res   : dictionary containing ort_session
        """
        qacc_file_logger.debug("TensorflowInferenceEngine start execution")
        # capture inference time
        inf_time = 0
        res = {}
        do_profile = False
        save_intermediate_outputs = True
        profile_data = None
        if self.extra_params:
            profile_data = {}
            if '-profile' in self.extra_params:
                do_profile = self.extra_params['-profile']
            if '-save-intermediate-outputs' in self.extra_params:
                save_intermediate_outputs = self.extra_params['-save-intermediate-outputs']

        # Making sure input-info and output-info are passed
        if self.input_info is None:
            qacc_file_logger.error("Please pass input-info")
            raise ce.ConfigurationException("input-info is empty")
        if self.output_info is None:
            qacc_file_logger.error("Please pass output-info")
            raise ce.ConfigurationException("output-info is empty")

        # assigning the session passed to self.sess
        self.sess = self.model_path

        # Create a list of input nodes so that it can be used to get node dtype and shapes while
        # running the inference
        graph_def = self.sess.graph.as_graph_def(add_shapes=True)
        inp_nodes = [node for node in graph_def.node if node.op == "Placeholder"]

        # Creating a list of output tensors to be used while running inference
        output_names = self.output_info.keys()
        try:
            output_tensors = [
                self.sess.graph.get_tensor_by_name(out_name + ':0') for out_name in output_names
            ]
        except Exception as e:
            qacc_file_logger.error('Creating output tensors for inference failed.')
            qacc_file_logger.exception(e)
            raise ce.ConfigurationException("output-info passed is incorrect", e)

        # Create the output file if requested.
        out_list_file = None
        if self.gen_out_file:
            out_list_file = open(self.gen_out_file, 'w')

        # Create input dictionary and run the inference for each input.
        with open(self.input_path) as f:
            for iter, line in enumerate(f):
                input_dict = {}
                inps = line.strip().split(',')
                inps = [inp.strip() for inp in inps if inp.strip()]

                for idx, inp in enumerate(inps):
                    if inp_nodes[idx].name not in self.input_info:
                        qacc_file_logger.error(
                            'Input info name not valid for this model. expected: {} '.format(
                                inp_nodes[idx].name))
                        raise ce.ConfigurationException("Invalid Configuration")
                    input_np = np.fromfile(inp, dtype=(
                        Helper.get_np_dtype(self.input_info[inp_nodes[idx].name] \
                                                [0], map_tf=True))).reshape(
                        self.input_info[inp_nodes[idx].name][1])
                    input_dict[self.sess.graph.get_tensor_by_name(inp_nodes[idx].name +
                                                                  ':0')] = input_np

                # Run inference
                try:
                    start_time = time.time()
                    outputs = self.sess.run(output_tensors, feed_dict=input_dict)
                    inf_time = time.time() - start_time

                except Exception as e:
                    qacc_file_logger.error('inference.run failed')
                    qacc_file_logger.exception(e)
                    raise ce.InferenceEngineException("inference failed", e)

                if self.output_path:
                    path = self.output_path + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # Write output files and get profile data.
                    profile_data, _paths = self.save_outputs_and_profile(
                        [Helper.sanitize_node_names(name) for name in output_names], outputs, iter,
                        True, do_profile)

                    # generate output text file for each of the inputs.
                    if self.gen_out_file:
                        out_list_file.write(','.join(_paths) + '\n')

        if self.gen_out_file:
            out_list_file.close()

        #For generating histogram profile
        output_dtypes = [op.dtype for op in outputs]
        output_array_map = list(zip(output_names, output_dtypes, outputs))

        res['tf_session'] = self.sess
        res['profile'] = profile_data

        if do_profile:
            qacc_logger.info('Captured tf profile')

        qacc_file_logger.debug("TensorflowInferenceEngine execution success")
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
        tf = Helper.safe_import_package("tensorflow")
        qacc_file_logger.debug("TensorflowInferenceEngine validation")
        # check the existence of model path and its authenticity
        # if not os.path.exists(self.model_path):
        #     qacc_file_logger.error('Model path : {} does not exist '.format(self.model_path))
        #     raise ce.InferenceEngineException('Model path : {} does not exist '.format(
        #         self.model_path))

        if not isinstance(self.model_path, tf.compat.v1.Session):
            qacc_file_logger.error(
                'Model passed is not an instance of tensorflow.compat.v1.Session')
            raise ce.InferenceEngineException(
                'Model passed is not an instance of tensorflow.compat.v1.Session')

        # check whether the output path exists and create the path otherwise
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # check the existence of input path
        if not os.path.exists(self.input_path):
            qacc_logger.error('Input path : {} does not exist '.format(self.input_path))
            raise ce.InferenceEngineException('Input path : {} does not exist '.format(
                self.input_path))
        qacc_file_logger.debug("TensorflowInferenceEngine validation success")
        return True
