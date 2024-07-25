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


class TorchScriptInferenceEngine(InferenceEngine):
    """TorchScriptInferenceEngine class takes required inputs supplied by user
    from commandline options and calls validate and execute methods.

    To use:
    >>> engine = TorchScriptInferenceEngine(model, inputlistfile, output_path, multithread, input_info,
                output_info, gen_out_file, extra_params)
    >>> engine.validate()
    >>> engine.execute()
    """

    def __init__(self, model, inputlistfile, output_path, multithread, input_info, output_info=None,
                 gen_out_file=None, extra_params=None):
        super().__init__(model, inputlistfile, output_path, multithread, input_info, output_info,
                         gen_out_file, extra_params)
        torch = Helper.safe_import_package("torch")
        self.validate()

    def execute(self):
        """
        This method runs the given TorchScript model and returns session and status of execution
        Returns:
            status: execution status
            res   : dictionary containing session
        """

        # This method runs the given torchscript model and returns session and status of
        # execution
        torch = Helper.safe_import_package("torch")
        qacc_file_logger.debug("TorchScriptInferenceEngine start execution")
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

        # Load torchscript model
        loaded_model = torch.jit.load(self.model_path)
        loaded_model.eval()

        # Create a list of input nodes so that it can be used to get node dtype and shapes while
        # running the session
        inp_nodes = [
            ip.debugName().split('.')[0] for ip in list(loaded_model.graph.inputs())
            if str(ip.type()) == 'Tensor'
        ]

        # Create the output file if requested.
        out_list_file = None
        if self.gen_out_file:
            out_list_file = open(self.gen_out_file, 'w')

        start_time = time.time()

        # Create input dictionary for torchscript session and run the session for each input.
        with open(self.input_path) as f:
            for iter, line in enumerate(f):
                input_list = []
                inps = line.strip().split(',')
                inps = [inp.split(':=')[-1].strip() for inp in inps if inp.strip()]

                for idx, inp in enumerate(inps):
                    if self.input_info is None:
                        # When input shapes and dtypes are not passed by user,inputs dictionary
                        # is formed using inp_nodes list
                        try:
                            input_np = np.fromfile(inp, dtype=Helper.get_np_dtype(
                                inp_nodes[idx].type)).reshape(inp_nodes[idx].shape)
                        except Exception as e:
                            qacc_logger.error('Unable to extract input info from model.Please try '
                                              'passing input-info ')
                            qacc_file_logger.exception(e)
                            raise ce.InferenceEngineException(
                                "Unable to extract input info from model.Please try "
                                "passing input-info", e)
                    else:
                        if inp_nodes[idx] not in self.input_info:
                            qacc_file_logger.error(
                                'Input info name not valid for this model. expected: {} '.format(
                                    inp_nodes[idx]))
                            raise ce.ConfigurationException("Invalid Configuration")

                        input_np = np.fromfile(inp, dtype=(
                            Helper.get_np_dtype(self.input_info[inp_nodes[idx]] \
                                                    [0]))).reshape(
                            self.input_info[inp_nodes[idx]][1])
                        input_list.append(torch.from_numpy(input_np))

                # Run torchscript session
                try:
                    with torch.no_grad():
                        outputs = loaded_model(*input_list)
                except Exception as e:
                    qacc_file_logger.error('TorchScript run failed')
                    qacc_file_logger.exception(e)
                    raise ce.InferenceEngineException("TorchScript run failed", e)

                if not isinstance(outputs, tuple):
                    outputs = tuple(outputs)
                #Convert torch.Tensor to numpy array.
                outputs = tuple(op.numpy() for op in outputs)

                # Store the names of the outputs. Use the same for raw file name generation.
                if self.output_info:
                    output_names = list(self.output_info.keys())
                else:
                    output_names = []
                    for op in loaded_model.graph.outputs():
                        output_names.append(op.debugName())

                if self.output_path:
                    path = self.output_path + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # Check if the output info is configured correctly.
                    if len(output_names) != len(outputs):
                        raise ce.ConfigurationException('The number of outputs in config file'
                                                        '({}) does not match with torchscript model'
                                                        ' outputs ({})'.format(
                                                            len(output_names), len(outputs)))

                    # Write output files and get profile data.
                    profile_data, _paths = self.save_outputs_and_profile(
                        output_names, outputs, iter, save_intermediate_outputs, do_profile)

                    # generate output text file for each of the inputs.
                    if self.gen_out_file:
                        out_list_file.write(','.join(_paths) + '\n')

        #For generating histogram profile
        output_dtypes = [op.dtype for op in outputs]
        output_array_map = list(zip(output_names, output_dtypes, outputs))

        if self.gen_out_file:
            out_list_file.close()

        res['profile'] = profile_data

        if do_profile:
            qacc_logger.info('Captured TorchScript profile')

        inf_time = time.time() - start_time
        qacc_file_logger.debug("TorchScriptInferenceEngine execution success")
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
        qacc_file_logger.debug("TorchScriptInferenceEngine validation")
        # check the existence of model path and its authenticity
        if not os.path.exists(self.model_path):
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
        qacc_file_logger.debug("TorchScriptInferenceEngine validation success")
        return True
