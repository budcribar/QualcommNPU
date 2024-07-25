# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import os
import logging

from lib.utils.nd_errors import get_message, get_warning_message
from lib.utils.nd_exceptions import DeepAnalyzerError


class ReferenceAnalyzer:
    # Reference Code Analyzer class that calls the special route of qnn-net-run to enable reference code for specifc ops
    def __init__(self, args, logger, envToolConfig):
        # type: str
        """
        param verifier:
        param engine:
        param engine_version:
        param engine_path:
        param ndk_path:
        param framework:
        param framework_version:
        param golden_tensor_paths:
        param golden_to_inference_mapping:
        param inference_to_golden_mapping:
        """
        #TODO implement validate_parameters for specific reference_code_analyzer cases.
        def validate_parameters():
            pass

        self.verifier = args.default_verifier
        self.engine = args.engine
        self.engine_version = args.engine_version
        self.engine_path = args.engine_path
        self.framework = args.framework
        self.framework_version = args.framework_version
        self.ndk_path = args.ndk_path
        self.model_path = args.model_path
        self.working_dir = args.working_dir
        self.output_dir = args.output_dir
        self.input_tensor = args.input_tensor
        self.output_tensor = args.output_tensor
        self.input_list = args.input_list
        self.framework_results = args.framework_results
        self.inference_results = args.inference_results
        self.tensor_mapping = {}
        self.tensor_mapping_path = None
        if os.path.exists(args.tensor_mapping):
            self.tensor_mapping_path = args.tensor_mapping
            with open(self.tensor_mapping_path) as tensor_mapping:
                self.tensor_mapping = json.load(tensor_mapping)
        self.graph_struct_path = None
        if os.path.exists(args.graph_struct):
            self.graph_struct_path = args.graph_struct
        self.config_file_path = args.config_file_path
        self.logger = logger
        self.target_device = args.target_device
        self.runtime = args.runtime
        self.architecture = args.architecture
        self.envToolConfig = envToolConfig
        self.deviceId = args.deviceId

    def executeModelwithRefCode(self):
        args = {
            'framework': '{} {}'.format(self.framework, (self.framework_version if self.framework_version else '')),
            'engine_path': self.engine_path,
            'runtime': self.runtime,
            'working_dir': os.path.join(self.output_dir, 'deepAnalyzer', 'refcode_ran_inference_engine'),
            'input_list': self.input_list,
            'deviceId': self.deviceId,
            'target_device': self.target_device,
            'model_path': self.model_path,
            'model_inputs': ' '.join(self.input_tensor),
            'model_outputs': self.output_tensor,
            'target_architecture': self.architecture,
            'ndk_path': self.ndk_path,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else ''),
            'config_file_path': self.config_file_path
        }

        inference_args = (
            ' --framework {args[framework]}'
            ' --engine_path {args[engine_path]}'
            ' --runtime {args[runtime]}'
            ' --working_dir {args[working_dir]}'
            ' --input_list {args[input_list]}'
            ' --deviceId {args[deviceId]}'
            ' --target_device {args[target_device]}'
            ' --model_path {args[model_path]}'
            ' --architecture {args[target_architecture]}'
            ' --ndk_path {args[ndk_path]}'
            ' --qnn_netrun_config_file {args[config_file_path]}'
            ' --input_tensor {args[model_inputs]}'
            ' --output_tensor {args[model_outputs]}'
            '{args[verbose]}'
        ).format(args=args)

        # configs and spawns run_qnn_inference_engine sub-process
        self.logger.debug("Running nd_run_qnn_inference_engine.py with parameters: {}".format(inference_args))
        ret_inference_engine = self.envToolConfig.run_qnn_inference_engine(inference_args.split())
        if ret_inference_engine != 0:
            raise DeepAnalyzerError("Subprocess finished with exit code {}".format(ret_inference_engine))
        self.inference_results=os.path.join(args['working_dir'], 'inference_engine', 'latest', 'output', 'Result_0')

    def validateAccuracy(self):
        """Calls Run verification based on the results generated from reference run
        """

        # Get new accuracies run with golden input, compare to original accuracies
        args = {
            'default_verifier': self.verifier,
            'framework_results': self.framework_results,
            'inference_results': self.inference_results,
            'working_dir': os.path.join(self.output_dir,'deepAnalyzer','refcode_ran_verification'),
            'tensor_mapping': self.tensor_mapping_path,
            'graph_struct': self.graph_struct_path,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else '')
        }

        verification_args = (
            ' --default_verifier {args[default_verifier]}'
            ' --framework_results {args[framework_results]}'
            ' --inference_results {args[inference_results]}'
            ' --working_dir {args[working_dir]}'
            ' --tensor_mapping {args[tensor_mapping]}'
            ' --graph_struct {args[graph_struct]}'
            '{args[verbose]}'
        ).format(args=args)

        self.logger.debug("Running nd_run_verification.py with parameters: {}".format(verification_args))
        ret_verifier = self.envToolConfig.run_verifier(verification_args.split())
        if ret_verifier != 0:
            raise DeepAnalyzerError("Subprocess finished with exit code {}".format(ret_verifier))
