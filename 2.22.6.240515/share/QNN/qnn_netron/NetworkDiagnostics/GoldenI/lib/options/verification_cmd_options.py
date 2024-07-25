# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError
from lib.utils.nd_logger import setup_logger
from lib.utils.nd_symlink import symlink
from lib.verifier.nd_verification import Verification
from lib.utils.nd_path_utility import get_absolute_path
from lib.options.cmd_options import CmdOptions
from lib.utils.nd_exceptions import ParameterError

import argparse
import os
import json
from datetime import datetime


class VerificationCmdOptions(CmdOptions):

    def __init__(self, args):
        super().__init__('verification', args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to run verification."
        )

        # Workaround to list required arguments before optional arguments
        self.parser._action_groups.pop()

        required = self.parser.add_argument_group('required arguments')

        # default_verifier will verify in nd_verifier_factory.py
        required.add_argument('--default_verifier', type=str.lower, required=True, nargs='+', action="append",
                            help='Default verifier used for verification. The options '
                                '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "MeanIOU", "ScaledDiff" are supported. '
                                'An optional list of hyperparameters can be appended. For example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0,01'
                                'to use multiple verifiers, add additional --default_verifier CosineSimilarity')
        required.add_argument('--framework_results', type=str, required=True,
                            help="Path to root directory generated from framework diagnosis. "
                                "Paths may be absolute, or relative to the working directory.")
        required.add_argument('--inference_results', type=str, required=True,
                            help="Path to root directory generated from inference engine diagnosis. "
                                "Paths may be absolute, or relative to the working directory.")

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument('--tensor_mapping', type=str, required=False, default=None,
                            help='Path to the file describing the tensor name mapping '
                                'between inference and golden tensors.'
                                'can be generated with nd_run_{engine}_inference_engine')
        optional.add_argument('--verifier_config', type=str, default=None, help='Path to the verifiers\' config file')
        optional.add_argument('--graph_struct', type=str, default=None,
                            help='Path to the inference graph structure .json file.')
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                            help="Verbose printing")
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--layout', type=str.upper, required=False, choices=['NHWC', 'NCHW'],
                            default='NHWC', help="The data layout used by the framework.")

        tensor_mapping = self.parser.add_argument_group('arguments for generating a new tensor_mapping.json')

        tensor_mapping.add_argument('-m', '--model_path', type=str, required=False,
                                  help='path to original model for tensor_mapping uses here.')
        tensor_mapping.add_argument('-e', '--engine', nargs='+', type=str, required=False, default=None,
                                  metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
                                  help='Name of engine that will be running inference, '
                                      'optionally followed by the engine version. Used here for tensor_mapping.')
        tensor_mapping.add_argument('-f', '--framework', nargs='+', type=str, default=None, required=False,
                                  help="Framework type to be used, followed optionally by framework "
                                      "version. Used here for tensor_mapping.")
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        valid_verifier=["rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae", "sqnr", "meaniou", "scaleddiff"]

        for verifier in parsed_args.default_verifier:
            if verifier[0] not in valid_verifier:
                raise ParameterError("--default_verifier invalid verifier error")
        parsed_args.verify_types = parsed_args.default_verifier
        parsed_args.framework_results = get_absolute_path(parsed_args.framework_results)
        parsed_args.inference_results = get_absolute_path(parsed_args.inference_results)
        parsed_args.tensor_mapping = get_absolute_path(parsed_args.tensor_mapping)
        parsed_args.verifier_config = get_absolute_path(parsed_args.verifier_config)
        parsed_args.graph_struct = get_absolute_path(parsed_args.graph_struct)

        #get framework and framework version
        parsed_args.framework_version = None
        if parsed_args.framework is not None:
            if len(parsed_args.framework) > 2:
                raise ParameterError("Maximum two arguments required for framework.")
            elif len(parsed_args.framework) == 2:
                parsed_args.framework_version = parsed_args.framework[1]
            parsed_args.framework = parsed_args.framework[0]

        #get engine and engine version
        parsed_args.engine_version = None
        if parsed_args.engine is not None:
            if len(parsed_args.engine) > 2:
                raise ParameterError("Maximum two arguments required for inference engine.")
            elif len(parsed_args.engine) == 2:
                parsed_args.engine_version = parsed_args.engine[1]
            parsed_args.engine = parsed_args.engine[0]

        return parsed_args
