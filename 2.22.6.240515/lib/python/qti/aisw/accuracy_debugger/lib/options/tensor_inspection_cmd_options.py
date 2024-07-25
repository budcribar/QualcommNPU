# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path

import argparse
import os
from datetime import datetime


class TensorInspectionCmdOptions(CmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__('tensor_inspection', args, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to compare given target outputs with reference outputs")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument(
            '--golden_data', type=str, required=True,
            help="Path to Golden/Framework outputs folder."
            "Paths may be absolute, or relative to the working directory.")
        required.add_argument(
            '--target_data', type=str, required=True, help="Path to Target outputs folder."
            "Paths may be absolute, or relative to the working directory.")
        required.add_argument(
            '--verifier', type=str.lower, required=True, nargs='+', action="append",
            help='Verifier used for verification. The options '
            '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "MeanIOU", "ScaledDiff" are supported. '
            'An optional list of hyperparameters can be appended. For example: --verifier rtolatol,rtolmargin,0.01,atolmargin,0,01. '
            'An optional list of placeholders can be appended. For example: --verifier CosineSimilarity param1 1 param2 2. '
            'to use multiple verifiers, add additional --verifier CosineSimilarity')

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument('--data_type', type=str, default="float32",
                              choices=['int8', 'uint8', 'int16', 'uint16',
                                       'float32'], help="DataType of the output tensor.")
        optional.add_argument('--target_encodings', type=str, default=None,
                              help="Path to target encodings json file.")
        optional.add_argument(
            '--working_dir', type=str, required=False,
            default='working_directory',
            help='Working directory for the {} to store temporary files. '.format(self.component) + \
                'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Verbose printing")
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.golden_data = get_absolute_path(parsed_args.golden_data)
        parsed_args.target_data = get_absolute_path(parsed_args.target_data)

        valid_verifier = [
            "rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae",
            "sqnr", "meaniou", "scaleddiff"
        ]
        for verifier in parsed_args.verifier:
            verifier_name = verifier[0].split(',')[0]
            if verifier_name not in valid_verifier:
                raise ParameterError(f"--verifier '{verifier_name}' is not a valid verifier.")

        if parsed_args.target_encodings and not parsed_args.target_encodings.endswith('.json'):
            raise ParameterError("Expected .json for -target_encodings argument")

        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]
