# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError

import argparse
import os
import json
from datetime import datetime


class CompareEncodingsCmdOptions(CmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__('compare_encodings', args, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to compare QNN encodings with AIMET encodings")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('--input', type=str, required=True,
                              help='Path to either QNN model net json or SNPE dlc')
        required.add_argument('--aimet_encodings_json', type=str, required=True,
                              help='Path to AIMET encodings json file')

        optional = self.parser.add_argument_group('optional arguments')
        # default value for precision is set to 17 since that is the max decimal places used in the encodings
        optional.add_argument(
            '--precision', type=int, required=False, default=17,
            help='number of decimal places upto which comparison will be done (default: 17)')
        # bias comes under params section
        optional.add_argument('--params_only', required=False, action="store_true", default=False,
                              help='Compare only params in the encodings')
        optional.add_argument('--activations_only', required=False, action="store_true",
                              default=False, help='Compare only activations in the encodings')
        optional.add_argument('--specific_node', type=str, required=False, default=None,
                              help='Display encoding difference for given node name')
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
        if not (parsed_args.input.endswith('.json') or parsed_args.input.endswith('.dlc')):
            raise ParameterError("Expected json/dlc file for --input argument")

        if not parsed_args.aimet_encodings_json.endswith('.json'):
            raise ParameterError("Expected json file for --aimet_encodings_json argument")

        if not os.path.exists(parsed_args.input):
            raise ParameterError(f"{parsed_args.input} does not exist!")

        if not os.path.exists(parsed_args.aimet_encodings_json):
            raise ParameterError(f"{parsed_args.aimet_encodings_json} does not exist!")

        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]