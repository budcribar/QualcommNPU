# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.options.cmd_options import CmdOptions
from lib.utils.nd_constants import Engine

import argparse
import os
from datetime import datetime


class WrapperCmdOptions(CmdOptions):

    def __init__(self, args):
        super().__init__('wrapper', args)

    def initialize(self):
        """
        type: (List[str]) -> argparse.Namespace

        Parses first cmd line argument to determine which tool to run
        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script that runs Framework Diagnosis, Inference Engine and Verification "
                        "consecutively."
        )

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-e', '--engine', type=str, required=True, choices=['QNN','SNPE'],
                            help='Engine type, optionally followed by the engine version.')
        required.add_argument('-f', '--framework', nargs='+', type=str, required=True,
                            help='Framework type and version, version is optional. For example '
                                '"caffe2 0.8.0".')
        required.add_argument('-m', '--model_path', type=str, required=True,
                            help='Path to the model file(s).')
        required.add_argument('-i', '--input_tensor', nargs="+", action='append', required=True,
                            help='The name, dimensions, raw data, and optionally data type of the '
                                'network input tensor(s) specified'
                                'in the format "input_name" comma-separated-dimensions '
                                'path-to-raw-file, '
                                'for example: "data" 1,224,224,3 data.raw float32. Note that the '
                                'quotes should always be included in order to handle special '
                                'characters, spaces, etc. For multiple inputs specify multiple '
                                '--input_tensor on the command line like: --input_tensor "data1" '
                                '1,224,224,3 data1.raw --input_tensor "data2" 1,50,100,3 data2.raw '
                                'float32.')
        required.add_argument('-o', '--output_tensor', type=str, required=True, action='append',
                            help='Name of the graph\'s specified output tensor(s).')
        required.add_argument('-t', '--target_device', type=str, required=True, choices=['x86', 'android'],
                            help='The device that will be running inference.')
        required.add_argument('-r', '--runtime', type=str.lower, required=True,
                                    help="Runtime to be used for inference.")
        required.add_argument('--default_verifier', type=str.lower, required=True, nargs='+',
                            choices=["rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae", "sqnr", "meaniou", "scaleddiff"],
                            help='Default verifier used for verification. The options '
                                '["RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "MeanIOU", "ScaledDiff"] are supported. ')
        required.add_argument('-a', '--architecture', type=str, required=True,
                                choices=['aarch64-android', 'x86_64-linux-clang','aarch64-android-clang6.0'],
                                help='Name of the architecture to use for inference engine.')
        required.add_argument('-p', '--engine_path', type=str, required=True,
                                help="Path to the inference engine.")
        required.add_argument('-l', '--input_list', type=str, required=True,
                                help="Path to the input list text.")

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                            help="Verbose printing")
        optional.add_argument('-n', '--ndk_path', type=str, required='QNN' in self.args,
                                help="Path to the Android NDK.")
        optional.add_argument('--deviceId', required=False, default=None,
                             help='The serial number of the device to use. If not available, '
                                 'the first in a list of queried devices will be used for validation.')
        optional.add_argument('--host_device', type=str, required=False, default='x86', choices=['x86'],
                            help='The device that will be running conversion. Set to x86 by default.')
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                            default='working_directory',
                            help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                'Creates a new directory if the specified working directory does not exitst.')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--verifier_config', type=str, default=None,
                            help='Path to the verifiers\' config files')
        optional.add_argument('--tensor_mapping', type=str, default=None,
                            help='Path to the file describing the tensor name mapping '
                                'between inference and golden tensors.')
        optional.add_argument('--graph_struct', type=str, default=None,
                            help='Path to the inference graph structure .json file.')
        optional.add_argument('--layout', type=str, required=False, choices=['NHWC', 'NCHW'],
                            default='NHWC', help="The data layout used by the framework.")
        optional.add_argument( '--engine_version', type=str, required=False,
                            help='engine version, will retrieve the latest available if not specified')

        optional.add_argument('--deep_analyzer', type=str, required=False, default=None,
                            choices= ['modelDissectionAnalyzer'],
                            help='Deep Analyzer to perform deep analysis')
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        # Parse version since it is an optional argument that is combined with framework
        parsed_args.framework_version = None
        if len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]
        parsed_args.framework = parsed_args.framework[0]
        return parsed_args
