# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeepAnalyzerError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Device_type, Engine, Runtime, Framework, Android_Architectures, X86_Architectures, Architecture_Target_Types
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions

import argparse


class AccuracyDeepAnalyzerCmdOptions(CmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__('deep_analysis', args, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run deep_analyzer.")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-m', '--model_path', type=str, required=True,
                              help='path to original model that needs to be dissected.')
        required.add_argument(
            '--deep_analyzer',
            type=str,
            required=True,
            choices=['modelDissectionAnalyzer'
                     ],  #'referencecodeAnalyzer' will be supported in future
            help='Deep Analyzer to perform deep analysis')
        required.add_argument(
            '--default_verifier', type=str.lower, required=True, choices=[
                "rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae",
                "sqnr", "meaniou"
            ], help='verifier used for verification summary.')
        required.add_argument(
            '--inference_results', type=str, required=True,
            help='Path to root directory generated from inference engine diagnosis. '
            'Paths may be absolute, or relative to the working directory.')
        required.add_argument(
            '--framework_results', type=str, required=True,
            help='Path to root directory generated from inference engine diagnosis. '
            'Paths may be absolute, or relative to the working directory.')
        required.add_argument(
            '-f', '--framework', nargs='+', type=str, required=True,
            help='Framework type to be used, followed optionally by framework '
            'version.')
        required.add_argument(
            '-e', '--engine', nargs='+', type=str, required=True,
            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
            help='Name of engine that will be running inference, '
            'optionally followed by the engine version.')
        required.add_argument('-p', '--engine_path', type=str, required=True,
                              help='Path to the inference engine.')

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument('-r', '--runtime', type=str.lower, default=Runtime.dspv68.value,
                              choices=[r.value for r in Runtime], help="Runtime to be used.")
        optional.add_argument('-a', '--architecture', type=str.lower, default='x86_64-linux-clang',
                              choices=Architecture_Target_Types.target_types.value,
                              help='Name of the architecture to use for inference engine.')
        optional.add_argument(
            '--deviceId', required=False, default=None,
            help='The serial number of the device to use. If not available, '
            'the first in a list of queried devices will be used for validation.')
        optional.add_argument(
            '--result_csv', type=str, required=False,
            help='Path to the csv summary report comparing the inference vs framework'
            'Paths may be absolute, or relative to the working directory.'
            'if not specified, then a --problem_inference_tensor must be specified')
        optional.add_argument(
            '--problem_inference_tensor', type=str, default=None,
            required='modelDissectionAnalyzer' in self.args and '--result_csv' not in self.args,
            help='Manually specify problematic tensor name to start partitioner iteration.')
        optional.add_argument(
            '--maximum_dissection_iterations', type=int, default=7,
            help='Specify maximum number of iterations allowed for modelDissectionAnalyzer.')
        optional.add_argument(
            '--auto_stop_iterations', action='store_true', default=False,
            help='Automatically stop modelDissectionAnalyzer dissection iteration.')
        optional.add_argument('--verifier_threshold', type=float, default=None,
                              help='Verifier threshold for problematic tensor to be chosen.')
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist.')
        optional.add_argument('--verifier_config', type=str, default=None,
                              help='Path to the verifiers\' config file')
        optional.add_argument(
            '--tensor_mapping', type=str, default=None,
            help='Path to the file describing the tensor name mapping '
            'between inference and golden tensors.')
        optional.add_argument('--graph_struct', type=str, default=None,
                              required="modelDissectionAnalyzer" in self.args,
                              help='Path to the model graph structure json.')
        optional.add_argument('-v', '--verbose', action='store_true', default=False,
                              help='Verbose printing')
        optional.add_argument(
            '--partition_override', type=str, default=None,
            help='accepting a .json file that would override the automatic partitioning process.')
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.result_csv = get_absolute_path(parsed_args.result_csv)
        parsed_args.framework_results = get_absolute_path(parsed_args.framework_results)
        parsed_args.inference_results = get_absolute_path(parsed_args.inference_results)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)

        parsed_args.verifier_config = get_absolute_path(parsed_args.verifier_config)
        parsed_args.tensor_mapping = get_absolute_path(parsed_args.tensor_mapping)
        parsed_args.graph_struct = get_absolute_path(parsed_args.graph_struct)

        # get engine and engine version if possible
        parsed_args.engine_version = None
        if len(parsed_args.engine) > 2:
            raise ParameterError("Maximum two arguments required for inference engine.")
        elif len(parsed_args.engine) == 2:
            parsed_args.engine_version = parsed_args.engine[1]

        parsed_args.engine = parsed_args.engine[0]

        # get framework and framework version if possible
        parsed_args.framework_version = None
        if len(parsed_args.framework) > 2:
            raise ParameterError("Maximum two arguments required for framework.")
        elif len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]

        parsed_args.framework = parsed_args.framework[0]

        # verify that runtime and architecture align
        arch = parsed_args.architecture

        if (arch in [a.value for a in Android_Architectures]):
            target_device = Device_type.android.value
        elif (arch in [a.value for a in X86_Architectures]):
            target_device = Device_type.x86.value
        else:
            raise ParameterError("Invalid architecture.")
        parsed_args.target_device = target_device

        is_linux_target = target_device == Device_type.x86.value
        if is_linux_target and parsed_args.runtime == Runtime.dspv66.value:
            raise ParameterError("Engine and runtime mismatch.")
        if parsed_args.engine == Engine.SNPE.value:
            if parsed_args.runtime not in ["cpu", "dsp", "gpu", "aip"]:
                raise ParameterError("Engine and runtime mismatch.")
        else:
            if parsed_args.runtime not in [
                    "cpu", "dsp", "dspv66", "dspv68", "dspv69", "dspv73", "dspv75", "gpu"
            ]:
                raise ParameterError("Engine and runtime mismatch.")

            # set default dsp version if 'dsp' is selected as runtime (NOTE: exclude dspv75 as default for now)
            dspArchs = [
                r.value for r in Runtime
                if r.value.startswith("dsp") and r.value != "dsp" and r.value != "dspv75"
            ]
            default_dsp = max(dspArchs)
            if parsed_args.runtime == "dsp": parsed_args.runtime = default_dsp

        # model dissection specific arguments
        if parsed_args.deep_analyzer == 'modelDissectionAnalyzer':
            if not (parsed_args.engine == Engine.QNN.value and
                    (parsed_args.framework == Framework.tensorflow.value
                     or parsed_args.framework == Framework.onnx.value)):
                raise ParameterError(
                    "Currently only QNN engine and TensorFlow/Onnx framework is supported.")

        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]
