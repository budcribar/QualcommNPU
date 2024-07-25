# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import DeepAnalyzerError
from lib.utils.nd_constants import Engine, Runtime, Framework
from lib.utils.nd_exceptions import ParameterError
from lib.utils.nd_namespace import Namespace
from lib.utils.nd_path_utility import get_absolute_path
from lib.options.cmd_options import CmdOptions

import argparse

class AccuracyDeepAnalyzerCmdOptions(CmdOptions):

    def __init__(self, args):
        super().__init__('deep_analysis', args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to run deep_analyzer."
        )

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-m', '--model_path', type=str, required=True,
                            help='path to original model that needs to be dissected.')
        required.add_argument('--deep_analyzer', type=str, required=True,
                            choices=['modelDissectionAnalyzer'], #'referencecodeAnalyzer' will be supported in future
                            help='Deep Analyzer to perform deep analysis')
        required.add_argument('--default_verifier', type=str.lower, required=True,
                            choices=["rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae", "sqnr", "meaniou"],
                            help='verifier used for verification summary.')
        required.add_argument('--inference_results', type=str, required=True,
                            help='Path to root directory generated from inference engine diagnosis. '
                                'Paths may be absolute, or relative to the working directory.')
        required.add_argument('--framework_results', type=str, required=True,
                            help='Path to root directory generated from inference engine diagnosis. '
                                'Paths may be absolute, or relative to the working directory.')
        required.add_argument('-f', '--framework', nargs='+', type=str, required=True,
                            help='Framework type to be used, followed optionally by framework '
                                'version.')
        required.add_argument('-e', '--engine', nargs='+', type=str, required=True,
                            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
                            help='Name of engine that will be running inference, '
                                'optionally followed by the engine version.')
        required.add_argument('-p', '--engine_path', type=str, required=True,
                                help='Path to the inference engine.')

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument('-r', '--runtime', type=str.lower, default=Runtime.dspv68.value,
                                choices=[r.value for r in Runtime], help="Runtime to be used.")
        optional.add_argument('-t', '--target_device', type=str.lower, default='x86',
                                choices=['x86', 'android', 'linux-embedded'],
                                help='The device that will be running inference.')
        optional.add_argument('-a', '--architecture', type=str.lower, default='x86_64-linux-clang',
                                choices=['x86_64-linux-clang', 'aarch64-android'],
                                help='Name of the architecture to use for inference engine.')
        optional.add_argument('--deviceId', required=False, default=None,
                             help='The serial number of the device to use. If not available, '
                                 'the first in a list of queried devices will be used for validation.')
        optional.add_argument('--result_csv', type=str, required=False,
                                help='Path to the csv summary report comparing the inference vs framework'
                                'Paths may be absolute, or relative to the working directory.'
                                'if not specified, then a --problem_inference_tensor must be specified')
        optional.add_argument('--problem_inference_tensor', type=str, default=None,
                            required='modelDissectionAnalyzer' in self.args and '--result_csv' not in self.args,
                            help='Manually specify problematic tensor name to start partitioner iteration.')
        optional.add_argument('--maximum_dissection_iterations', type=int, default=7,
                            help='Specify maximum number of iterations allowed for modelDissectionAnalyzer.')
        optional.add_argument('--auto_stop_iterations', action='store_true', default=False,
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
        optional.add_argument('--verifier_config', type=str, default=None, help='Path to the verifiers\' config file')
        optional.add_argument('--tensor_mapping', type=str, default=None,
                            help='Path to the file describing the tensor name mapping '
                                'between inference and golden tensors.')
        optional.add_argument('--graph_struct', type=str, default=None, required="modelDissectionAnalyzer" in self.args,
                                help='Path to the model graph structure json.')
        optional.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='Verbose printing')
        optional.add_argument('-n', '--ndk_path', type=str, default=None, required="modelDissectionAnalyzer" in self.args,
                                help='Path to the Android NDK.')
        optional.add_argument('--partition_override', type=str, default=None,
                                help='accepting a .json file that would override the automatic partitioning process.')
        #TODO:Temporary Disabled Reference Code Analyzer for this release.
        # optional.add_argument('--input_list', type=str, required=False, default=None,
        #                         help="Path to the input list text.")
        # optional.add_argument('--input_tensor', nargs='+', type=str, required=False, default=None,
        #                             help='The name, dimension, and raw data of the network input tensor(s) '
        #                                 'specified in the format "input_name" comma-separated-dimensions '
        #                                 'path-to-raw-file, for example: "data" 1,224,224,3 data.raw. '
        #                                 'Note that the quotes should always be included in order to '
        #                                 'handle special characters, spaces, etc. For multiple inputs '
        #                                 'specify multiple --input_tensor on the command line like: '
        #                                 '--input_tensor "data1" 1,224,224,3 data1.raw '
        #                                 '--input_tensor "data2" 1,50,100,3 data2.raw.')
        # optional.add_argument('--output_tensor', type=str, required=False, default=None,
        #                             help='Name of the graph\'s output tensor(s).')
        # optional.add_argument('--config_file_path', type=str, required=False, default=None,
        #                             help="path for the backend extension .conf file, the config.json file is automatically generated")
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        #TODO: make the validate and Update parsed argument as a separate function.

        parsed_args.result_csv = get_absolute_path(parsed_args.result_csv)
        parsed_args.framework_results = get_absolute_path(parsed_args.framework_results)
        parsed_args.inference_results = get_absolute_path(parsed_args.inference_results)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)


        parsed_args.verifier_config = get_absolute_path(parsed_args.verifier_config)
        parsed_args.tensor_mapping = get_absolute_path(parsed_args.tensor_mapping)
        parsed_args.ndk_path = get_absolute_path(parsed_args.ndk_path)
        parsed_args.graph_struct = get_absolute_path(parsed_args.graph_struct)

        #TODO: this section was used for referenceCodeAnalyzer
        # parsed_args.input_list = get_absolute_path(parsed_args.input_list)
        # if parsed_args.input_list != None:
        #     with open(parsed_args.input_list, 'r') as f:
        #         model_inputs = f.readlines()
        #     if len(model_inputs) != 1:
        #         raise DeepAnalyzerError("--input_list parameter only accepts input(s) for one test sample."
        #                                 " Multiple inputs for one test sample must be separated by spaces, not new lines.")

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

        # verify that target_device and architecture align
        arch = parsed_args.architecture
        linux_target, android_target = (parsed_args.target_device == 'x86' or parsed_args.target_device == 'linux_embedded'), parsed_args.target_device == 'android'
        if linux_target and parsed_args.runtime==Runtime.dspv66.value: raise ParameterError("Engine and runtime mismatch.")
        linux_arch = android_arch = None
        if parsed_args.engine == Engine.SNPE.value:
            linux_arch, android_arch = arch == 'x86_64-linux-clang', arch.startswith('aarch64-android-clang')
            if parsed_args.runtime not in ["cpu","dsp","gpu","aip"]:
                raise ParameterError("Engine and runtime mismatch.")
        else:
            linux_arch, android_arch = arch == 'x86_64-linux-clang', arch == 'aarch64-android'
            if parsed_args.runtime not in ["cpu","dsp","dspv66","dspv68","dspv69","dspv73","gpu"]:
                raise ParameterError("Engine and runtime mismatch.")
            dspArchs=[r.value for r in Runtime if r.value.startswith("dsp") and r.value != "dsp"]
            if parsed_args.runtime == "dsp": parsed_args.runtime=max(dspArchs)
        if not ((linux_target and linux_arch) or (android_target and android_arch)):
            raise ParameterError("Target device and architecture mismatch.")


        # model dissection specific arguments
        if parsed_args.deep_analyzer == 'modelDissectionAnalyzer':
            if not (parsed_args.engine == Engine.QNN.value and (parsed_args.framework == Framework.tensorflow.value or parsed_args.framework == Framework.onnx.value)):
                raise ParameterError("Currently only QNN engine and TensorFlow/Onnx framework is supported.")

        # TODO: temporarily disabling referenceCodeAnalyzer
        # elif parsed_args.deep_analyzer == 'referenceCodeAnalyzer':
        #     if not parsed_args.input_tensor:
        #         raise ParameterError("--input_tensor must be set in order to use referenceCodeAnalyzer.")
        #     if not parsed_args.output_tensor:
        #         raise ParameterError("--output_tensor must be set in order to use referenceCodeAnalyzer.")
        #     if not parsed_args.tensor_mapping:
        #         raise ParameterError("--tensor_mapping must be set in order to use referenceCodeAnalyzer.")
        #     if not parsed_args.graph_struct:
        #         raise ParameterError("--graph_struct must be set in order to use referenceCodeAnalyzer.")
        #     if not parsed_args.config_file_path:
        #         raise ParameterError("--config_file_path must be set in order to use referenceCodeAnalyzer.")



        return parsed_args
