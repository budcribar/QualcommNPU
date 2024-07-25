# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework, DebuggingAlgorithm, Engine, \
    Android_Architectures, X86_Architectures, \
    X86_windows_Architectures, Qnx_Architectures, Windows_Architectures, Architecture_Target_Types
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError


class AccDebuggerCmdOptions(CmdOptions):

    def __init__(self, engine, args, validate_args=True):
        super().__init__('wrapper', args, engine, validate_args=validate_args)

    def initialize(self):
        """
        type: (List[str]) -> argparse.Namespace

        Parses first cmd line argument to determine which tool to run
        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Options for running the Accuracy Debugger components")

        common_args = self.parser.add_argument_group(
            'Arguments required by both Framework Diagnosis and Inference Engine')
        common_args.add_argument(
            '-f', '--framework', nargs='+', type=str, required=True,
            help='Framework type and version, version is optional. '
            'Currently supported frameworks are [' + ', '.join([f.value for f in Framework]) + ']. '
            'For example, tensorflow 2.10.1')
        common_args.add_argument('-m', '--model_path', type=str, required=True,
                                 help='Path to the model file(s).')
        common_args.add_argument(
            '-i', '--input_tensor', nargs="+", action='append', required=True,
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
        common_args.add_argument('-o', '--output_tensor', type=str, required=True, action='append',
                                 help='Name of the graph\'s specified output tensor(s).')

        inference_args = self.parser.add_argument_group('Arguments required by Inference Engine')
        inference_args.add_argument('-r', '--runtime', type=str.lower, required=True,
                                    help="Runtime to be used for inference.")
        inference_args.add_argument('-l', '--input_list', type=str, required=True,
                                    help="Path to the input list text.")

        verification_args = self.parser.add_argument_group('Arguments required by Verification')
        verification_args.add_argument(
            '--default_verifier', type=str.lower, required=True, nargs='+', action="append",
            help='Default verifier used for verification. The options '
            '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "MeanIOU", "ScaledDiff" are supported. '
            'An optional list of hyperparameters can be appended. For example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0,01. '
            'An optional list of placeholders can be appended. For example: --default_verifier CosineSimilarity param1 1 param2 2. '
            'to use multiple verifiers, add additional --default_verifier CosineSimilarity')

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Verbose printing")
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                              default='working_directory',
                              help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                   'Creates a new directory if the specified working directory does not exitst.')
        optional.add_argument('--output_dirname', type=str, required=False,
                              default='<curr_date_time>',
                              help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(
                                  self.component, self.component) + \
                                   'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--deep_analyzer', type=str, required=False, default=None,
                              choices=['modelDissectionAnalyzer'],
                              help='Deep Analyzer to perform deep analysis')
        optional.add_argument(
            '--golden_output_reference_directory', dest='golden_output_reference_directory',
            type=str, required=False,
            help='Optional parameter to indicate the directory of the golden reference. '
            'When this option provided, framework diagnosis stage is skipped and, in '
            'verification step, outputs from this directory compared with outputs produced '
            'by inference engine step.')
        optional.add_argument(
            '--enable_tensor_inspection', action="store_true", default=False,
            help="Plots graphs like line, scatter, CDF etc., for each layers output. "
            "Additionally, summary sheet will have more details like golden min/max, target min/max etc.,"
        )

        if self.engine == Engine.QNN.value:
            optional.add_argument(
                '--debugging_algorithm', type=str, required=False, default='oneshot-layerwise',
                choices=[
                    DebuggingAlgorithm.cumulative_layerwise.value,
                    DebuggingAlgorithm.oneshot_layerwise.value, DebuggingAlgorithm.layerwise.value
                ], help=
                'Performs network debugging in layerwise, cumulative-layerwise or oneshot-layerwise based on choice.'
            )
            # TODO: Fix the target arch name wos-remote to arm64x-windows once libs and bins are shipped in arm64x arch
            inference_args.add_argument(
                '-a', '--architecture', type=str, required=True,
                choices=Architecture_Target_Types.target_types.value,
                help='Name of the architecture to use for inference engine.')
        else:
            inference_args.add_argument(
                '-a', '--architecture', type=str, required=True, choices=[
                    'aarch64-android', 'x86_64-linux-clang', 'aarch64-android-clang6.0',
                    'wos-remote', 'x86_64-windows-msvc', 'wos'
                ], help='Name of the architecture to use for inference engine.')

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        # Parse version since it is an optional argument that is combined with framework
        parsed_args.framework_version = None
        if len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]
        parsed_args.framework = parsed_args.framework[0]
        if parsed_args.golden_output_reference_directory:
            parsed_args.golden_output_reference_directory = get_absolute_path(
                parsed_args.golden_output_reference_directory)
        if parsed_args.enable_tensor_inspection and self.engine == Engine.QNN.value and parsed_args.debugging_algorithm != 'oneshot-layerwise':
            raise ParameterError(
                "Tensor Inspection is supported only for oneshot-layerwise debugging algorithm.")

        return parsed_args

    def get_all_associated_parsers(self):
        """
        :returns: All the parsers for modules that will be used by wrapper
        """
        parsers_to_be_validated = [self.parser]
        option_classes = []
        if '--golden_output_reference_directory' not in self.args:
            from qti.aisw.accuracy_debugger.lib.options.framework_diagnosis_cmd_options import FrameworkDiagnosisCmdOptions
            option_classes.append(FrameworkDiagnosisCmdOptions(self.args, False))

        if '--debugging_algorithm' in self.args:
            from qti.aisw.accuracy_debugger.lib.options.layerwise_snooping_cmd_options import LayerwiseSnoopingCmdOptions
            option_classes.append(LayerwiseSnoopingCmdOptions(self.args, None, False))

        if '--deep_analyzer' in self.args:
            from qti.aisw.accuracy_debugger.lib.options.accuracy_deep_analyzer_cmd_options import AccuracyDeepAnalyzerCmdOptions
            option_classes.append(AccuracyDeepAnalyzerCmdOptions(self.args, False))

        from qti.aisw.accuracy_debugger.lib.options.verification_cmd_options import VerificationCmdOptions
        from qti.aisw.accuracy_debugger.lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions

        option_classes.append(VerificationCmdOptions(self.args, False))
        option_classes.append(InferenceEngineCmdOptions(self.engine, self.args, False))

        for option_class in option_classes:
            option_class.initialize()
            parsers_to_be_validated.extend(option_class.get_all_associated_parsers())

        return parsers_to_be_validated
