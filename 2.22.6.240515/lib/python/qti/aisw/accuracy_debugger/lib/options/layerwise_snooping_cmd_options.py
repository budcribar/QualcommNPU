# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from packaging import version

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Architecture_Target_Types, Engine, Runtime, \
    Android_Architectures, X86_Architectures, \
    Device_type, Qnx_Architectures, Windows_Architectures, X86_windows_Architectures, Aarch64_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import update_layer_options


class LayerwiseSnoopingCmdOptions(CmdOptions):

    def __init__(self, args, snooper='layerwise', validate_args=True):
        if snooper == 'cumulative_layerwise':
            super().__init__('cumulative_layerwise_snooping', args, validate_args=validate_args)
        else:
            super().__init__('layerwise_snooping', args, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to run layerwise and cumulative layerwise snooping.")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-m', '--model_path', type=str, required=True,
                              help='path to original model that needs to be dissected.')
        required.add_argument(
            '--default_verifier', type=str.lower, required=True, nargs='+', action="append",
            help='Default verifier used for verification. The options '
            '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "MeanIOU", "ScaledDiff" are supported. '
            'An optional list of hyperparameters can be appended. For example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0.01 '
            'An optional list of placeholders can be appended. For example: --default_verifier CosineSimilarity param1 1 param2 2. '
            'to use multiple verifiers, add additional --default_verifier CosineSimilarity')
        required.add_argument(
            '--golden_output_reference_directory', '--framework_results',
            dest='golden_output_reference_directory', type=str, required=True,
            help='Path to root directory generated from framework diagnosis. '
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
        required.add_argument('-l', '--input_list', type=str, required=True,
                              help="Path to the input list text.")

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument(
            '-r', '--runtime', type=str.lower, default=Runtime.dspv68.value,
            choices=[r.value for r in Runtime], help='Runtime to be used.Please '
            'use htp runtime for emulation on x86 host')
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
        optional.add_argument('-v', '--verbose', action='store_true', default=False,
                              help='Verbose printing')
        optional.add_argument(
            '--host_device', type=str, required=False, default='x86', choices=['x86', 'x86_64-windows-msvc', 'wos'],
            help='The device that will be running conversion. Set to x86 by default.')
        optional.add_argument(
            '--start_layer', type=str, default=None, required=False,
            help="Extracts the given model from mentioned start layer output name")
        optional.add_argument('--end_layer', type=str, default=None, required=False,
                              help="Extracts the given model from mentioned end layer output name")
        optional.add_argument('--precision', choices=['int8', 'fp16'], default='int8',
                              help='select precision')
        optional.add_argument('--compiler_config', type=str, default=None, required=False,
                              help="Path to the compiler config file.")
        optional.add_argument(
            '-bbw', '--bias_bitwidth', type=int, required=False, default=8, choices=[8, 32],
            help="option to select the bitwidth to use when quantizing the bias. default 8")
        optional.add_argument(
            '-abw', '--act_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the activations. default 8")
        optional.add_argument(
            '-wbw', '--weights_bitwidth', type=int, required=False, default=8, choices=[8], help=
            "option to select the bitwidth to use when quantizing the weights. Only support 8 atm")
        optional.add_argument('-pq', '--param_quantizer', type=str.lower, required=False,
                              default='tf', choices=['tf', 'enhanced', 'adjusted', 'symmetric'],
                              help="Param quantizer algorithm used.")

        optional.add_argument('-qo', '--quantization_overrides', type=str, required=False,
                              default=None, help="Path to quantization overrides json file.")

        optional.add_argument('--act_quantizer', type=str, required=False, default='tf',
                              choices=['tf', 'enhanced', 'adjusted', 'symmetric'],
                              help="Optional parameter to indicate the activation quantizer to use")

        optional.add_argument(
            '--algorithms', type=str, required=False, default=None, help=
            "Use this option to enable new optimization algorithms. Usage is: --algorithms <algo_name1> ... \
                                        The available optimization algorithms are: 'cle ' - Cross layer equalization includes a number of methods for \
                                        equalizing weights and biases across layers in order to rectify imbalances that cause quantization errors."
        )

        optional.add_argument(
            '--ignore_encodings', action="store_true", default=False, help=
            "Use only quantizer generated encodings, ignoring any user or model provided encodings."
        )

        optional.add_argument('--per_channel_quantization', action="store_true", default=False,
                              help="Use per-channel quantization for convolution-based op weights.")
        optional.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional convereter arguments in a string. \
                                          example: --extra_converter_args input_dtype=data float;input_layout=data1 NCHW"
        )
        optional.add_argument(
            '--extra_runtime_args', type=str, required=False, default=None,
            help="additional convereter arguments in a quoted string. \
                                        example: --extra_runtime_args profiling_level=basic;log_level=debug"
        )
        optional.add_argument(
            '--add_layer_outputs', default=[], help="Output layers to be dumped. \
                                    example:1579,232")
        optional.add_argument(
            '--add_layer_types', default=[],
            help='outputs of layer types to be dumped. e.g :Resize,Transpose.\
                                All enabled by default.')
        optional.add_argument(
            '--skip_layer_types', default=[],
            help='comma delimited layer types to skip snooping. e.g :Resize, Transpose')
        optional.add_argument(
            '--skip_layer_outputs', default=[],
            help='comma delimited layer output names to skip debugging. e.g :1171, 1174')
        optional.add_argument('--remote_server', type=str, required=False, default=None,
                              help="ip address of remote machine")
        optional.add_argument('--remote_username', type=str, required=False, default=None,
                              help="username of remote machine")
        optional.add_argument('--remote_password', type=str, required=False, default=None,
                              help="password of remote machine")
        optional.add_argument(
            '-nif', '--use_native_input_files', action="store_true", default=False, required=False,
            help="Specifies that the input files will be parsed in the data type native to the graph.\
                                    If not specified, input files will be parsed in floating point."
        )
        optional.add_argument(
            '-nof', '--use_native_output_files', action="store_true", default=False, required=False,
            help="Specifies that the output files will be generated in the data \
                                    type native to the graph. If not specified, output files will \
                                    be generated in floating point.")

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.result_csv = get_absolute_path(parsed_args.result_csv)
        parsed_args.golden_output_reference_directory = get_absolute_path(
            parsed_args.golden_output_reference_directory)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)
        parsed_args.compiler_config = get_absolute_path(parsed_args.compiler_config)

        if parsed_args.framework[0] == 'onnx':
            onnx_env_available = False
            try:
                import onnx
                onnx_env_available = True
            except:
                pass
            if onnx_env_available and version.parse(onnx.__version__) < version.parse('1.8.0'):
                raise UnsupportedError(
                    "Layerwise and Cumulative Layerwise snooping requires onnx version >= 1.8.0")

        # get engine and engine version if possible
        parsed_args.engine_version = None
        if len(parsed_args.engine) > 2:
            raise ParameterError("Maximum two arguments required for inference engine.")
        elif len(parsed_args.engine) == 2:
            parsed_args.engine_version = parsed_args.engine[1]

        parsed_args.engine = parsed_args.engine[0]
        if parsed_args.engine != 'QNN':
            raise UnsupportedError("Layerwise and Cumulative Layerwise snooping supports only QNN")

        # get framework and framework version if possible
        parsed_args.framework_version = None
        if len(parsed_args.framework) > 2:
            raise ParameterError("Maximum two arguments required for framework.")
        elif len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]

        parsed_args.framework = parsed_args.framework[0]
        if parsed_args.framework != 'onnx':
            raise UnsupportedError("Layerwise snooping supports only onnx framework")

        if parsed_args.runtime == Runtime.htp.value and parsed_args.architecture != 'x86_64-linux-clang':
            raise ParameterError("Runtime htp supports only x86_64-linux-clang architecture")

        # verify that target_device and architecture align
        if hasattr(parsed_args, 'architecture'):
            arch = parsed_args.architecture
            if (arch in [a.value for a in Android_Architectures]):
                target_device = Device_type.android.value
            elif (arch in [a.value for a in X86_Architectures]):
                target_device = Device_type.x86.value
            elif (arch in [a.value for a in X86_windows_Architectures]):
                target_device = Device_type.x86_64_windows_msvc.value
            elif (arch in [a.value for a in Aarch64_windows_Architectures]):
                target_device = Device_type.wos.value
            elif (arch in [a.value for a in Qnx_Architectures]):
                target_device = Device_type.qnx.value
            elif (arch in [a.value for a in Windows_Architectures]):
                target_device = Device_type.wos_remote.value
            else:
                raise ParameterError("Invalid architecture.")
            parsed_args.target_device = target_device
        linux_target, android_target, x86_64_windows_msvc_target, wos_target = (parsed_args.target_device == 'x86'
                                        or parsed_args.target_device
                                        == 'linux_embedded'), parsed_args.target_device == 'android',\
            parsed_args.target_device == Device_type.x86_64_windows_msvc.value, parsed_args.target_device == Device_type.wos.value
        if linux_target and parsed_args.runtime == Runtime.dspv66.value:
            raise ParameterError("Engine and runtime mismatch.")

        if arch == "aarch64-qnx" and parsed_args.runtime not in ["dspv68", "dspv73", "dspv75"]:
            raise ParameterError("Invalid runtime for aarch64-qnx")
        if arch == "aarch64-qnx" and (parsed_args.remote_server is None
                                      or parsed_args.remote_username is None):
            raise ParameterError(
                "Remote server and username options are required for aarch64-qnx architecture.")

        linux_arch = android_arch = x86_64_windows_msvc_arch = wos_arch = None
        if parsed_args.engine == Engine.SNPE.value:
            linux_arch, android_arch = arch == 'x86_64-linux-clang', arch.startswith(
                'aarch64-android-clang')
            if parsed_args.runtime not in ["cpu", "dsp", "gpu", "aic"]:
                raise ParameterError("Engine and runtime mismatch.")
        else:
            linux_arch, android_arch, x86_64_windows_msvc_arch, wos_arch = arch == 'x86_64-linux-clang', arch == 'aarch64-android', arch == 'x86_64-windows-msvc', arch == 'wos'
            if parsed_args.runtime not in [
                    "cpu", "dsp", "dspv66", "dspv68", "dspv69", "dspv73", "dspv75", "gpu", "aic",
                    "htp"
            ]:
                raise ParameterError("Engine and runtime mismatch.")
            dspArchs = [r.value for r in Runtime if r.value.startswith("dsp") and r.value != "dsp"]
            if parsed_args.runtime == "dsp": parsed_args.runtime = max(dspArchs)
        if not ((linux_target and linux_arch) or
                (android_target and android_arch)
                or (x86_64_windows_msvc_target and x86_64_windows_msvc_arch) or (wos_target and wos_arch)) and arch != "aarch64-qnx":
            raise ParameterError("Target device and architecture mismatch.")

        if parsed_args.add_layer_types:
            parsed_args.add_layer_types = parsed_args.add_layer_types.split(',')

        if parsed_args.skip_layer_types:
            parsed_args.skip_layer_types = parsed_args.skip_layer_types.split(',')

        if parsed_args.add_layer_types or parsed_args.skip_layer_types or parsed_args.skip_layer_outputs:
            parsed_args.add_layer_outputs = update_layer_options(parsed_args)

        supported_verifiers = [
            "rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae",
            "sqnr", "meaniou", "scaleddiff"
        ]
        for verifier in parsed_args.default_verifier:
            verifier_name = verifier[0].split(',')[0]
            if verifier_name not in supported_verifiers:
                raise ParameterError(
                    f"--default_verifier '{verifier_name}' is not a supported verifier.")

        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]
