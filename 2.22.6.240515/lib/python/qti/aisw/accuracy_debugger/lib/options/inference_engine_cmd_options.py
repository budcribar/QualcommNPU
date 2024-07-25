# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import os
import json

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Runtime, Android_Architectures, X86_Architectures, \
    X86_windows_Architectures, Device_type, Architecture_Target_Types, Qnx_Architectures, Windows_Architectures, \
    Aarch64_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, format_args
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import update_layer_options


class InferenceEngineCmdOptions(CmdOptions):

    def __init__(self, engine, args, validate_args=True):
        super().__init__('inference_engine', args, engine, validate_args=validate_args)

    def _get_engine(self, engine):
        if engine == Engine.SNPE.value:
            return True, False, False
        elif engine == Engine.QNN.value:
            return False, True, False
        elif engine == Engine.ANN.value:
            return False, False, True
        else:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND")(engine))

    def initialize(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run QNN inference engine.")

        snpe, qnn, _ = self._get_engine(self.engine)

        core = self.parser.add_argument_group('Core Arguments')

        core.add_argument(
            '--stage', type=str.lower, required=False, choices=['source', 'converted', 'compiled'],
            default='source', help='Specifies the starting stage in the Golden-I pipeline.'
            'Source: starting with a source framework.'
            'Converted: starting with a model\'s .cpp and .bin files.'
            'Compiled: starting with a model\'s .so binary')

        core.add_argument('-l', '--input_list', type=str, required=True,
                          help="Path to the input list text.")
        if qnn:
            core.add_argument(
                '-r', '--runtime', type=str.lower, required=True,
                choices=[r.value for r in Runtime], help='Runtime to be used. Please'
                'use htp runtime for emulation on x86 host')
            core.add_argument('-a', '--architecture', type=str, required=True,
                              choices=Architecture_Target_Types.target_types.value,
                              help='Name of the architecture to use for inference engine.')

        if snpe:
            core.add_argument('-r', '--runtime', type=str.lower, required=True,
                              choices=[r.value for r in Runtime if r.value not in ['aic', 'htp']],
                              help="Runtime to be used.")
            core.add_argument('-a', '--architecture', type=str, required=True,
                              choices=Architecture_Target_Types.target_types.value,
                              help='Name of the architecture to use for inference engine.')

        source_stage = self.parser.add_argument_group('Arguments required for SOURCE stage')

        source_stage.add_argument(
            '-i', '--input_tensor', nargs='+', action='append', required=False,
            help='The name, dimension, and raw data of the network input tensor(s) '
            'specified in the format "input_name" comma-separated-dimensions '
            'path-to-raw-file, for example: "data" 1,224,224,3 data.raw. '
            'Note that the quotes should always be included in order to '
            'handle special characters, spaces, etc. For multiple inputs '
            'specify multiple --input_tensor on the command line like: '
            '--input_tensor "data1" 1,224,224,3 data1.raw '
            '--input_tensor "data2" 1,50,100,3 data2.raw.')

        source_stage.add_argument('-o', '--output_tensor', type=str, required=False,
                                  action='append', help='Name of the graph\'s output tensor(s).')

        source_stage.add_argument('-m', '--model_path', type=str, default=None, required=False,
                                  help="Path to the model file(s).")
        source_stage.add_argument(
            '-f', '--framework', nargs='+', type=str.lower, default=None, required=False,
            help="Framework type to be used, followed optionally by framework "
            "version.")

        if snpe:
            notSource = self.parser.add_argument_group(
                'Arguments required for CONVERTED or COMPILED stage')
            notSource.add_argument(
                '--static_model', type=str, required='--stage' in self.args
                and ('compiled' in self.args or 'converted' in self.args), default=None,
                help='Path to the converted model.')
        elif qnn:
            converted_stage = self.parser.add_argument_group(
                'Arguments required for CONVERTED stage')
            compiled_stage = self.parser.add_argument_group('Arguments required for COMPILED stage')
            converted_stage.add_argument(
                '-qmcpp', '--qnn_model_cpp_path', type=str, required='--stage' in self.args
                and 'converted' in self.args, help="Path to the qnn model .cpp file")

            converted_stage.add_argument('-qmbin', '--qnn_model_bin_path', type=str, required=False,
                                         help="Path to the qnn model .bin file")

            compiled_stage.add_argument('-qmb', '--qnn_model_binary_path', type=str,
                                        required='--stage' in self.args and 'compiled' in self.args,
                                        help="Path to the qnn model .so binary.")

        optional = self.parser.add_argument_group('Optional Arguments')

        optional.add_argument('-p', '--engine_path', type=str, required=False,
                              help="Path to the inference engine.")

        optional.add_argument(
            '-e', '--engine', nargs='+', type=str, required=False, default=None,
            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
            help='Name of engine that will be running inference, '
            'optionally followed by the engine version. Used here for tensor_mapping.')

        optional.add_argument(
            '--deviceId', required=False, default=None,
            help='The serial number of the device to use. If not available, '
            'the first in a list of queried devices will be used for validation.')

        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Verbose printing")

        optional.add_argument(
            '--host_device', type=str, required=False, default='x86',
            choices=['x86', 'x86_64-windows-msvc', 'wos'],
            help='The device that will be running conversion. Set to x86 by default.')

        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{} .'.format(self.component,self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')

        optional.add_argument('--debug_mode_off', dest="debug_mode", action="store_false",
                              required=False, help="Specifies if wish to turn off debug_mode mode.")

        optional.set_defaults(debug_mode=True)

        optional.add_argument(
            '-bbw', '--bias_bitwidth', type=int, required=False, default=8, choices=[8, 32],
            help="option to select the bitwidth to use when quantizing the bias. default 8")
        optional.add_argument(
            '-abw', '--act_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the activations. default 8")
        optional.add_argument(
            '-wbw', '--weights_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the weights. default 8")
        optional.add_argument(
            '--float_bitwidth', type=int, required=False, default=32, choices=[32, 16],
            help="Use the --float_bitwidth option to select the bitwidth to use when using float \
                  for parameters(weights/bias) and activations for all ops or specific \
                  Op (via encodings) selected through encoding, either 32 (default) or 16.")
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
        optional.add_argument('-qo', '--quantization_overrides', type=str, required=False,
                              default=None, help="Path to quantization overrides json file.")
        optional.add_argument(
            '--golden_output_reference_directory', '--golden_dir_for_mapping',
            dest='golden_output_reference_directory', type=str, required=False, default=None,
            help="Optional parameter to indicate the directory of the goldens, \
                it's used for tensor mapping without framework.")
        optional.add_argument('-mn', '--model_name', type=str, required=False, default="model",
                              help='Name of the desired output sdk specific model')
        optional.add_argument(
            '--args_config', type=str, required=False,
            help="Path to a config file with arguments.  This can be used to feed arguments to "
            "the AccuracyDebugger as an alternative to supplying them on the command line.")

        optional.add_argument('--print_version', type=bool, required=False, default=False,
                              help=f"Print the {self.engine} SDK version alongside the output.")
        optional.add_argument(
            '--perf_profile', type=str.lower, required=False, default="balanced", choices=[
                'low_balanced', 'balanced', 'high_performance', 'sustained_high_performance',
                'burst', 'low_power_saver', 'power_saver', 'high_power_saver', 'system_settings'
            ])
        optional.add_argument('--offline_prepare', action="store_true", default=False,
                              help=f"Use offline prepare to run {self.engine} model.")
        optional.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional converter arguments in a quoted string. \
                 example: --extra_converter_args 'input_dtype=data float;input_layout=data1 NCHW'")

        optional.add_argument(
            '--extra_runtime_args', type=str, required=False, default=None,
            help="additional net runner arguments in a quoted string. \
                                                    example: --extra_runtime_args 'arg1=value1;arg2=value2'"
        )

        optional.add_argument('--remote_server', type=str, required=False, default=None,
                              help="ip address of remote machine")
        optional.add_argument('--remote_username', type=str, required=False, default=None,
                              help="username of remote machine")
        optional.add_argument('--remote_password', type=str, required=False, default=None,
                              help="password of remote machine")

        if snpe:
            optional.add_argument('--profiling_level', type=str.lower, required=False, default=None,
                                  choices=['off', 'basic', 'moderate', 'detailed',
                                           'linting'], help="Enables profiling and sets its level.")
            optional.add_argument(
                '--extra_quantizer_args', type=str, required=False, default=None,
                help="additional dlc quantizer arguments in a quoted string. \
                     example: --extra_runtime_args 'arg1=value1;arg2=value2'")

            optional.add_argument(
                '--fine_grain_mode', type=str, required=False, default=None,
                help='Path to the model golden outputs required to run inference '
                'engine using fine-grain mode.')

            optional.add_argument(
                '--no_weight_quantization', action="store_true", default=False, help=
                "Generate and add the fixed-point encoding metadata but keep the weights in floating point"
            )
            optional.add_argument(
                '--use_symmetric_quantize_weights', action="store_true", default=False,
                help="Use the symmetric quantizer feature when quantizing the weights of the model")

            optional.add_argument(
                '--use_enhanced_quantizer', action="store_true", default=False,
                help="Use the enhanced quantizer feature when quantizing the model")
            optional.add_argument('--htp_socs', type=str, default="sm8350",
                                  help="Specify SoC to generate HTP Offline Cache for.")

            optional.add_argument(
                '--use_adjusted_weights_quantizer', action="store_true", default=False,
                help="Use the adjusted tf quantizer for quantizing the weights only")
            optional.add_argument(
                '--override_params', action="store_true", default=False, help=
                "Use this option to override quantization parameters when quantization was provided from \
                the original source framework")
        elif qnn:
            optional.add_argument('--profiling_level', type=str.lower, required=False, default=None,
                                  choices=['basic', 'detailed',
                                           'backend'], help="Enables profiling and sets its level.")

            optional.add_argument('--lib_name', type=str, required=False, default=None,
                                  help='Name to use for model library (.so file or .dll file)')

            optional.add_argument(
                '-bd', '--binaries_dir', type=str, required=False, default='qnn_model_binaries',
                help="Directory to which to save model binaries, if they don't yet exist.")

            optional.add_argument('-pq', '--param_quantizer', type=str.lower, required=False,
                                  default='tf', choices=['tf', 'enhanced', 'adjusted', 'symmetric'],
                                  help="Param quantizer algorithm used.")

            optional.add_argument(
                '--act_quantizer', type=str, required=False, default='tf',
                choices=['tf', 'enhanced', 'adjusted', 'symmetric'],
                help="Optional parameter to indicate the activation quantizer to use")

            optional.add_argument(
                '--act_quantizer_calibration', type=str.lower, required=False, default=None,
                choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
                help="Specify which quantization calibration method to use for activations. \
                                        This option has to be paired with --act_quantizer_schema.")

            optional.add_argument(
                '--param_quantizer_calibration', type=str.lower, required=False, default=None,
                choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
                help="Specify which quantization calibration method to use for parameters.\
                                        This option has to be paired with --param_quantizer_schema."
            )

            optional.add_argument(
                '--act_quantizer_schema', type=str.lower, required=False, default='asymmetric',
                choices=['asymmetric', 'symmetric', 'unsignedsymmetric'],
                help="Specify which quantization schema to use for activations. \
                                        Can not be used together with act_quantizer.")

            optional.add_argument(
                '--param_quantizer_schema', type=str.lower, required=False, default='asymmetric',
                choices=['asymmetric', 'symmetric', 'unsignedsymmetric'],
                help="Specify which quantization schema to use for parameters. \
                                        Can not be used together with param_quantizer.")

            optional.add_argument('--percentile_calibration_value', type=float, required=False,
                                  default=99.99, help="Value must lie between 90-100")

            optional.add_argument(
                '-fbw', '--float_bias_bitwidth', type=int, required=False, default=32,
                choices=[16, 32],
                help="option to select the bitwidth to use when biases are in float. default 32")
            optional.add_argument(
                '-rqs', '--restrict_quantization_steps', type=str, required=False,
                help="ENCODING_MIN, ENCODING_MAX \
                    Specifies the number of steps to use for computing quantization encodings \
                    such that scale = (max - min) / number of quantization steps.\
                    The option should be passed as a space separated pair of hexadecimal string \
                    minimum and maximum values. i.e. --restrict_quantization_steps 'MIN MAX'. \
                    Please note that this is a hexadecimal string literal and not a signed \
                    integer, to supply a negative value an explicit minus sign is required. \
                    E.g.--restrict_quantization_steps '-0x80 0x7F' indicates an example 8 bit \
                    range, --restrict_quantization_steps '-0x8000 0x7F7F' indicates an example 16 \
                    bit range.")
            optional.add_argument(
                '--algorithms', type=str, required=False, default=None,
                help="Use this option to enable new optimization algorithms. Usage is: \
                --algorithms <algo_name1> ... \
                The available optimization algorithms are: 'cle ' - Cross layer equalization \
                includes a number of methods for equalizing weights and biases across layers \
                in order to rectify imbalances that cause quantization errors.")

            optional.add_argument(
                '--ignore_encodings', action="store_true", default=False,
                help="Use only quantizer generated encodings, ignoring any user or model\
                      provided encodings.")

            optional.add_argument(
                '--per_channel_quantization', action="store_true", default=False,
                help="Use per-channel quantization for convolution-based op weights.")

            optional.add_argument('--log_level', type=str.lower, required=False, default=None,
                                  choices=['error', 'warn', 'info', 'debug',
                                           'verbose'], help="Enable verbose logging.")

            optional.add_argument(
                '--qnn_model_net_json', type=str, required=False, help=
                "Path to the qnn model net json. Only necessary if it's being run from the converted stage. \
                                        It has information about what structure the data is in within the framework_diagnosis and inference_engine \
                                        steps. This file is required to generate the model_graph_struct.json file which is used by the \
                                        verification stage.")

            optional.add_argument(
                '--qnn_netrun_config_file', type=str, required=False, default=None,
                help="allow backend_extention features to be applied during qnn-net-run")

            optional.add_argument('--compiler_config', type=str, default=None, required=False,
                                  help="Path to the compiler config file.")

            optional.add_argument('--precision', choices=['int8', 'fp16'], default='int8',
                                  help='select precision')
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
            optional.add_argument(
                '--extra_contextbin_args', type=str, required=False, default=None,
                help="additional context binary generator arguments in a quoted string. \
                                                                example: --extra_contextbin_args 'arg1=value1;arg2=value2'"
            )
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        #CHECKING Core Arguments
        # change the relpath to abspath
        # get engine
        parsed_args.engine_version = None
        if hasattr(parsed_args, 'engine'):
            if parsed_args.engine is None:
                parsed_args.engine = self.engine
            else:
                if len(parsed_args.engine) > 2:
                    raise ParameterError("Maximum two arguments allowed for inference engine.")
                elif len(parsed_args.engine) == 2:
                    parsed_args.engine_version = parsed_args.engine[1]
                parsed_args.engine = parsed_args.engine[0]
        else:
            parsed_args.engine = self.engine

        snpe, qnn, _ = self._get_engine(self.engine)
        source_stage, converted_stage, compiled_stage = parsed_args.stage == 'source', \
                                                    parsed_args.stage == 'converted', parsed_args.stage == 'compiled'

        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)
        parsed_args.input_list = get_absolute_path(parsed_args.input_list)
        if qnn:
            if parsed_args.qnn_netrun_config_file:
                parsed_args.qnn_netrun_config_file = get_absolute_path(
                    parsed_args.qnn_netrun_config_file)

        # verify architecture, and that runtime and architecture align
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

            if target_device == Device_type.qnx.value and (parsed_args.remote_server is None
                                                           or parsed_args.remote_username is None):
                raise ParameterError(
                    "Remote server and username options are required for aarch64-qnx architecture.")

            is_linux_target = target_device == Device_type.x86.value
            is_windows_x86_target = target_device == Device_type.x86_64_windows_msvc.value
            wos_target = target_device == Device_type.wos.value
            if is_linux_target and parsed_args.runtime == Runtime.dspv66.value:
                raise ParameterError("Engine and runtime mismatch.")

            if self.engine == Engine.SNPE.value:
                if parsed_args.runtime not in [
                        "cpu", "dsp", "dspv66", "dspv68", "dspv69", "dspv73", "dspv75", "gpu", "aip"
                ]:
                    raise ParameterError("Engine and runtime mismatch.")
                # set default dsp version if 'dsp' is selected as runtime (NOTE: exclude dspv75,dspv79 as default for now)
                dspArchs = [
                    r.value for r in Runtime
                    if r.value.startswith("dsp") and r.value != "dsp" and r.value not in ["dspv75", "dspv79"]
                ]
                if parsed_args.runtime == "dsp": parsed_args.runtime = max(dspArchs)

                if parsed_args.architecture == 'aarch64-qnx':
                    raise ParameterError(
                        f"{parsed_args.runtime} runtime is unsupported for {Qnx_Architectures.aarch64_qnx.value} architecture."
                    )
                if parsed_args.architecture == X86_windows_Architectures.x86_64_windows_msvc.value\
                                                            and parsed_args.runtime not in ["cpu"]:
                    raise ParameterError(
                        f"{parsed_args.runtime} runtime is unsupported for {X86_windows_Architectures.x86_64_windows_msvc.value} architecture."
                    )
                if parsed_args.architecture == Aarch64_windows_Architectures.wos.value\
                                                            and parsed_args.runtime not in ["cpu"]:
                    raise ParameterError(
                        f"{parsed_args.runtime} runtime is unsupported for {Aarch64_windows_Architectures.wos.value} architecture."
                    )
            else:
                if target_device == Device_type.qnx.value and parsed_args.runtime not in [
                        "dspv68", "dspv73", "dspv75"
                ]:
                    raise ParameterError("Invalid runtime for aarch64-qnx")

                excluded_QNN_runtimes = ['aip', 'dspv65']
                valid_QNN_runtimes = [
                    r.value for r in Runtime if r.value not in excluded_QNN_runtimes
                ]
                if parsed_args.runtime not in valid_QNN_runtimes:
                    raise ParameterError("Engine and runtime mismatch.")

                # set default dsp version if 'dsp' is selected as runtime (NOTE: exclude dspv75,dspv79 as default for now)
                dspArchs = [
                    r.value for r in Runtime
                    if r.value.startswith("dsp") and r.value != "dsp" and r.value not in ["dspv75", "dspv79"]
                ]
                default_dsp = max(dspArchs)
                if parsed_args.runtime == "dsp": parsed_args.runtime = default_dsp

                if parsed_args.architecture == X86_windows_Architectures.x86_64_windows_msvc.value\
                                                            and parsed_args.runtime not in ["cpu"]:
                    raise ParameterError(
                        f"{parsed_args.runtime} runtime is unsupported for {X86_windows_Architectures.x86_64_windows_msvc.value} architecture."
                    )
                if parsed_args.architecture == Aarch64_windows_Architectures.wos.value\
                                                            and parsed_args.runtime not in ["cpu"]:
                    raise ParameterError(
                        f"{parsed_args.runtime} runtime is unsupported for {Aarch64_windows_Architectures.wos.value} architecture."
                    )

        # CHECKING Stage related Arguments
        if source_stage:
            source_attr = ["model_path", "framework", "input_tensor", "output_tensor"]
            for attr in source_attr:
                if getattr(parsed_args, attr) is None:
                    raise ParameterError("stage is at SOURCE, missing --{} argument".format(attr))

            parsed_args.model_path = get_absolute_path(parsed_args.model_path)

        #get framework and framework version even when not in source stage
        parsed_args.framework_version = None
        if parsed_args.framework is not None:
            if len(parsed_args.framework) > 2:
                raise ParameterError("Maximum two arguments required for framework.")
            elif len(parsed_args.framework) == 2:
                parsed_args.framework_version = parsed_args.framework[1]
            parsed_args.framework = parsed_args.framework[0]

        if snpe:
            if not source_stage:
                if not parsed_args.static_model:
                    raise ParameterError("stage is NOT at SOURCE, missing --static_model parameter")
                else:
                    parsed_args.static_model = get_absolute_path(parsed_args.static_model)

        if qnn:
            if converted_stage:
                # check if cpp exist(bin is not always required)
                if not parsed_args.qnn_model_cpp_path:
                    raise ParameterError("stage is at CONVERTED, missing qnn_model_cpp")
                parsed_args.qnn_model_cpp_path = get_absolute_path(parsed_args.qnn_model_cpp_path)
                if parsed_args.qnn_model_bin_path is not None:
                    parsed_args.qnn_model_bin_path = get_absolute_path(
                        parsed_args.qnn_model_bin_path)
                cpp_file_name = os.path.splitext(os.path.basename(
                    parsed_args.qnn_model_cpp_path))[0]
                parsed_args.qnn_model_name = cpp_file_name  # which is also equal to bin_file_name
            if compiled_stage:
                if not parsed_args.qnn_model_binary_path:
                    raise ParameterError("stage is at COMPILED, missing --qnn_model_binary_path")
                else:
                    parsed_args.qnn_model_binary_path = get_absolute_path(
                        parsed_args.qnn_model_binary_path)
            if parsed_args.restrict_quantization_steps:
                parsed_args.restrict_quantization_steps = f'"{parsed_args.restrict_quantization_steps}"'

        if parsed_args.input_tensor is not None:
            # get proper input_tensor format
            for tensor in parsed_args.input_tensor:
                if len(tensor) < 3:
                    raise argparse.ArgumentTypeError(
                        "Invalid format for input_tensor, format as "
                        "--input_tensor \"INPUT_NAME\" INPUT_DIM INPUT_DATA.")
                tensor[2] = get_absolute_path(tensor[2],
                                              pathPrepend=os.path.dirname(parsed_args.input_list))
                tensor[:] = tensor[:3]

        #The last data type gets shaved off
        if parsed_args.engine == Engine.QNN.value:
            if parsed_args.input_tensor is not None:
                tensor_list = []
                for tensor in parsed_args.input_tensor:
                    #this : check acts differently on snpe vs qnn on tensorflow models.
                    if ":" in tensor[0]:
                        tensor[0] = tensor[0].split(":")[0]
                    tensor_list.append(tensor)
                parsed_args.input_tensor = tensor_list

            if parsed_args.output_tensor is not None:
                tensor_list = []
                for tensor in parsed_args.output_tensor:
                    if ":" in tensor:
                        tensor = tensor.split(":")[0]
                    tensor_list.append(tensor)
                parsed_args.output_tensor = tensor_list

        #QNN related optional parameters check
        if qnn:
            # modification below converts the list to a comma-separated string wrapped in quotes,
            # which enables proper processing by model-lib-generator

            valid_runtimes = ['aic', 'cpu', 'htp', 'dspv68', 'dspv73', 'dspv75', 'dspv79']
            if parsed_args.runtime in valid_runtimes and target_device in [
                    Device_type.x86.value, Device_type.qnx.value
            ]:
                parsed_args.lib_target = [X86_Architectures.x86_64_linux_clang.value]
            elif parsed_args.runtime in valid_runtimes and target_device in [
                    Device_type.x86_64_windows_msvc.value
            ]:
                parsed_args.lib_target = [X86_windows_Architectures.x86_64_windows_msvc.value]
            elif parsed_args.runtime in ['dspv68', 'dspv73', 'dspv75'] and not parsed_args.offline_prepare and target_device in [
                    Device_type.android.value
            ]:
                parsed_args.lib_target = [Android_Architectures.aarch64_android.value]
            elif parsed_args.runtime in valid_runtimes and target_device in [
                Device_type.wos.value
            ]:
                parsed_args.lib_target = [Aarch64_windows_Architectures.wos.value]
            else:
                parsed_args.lib_target = [X86_Architectures.x86_64_linux_clang.value]
            if len(parsed_args.lib_target) > 1:
                raise ParameterError("Maximum one argument required for library targets.")
            parsed_args.lib_target = '\'' + ' '.join(parsed_args.lib_target) + '\''
            if parsed_args.qnn_model_net_json:
                parsed_args.qnn_model_net_json = get_absolute_path(parsed_args.qnn_model_net_json)
            if parsed_args.extra_converter_args:
                converter_ignore_list = [
                    'input_network', 'input_dim', 'out_node', 'output_path',
                    'quantization_overrides', 'input_list', 'param_quantizer', 'act_quantizer',
                    'weight_bw', 'bias_bw', 'act_bw', 'float_bias_bw',
                    'restrict_quantization_steps', 'algorithms', 'ignore_encodings',
                    'use_per_channel_quantization', 'disable_batchnorm_folding', 'i', 'b',
                    'act_quantizer_calibration', 'param_quantizer_calibration',
                    'act_quantizer_schema', 'param_quantizer_schema', 'percentile_calibration_value'
                ]
                parsed_args.extra_converter_args = format_args(parsed_args.extra_converter_args,
                                                               converter_ignore_list)
            if parsed_args.extra_contextbin_args:
                context_binary_ignore_list = [
                    'model', 'backend', 'binary_file', 'model_prefix', 'output_dir', 'config_file',
                    'enable_intermediate_outputs', 'set_output_tensors'
                ]
                parsed_args.extra_contextbin_args = format_args(parsed_args.extra_contextbin_args,
                                                                context_binary_ignore_list)
            if parsed_args.extra_runtime_args:
                runtime_ignore_list = [
                    'model', 'retrieve_context', 'input_list', 'backend', 'output_dir', 'debug',
                    'use_native_input_files', 'use_native_output_files', 'profiling_level',
                    'perf_profile', 'config_file', 'log_level', 'version'
                ]
                parsed_args.extra_runtime_args = format_args(parsed_args.extra_runtime_args,
                                                             runtime_ignore_list)
            if parsed_args.add_layer_types:
                parsed_args.add_layer_types = parsed_args.add_layer_types.split(',')

            if parsed_args.skip_layer_types:
                parsed_args.skip_layer_types = parsed_args.skip_layer_types.split(',')

            if parsed_args.add_layer_types or parsed_args.skip_layer_types or parsed_args.skip_layer_outputs:
                parsed_args.add_layer_outputs = update_layer_options(parsed_args)

            # Handle the user passed --precision fp16 for aic runtime
            if parsed_args.precision == 'fp16' and parsed_args.runtime == 'aic':
                handle_at_converter_stage = True
                if parsed_args.compiler_config:
                    with open(parsed_args.compiler_config, 'r') as file:
                        compiler_config = json.load(file)
                        if "compiler_convert_to_FP16" in compiler_config and compiler_config[
                                "compiler_convert_to_FP16"] is True:
                            # Everything okay, fp16 conversion will be taken care of at the context_binary stage
                            handle_at_converter_stage = False
                # Now handle at the converter stage if compiler_config not passed or "compiler_convert_to_FP16"
                # not in config_file or False
                if handle_at_converter_stage:
                    if parsed_args.extra_converter_args is None:
                        parsed_args.extra_converter_args = '--float_bitwidth 16'
                    elif '--float_bitwidth 16' not in parsed_args.extra_converter_args:
                        if '--float_bitwidth' in parsed_args.extra_converter_args:
                            raise (ParameterError(
                                "Please check your extra_converter_args, with precision = fp16, float_bitwidth must be 16"
                            ))
                        else:
                            parsed_args.extra_converter_args += '--float_bitwidth 16'

            # If newer quantization scheme is used then set the legacy default initialised scheme to None
            if parsed_args.act_quantizer_calibration is not None or parsed_args.param_quantizer_calibration is not None:
                parsed_args.act_quantizer = None
                parsed_args.param_quantizer = None

        if snpe:
            if parsed_args.extra_converter_args:
                converter_ignore_list = [
                    'input_network', 'input_dim', 'out_node', 'output_path',
                    'quantization_overrides', 'input_list'
                ]

                parsed_args.extra_converter_args = format_args(parsed_args.extra_converter_args,
                                                               converter_ignore_list)
            if parsed_args.extra_quantizer_args:
                parsed_args.extra_quantizer_args.replace(";", " ")
            if parsed_args.extra_runtime_args:
                runtime_ignore_list = [
                    'model', 'input_list', 'backend', 'output_dir', 'debug',
                    'use_native_input_files', 'use_native_output_files', 'profiling_level',
                    'perf_profile', 'log_level', 'version'
                ]
                parsed_args.extra_runtime_args = format_args(parsed_args.extra_runtime_args,
                                                             runtime_ignore_list)

        return parsed_args

    def get_all_associated_parsers(self):
        self._validate_quantizer_args()
        return [self.parser]

    def _validate_quantizer_args(self):
        # Make sure (--activation_quantizer, --param_quantizer) and
        # (--act_quantizer_calibration, --param_quantizer_calibration) not in
        # args passed by the user
        if ("--param_quantizer_calibration" in self.args or "--act_quantizer_calibration"
                in self.args) and ("--act_quantizer" in self.args
                                   or "--param_quantizer" in self.args or "-pq" in self.args):
            raise ParameterError(
                "Invalid combination: legacy quantizer options: --act_quantizer or --param_quantizer cannot be combined with --act_quantizer_calibration or --param_quantizer_calibration"
            )
