# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys

try:
    from qti.aisw.converters.common import ir_quantizer
    from qti.aisw.converters.common import encodings_json_serializer
except ImportError as ie:
    print("Failed to find necessary quantization packages:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

try:
    from qti.aisw.converters.common import modeltools
except ImportError as ie:
    from qti.aisw.dlc_utils import modeltools

try:
    from qti.aisw.converters.aimet.qnn_quantsim_interface import aimet_dlc_quantizer, AimetQuantizerOpts
except:
    pass

from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
import argparse
import os

class DLCQuantizer(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(DLCQuantizer.ArgParser, self).__init__(**kwargs)
            self.add_required_argument('--input_dlc', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help='Path to the dlc container containing the model for which '
                                            'fixed-point encoding metadata should be generated. This argument is required')

            self.add_optional_argument('--output_dlc', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=False,
                                                                                     create_missing_directory=True),
                                       help='Path at which the metadata-included quantized model container should be written.'
                                            'If this argument is omitted, the quantized model will be written at '
                                            '<unquantized_model_name>_quantized.dlc')

            self.add_optional_argument('--input_list', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help='Path to a file specifying the input data. This file should be a plain text '
                                            'file, containing one or more absolute file paths per line. Each path is '
                                            'expected to point to a binary file containing one input in the "raw" format, '
                                            'ready to be consumed by the quantizer without any further preprocessing. '
                                            'Multiple files per line separated by spaces indicate multiple inputs to the '
                                            'network. See documentation for more details. Must be specified for quantization. '
                                            'All subsequent quantization options are ignored when this is not provided.')

            self.add_optional_argument('--float_fallback', action='store_true', default=False,
                                       help='Use this option to enable fallback to floating point (FP) instead of fixed point. \n'
                                            'This option can be paired with --float_bitwidth to indicate the bitwidth for FP (by default 32). \n'
                                            'If this option is enabled, then input list must not be provided and --ignore_encodings must not be provided.\n'
                                            'The external quantization encodings (encoding file/FakeQuant encodings) might be missing quantization parameters for some interim tensors. \n'
                                            'First it will try to fill the gaps by propagating across math-invariant functions. If the quantization params are still missing, \n'
                                            'then it will apply fallback to nodes to floating point. \n')

            self.add_optional_argument('--param_quantizer', type=str,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--act_quantizer', type=str, default='tf',
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--algorithms', type=str, nargs='+', default=[],
                                       help='Use this option to enable new optimization algorithms. Usage is: '
                                            '--algorithms <algo_name1> ... '
                                            'The available optimization algorithms are: '
                                            '"cle" - Cross layer equalization includes a number of methods for equalizing '
                                            'weights and biases across layers in order to rectify imbalances that cause '
                                            'quantization errors. ')

            self.add_optional_argument('--bias_bitwidth', type=int, default=8,
                                       help='Use the --bias_bitwidth option to select the bitwidth to use when quantizing the '
                                            'biases, either 8 (default) or 32.')

            self.add_optional_argument('--act_bitwidth', type=int, default=8,
                                       help='Use the --act_bitwidth option to select the bitwidth to use when quantizing the '
                                            'activations, either 8 (default) or 16.')

            self.add_optional_argument('--weights_bitwidth', type=int, default=8,
                                       help='Use the --weights_bitwidth option to select the bitwidth to use when quantizing '
                                            'the weights, either 4 or 8 (default).')

            self.add_optional_argument('--float_bitwidth', type=int, default=32,
                                       help='Use the --float_bitwidth option to select the bitwidth to use for float tensors,'
                                            'either 32 (default) or 16.')

            self.add_optional_argument('--float_bias_bitwidth', type=int, default=0,
                                       help='Use the --float_bias_bitwidth option to select the bitwidth to use when biases '
                                            'are in float, either 32 or 16.')

            self.add_optional_argument('--ignore_encodings', action='store_true', default=False,
                                       help='Use only quantizer generated encodings, ignoring any user or model provided '
                                            'encodings.\n'
                                            'Note: Cannot use --ignore_encodings with --quantization_overrides')

            self.add_optional_argument('--use_per_channel_quantization', action='store_true', default=False,
                                       help='Use this option to enable per-channel quantization for convolution-based op weights. \n'
                                            'Note: This will replace built-in model QAT encodings when used for a given weight.')

            self.add_optional_argument('--use_per_row_quantization', action='store_true', default=False,
                                       help='Use this option to enable rowwise quantization of Matmul and FullyConnected ops.')


            self.add_optional_argument('--use_native_input_files', action='store_true', default=False,
                                       help='Boolean flag to indicate how to read input files:\n'
                                            '1. float (default): reads inputs as floats and quantizes if necessary based on quantization parameters in the model.\n'
                                            '2. native:          reads inputs assuming the data type to be native to the model. For ex., uint8_t.\n')


            self.add_optional_argument('--use_native_output_files', action='store_true', default=False,
                                       help='Use this option to indicate the data type of the output files\n'
                                            '1. float (default): output the file as floats.\n'
                                            '2. native:          outputs the file that is native to the model. For ex., uint8_t.\n')

            self.add_optional_argument('--restrict_quantization_steps', type=validation_utils.two_hex, action = "store",
                                       help='Specifies the number of steps to use for computing quantization encodings such that '
                                            'scale = (max - min) / number of quantization steps.\n'
                                            'The option should be passed as a space separated pair of hexadecimal string minimum and maximum values'
                                            'i.e. --restrict_quantization_steps "MIN MAX".  \n Please note that this is a hexadecimal string literal'
                                            ' and not a signed integer, to supply a negative value an explicit minus sign is required.\n'
                                            'E.g.--restrict_quantization_steps "-0x80 0x7F" indicates an example 8 bit range,\n'
                                            '    --restrict_quantization_steps "-0x8000 0x7F7F" indicates an example 16 bit range.\n'
                                            'This argument is required for 16-bit Matmul operations.\n',
                                       metavar="ENCODING_MIN, ENCODING_MAX", default=[])

            # TODO: Remove this flag once we fully support 16-bit
            self.add_optional_argument('--use_dynamic_16_bit_weights', action='store_true', default=False,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--pack_4_bit_weights', action='store_true', default=False,
                                       help=argparse.SUPPRESS)

            self.add_optional_argument('--act_quantizer_calibration', type=str, default="min-max",
                                       help='Specify which quantization calibration method to use for activations\n'
                                            'supported values: min-max (default), sqnr, entropy, mse, percentile\n'
                                            'This option can be paired with --act_quantizer_schema to override the quantization\n'
                                            'schema to use for activations otherwise default schema(asymmetric) will be used\n')

            self.add_optional_argument('--param_quantizer_calibration', type=str, default="min-max",
                                       help='Specify which quantization calibration method to use for parameters\n'
                                            'supported values: min-max (default), sqnr, entropy, mse, percentile\n'
                                            'This option can be paired with --param_quantizer_schema to override the quantization\n'
                                            'schema to use for parameters otherwise default schema(asymmetric) will be used\n')

            self.add_optional_argument('--act_quantizer_schema', type=str, default="asymmetric",
                                       help='Specify which quantization schema to use for activations\n'
                                            'supported values: asymmetric (default), symmetric\n')

            self.add_optional_argument('--param_quantizer_schema', type=str, default="asymmetric",
                                       help='Specify which quantization schema to use for parameters\n'
                                            'supported values: asymmetric (default), symmetric\n')

            self.add_optional_argument('--percentile_calibration_value', type=float, default=99.99,
                                       help='Specify the percentile value to be used with Percentile calibration method\n'
                                            'The specified float value must lie within 90 and 100, default: 99.99\n')

            self.add_optional_argument("--use_aimet_quantizer",
                                       action="store_true",
                                       help='Use AIMET for Quantization instead of QNN IR quantizer',
                                       default=False)

            self.add_optional_argument('--op_package_lib', '-opl', type=str, default="",
                                       help='Use this argument to pass an op package library for quantization. '
                                            'Must be in the form <op_package_lib_path:interfaceProviderName> and'
                                            ' be separated by a comma for multiple package libs')

            self.add_optional_argument('--dump_encoding_json', action='store_true', default=False,
                                       help="Use this argument to dump encoding of all the tensors in a json file")

            self.add_optional_argument('--include_data_invariant_ops', action='store_true', help=argparse.SUPPRESS,
                                       default=False)

        @classmethod
        def validate_and_convert_args(cls, args):
            # check if legacy quantizer options are used with new quantizer options.
            if (("--param_quantizer_calibration" in sys.argv or "--act_quantizer_calibration" in sys.argv) and
                    ("--param_quantizer" in sys.argv or "--act_quantizer" in sys.argv)):
                raise Exception("Invalid combination: legacy quantizer options: --act_quantizer or --param_quantizer cannot be "
                                "combined with --act_quantizer_calibration or --param_quantizer_calibration")

            if (("--param_quantizer_schema" in sys.argv or "--act_quantizer_schema" in sys.argv) and
                    ("--param_quantizer" in sys.argv or "--act_quantizer" in sys.argv)):
                raise Exception("Invalid combination: legacy quantizer options: --act_quantizer or --param_quantizer cannot be "
                                "combined with --act_quantizer_schema or --param_quantizer_schema. "
                                "To create quantizer with different quantization schema use --act_quantizer_calibration or "
                                "--param_quantizer_calibration with --act_quantizer_schema or --param_quantizer_schema respectively")


            if "--param_quantizer_schema" in sys.argv and args.param_quantizer_schema not in ["symmetric", "asymmetric"]:
                raise Exception("Invalid param quantizer schema: ", args.param_quantizer_schema)

            if "--act_quantizer_schema" in sys.argv and args.act_quantizer_schema not in ["symmetric", "asymmetric"]:
                raise Exception("Invalid activation quantizer schema: ", args.act_quantizer_schema)
            # If percentile_calibration value is passed check if the calibration method selected is percentile.
            if ("--percentile_calibration_value" in sys.argv and
                    (args.act_quantizer_calibration != "percentile" and args.param_quantizer_calibration != "percentile")):
                raise Exception("Invalid combination: --percentile_calibration_value option should be used with "
                                "--act_quantizer_calibration percentile or --param_quantizer_calibration percentile options")

            # Throw error if an argument is provided that is not supported by AIMET Quantizer
            if "--use_aimet_quantizer" in sys.argv:
                args_not_supported_by_aimet = tuple(["--float_bw", "--float_bitwidth", "--float_bias_bw", "--float_bias_bitwidth",
                                                     "--disable_relu_squashing", "--restrict_quantization_steps", "--float_fallback",
                                                     "--use_dynamic_16_bit_weights", "--pack_4_bit_weights", "--op_package_lib"])
                args_provided_by_user = [arg for arg in sys.argv if arg[0:2]=="--"]
                args_provided_by_user_not_supported_by_aimet = [arg for arg in args_provided_by_user if arg in args_not_supported_by_aimet]
                if len(args_provided_by_user_not_supported_by_aimet)!=0:
                    raise Exception(f"AIMET Quantizer doesn't support the following options currently: "
                                    f"{args_provided_by_user_not_supported_by_aimet}")


            args_dict = vars(args).copy()

            args_dict['disable_legacy_quantizer'] = True
            # If any one of legacy quantizer options is passed then enable the legacy quantizer
            if ("--param_quantizer" in sys.argv or "--act_quantizer" in sys.argv):
                args_dict['disable_legacy_quantizer'] = False

            return args_dict


    def __init__(self,
                 input_dlc,
                 output_dlc=None,
                 input_list="",
                 float_fallback=False,
                 param_quantizer="tf",
                 act_quantizer="tf",
                 algorithms=[],
                 bias_bitwidth=8,
                 act_bitwidth=8,
                 weights_bitwidth=8,
                 float_bitwidth=32,
                 float_bias_bitwidth=0,
                 ignore_encodings=False,
                 use_per_channel_quantization=False,
                 use_per_row_quantization=False,
                 use_native_input_files=False,
                 use_native_output_files=False,
                 restrict_quantization_steps=[],
                 use_dynamic_16_bit_weights=False,
                 pack_4_bit_weights=False,
                 act_quantizer_calibration="min-max",
                 param_quantizer_calibration="min-max",
                 act_quantizer_schema="asymmetric",
                 param_quantizer_schema="asymmetric",
                 percentile_calibration_value=99.99,
                 use_aimet_quantizer=False,
                 op_package_lib="",
                 disable_legacy_quantizer=False,
                 dump_encoding_json=False,
                 include_data_invariant_ops=False):
        self.input_dlc = input_dlc

        self.use_aimet_quantizer = use_aimet_quantizer

        if not self.use_aimet_quantizer:
            # Deserialize the DLC
            self.dlc_reader = modeltools.IrDlcReader(input_dlc, disableLazyWeightLoading=True)
            self.ir_graph = self.dlc_reader.get_ir_graph()

            # store DLC metadata
            self.model_version = self.dlc_reader.custom_model_version
            self.copyright_str = self.dlc_reader.copyright
            self.converter_command = self.dlc_reader.converter_command
        else:
            self.dlc_reader = None
            self.ir_graph = None
            self.model_version = None
            self.copyright_str = None
            self.converter_command = None

        # store the output path for the quantized DLC
        if output_dlc is None:
            filename, _ = os.path.splitext(os.path.realpath(input_dlc))
            self.output_path = filename + "_quantized.dlc"
            self.output_encoding_json_path = filename + "_quantized_encoding.json"
        else:
            self.output_path = output_dlc
            self.output_encoding_json_path = os.path.splitext(os.path.realpath(self.output_path))[0] + "_encoding.json"

        self.dump_encoding_json = dump_encoding_json
        self.include_data_invariant_ops = include_data_invariant_ops

        # Set Quantizer option
        self.opts = ir_quantizer.IrQuantizerOpts()

        if (input_list is None and not float_fallback):
            self.should_quantize = False
            self.use_fallback_to_float = False
        elif (input_list is None and float_fallback):
            log_warning("Quantization is disabled as --float_fallback flag is provided "
                        "Some Ops may fallback to float datatype")
            self.should_quantize = True
            self.use_fallback_to_float = True
        elif (input_list is not None and float_fallback):
            raise Exception("Invalid combination: --input_list and --float_fallback "
                            "cannot be provided at the same time.")
        else:
            self.should_quantize = True
            self.opts.input_list = input_list
            self.use_fallback_to_float = False

        self.opts.disable_legacy_quantizer = False

        if not self.should_quantize:
            return

        if self.use_fallback_to_float and ignore_encodings:
            raise Exception("Cannot determine quantization encodings for any tensor. "
                            "--ignore_encodings cannot be provided with --float_fallback flag")

        if percentile_calibration_value < 90 or percentile_calibration_value > 100:
            raise Exception("--percentile_calibration_value must lie with 90 and 100")

        # Set default values for act_quantizer and param_quantizer
        if not param_quantizer:
            if weights_bitwidth == 16:
                param_quantizer = "symmetric"
            else:
                param_quantizer = "tf"

        self.opts.param_quantizer = param_quantizer
        self.opts.act_quantizer = act_quantizer
        self.opts.param_quantizer_calibration = param_quantizer_calibration
        self.opts.act_quantizer_calibration = act_quantizer_calibration
        self.opts.param_quantizer_schema = param_quantizer_schema
        self.opts.act_quantizer_schema = act_quantizer_schema
        self.opts.percentile_calibration_value = percentile_calibration_value
        self.opts.algorithms = algorithms


        self.opts.bias_bw = bias_bitwidth
        self.opts.act_bw = act_bitwidth
        self.opts.weight_bw = weights_bitwidth
        self.opts.float_bw = float_bitwidth
        self.opts.float_bias_bw = float_bias_bitwidth
        self.opts.optimizations = True
        self.opts.op_package_lib = op_package_lib
        self.opts.ignore_encodings = ignore_encodings
        self.opts.use_per_row_quantization = use_per_row_quantization
        self.opts.use_per_channel_quantization = use_per_channel_quantization
        self.opts.use_native_input_dtype = use_native_input_files
        self.opts.use_native_output_dtype = use_native_output_files
        self.opts.reset_irgraph_maps = True
        self.opts.enable_qnn_quantizer = True
        self.opts.use_dynamic_16_bit_weights = use_dynamic_16_bit_weights
        self.opts.pack_4_bit_weights = pack_4_bit_weights
        self.opts.disable_legacy_quantizer = disable_legacy_quantizer
        self.opts.disable_relu_squashing = True

        if restrict_quantization_steps:
            if self.opts.param_quantizer == "d" or self.opts.use_per_channel_quantization or self.opts.use_per_row_quantization:
                self.opts.quantization_step_min = restrict_quantization_steps[0]
                self.opts.quantization_step_max = restrict_quantization_steps[1]
                log_info("Restricting number of quantization steps to: min: {} - max: {}".format(self.opts.quantization_step_min,
                                                                                                 self.opts.quantization_step_max))
            else:
                log_warning("Restrict_quantization_steps is only supported for --param_quantizer = symmetric"
                            " or per channel/row quantization. Value will be ignored.")

        self.quant_schemes = None
        if self.use_aimet_quantizer:
            self.quant_schemes = {}
            if not self.opts.disable_legacy_quantizer:
                self.quant_schemes["param_quant"] = self.opts.param_quantizer
                self.quant_schemes["act_quant"] = self.opts.act_quantizer
            else:
                self.quant_schemes["param_quant"] = {"calibration": self.opts.param_quantizer_calibration,
                                                           "schema": self.opts.param_quantizer_schema}
                self.quant_schemes["act_quant"] = {"calibration":self.opts.act_quantizer_calibration,
                                                         "schema": self.opts.act_quantizer_schema}



    def quantize(self):
        """
        This method quantize the IR graph (inplace) generated from the DLC.
        :return: None
        """

        if not self.should_quantize:
            log_info('Skipping quantization, no input_list provided')
            return

        if self.use_aimet_quantizer:
            opts = AimetQuantizerOpts(input_network=self.input_dlc,
                                      output_path=self.output_path,
                                      input_list=self.opts.input_list,
                                      quant_schemes=self.quant_schemes,
                                      disable_legacy_quant_scheme_opts=self.opts.disable_legacy_quantizer,
                                      algorithms=self.opts.algorithms,
                                      act_bitwidth=self.opts.act_bw,
                                      weights_bitwidth=self.opts.weight_bw,
                                      bias_bitwidth=self.opts.bias_bw,
                                      percentile_calibration_value=self.opts.percentile_calibration_value,
                                      ignore_encodings=self.opts.ignore_encodings,
                                      use_per_channel_quantization=self.opts.use_per_channel_quantization,
                                      use_per_row_quantization=self.opts.use_per_row_quantization,
                                      use_native_input_files=self.opts.use_native_input_dtype,
                                      use_native_output_files=self.opts.use_native_output_dtype)
            # AIMET saves quantized DLC to self.output_path
            aimet_dlc_quantizer(opts)
            return

        # Quantize IR graph
        quantizer = ir_quantizer.IrQuantizer(self.opts, self.ir_graph)
        quantizer.quantize()
        log_info(code_to_message.get_progress_message("Quantization completed successfully"))

    def save(self, quantizer_command=""):
        """
        This method saves the quantized model to the output path specifies
        during instantiation. If nothing specifies, the quantized model will
        be stored at the same location as of input dlc.
        :return: None
        """
        if not self.use_aimet_quantizer:
            dlc_writer = modeltools.IrDlcSerializer(self.output_path,
                                                    self.copyright_str,
                                                    self.model_version,
                                                    self.converter_command,
                                                    quantizer_command)
            dlc_writer.initialize()
            dlc_writer.serialize(self.ir_graph)
            dlc_writer.finish()

            # Serialize QNNIR to ENCODING JSON
            if self.dump_encoding_json:
                self.encodings_json_serializer = encodings_json_serializer.IrEncodingsJsonSerializer(
                    self.output_encoding_json_path, self.include_data_invariant_ops)
                self.encodings_json_serializer.serialize(self.ir_graph)
                encoding_json = self.encodings_json_serializer.get_graph_json()
                with open(self.output_encoding_json_path, "w") as json_file:
                    json_file.write(encoding_json)
                log_info("encodings JSON saved at: %s " % self.output_encoding_json_path)

        log_info("Quantized Model saved at: %s " % self.output_path)
