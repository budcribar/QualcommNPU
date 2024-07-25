# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, format_args
from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework, Engine, Runtime
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message

import argparse
import os
import json
from datetime import datetime
import numpy as np


class QuantCheckerCmdOptions(CmdOptions):

    def __init__(self, args, engine, validate_args=True):
        super().__init__('quant_checker', args, engine, validate_args=validate_args)

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
        """
        type: (List[str]) -> argparse.Namespace

        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        snpe, qnn, _ = self._get_engine(self.engine)
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script that runs quant_checker")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-m', '--model_path', type=str, required=True,
                              help='Path to the model file(s).')
        required.add_argument('-l', '--input_list', type=str, required=True,
                              help="Path to the input list text.")

        required.add_argument(
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
        required.add_argument(
            '-f', '--framework', nargs='+', type=str.lower, required=True,
            help='Framework type and version, version is optional. '
            'Currently supported frameworks are [' + ', '.join([f.value for f in Framework]) + ']. '
            'For example, tensorflow 2.3.0')
        required.add_argument(
            '-c', '--config_file', required=True, type=str,
            help='Config file specifying all possible options required for execution.')
        required.add_argument('-o', '--output_tensor', type=str, required=True, action='append',
                              help='Name of the graph\'s output tensor(s).')

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument(
            '--host_device', type=str, required=False, default='x86',
            choices=['x86', 'x86_64-windows-msvc', 'wos'],
            help='The device that will be running conversion. Set to x86 by default.')
        optional.add_argument(
            '--deviceId', required=False, default=None,
            help='The serial number of the device to use. If not available, '
            'the first in a list of queried devices will be used for validation.')
        optional.add_argument('--bias_width', type=str, required=False, default=8,
                              help='Bit-width to use for biases. E.g., 8, 32. Default is 8.')
        optional.add_argument('--weight_width', type=str, required=False, default=8,
                              help='Bit-width to use for weights. E.g., 8. Default is 8.')

        optional.add_argument('--generate_csv', action='store_true', required=False,
                              help='Output analysis data to a csv file in the output directory.')
        optional.add_argument(
            '--generate_plots', nargs="+", required=False,
            help='Generate plot analysis for weights/biases. Default is to skip plot generation.'
            'There are 4 different plots supported, user can pass the plots which they want to generate'
            ' For example --generate_plots histogram cdf diff min_max_distribution')
        optional.add_argument('--generate_html', action='store_true', required=False,
                              help='Output analysis data to a html file in the output directory.')
        optional.add_argument(
            '--per_channel_plots', action='store_true', required=False, help=
            'Generate per channel plots analysis for weights/biases. Default is to skip per channel plots generation.'
        )
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Verbose printing")
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                            default='working_directory',
                            help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                'Creates a new directory if the specified working directory does not exitst.')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('-p', '--engine_path', type=str, required=False,
                              help="Path to the inference engine.")
        optional.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional converter arguments in a quoted string. \
                 example: --extra_converter_args 'input_dtype=data float;input_layout=data1 NCHW'")
        optional.add_argument('-qo', '--quantization_overrides', type=str, required=False,
                              default=None, help="Path to quantization overrides json file.")
        optional.add_argument(
            '--golden_output_reference_directory', dest='golden_output_reference_directory',
            type=str, required=False,
            help='Optional parameter to indicate the directory of the golden reference. '
            'When this option provided, framework diagnosis stage is skipped. '
            'Provide the path to directory which contains the fp32 outputs for '
            'all inputs mentioned in the input_list.txt. ')
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.model_path = get_absolute_path(parsed_args.model_path)
        parsed_args.input_list = get_absolute_path(parsed_args.input_list)
        parsed_args.config_file = get_absolute_path(parsed_args.config_file)
        parsed_args.quantization_overrides = get_absolute_path(parsed_args.quantization_overrides)
        if parsed_args.golden_output_reference_directory:
            parsed_args.golden_output_reference_directory = get_absolute_path(
                parsed_args.golden_output_reference_directory)
        snpe, qnn, _ = self._get_engine(self.engine)
        # Parse version since it is an optional argument that is combined with framework
        parsed_args.framework_version = None
        if len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]
        parsed_args.framework = parsed_args.framework[0]
        with open(parsed_args.config_file) as configFile:
            configInfosJson = json.load(configFile)

        comparison_algorithms = {}
        if "WEIGHT_COMPARISON_ALGORITHMS" in configInfosJson:
            weight_comparison_algorithms = configInfosJson["WEIGHT_COMPARISON_ALGORITHMS"]
        else:
            #Put comments for each algo comments
            weight_comparison_algorithms = [{
                "algo_name": "minmax",
                "threshold": "10"
            }, {
                "algo_name": "maxdiff",
                "threshold": "10"
            }, {
                "algo_name": "sqnr",
                "threshold": "26"
            }, {
                "algo_name": "stats",
                "threshold": "2"
            }, {
                "algo_name": "data_range_analyzer"
            }, {
                "algo_name": "data_distribution_analyzer",
                "threshold": "0.6"
            }]
        comparison_algorithms['weight_comparison_algorithms'] = weight_comparison_algorithms

        if "BIAS_COMPARISON_ALGORITHMS" in configInfosJson:
            bias_comparison_algorithms = configInfosJson["BIAS_COMPARISON_ALGORITHMS"]
        else:
            bias_comparison_algorithms = [{
                "algo_name": "minmax",
                "threshold": "10"
            }, {
                "algo_name": "maxdiff",
                "threshold": "10"
            }, {
                "algo_name": "sqnr",
                "threshold": "26"
            }, {
                "algo_name": "stats",
                "threshold": "2"
            }, {
                "algo_name": "data_range_analyzer"
            }, {
                "algo_name": "data_distribution_analyzer",
                "threshold": "0.6"
            }]
        comparison_algorithms['bias_comparison_algorithms'] = bias_comparison_algorithms

        if "ACT_COMPARISON_ALGORITHMS" in configInfosJson:
            act_comparison_algorithms = configInfosJson["ACT_COMPARISON_ALGORITHMS"]
        else:
            act_comparison_algorithms = [{"algo_name": "minmax", "threshold": "10"}]
        comparison_algorithms['act_comparison_algorithms'] = act_comparison_algorithms

        if "INPUT_DATA_ANALYSIS_ALGORITHMS" in configInfosJson:
            input_data_analysis_algorithms = configInfosJson["INPUT_DATA_ANALYSIS_ALGORITHMS"]
        else:
            input_data_analysis_algorithms = [{"algo_name": "stats", "threshold": "2"}]
        comparison_algorithms['input_data_analysis_algorithms'] = input_data_analysis_algorithms
        parsed_args.comparison_algorithms = comparison_algorithms

        if "QUANTIZATION_VARIATIONS" in configInfosJson:
            quantization_variations = configInfosJson["QUANTIZATION_VARIATIONS"]
        else:
            quantization_variations = ['unquantized', 'enhanced', 'tf', 'adjusted', 'symmetric']
        parsed_args.quantization_variations = quantization_variations

        if "QUANTIZATION_ALGORITHMS" in configInfosJson:
            quantization_algorithms = configInfosJson["QUANTIZATION_ALGORITHMS"]
        else:
            quantization_algorithms = ["None", 'cle']
        parsed_args.quantization_algorithms = quantization_algorithms

        parsed_args.engine = self.engine

        user_provided_dtypes = []

        if parsed_args.input_tensor is not None:
            # get proper input_tensor format
            for tensor in parsed_args.input_tensor:
                if len(tensor) < 3:
                    raise argparse.ArgumentTypeError(
                        "Invalid format for input_tensor, format as "
                        "--input_tensor \"INPUT_NAME\" INPUT_DIM INPUT_DATA.")
                elif len(tensor) == 3:
                    user_provided_dtypes.append('float32')
                elif len(tensor) == 4:
                    user_provided_dtypes.append(tensor[-1])
                tensor[2] = get_absolute_path(tensor[2])
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

        if qnn and parsed_args.extra_converter_args:
            converter_ignore_list = [
                'input_network', 'input_dim', 'out_node', 'output_path', 'quantization_overrides',
                'input_list', 'param_quantizer', 'act_quantizer', 'weight_bw', 'bias_bw', 'act_bw',
                'float_bias_bw', 'restrict_quantization_steps', 'algorithms', 'ignore_encodings',
                'use_per_channel_quantization', 'disable_batchnorm_folding', 'i', 'b'
            ]
            parsed_args.extra_converter_args = format_args(parsed_args.extra_converter_args,
                                                           converter_ignore_list)
        if snpe and parsed_args.extra_converter_args:
            converter_ignore_list = [
                'input_network', 'input_dim', 'out_node', 'output_path', 'quantization_overrides',
                'input_list'
            ]

            parsed_args.extra_converter_args = format_args(parsed_args.extra_converter_args,
                                                           converter_ignore_list)

        if snpe:
            # Since underlying snpe tools are not supporting 64 bit inputs but our framework
            # diagnosis works on 64 bit, if model accepts 64 bit inputs
            # we need to reconvert the user provided 64 bit inputs through the input_list
            # into 32 bits. Here, we loop over each line in the input_list.txt
            # and convert the 64 bit input and dump them into the working directory
            # and finally we create new input_list containing 32 bit input paths
            converted_input_file_dump_path = os.path.join(parsed_args.working_dir, "inputs_32")
            os.makedirs(converted_input_file_dump_path, exist_ok=True)
            new_input_list_file_path = os.path.join(converted_input_file_dump_path,
                                                    'input_list.txt')
            new_input_list_file = open(new_input_list_file_path, 'w')
            with open(parsed_args.input_list, 'r') as file:
                for line in file.readlines():
                    line = line.rstrip().lstrip().split('\n')[0]
                    if line:
                        new_file_name_and_path = []
                        file_name_and_paths = [
                            file_name_and_path.split(':=')
                            if ':=' in file_name_and_path else [None, file_name_and_path]
                            for file_name_and_path in line.split()
                        ]
                        for user_provided_dtype, file_name_and_path in zip(
                                user_provided_dtypes, file_name_and_paths):
                            user_provided_tensor = np.fromfile(file_name_and_path[1],
                                                               dtype=user_provided_dtype)
                            if user_provided_dtype == "int64":
                                converted_tensor = user_provided_tensor.astype(np.int32)
                            elif user_provided_tensor == "float64":
                                converted_tensor = user_provided_tensor.astype(np.float32)
                            else:
                                converted_tensor = user_provided_tensor
                            file_name = os.path.join(converted_input_file_dump_path,
                                                     os.path.basename(file_name_and_path[1]))
                            converted_tensor.tofile(file_name)
                            if file_name_and_path[0] is not None:
                                new_file_name_and_path.append(file_name_and_path[0] + ":=" +
                                                              file_name)
                            else:
                                new_file_name_and_path.append(file_name)
                        new_input_list_file.write(" ".join(new_file_name_and_path) + "\n")
            new_input_list_file.close()
            parsed_args.snpe_input_list = new_input_list_file_path

        if parsed_args.generate_plots:
            plots_requested = set()
            valid_plots = ["cdf", "histogram", "min_max_distribution", "diff"]
            for plot in parsed_args.generate_plots:
                if plot in valid_plots:
                    plots_requested.add(plot)
            parsed_args.generate_plots = plots_requested
        return parsed_args

    def get_all_associated_parsers(self):
        from qti.aisw.accuracy_debugger.lib.options.framework_diagnosis_cmd_options import FrameworkDiagnosisCmdOptions

        parsers_to_be_validated = [self.parser]
        if 'golden_output_reference_directory' not in self.args:
            option_class = FrameworkDiagnosisCmdOptions(self.args, False)
            option_class.initialize()
            parsers_to_be_validated.extend(option_class.get_all_associated_parsers())

        return parsers_to_be_validated
