# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import os
import shutil
import sys
import re

package_location = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../../../app.asar.unpacked/NetworkDiagnostics/GoldenI/')
build_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../NetworkDiagnostics/GoldenI/')
developer_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../NetworkDiagnostics/GoldenI/')
if os.path.exists(package_location):
    sys.path.append(package_location)
elif os.path.exists(build_location):
    sys.path.append(build_location)
elif os.path.exists(developer_location):
    sys.path.append(developer_location)
else:
    raise Exception("Unable to locate Golden-I")

from lib.wrapper.nd_tool_setup import ToolConfig
from lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions
from lib.options.verification_cmd_options import VerificationCmdOptions
from configparser import ConfigParser


class GoldenIWrapper:
    """
    This wrapper wraps the usage of Golden-I for this GraphVisualization tool.
    """

    def __init__(self, workspace):
        """
        Instantiates a GoldenIWrapper.
        """
        self.workspace = workspace
        if not os.path.isdir(self.workspace):
            os.makedirs(self.workspace)
        self.intermediate_dir = os.path.join(self.workspace, "intermediate_files")
        if not os.path.isdir(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
        self.first_raw_output_path = None
        self.second_raw_output_path = None
        self.verification_output_path = None

        self.profiling_file_name = 'profiling.csv'
        self.verification_file_name = 'summary.csv'

    @staticmethod
    def add_arg(flag, new_arg, args_list):
        """
        This function adds an argument, if it doesn't already exist, to an argument list.
        :param flag: parameter flag
        :param new_arg: argument to add
        :param args_list: argument list
        """
        if flag not in args_list:
            args_list.extend([flag, new_arg])

    @staticmethod
    def has_flag(flag, args_list):
        """
        This function determines if a parameter flag in an argument list.
        :param flag: parameter flag
        :param args_list: argument list
        """
        return flag in args_list

    @staticmethod
    def _parse_engine(args):
        """
        This function parses a list of arguments to get the specified engine and engine version.
        Updates engine option to only contain version to align with GoldenEye spec.
        :param args: list of arguments to parse
        :return: (engine, engine version)
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--engine', nargs='+', type=str, required=True,
                            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
                            help='Name of engine that will be running inference, '
                                 'optionally followed by the engine version.')
        parsed_args, _ = parser.parse_known_args(args)
        parsed_args.engine_version = None
        if len(parsed_args.engine) > 2:
            raise argparse.ArgumentError(parsed_args.engine, "Maximum two arguments required for inference engine.")
        elif len(parsed_args.engine) == 2:
            parsed_args.engine_version = parsed_args.engine[1]
            args[args.index('--engine')] = '--engine_version {}'.format(parsed_args.engine_version)
        else:
            del args[args.index('--engine')]

        parsed_args.engine = parsed_args.engine[0]

        return parsed_args.engine, parsed_args.engine_version

    def _create_devices_ini_file(self, args_list, file_name):
        """
        This method uses the target device and/or device id provided by the user
        to create a devices configuration file, which is necessary to run Golden-I.
        :param args_list: list of inference arguments
        """
        if os.path.splitext(file_name)[1] != '.ini':
            raise Exception("File extension must be .ini")

        target_device = args_list[args_list.index('--target_device') + 1]
        device_id_index = args_list.index('--devices_config_path') + 1

        config = ConfigParser()

        # config file in Golden-I must always have host device x86 listed
        config['x86'] = {}
        if target_device != 'x86':
            device_id = args_list[device_id_index]
            config[target_device] = {
                'device_id': device_id,
                'adb_path': '',
            }
        config_path = os.path.join(self.intermediate_dir, file_name)
        with open(config_path, 'w') as config_file:
            config.write(config_file)
        args_list[device_id_index] = config_path

    def prepare_inference_v_inference(self, inf1_args, inf2_args, verification_args):
        """
        This method runs inference twice, using a different set of inference arguments for each run,
        and augments each run's output path to a list of verification arguments.
        :param inf1_args: first list of inference arguments
        :param inf2_args: second list of inference arguments
        :param verification_args: list of verification arguments to augment
        """
        # prep inf engine 1
        self._create_devices_ini_file(inf1_args, 'devices_inference_1.ini')
        inf1_engine, inf1_engine_version = self._parse_engine(inf1_args)

        # prep inf engine 2
        self._create_devices_ini_file(inf2_args, 'devices_inference_2.ini')
        inf2_engine, _ = self._parse_engine(inf2_args)

        # run inf and parse inf args to get output dirs. Note: need to get relpath since user provides same output
        # dir for both inferences and latest will be overwritten
        self.run_inference(inf1_args)
        parsed_inf1_args = InferenceEngineCmdOptions(inf1_engine, inf1_args).parse()
        resolved_out1_path = os.readlink(os.path.join(os.path.dirname(parsed_inf1_args.output_dir), 'latest'))
        self.first_raw_output_path = os.path.join(resolved_out1_path, 'output')

        self.run_inference(inf2_args)
        parsed_inf2_args = InferenceEngineCmdOptions(inf2_engine, inf2_args).parse()
        resolved_out2_path = os.readlink(os.path.join(os.path.dirname(parsed_inf2_args.output_dir), 'latest'))
        self.second_raw_output_path = os.path.join(resolved_out2_path, 'output')

        # prep verification
        self.add_arg('--framework_results', self.first_raw_output_path, verification_args)
        self.add_arg('--inference_results', self.second_raw_output_path, verification_args)

    def prepare_golden_v_inference(self, inf_args, verification_args):
        """
        This method runs inference once and augments the inference's output path to
        a list of verification arguments.
        :param inf_args: list of inference arguments
        :param verification_args: list of verification arguments
        """
        # prep inf engine
        self._create_devices_ini_file(inf_args, 'devices_inference.ini')
        inf_engine, _ = self._parse_engine(inf_args)

        # run inf
        self.run_inference(inf_args)

        # parse inf args to get output dir
        parsed_inf_args = InferenceEngineCmdOptions(inf_engine, inf_args).parse()
        self.second_raw_output_path = os.path.join(os.path.dirname(parsed_inf_args.output_dir), 'latest', 'output')

        # prep verification
        self.add_arg('--inference_results', self.second_raw_output_path, verification_args)

    @staticmethod
    def sanitize_name(name):
        """
        This function modifies a given name to adhere with C++ naming standard as names (node or tensors) are used
        as variable name lookup in generated model.cpp.
        :param name: name to modify
        """
        # All separators should be _ to follow C++ variable naming standard
        name = re.sub(r'\W+', "_", name)
        # prefix needed as C++ variables cant start with numbers
        return name if name[0].isalpha() else "_" + name

    @staticmethod
    def run_inference(args):
        """
        This function executes inference using Golden-I.
        :param args: list of inference arguments
        """
        config = ToolConfig()
        ret = config.run_qnn_inference_engine(args)
        if ret != 0:
            exit(ret)

    @staticmethod
    def run_verification(args):
        """
        This function executes verification using Golden-I.
        :param args: list of verification arguments
        """
        config = ToolConfig()
        ret_verifier = config.run_verifier(args)
        if ret_verifier != 0:
            exit(ret_verifier)

    def prepare_verification(self, case, verification_args, verifier_config):
        """
        This method stores the output path of verification as well as the output path of
        any inference, if it hasn't already been set. Additionally, for the use cases of GOLDEN_V_INFERENCE
        and OUTPUT_V_OUTPUT, it calls a helper method to generate a tensor mapping required to run verification.
        Lastly, it creates a verification hyper-parameter mapping, in the case that a user specifies a verifier's
        hyper-parameter values.
        :param case: INFERENCE_V_INFERENCE, or GOLDEN_V_INFERENCE, or OUTPUT_V_OUTPUT
        :param verification_args: verification arguments
        :param verifier_config: mapping of verifier's hyper-parameters to their user-specified values
        """
        parsed_verification_args = VerificationCmdOptions(verification_args).parse()
        self.verification_output_path = os.path.join(os.path.dirname(parsed_verification_args.output_dir), 'latest')

        if case == "INFERENCE_V_INFERENCE":
            assert self.first_raw_output_path is not None and self.second_raw_output_path is not None
        if case == "GOLDEN_V_INFERENCE":
            assert self.first_raw_output_path is None and self.second_raw_output_path is not None
            self.first_raw_output_path = parsed_verification_args.framework_results
            self.create_tensor_mapping(verification_args)
        if case == "OUTPUT_V_OUTPUT":
            assert self.first_raw_output_path is None and self.second_raw_output_path is None
            self.first_raw_output_path = parsed_verification_args.framework_results
            self.second_raw_output_path = parsed_verification_args.inference_results
            self.create_tensor_mapping(verification_args)

        # pass in verifier config mapping if user specifies verification hyper-parameters
        if verifier_config is not None:
            filename = os.path.join(self.intermediate_dir, "verifier_config.json")
            self.write_to_json(verifier_config, filename)
            self.verification_output_path = os.path.join(self.verification_output_path,parsed_verification_args.default_verifier[0])
            self.add_arg('--verifier_config', filename, verification_args)


    @staticmethod
    def write_to_json(obj, json_filename):
        """
        This function converts a Python object into a json string and saves the json string to a file.
        :param obj: Python object to convert to json
        :param json_filename: filename of the json
        """
        json_string = json.dumps(obj)
        json_file = open(json_filename, "w")
        json_file.write(json_string)
        json_file.close()

    def create_tensor_mapping(self, verif_args):
        """
        This method creates a mapping between a tensor named in C++ format and the same tensor named in
        non-C++ format. This mapping is necessary to link tensors during the verification process.
        :param verif_args: list of verification arguments
        """
        tensor_mapping = {}
        for dirpath, _, goldens in os.walk(self.first_raw_output_path):
            for g in goldens:
                tensor_path, ext = os.path.splitext(
                    os.path.relpath(os.path.join(dirpath, g), self.first_raw_output_path))
                if ext == ".raw":
                    tensor_dir, tensor_name = os.path.dirname(tensor_path), os.path.basename(tensor_path)
                    tensor_mapping[self.sanitize_name(os.path.join(tensor_dir, tensor_name))] = tensor_path
        filename = os.path.join(self.intermediate_dir, "tensor_mapping.json")
        self.write_to_json(tensor_mapping, filename)
        self.add_arg('--tensor_mapping', filename, verif_args)

    def save_results(self):
        """
        This method saves the summary.csv from verification and the profiling.csv from each
        run of inference to {user-specified workspace}/output.
        """
        final_output_dir = os.path.join(self.workspace, 'csv_outputs')
        if not os.path.isdir(final_output_dir):
            os.makedirs(final_output_dir)

        # copying over CSV detailing verification accuracy
        shutil.copy(os.path.join(self.verification_output_path, self.verification_file_name),
                    os.path.join(final_output_dir, self.verification_file_name))

        # copying over CSVs detailing runtime performance
        src1_profiling = os.path.join(self.first_raw_output_path, self.profiling_file_name)
        src2_profiling = os.path.join(self.second_raw_output_path, self.profiling_file_name)
        if os.path.exists(src1_profiling) and os.path.exists(src2_profiling):
            shutil.copy(src1_profiling,
                        os.path.join(final_output_dir, 'inf1_profiling.csv'))
            shutil.copy(src2_profiling,
                        os.path.join(final_output_dir, 'inf2_profiling.csv'))

    @staticmethod
    def _is_null(value):
        """
        This function checks if a value is null, which is satisfied either when the value
        is literally None or is a string equivalent to 'null'.
        :param value: value to check if null
        :return: True if value is null or 'null'; False otherwise
        """
        return value is None or value == 'null'

    def set_working_dir(self, inf1, inf2, verif):
        """
        This method sets the working directory for the upcoming inference and verification runs.
        :param inf1: first set of inference arguments
        :param inf2: second set of inference arguments
        :param verif: set of verification arguments
        """
        for i in [inf1, inf2, verif]:
            if not self._is_null(i):
                self.add_arg('--working_dir', self.workspace, i)

    def _error_check_args(self, case, inf1, inf2, verif):
        """
        This method error checks the arguments of the run method based on use case.
        :param case: use case
        :param inf1: first set of inference arguments
        :param inf2: second set of inference arguments
        :param verif: set of verification arguments
        :raise Exception on invalid use case
        """
        required_inference_flags = ['--stage', '--engine', '--target_device', '--runtime', '--devices_config_path',
                                    '--architecture', '--engine_path', '--input_list', '--ndk_path',
                                    '--qnn_model_cpp_path']

        assert not self._is_null(verif), "Verification arguments must be provided"
        assert self.has_flag('--default_verifier', verif), \
            "A preferred verifier must be included among verification arguments"

        if case == "INFERENCE_V_INFERENCE":
            assert not self._is_null(inf1) and not self._is_null(inf2), \
                "INFERENCE_V_INFERENCE requires two sets of inference arguments"
            for f in required_inference_flags:
                assert self.has_flag(f, inf1), \
                    f"The first set of inference arguments is missing the following flag: {f}"
                assert self.has_flag(f, inf2), \
                    f"The second set of inference arguments is missing the following flag: {f}"
            assert not self.has_flag('--framework_results', verif), \
                "INFERENCE_V_INFERENCE does not accept golden outputs among its verification arguments"
            assert not self.has_flag('--inference_results', verif), \
                "INFERENCE_V_INFERENCE does not accept inference outputs among its verification arguments"
        elif case == "GOLDEN_V_INFERENCE":
            assert self._is_null(inf1) ^ self._is_null(inf2), \
                "GOLDEN_V_INFERENCE requires exactly one set of inference arguments."
            inf = inf1 if not self._is_null(inf1) else inf2
            for f in required_inference_flags:
                assert self.has_flag(f, inf), \
                    f"The set inference arguments is missing the following flag: {f}"
            assert self.has_flag('--framework_results', verif), \
                "GOLDEN_V_INFERENCE requires a path to golden outputs among its verification arguments"
            assert not self.has_flag('--inference_results', verif), \
                "GOLDEN_V_INFERENCE does not accept inference outputs among its verification arguments"
        elif case == "OUTPUT_V_OUTPUT":
            assert self._is_null(inf1) and self._is_null(inf2), \
                "OUTPUT_V_OUTPUT requires exactly zero sets of inference arguments"
            assert self.has_flag('--framework_results', verif), \
                "GOLDEN_V_INFERENCE requires a path to golden outputs among its verification arguments"
            assert self.has_flag('--inference_results', verif), \
                "GOLDEN_V_INFERENCE requires a path to inference outputs among its verification arguments"
        else:
            raise Exception("Invalid use case.")

    def run(self, use_case, inference_1_args, inference_2_args, verification_args, verifier_config=None):
        """
        This method runs the GoldenIWrapper.
        :param use_case: INFERENCE_V_INFERENCE, or GOLDEN_V_INFERENCE, or OUTPUT_V_OUTPUT
        :param inference_1_args: comma-separated list of of inference arguments
        :param inference_2_args: comma-separated list of inference arguments
        :param verification_args: comma-separated list of verification arguments
        :param verifier_config: json object of hyperparameters for the verifier specified in verification_args
        :raise Exception if an invalid use case is specified
        """
        # since we spawn a copy subprocess with a copy of our path, we have to manually modify the
        # env value
        os.environ["PYTHONPATH"] = os.pathsep.join([os.path.dirname(__file__)]
                                                   + os.environ.get("PYTHONPATH", "").split(os.pathsep))

        self._error_check_args(use_case, inference_1_args, inference_2_args, verification_args)
        self.set_working_dir(inference_1_args, inference_2_args, verification_args)

        if use_case == "INFERENCE_V_INFERENCE":
            self.prepare_inference_v_inference(inference_1_args, inference_2_args, verification_args)
        elif use_case == "GOLDEN_V_INFERENCE":
            inf_args = inference_1_args if not self._is_null(inference_1_args) else inference_2_args
            self.prepare_golden_v_inference(inf_args, verification_args)

        self.prepare_verification(use_case, verification_args, verifier_config)
        self.run_verification(verification_args)
        self.save_results()
