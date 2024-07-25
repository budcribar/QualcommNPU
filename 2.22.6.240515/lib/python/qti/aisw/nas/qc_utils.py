# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import sys
import json
import glob
import csv
import copy
import shutil
import argparse
import subprocess
import numpy as np
import multiprocessing
import subprocess
import shlex
import time

import logging

from pathlib import Path

# setting logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# setting target compiler (eg ANDROID_NDK_ROOT)
MY_CWD = os.getcwd()
COMPILER = '/qnn/compiler'
os.environ["ANDROID_NDK_ROOT"] = COMPILER
sys.path.append(COMPILER)

SDK_PATH = os.environ.get("QNN_SDK_ROOT")
if not SDK_PATH:
    SDK_PATH='/qnn/sdk'
    logger.error(
        "ERROR: Please set environment before running this script. Follow steps mentioned in README")
    exit(1)

# including files present in benchmark folder sdk
sys.path.append(os.path.join(SDK_PATH, "benchmarks/"))


from common_utils.constants import LOG_FORMAT
from common_utils import exceptions
from bm_utils.qnn import QNN
from bm_utils.error import Error
from bm_utils import bm_parser, bm_config, bm_bm, bm_md5, bm_writer, bm_device , bm_constants

from common_utils.adb import Adb
from common_utils.device import Device
from common_utils.common import execute

TNR_TEST_TIMEOUT = 300

class Converter(object):

    def __init__(self, config):
        self.cfg = config
        self.user_data = '/qnn/user_data' if os.path.isdir('/qnn/user_data') else '/qnn/output'

    def convert(self, model, output_path, backend='DSP'):
        """Select and run the proper converter"""

        if not model:
            logging.error("Invalid model %s received!", model)
            exit(1)

        cfg = self.cfg
        model_cfg =  cfg['Model']
        converter_cfg = model_cfg['Conversion']
        cmd = converter_cfg['Command']

        cmd_args = shlex.split(cmd)
        if backend == 'DSP':
            if 'InputList'not in model_cfg:
                logging.error("Must pass InputList arg for DSP backends")
                exit(1)
            else:
                input_list = os.path.join(self.user_data, model_cfg['InputList'])

            input_list_found = False
            for i,k in enumerate(cmd_args):
                # Check for the graph definition
                if '--input_list' == k:
                    cmd_args[i+1] = input_list
                    input_list_found = True
            if not input_list_found:
                cmd += ' --input_list ' + input_list
        else:
            for i,k in enumerate(cmd_args):
                if '--input_list' == k:
                    logging.error('Trying to quantize a model with "{} {}" for backend {} which does not support it.'
                        .format(k, cmd_args[i+1], backend))
                    exit(1)

        logging.debug("Original cmd: %s", cmd)
        converted_model_path = os.path.join(output_path,"qnn_model.cpp")
        found_graph = False
        found_output = False
        for i,k in enumerate(cmd_args):
            # Check for the graph definition
            if '--input_network' == k or '-i' == k:
                cmd_args[i+1] = model
                found_graph = True

            if '--output' == k:
                cmd_args[i+1] = converted_model_path


        if found_graph == False:
            cmd_args.append('-i')
            cmd_args.append(model)

        if found_output == False:
            cmd_args.append('--output')
            cmd_args.append(converted_model_path)


        logging.info("Modified converter cmd: %s", cmd_args)
        logging.info("*******************************************************************")
        logging.info("Converting and quantizing received model:\n%s\n", cmd)
        result = subprocess.run(cmd_args, cwd=self.user_data, check=True) #, capture_output=True, text=True, check=True)
        logging.debug("Model conversion output %s", result.stdout)
        if result.returncode != 0:
            logging.error("Model conversion failure w/error %s", result.stderr)
            return ""

        logging.info("Model artifacts generated: ")
        logging.info("%s",converted_model_path)
        logging.info("%s",Path(converted_model_path).with_suffix(".bin"))
        logging.info("*******************************************************************\n")

        logging.info("stdout: %s", result.stdout)
        logging.info("stderr: %s", result.stderr)
        return converted_model_path


    def compile(self, model, output_path, toolchain='aarch64-android'):

        # Ff a previous failure occurred there won't be a valid model
        if not model:
            return ""

        cmd = SDK_PATH+'/bin/x86_64-linux-clang/qnn-model-lib-generator'
        p_model = Path(model)
        compiled_model = ""
        cmd += ' -c ' + model + \
               ' -b ' + str(p_model.with_suffix('.bin')) + \
               ' -t ' + toolchain + \
               ' -o ' + output_path

        logging.info("*******************************************************************");
        logging.info("Compiling model for architechture %s:\n%s", toolchain, cmd);

        cmd_args = shlex.split(cmd)
        result = subprocess.run(cmd_args, cwd=self.user_data, check=True)

        for root, dirs, files in os.walk(output_path):
            for name in files:
                if name.endswith('.so'):
                    compiled_model = os.path.join(root, name)
                    break

        logging.info("Compiled model available here %s\n", compiled_model);
        logging.info("*******************************************************************\n");

        return compiled_model


class OutputVerifier(object):
    def __init__(self):
        self.checker_identifier = 'basic_verifier'
        pass

    def verify_output(self, inputs_dir, outputs_dir, expected_outputs_dir,
                      sanity=False, num_of_batches=1, output_data_type=None):
        if sanity:
            input_list_path = 'input_list_sanity.txt'
        else:
            input_list_path = 'input_list.txt'
        with open(os.path.join(inputs_dir, input_list_path)) as inputs_list_file:
            inputs_list = inputs_list_file.readlines()
            for inputs in inputs_list:
                if inputs.startswith('#'):
                    inputs_list.remove(inputs)

            # verify if the number of results is the same missing
            number_of_checks_passing = 0
            iterations_verification_info_list = []
            failure_list = []
            for iteration in range(0, len(inputs_list)):
                iteration_input_files = (
                    inputs_list[iteration].strip()).split(' ')
                iteration_input_files = [input_file.split(
                    ':=')[-1] for input_file in iteration_input_files]
                iteration_input_files = [
                    os.path.join(
                        inputs_dir,
                        f) for f in iteration_input_files]
                iteration_result_str = 'Result_' + str(iteration)
                is_passing, iterations_verification_info, failures = self._verify_iteration_output(
                    iteration_input_files,
                    outputs_dir,
                    expected_outputs_dir,
                    iteration_result_str,
                    num_of_batches,
                    output_data_type
                )

                if is_passing:
                    number_of_checks_passing += 1
                else:
                    failure_list.extend(failures)
                iterations_verification_info_list += iterations_verification_info

            return number_of_checks_passing, \
                   len(inputs_list), \
                   iterations_verification_info_list, \
                   failure_list

    def _verify_iteration_output(self, input_files, outputs_dir, expected_dir, iteration_result_str,
                                 num_of_batches=1, output_data_type=None):
        is_passing = True
        iterations_verification_info_list = []
        failure_list = []
        for root, dirnames, filenames in os.walk(os.path.join(expected_dir, iteration_result_str)):
            for output_layer in filenames:
                if "linux" in sys.platform:
                    output_result = root.split(iteration_result_str)[1].replace("/", "_")[1:]
                else:
                    output_result = root.split(iteration_result_str)[1].replace("\\", "_")[1:]
                if output_result:
                    output_name = output_result.replace(":", "_") + "_" + \
                                  output_layer.replace("-", "_").replace(":", "_")
                else:
                    output_name = output_layer.replace("-", "_").replace(":", "_")

                if output_data_type is not None:
                    output = output_name.split('.')
                    output_name = output[0] + "_" + \
                                  output_data_type.split('_')[0] + "." + output[1]
                expected_output_file = os.path.join(root, output_layer)

                output_file = os.path.join(outputs_dir, iteration_result_str, output_name)

                # hack around for splitter - needs to be removed
                # in the future
                if not os.path.exists(expected_output_file):
                    expected_output_file_intermediate = os.path.join(
                        expected_dir,
                        output_file[output_file.find('Result_'):]).split(iteration_result_str)

                    expected_output_file = expected_output_file_intermediate[0] \
                                           + iteration_result_str + '/' + \
                                           expected_output_file_intermediate[1]
                # end of hack
                # changing output name if raw file name starts with integer
                if not (os.path.exists(output_file)) and output_name[0].isdigit():
                    output_name = '_' + output_name
                    output_file = os.path.join(
                        outputs_dir,
                        iteration_result_str,
                        output_name
                    )
                iteration_verification_info = self._verify_iteration_output_file(
                    input_files,
                    output_file,
                    expected_output_file,
                    num_of_batches
                )
                iterations_verification_info_list.append(
                    iteration_verification_info[1])
                if not iteration_verification_info[0]:
                    is_passing = False
                    failure_list.append(expected_output_file)
        return is_passing, iterations_verification_info_list, failure_list

    def _verify_iteration_output_file(self, input_files, output_file, expected_output_file,
                                      num_of_batches=1):
        try:
            output = np.fromfile(output_file, dtype=np.float32)
            expected_output = np.fromfile(
                expected_output_file, dtype=np.float32)
        except IOError:
            raise Exception('Can not open the golden or predicted files (names does not match). '
                            'Please check both the output and golden directories contain: %s'
                            % os.path.basename(output_file))

        result = self._verify_iteration_output_helper(input_files, output, expected_output,
                                                      num_of_batches)
        if not result[0]:
            logger.error(
                'Failed to verify %s on %s' % (expected_output_file, self.checker_identifier))
        return result

    def _verify_iteration_output_helper(self, input_files, output, expected_output, num_of_batches):
        pass


class CosineSimilarity(OutputVerifier):
    def __init__(self, threshold = 0.9):
        super(CosineSimilarity, self).__init__()
        self.checker_identifier = 'CosineSimilarity'
        self.threshold = threshold

    def _verify_iteration_output_helper(
            self, input_files, output, expected_output, num_of_batches=1):
        output, expected_output = output.flatten(), expected_output.flatten()
        num = output.dot(expected_output.T)
        denom = np.linalg.norm(output) * np.linalg.norm(expected_output)
        if denom == 0:
            return [False, False]
        else:
            similarity_score = num / denom
            if similarity_score >= self.threshold:
                return [True, True]
            else:
                return [False, False]


class RtolAtolOutputVerifier(OutputVerifier):
    def __init__(self, rtolmargin=1e-2, atolmargin=1e-2):
        super(RtolAtolOutputVerifier, self).__init__()
        self.checker_identifier = 'RtolAtolVerifier_w_r_' + \
                                  str(rtolmargin) + '_a_' + str(atolmargin)
        self.rtolmargin = rtolmargin
        self.atolmargin = atolmargin

    def _calculate_margins(self, expected_output):
        return self.atolmargin, self.rtolmargin

    def _verify_iteration_output_helper(
            self, input_files, output, expected_output, num_of_batches=1):
        adjustedatolmargin, adjustedrtolmargin = self._calculate_margins(
            expected_output)
        match_array = np.isclose(
            output,
            expected_output,
            atol=adjustedatolmargin,
            rtol=adjustedrtolmargin)

        notclose = (len(match_array) - np.sum(match_array))
        if notclose == 0:
            return [True, False]
        else:
            return [False, False]


class RunTestPackage(object):

    def runner(self):
        pass

    @staticmethod
    def _run_command(cmd):
        try:
            logger.debug("Running - {}".format(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            return_code = process.returncode
            return out, err, return_code
        except Exception as err:
            logger.error("Error Observed: {}".format(err))
            return "", "Error", "1"

    @staticmethod
    def check_models(user_model):
        """
            This function will check model name.
            By default model_run will have all models.
        """

        if not user_model:
            model_run = MODELS
        elif isinstance(user_model, str):
            if user_model.lower() == "all":
                model_run = MODELS
            else:
                model_run = user_model.split(" ")
        else:
            model_run = MODELS
        logger.debug("Running test for {}".format(model_run))
        return model_run

    @staticmethod
    def check_dsp_type(chipset):
        """
        :description: This fucntion will return dsp type based on soc.
        :return: dsp_type
        """
        # add more mapping if required
        dsp_type = None
        if chipset in ["8350", "7350", "7325", "lahaina", "cedros", "kodiak"]:
            dsp_type = 'v68'
        elif chipset in ["8450", "waipio", "7450", '8475', 'fillmore', 'palima', '8415', 'alakai', 'SXR2230P' , 'halliday']:
            dsp_type = 'v69'
        elif chipset in ["6450", "netrani"]:
            dsp_type = 'v69-plus'
        elif chipset in ["kailua", '8550', '1230', 'sxr1230', '2115', 'ssg2115']:
            dsp_type = 'v73'
        elif chipset in ["6375", "4350", "strait", "mannar", "5165", 'qrb5165', '610', 'qcs610']:
            dsp_type = 'v66'
        elif chipset in ["lanai", "8650"]:
            dsp_type = 'v75'
        else:
            logger.error("Please provide --dsp_type argument value")
            exit(1)
        return dsp_type

    @staticmethod
    def check_mem_type(chipset):
        """
        :description: This function will return mem type based on soc.
        :return: mem_type
        """
        # add more mapping if required
        mem_type = None
        if chipset in ['8450', 'waipio', 'kailua', '8550', '8475', 'palima', '8415', 'alakai', 'lanai', '8650', 'SXR2230P' , 'halliday']:
            mem_type = '8m'
        elif chipset in ["8350", "lahaina"]:
            mem_type = '4m'
        elif chipset in ['7350', 'cedros', '7325', 'kodiak', '7450', 'fillmore', '1230', \
                         'sxr1230', '2115', 'ssg2115', "6450", "netrani"]:
            mem_type = '2m'

        return mem_type

    @staticmethod
    def check_toolchain(chipset):
        """
        :description: This function will return toolchain based on soc.
        :return: toolchain
        """
        # add more mapping if required
        mem_type = None
        if chipset in ['5165', 'qrb5165', '1230', 'sxr1230']:
            toolchain = 'aarch64-oe-linux-gcc9.3'
        elif chipset in ['610', 'qcs610']:
            toolchain = 'aarch64-oe-linux-gcc8.2'
        else:
            toolchain = 'aarch64-android'
        return toolchain

    @staticmethod
    def copy_sdk_artifacts(tmp_work_dir, dsp_type, toolchain):
        lib_dir = os.path.join(tmp_work_dir, toolchain, 'lib')
        bin_dir = os.path.join(tmp_work_dir, toolchain, 'bin')
        os.makedirs(lib_dir, exist_ok = True)
        os.makedirs(bin_dir, exist_ok = True)
        shutil.copy(os.path.abspath(os.path.join(
            SDK_PATH,
            'lib', toolchain, 'libQnnCpu.so')),
            lib_dir)
        shutil.copy(os.path.abspath(os.path.join(
            SDK_PATH,
            'bin', toolchain, 'qnn-profile-viewer')),
            bin_dir)
        shutil.copy(os.path.abspath(os.path.join(
            SDK_PATH,
            'lib', toolchain, 'libQnnGpu.so')),
            lib_dir)
        shutil.copy(os.path.abspath(os.path.join(
            SDK_PATH,
            'bin', toolchain, 'qnn-net-run')),
            bin_dir)
        shutil.copy(os.path.abspath(os.path.join(
            COMPILER,
            'toolchains', 'llvm', 'prebuilt', 'linux-x86_64', 'sysroot', 'usr', 'lib', 'aarch64-linux-android', 'libc++_shared.so')),
            lib_dir)
        if dsp_type == 'v68' or dsp_type == 'v69' or dsp_type == 'v69-plus' or dsp_type == 'v73' or dsp_type == 'v75':
            shutil.copy(os.path.abspath(os.path.join(
                SDK_PATH,
                'lib', toolchain, 'libQnnHtpNetRunExtensions.so')),
                lib_dir)
            if dsp_type == 'v69':
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtp.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpV69Stub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpPrepare.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v69', 'unsigned', 'libQnnHtpV69Skel.so')),
                    lib_dir)
            elif dsp_type == 'v69-plus':
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtp.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpV69PlusStub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpPrepare.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v69-plus', 'unsigned', 'libQnnHtpV69PlusSkel.so')),
                    lib_dir)
            elif dsp_type == 'v73':
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtp.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpV73Stub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpPrepare.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v73', 'unsigned', 'libQnnHtpV73Skel.so')),
                    lib_dir)
            elif dsp_type == 'v75':
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtp.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpV75Stub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpPrepare.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v75', 'unsigned', 'libQnnHtpV75Skel.so')),
                    lib_dir)
            else:
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtp.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpV68Stub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnHtpPrepare.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v68', 'unsigned', 'libQnnHtpV68Skel.so')),
                    lib_dir)
        else:
            shutil.copy(os.path.abspath(os.path.join(
                SDK_PATH,
                'lib', toolchain, 'libQnnDspNetRunExtensions.so')),
                lib_dir)
            shutil.copy(os.path.abspath(os.path.join(
                SDK_PATH,
                'lib', toolchain, 'libQnnDsp.so')),
                lib_dir)
            shutil.copy(os.path.abspath(os.path.join(
                MY_CWD,
                'backend_configs', 'signed_pd_session.json')),
                tmp_work_dir)
            if dsp_type == 'v66':
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnDspV66Stub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v66', 'unsigned', 'libQnnDspV66Skel.so')),
                    lib_dir)

            if dsp_type == 'v65':
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', toolchain, 'libQnnDspV65Stub.so')),
                    lib_dir)
                shutil.copy(os.path.abspath(os.path.join(
                    SDK_PATH,
                    'lib', 'hexagon-v65', 'unsigned', 'libQnnDspV65Skel.so')),
                    lib_dir)

    @staticmethod
    def push_to_device(adb, inputs, on_device_path):
        ret = True
#        logger.info('Pushing artifacts {} to device: {}'.format(inputs, adb.device))
        for inp in inputs:
            if os.path.isdir(inp):
                logger.info("Pushing {} to {} on device {}".format(inp, on_device_path, adb._adb_device))
            else:
                logger.debug(os.path.join(on_device_path, os.path.basename(inp)))
                adb.shell('rm -rf {0}'.format(os.path.join(on_device_path, os.path.basename(inp))))
                logger.debug('Existing test directory removed from device.')
            return_code, out, err = adb.push(inp, on_device_path)
            if return_code:
                ret = False
                logger.error("ADB Push Failed!!")
                break
        return ret

    @staticmethod
    def pull_from_device(adb, on_device_inputs, host_dest_path):
        ret = True
        for inp in on_device_inputs:
            return_code, out, err = adb.pull(inp, host_dest_path)
            if return_code:
                ret = False
                logger.error("ADB Pull Failed!!")
                break
        return ret

    def check_deviceid(self, device, hostname):
        """
        :description:
            This function will fetch the devices connected to local host.
        :return:
            device id
        """
        if not device and hostname == "localhost":
            cmd = "adb devices | awk '{print $1;}'"
            out, err, code = self._run_command(cmd)
            if code or err:
                logger.error("Error Observed: {}".format(err))
                logger.error("Check if device is detectable and provide device id as argument")
                return

            if not isinstance(out, str):
                out = out.decode('utf-8')
            devices = out.split("\n")
            device_id = ""
            if devices[1]:
                device_id = devices[1]
            return device_id

        elif not device and not hostname == "localhost":
            logger.error("Please provide device id")
            exit(1)
        else:
            return device

    def _check_verifier(self, conf_file_path, model, backend_option, device_id, host, result_summary):
        # verify output with goldens
        with open(conf_file_path, 'r') as f:
            input_config = json.load(f)

        source = '/'.join([input_config["DevicePath"], input_config["Model"]["Name"], 'output'])
        dest = os.path.join(MY_CWD, model)
        output_location = os.path.join(dest, 'output')

        if os.path.isdir(output_location):
            shutil.rmtree(os.path.join(dest, 'output'))

        logger.debug('Netrun passed on Model: %s for Runtime: %s' % (
            input_config["Model"]["Name"],
            backend_option
        ))

        adb_cmd = 'adb -s ' + device_id + ' -H ' + host + ' pull ' + source + ' ' + dest
        out, err, code = self._run_command(adb_cmd)
        if err or code:
            result_summary[model][backend_option] = 'Fail'
            logger.error("Error observed while running pulling output in verifier")

        if not code:
            check_result = True
            for checker in input_config["Model"]["Verifier"]:
                logger.info('Running  verification  on Model:%s with verifier : %s' % (
                    input_config["Model"]["Name"],
                    eval(checker).checker_identifier
                ))
                comparision_result = eval(checker).verify_output(
                    os.path.join(MY_CWD, input_config["Model"]["Data"][0], '..'),
                    os.path.join(dest, 'output'),
                    os.path.join(dest, 'Goldens')
                )
                logger.info("----------------------------------------")
                if comparision_result[0] == comparision_result[1]:
                    logger.info('All(%s/%s) test cases passed on Model:%s with Verifier: %s' % (
                        comparision_result[0],
                        comparision_result[1],
                        input_config["Model"]["Name"],
                        eval(checker).checker_identifier
                    ))
                    check_result = check_result and True
                else:
                    check_result = check_result and False
                    logger.warning('%s Verifier Accuracy:%s/%s ' % (
                        eval(checker).checker_identifier,
                        comparision_result[0],
                        comparision_result[1]
                    ))
                logger.info("------------------------------------------")

            if check_result:
                result_summary[model][backend_option] = 'PASS'
            else:
                result_summary[model][backend_option] = 'Fail'
            logger.info("##############################################################")
        else:
            result_summary[model][backend_option] = 'Fail'
            logger.error('Not able to pull output folder. Error :%s' % out)

        return result_summary

    @staticmethod
    def generate_config_file(conf_file_path):
        filename = os.path.basename(conf_file_path)
        temp_json = os.path.join(os.path.dirname(conf_file_path), 'windows_' + filename)
        conf_data = json.load(open(conf_file_path))
        conf_data["Model"]["qnn_model"] = conf_data["Model"]["qnn_model"].replace('/', '\\')
        conf_data["Model"]["InputList"] = conf_data["Model"]["InputList"].replace('/', '\\')
        data = conf_data["Model"]["Data"]
        conf_data["Model"]["Data"] = []
        for i in data:
            conf_data["Model"]["Data"].append(i.replace('/', '\\'))

        with open(temp_json, 'w') as f:
            json.dump(conf_data, f, indent=2)

        return temp_json


    def run_test(self, itr, backends, run_model, device_id, host, soc, toolchain,
                 dsp, mem_type, verify=None, perf="burst", shared_buffer=False):
        """
        :description:
            This function will run benchmark script.
        :arguments:
            itr: number of iterations
            backends: available backends where test will run.
            run_model: number of models where test will run.
            model_type: type of model. Either quantized, float or both.
            device_id: device id on which test will run.
            host: host machine where device is connected. By default- localhost.
            soc: chipset number. 8350 8150 845
            dsp: dsp type based on soc number. Example - v68, v66
        """
        if sys.platform.startswith('win'):
            cmd = 'python {0} -json -p {1} -o output_{2} --arm_prepare'.format(BENCH_FILE, perf, soc)
        else:
            cmd = 'python3 {0} -json -p {1} -o output_{2} --arm_prepare'.format(BENCH_FILE, perf, soc)

        if dsp and ("DSP" in backends or "DSP_FP16" in backends):
            cmd += ' --dsp_type {}'.format(dsp)
        if device_id:
            cmd += ' -v {}'.format(device_id)
        if host:
            cmd += ' -r {}'.format(host)
        if shared_buffer:
            cmd += ' --shared_buffer'
        if toolchain != 'aarch64-android':
            cmd += ' -t {}'.format(toolchain)
        conf_file = 'tmp_config.json'
        # running qnn_bench script for each model and model type
        try:
            result_summary = dict()
            for model in run_model:
                result_summary[model] = dict()
                for backend in backends:
                    if backend == 'DSP_FP16' and dsp in ['v65', 'v66', 'v68']:continue
                    # check for configuration and backend
                    _cmd = cmd
                    if backend == "DSP":
                        model_type = 'quantized'
                        out_name = self.set_config(model, model_type, backend, itr, mem_type, verify)
                        if dsp == 'v68' or dsp == 'v69' or dsp == 'v69-plus' or dsp == 'v73':
                            _cmd += ' --backend_config backend_configs/htp_config.json'
                    else:
                        model_type = 'float'
                        out_name = self.set_config(model, model_type, backend, itr, mem_type, verify)
                        if backend == 'DSP_FP16':
                            _cmd += ' --backend_config backend_configs/htp_fp16_config.json'
                        elif backend == 'GPU_FP16':
                            _cmd += ' --backend_config backend_configs/gpu_config.json'

                    conf_file_path = os.path.join(
                        MY_CWD,
                        model,
                        conf_file
                    )

                    if "linux" not in sys.platform:
                        conf_file_path = self.generate_config_file(conf_file_path)
                    final_cmd = _cmd + ' -be {} -c {}'.format(backend, conf_file_path)

                    logger.debug("Running command: \n{}".format(final_cmd))
                    out, err, code = self._run_command(final_cmd)
                    if code:
                        logger.info(code)
                        logger.info("--------------------------------------------")
                        logger.info('FAIL : {0} Test Failed for {1} on {2}'.format(self.output_file_name[:-1].upper(), model, backend))
                        logger.info("--------------------------------------------")
                        logger.info('################################################')
                        result_summary[model][backend] = "Fail"
                    else:
                        if verify:
                            result_summary = self._check_verifier(conf_file_path, model, backend, device_id, host, result_summary)
                        else:
                            logger.info("--------------------------------------------")
                            logger.info('PASS : {0} Test Ran Sucessfully for {1} on {2}'.format(self.output_file_name[:-1].upper(), model, backend))
                            logger.info("--------------------------------------------")
                            result_summary[model][backend] = "PASS"
                            if self.output_file_name[:-1].upper() == "BENCHMARK":
                                out_path = os.path.join(os.getcwd(), 'output_' + soc, out_name, 'results','latest_results')
                                if not os.path.exists(out_path):
                                    logger.info('################################################')
                                    continue
                                out_file_name = [f for f in os.listdir(out_path) if f.endswith('.json')]
                                if len(out_file_name) > 0:
                                    out_file_name = out_file_name[0]

                                out_file = os.path.join(out_path, out_file_name)
                                new_out_file = os.path.join(os.getcwd(), soc + "_" + backend + "_" + out_file_name)
                                if os.path.exists(out_file):
                                    shutil.copyfile(out_file, new_out_file)
                                    with open(out_file, 'r') as hdl:
                                        data = json.load(hdl)
                                        logger.info("Benchmark Profiling Metrics: ")
                                        logger.info("--------------------------------------------")
                                        logger.info("Total Inference Time [NetRun]: ")
                                        logger.info(json.dumps(data['Execution_Data'][backend]["Total Inference Time [NetRun]"], indent=4, sort_keys=True))
                                        logger.info("--------------------------------------------")
                                        logger.info("Detailed Result File can be found at {0}".format(new_out_file))
                                        return {"latency_in_milliseconds": json.dumps(data['Execution_Data'][backend]["Total Inference Time [NetRun]"])}

                            logger.info('################################################')


                    if "linux" not in sys.platform:
                        os.remove(conf_file_path)
        except Exception as e:
            logger.error("Error while running qnn_bench script. Error: {}".format(e))

        with open(self.output_file_name + soc + ".json", "w") as outfile:
            json.dump(result_summary, outfile)
        logger.info('{0} Results Summary : '.format(self.output_file_name[:-1].upper()))
        logger.info(json.dumps(result_summary, indent=4, sort_keys=True))

    def run_throughput(self, device, hostname, soc, toolchain, mem, dsp_type, conf_files_list):
        """
        :description:
            This function will run throughput app.
        :arguments:
            device_id: device id on which test will run.
            host: host machine where device is connected. By default- localhost.
            soc: chipset number. 8350 8150 845
            conf_files_list: configuration files list to run
        """
        tmp_dir = os.path.join(MY_CWD, "tmp_work")
        results_dir = os.path.join(MY_CWD, "output_" + str(soc))

        adb = Adb('adb', device, hostname=hostname)
        logger.info("Starting Test Runs on Device")
        result_summary = dict()
        for test_cfg in conf_files_list:
            logger.info('########################################################################################')
            logger.info('Config File: {0}'.format(test_cfg))
            try:
                if sys.platform.startswith('win'):
                    test_cfg = self.generate_throughput_config_file(test_cfg)
                gen_test_inputs_path = self.prepare_inputs(test_cfg, mem, dsp_type, toolchain, tmp_dir)
            except Exception as e:
                logger.error("Error observed while preparing inputs.")
                continue

            logger.debug(
                "Pushing the generated model inputs to device - {0}".format(adb._adb_device))

            ret_status = self.push_to_device(adb, [gen_test_inputs_path], '/data/local/tmp/')
            file_name = os.path.basename(gen_test_inputs_path)
            file_name = "_".join(file_name.split("_")[1:])
            if ret_status:
                run_status = self.run_test_config(adb, gen_test_inputs_path, '/data/local/tmp/',
                                                  results_dir)
                if run_status:
                    result_summary[file_name] = "Fail"
                else:
                    result_summary[file_name] = "PASS"
            else:
                result_summary[file_name] = "Fail"
                logger.error("Device not detectable")
            logger.info('########################################################################################')

        with open(self.output_file_name + soc + ".json", "w") as outfile:
            json.dump(result_summary, outfile)
        logger.info('{0} Results Summary : '.format(self.output_file_name[:-1].upper()))
        logger.info(json.dumps(result_summary, indent=4, sort_keys=True))

        #shutil.rmtree(tmp_dir)
        return

    def run_qnn_platform_validator(self, device, hostname, dsp_type, socName, soc):
        """
        :description:
            This function will run qnn platform validator app.
        :arguments:
            device_id: device id on which test will run.
            host: host machine where device is connected. By default- localhost.
            dsp_type: DSP architecture
        """

        cmd = 'qnn-platform-validator' \
              ' --backend all' \
              ' --directory ' + os.environ['QNN_SDK_ROOT'] + \
              ' --dsp_type ' + dsp_type + \
              ' --testBackend --deviceId ' + device + \
              ' --coreVersion --libVersion' \
              ' --remoteHost ' + hostname + \
              ' --debug'
        if socName is not None and soc is not None:
            cmd += ' --socName ' + socName + ' --socId ' + soc

        if sys.platform.startswith('win'):
            exp_cmd = 'set PYTHONPATH={}\\benchmarks'.format(os.environ['QNN_SDK_ROOT'])
            py_cmd = 'python {}\\bin\\x86_64-linux-clang\\'.format(os.environ['QNN_SDK_ROOT'])
            exe_cmd = exp_cmd + ' && ' + py_cmd + cmd
        else:
            exp_cmd = 'export PYTHONPATH=$PYTHONPATH:$QNN_SDK_ROOT/benchmarks'
            exe_cmd = exp_cmd + ';' + cmd

        try:

            logger.info("Starting QNN Platform Validator, cmd: " + exe_cmd)
            out, err, code = self._run_command(exe_cmd)
            if code or err:
                logger.error("Error Observed: {}".format(err))
            else:
                logger.info(
                    "QNN Platform Validator executed Successfully and Reports available at :: " +
                    os.path.join(os.getcwd(), 'output', 'Result_' + device + '.csv'))

                with open(os.path.join(os.getcwd(), 'output', 'Result_' + device + '.csv'), 'r') as f:
                    result = csv.reader(f, delimiter=',')
                    for res in result:
                        logger.info(res)
                logger.info('############################################################')
        except Exception as e:
            logger.error("Error while running qnn platform Validator. Error: {}".format(e))


class RunBenchmark(RunTestPackage):

    def __init__(self, args, latency_worker_id):

        self.converter = Converter(args)
        self.local_path=args["HostRootPath"]
        self.device_path=args['DevicePath']
        self.deviceId = self.check_deviceid(args['Devices'][latency_worker_id], 'localhost')
        self.chipset =  args['Chipset'] if 'Chipset' in args else None
        self.dsp_type = self.check_dsp_type(self.chipset)
        self.mem_type = self.check_mem_type(self.chipset)
        self.toolchain = self.check_toolchain(self.chipset)
        adb = Adb('adb', self.deviceId)
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

        self.device = Device(self.deviceId, [self.toolchain], self.device_path)
        self.device.init_env(self.device_path, False)
        self.artifacts=os.path.join('/tmp', self.deviceId)
        if os.path.exists(self.artifacts):
            shutil.rmtree(self.artifacts)
        os.makedirs(self.artifacts)
        self._setup_env()
        self.copy_sdk_artifacts(self.artifacts, self.dsp_type, self.toolchain)

        self.hostname = 'localhost'
        logger.debug("Device id on which test will run: {}".format(self.deviceId))

        self.backend = args['Backends']

        self.itr = 1 #int(args.iterations)
        self.verify = False
        self.output_file_name = "benchmark_"
        self.perf = 'perf'
        self.shared_buffer = False #args.shared_buffer

        self.input_list = args["Model"]["InputList"] if os.path.isabs(args["Model"]["InputList"]) else os.path.join(self.local_path, args["Model"]["InputList"])
        self.inputs = args["Model"]["Data"] if os.path.isabs(args["Model"]["Data"]) else os.path.join(self.local_path, args["Model"]["Data"])
        self.device.push_data(os.path.join(self.artifacts, self.toolchain), self.device_path)
        self.device.push_data(self.input_list, self.device_path)
        self.device.push_data(self.inputs, self.device_path)

        return

    def _setup_env(self):
#        logger.debug(           '[{}] Pushing envsetup scripts'.format(                self._adb._adb_device))
        lib_dir = os.path.join(self.device_path, self.toolchain, 'lib')
        bin_dir = os.path.join(self.device_path, self.toolchain, 'bin')

        commands = [
            'export LD_LIBRARY_PATH={}:/vendor/lib64/:$LD_LIBRARY_PATH'.format(lib_dir),
            'export ADSP_LIBRARY_PATH="{};/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/dsp"'.format(
                lib_dir),
            'export PATH={}:$PATH'.format(bin_dir)
        ]
        script_name = os.path.join(self.artifacts, '{}_{}_{}'.format(self.deviceId, self.toolchain, 'env.sh'))
        with open(script_name, 'w') as f:
            f.write('\n'.join(commands) + '\n')
        self.device.push_data(script_name, self.device_path)
           # self._adb.push(script_name, self._device_root)
        self.env_script = os.path.join(self.device_path, os.path.basename(script_name))

    def execute_bm(self, model):

        bm = bm_constants.BmConstants()
        output_dir = os.path.join(self.device_path, "output")
        cmd  = "source {} && ".format(self.env_script)
        cmd += "cd {} && ".format(self.device_path)
        cmd += "{} ".format("qnn-net-run")
        cmd += "{} ".format(bm.RUNTIMES[self.backend])
        cmd += "--input_list {} ".format(os.path.join(self.device_path, os.path.basename(self.input_list)))
        cmd += "--profiling_level {} ".format("basic")
        cmd += "--perf_profile {} ".format("burst")
        cmd += "--output_dir {} ".format(output_dir)
        cmd += "--model {} ".format(os.path.join(self.device_path, os.path.basename(model)))
        logging.info("\nRunning benchmark:\n{}\n".format(cmd))
        return_code, output, err_msg = self.device.adb_helper._execute('shell', shlex.split(cmd))
        if return_code != 0:
            logging.error("Benchmark FAILED:\n{}".format(err_msg))
            exit(1)

        logging.info("Benchmark SUCCESS:\n{}".format(output))
        logging.info("Copying results from {} to {}".format(output_dir, os.path.dirname(model)))
        self.device.adb_helper.pull(output_dir, self.current_hil_path)
        return True

    def convert(self, model, output_path):
        self.current_hil_path = output_path
        converted_model = self.converter.convert(model, output_path, self.backend)
        return self.converter.compile(converted_model, output_path)


    def runner(self, model):
        # running benchmark
        logging.info("*******************************************************************\n");
        logging.info("Preparing to run inference, pushing model")
        self.device.push_data(model, self.device_path)

        self.execute_bm(model)
        stats = self.process_results()
        logging.info("*******************************************************************\n");
        return stats


    def process_results(self):
        stats = {}
        profiling_file = os.path.join(self.current_hil_path, 'output/qnn-profiling-data.log')
        if not os.path.exists(profiling_file):
            logging.error("Invalid profiling file {}".format(profiling_file))
            exit(1)

        parsed_file = str(Path(profiling_file).with_suffix('.txt'))
        cmd = SDK_PATH+'/bin/x86_64-linux-clang/qnn-profile-viewer'
        cmd += ' --input_log ' + profiling_file # + '  > ' + parsed_file

        logging.info("*******************************************************************");
        logging.info("Parsing stats %s:\n", profiling_file);

        cmd_args = shlex.split(cmd)
        result = subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            logging.error("Couldn't process statistics result file {}".format(profiling_file));
            exit(1)

        print(str(result.stdout))
        text_file = open(parsed_file, "wt")
        text_file.write(result.stdout)
        text_file.close()

#        print(result.stderr)

        init_times = {'NetRun':[],
                      'Backend (RPC (load binary) time)':[],
                      'Backend (QNN accelerator (load binary) time)':[],
                      'Backend (Accelerator (load binary) time)':[],
                      'Backend (QNN (load binary) time)':[]}
        deinit_times = {'NetRun':[],
                        'Backend (RPC (deinit) time)':[],
                        'Backend (QNN Accelerator (deinit) time)':[],
                        'Backend (Accelerator (deinit) time)':[],
                        'Backend (QNN (deinit) time)':[]}
        exec_times = {'NetRun':[],
                         'Backend (RPC (execute) time)':[],
                         'Backend (QNN accelerator (execute) time)':[],
                         'Backend (Accelerator (execute) time)':[],
                         'Backend (QNN (execute) time)':[]}
        finalize_times = {'NetRun':[]}
        with open(parsed_file, 'r') as hdl:
            data = hdl.readlines()
        init_stats, deinit_stats, finalize_stats, exec_stats = False, False, False, False
        for log_line in data:
            if log_line.startswith('Execute Stats (Average):'):
                exec_stats = True
                init_stats = False
                finalize_stats = False
                deinit_stats = False
            elif log_line.startswith('Init Stats:'):
                exec_stats = False
                init_stats = True
                finalize_stats = False
                deinit_stats = False
            elif log_line.startswith('Finalize Stats:'):
                exec_stats = False
                init_stats = False
                finalize_stats = True
                deinit_stats = False
            elif log_line.startswith('De-Init Stats:'):
                exec_stats = False
                init_stats = False
                finalize_stats = False
                deinit_stats = True
            elif 'IPS' in log_line:
                continue
            if init_stats:
                if 'NetRun' in log_line:
                    init_times["NetRun"].append(int(log_line.split('NetRun:')[1][:-3].strip()))
                elif 'Backend (RPC (load binary) time)' in log_line:
                    init_times["Backend (RPC (load binary) time)"].append(
                    int(log_line.split('Backend (RPC (load binary) time):')[1][:-3].strip()))
                elif 'Backend (QNN accelerator (load binary) time)' in log_line:
                    init_times["Backend (QNN accelerator (load binary) time)"].append(
                    int(log_line.split('Backend (QNN accelerator (load binary) time):')[1][:-3].strip()))
                elif 'Backend (Accelerator (load binary) time)' in log_line:
                    init_times["Backend (Accelerator (load binary) time)"].append(
                    int(log_line.split('Backend (Accelerator (load binary) time):')[1][:-3].strip()))
                elif 'Backend (QNN (load binary) time)' in log_line:
                    init_times["Backend (QNN (load binary) time)"].append(
                    int(log_line.split('Backend (QNN (load binary) time):')[1][:-3].strip()))
            if exec_stats:
                if 'NetRun' in log_line:
                    exec_times["NetRun"].append(int(log_line.split('NetRun:')[1][:-3].strip()))
                elif 'Backend (RPC (execute) time)' in log_line:
                    exec_times["Backend (RPC (execute) time)"].append(
                    int(log_line.split('Backend (RPC (execute) time):')[1][:-3].strip()))
                elif 'Backend (QNN accelerator (execute) time)' in log_line:
                    exec_times["Backend (QNN accelerator (execute) time)"].append(
                    int(log_line.split('Backend (QNN accelerator (execute) time):')[1][:-3].strip()))
                elif 'Backend (Accelerator (execute) time)' in log_line:
                    exec_times["Backend (Accelerator (execute) time)"].append(
                    int(log_line.split('Backend (Accelerator (execute) time):')[1][:-3].strip()))
                elif 'Backend (QNN (execute) time)' in log_line:
                    exec_times["Backend (QNN (execute) time)"].append(
                    int(log_line.split('Backend (QNN (execute) time):')[1][:-3].strip()))
            if finalize_stats:
                if 'NetRun' in log_line:
                    finalize_times["NetRun"].append(int(log_line.split('NetRun:')[1][:-3].strip()))
            if deinit_stats:
                if 'NetRun' in log_line:
                    deinit_times["NetRun"].append(int(log_line.split('NetRun:')[1][:-3].strip()))
                elif 'Backend (RPC (deinit) time)' in log_line:
                    deinit_times["Backend (RPC (deinit) time)"].append(
                    int(log_line.split('Backend (RPC (deinit) time):')[1][:-3].strip()))
                elif 'Backend (QNN Accelerator (deinit) time)' in log_line:
                    deinit_times["Backend (QNN Accelerator (deinit) time)"].append(
                    int(log_line.split('Backend (QNN Accelerator (deinit) time):')[1][:-3].strip()))
                elif 'Backend (Accelerator (deinit) time)' in log_line:
                    deinit_times["Backend (Accelerator (deinit) time)"].append(
                    int(log_line.split('Backend (Accelerator (deinit) time):')[1][:-3].strip()))
                elif 'Backend (QNN (deinit) time)' in log_line:
                    deinit_times["Backend (QNN (deinit) time)"].append(
                    int(log_line.split('Backend (QNN (deinit) time):')[1][:-3].strip()))
        for key in init_times.keys():
            if init_times[key]:
                init_times[key] = np.mean(init_times[key])
        for key in deinit_times.keys():
            if deinit_times[key]:
                deinit_times[key] = np.mean(deinit_times[key])
        for key in exec_times.keys():
            if exec_times[key]:
                exec_times[key] = np.mean(exec_times[key])
        for key in finalize_times.keys():
            if finalize_times[key]:
                finalize_times[key] = np.mean(finalize_times[key])


        execute_time = exec_times['Backend (QNN accelerator (execute) time)']
        logger.info("-----------------------------------------------------------")
        logger.info("[Backend QNN Execute Time (Avg)]: {} us".format(execute_time))
        logger.info("-----------------------------------------------------------")

        # Return QNN backend latency (in microseconds)
        stats = { 'latency_in_us' : execute_time, 'model_memory' : 0.0 }
        return stats


class RunNetrun(RunTestPackage):

    def __init__(self, args):

        level = args.level.upper()
        if level == "DEBUG":
            logger.setLevel(logging.DEBUG)
        # check for device and hostname
        self.device = args.device
        self.hostname = args.hostname
        self.device = self.check_deviceid(self.device, self.hostname)
        logger.debug("Device id on which test will run: {}".format(self.device))

        self.model_run = self.check_models(args.model)

        self.backends = args.backends.upper().split(",")

        self.chipset = args.chipset
        self.dsp_type = args.dsp_type if args.dsp_type else self.check_dsp_type(self.chipset)
        self.mem_type = args.mem_type if args.mem_type else self.check_mem_type(self.chipset)
        self.toolchain = self.check_toolchain(self.chipset)

        # set all the images
        imgs = str(args.image_count)
        if imgs.lower() == 'all':
            imgs = 0
        self.img_cnt = int(imgs)
        self.unset_images()
        self.set_images(self.img_cnt)

        self.time = args.time
        self.itr = int(args.iterations)
        self.verify = False
        self.output_file_name = "netrun_"
        self.perf = args.perf

        adb = Adb('adb', self.device, hostname=self.hostname)
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

    def run_with_time(self):
        if self.time:
            # converting time from minutes to seconds.
            time = int(self.time) * 60
            logger.debug("Timeout set for {} seconds.".format(time))
            if self.itr <= 1:
                # this is maximum number of iteration number
                itr = time * 100
            else:
                itr = self.itr

            # running qnn_bench script with thread process for specified time
            proc = multiprocessing.Process(
                target=self.run_test,
                args=(itr, self.backends, self.model_run,
                      self.device, self.hostname, self.chipset, self.toolchain,
                      self.dsp_type, self.mem_type, self.verify, self.perf)
            )
            # starting process
            proc.start()
            proc.join(timeout=time)
            # If thread is still active
            if proc.is_alive():
                logger.info("running netrun... timeout... let's kill it...")
                proc.terminate()

    def runner(self):
        if self.time:
            self.run_with_time()

        elif self.itr:
            self.run_test(self.itr, self.backends, self.model_run,
                          self.device, self.hostname, self.chipset, self.toolchain,
                          self.dsp_type, self.mem_type, self.verify, self.perf)

        # reset image list for image data.
        if self.img_cnt:
            self.unset_images()


class RunVerifier(RunTestPackage):

    def __init__(self, args):
        level = args.level.upper()
        if level == "DEBUG":
            logger.setLevel(logging.DEBUG)

        # check for device and hostname
        self.device = args.device
        self.hostname = args.hostname
        self.device = self.check_deviceid(self.device, self.hostname)
        logger.debug("Device id on which test will run: {}".format(self.device))

        self.model_run = self.check_models(args.model)

        self.backends = args.backends.upper().split(",")

        self.chipset = args.chipset
        self.dsp_type = args.dsp_type if args.dsp_type else self.check_dsp_type(self.chipset)
        self.mem_type = args.mem_type if args.mem_type else self.check_mem_type(self.chipset)
        self.toolchain = self.check_toolchain(self.chipset)
        self.itr = 1
        self.verify = True
        self.output_file_name = "verifier_"
        self.perf = args.perf

        adb = Adb('adb', self.device, hostname=self.hostname)
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

    def runner(self):
        # running verifier
        self.run_test(self.itr, self.backends, self.model_run,
                      self.device, self.hostname, self.chipset, self.toolchain,
                      self.dsp_type, self.mem_type, self.verify, self.perf)


class RunConcurrency(RunTestPackage):

    def __init__(self, args):
        level = args.level.upper()
        if level == "DEBUG":
            logger.setLevel(logging.DEBUG)

        # check for device and hostname
        device = args.device
        self.hostname = args.hostname
        self.device = self.check_deviceid(device, self.hostname)
        self.chipset = args.chipset
        self.output_file_name = "concurrency_"
        self.dsp_type = args.dsp_type if args.dsp_type else self.check_dsp_type(self.chipset)
        self.mem_type = args.mem_type if args.mem_type else self.check_mem_type(self.chipset)
        self.toolchain = self.check_toolchain(self.chipset)
        self.conf_files_list = [os.path.join(MY_CWD, "throughput_configs", x) for x in
                                os.listdir(os.path.join(MY_CWD, "throughput_configs")) if
                                x.startswith("concurrency")]
        config_run = args.config.split(" ")
        if isinstance(config_run, list):
            config_run_path = [os.path.join(MY_CWD, "throughput_configs", x) for x in config_run]

        if args.config != 'all':
            for conf in config_run_path:
                if conf not in self.conf_files_list:
                    logger.error(
                        "Invalid Config name provided...Please provide the correct config name {0}  {1}".format(
                            conf, self.conf_files_list))
                    exit(1)
            self.conf_files_list = config_run_path
        logger.debug("Device id on which test will run: {}".format(self.device))

        adb = Adb('adb', self.device, hostname=self.hostname)
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

    def runner(self):
        # running concurrency
        self.run_throughput(self.device, self.hostname, self.chipset, self.toolchain,
                            self.mem_type, self.dsp_type, self.conf_files_list)


class RunStress(RunTestPackage):

    def __init__(self, args):
        level = args.level.upper()
        if level == "DEBUG":
            logger.setLevel(logging.DEBUG)

        # check for device and hostname
        device = args.device
        self.hostname = args.hostname
        self.device = self.check_deviceid(device, self.hostname)
        self.chipset = args.chipset
        self.output_file_name = "stress_"
        self.dsp_type = args.dsp_type if args.dsp_type else self.check_dsp_type(self.chipset)
        self.mem_type = args.mem_type if args.mem_type else self.check_mem_type(self.chipset)
        self.toolchain = self.check_toolchain(self.chipset)
        self.conf_files_list = [os.path.join(MY_CWD, "throughput_configs", x) for x in
                                os.listdir(os.path.join(MY_CWD, "throughput_configs")) if
                                x.startswith("stress")]
        config_run = args.config.split(" ")
        if isinstance(config_run, list):
            config_run_path = [os.path.join(MY_CWD, "throughput_configs", x) for x in config_run]

        if args.config != 'all':
            for conf in config_run_path:
                if conf not in self.conf_files_list:
                    logger.error(
                        "Invalid Config name provided...Please provide the correct config name {0}  {1}".format(
                            conf, self.conf_files_list))
                    exit(1)
            self.conf_files_list = config_run_path
        logger.debug("Device id on which test will run: {}".format(self.device))

        adb = Adb('adb', self.device, hostname=self.hostname)
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

    def runner(self):
        # running stress
        self.run_throughput(self.device, self.hostname, self.chipset, self.toolchain,
                            self.mem_type, self.dsp_type, self.conf_files_list)


class RunQNNPlatformValidator(RunTestPackage):

    def __init__(self, args):
        level = args.level.upper()
        if level == "DEBUG":
            logger.setLevel(logging.DEBUG)

        # check for device and hostname
        device = args.device
        self.hostname = args.hostname
        self.device = self.check_deviceid(device, self.hostname)
        self.chipset = args.chipset
        self.dsp_type = args.dsp_type if args.dsp_type else self.check_dsp_type(self.chipset)
        self.socName = args.socName

        adb = Adb('adb', self.device, hostname=self.hostname)
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

    def runner(self):
        # running platform validator
        self.run_qnn_platform_validator(self.device, self.hostname, self.dsp_type, self.socName,
                                        self.chipset)


def benchmark(args):
    """
    :description: This function creates benchmark object.
    :arguments: args: input arguments added by user
    """
    RunBenchmark(args).runner()


def netrun(args):
    """
    :description: This function creates netrun object.
    :arguments: args: input arguments added by user
    """
    RunNetrun(args).runner()


def verifier(args):
    """
    :description: This function creates verifier object.
    :arguments: args: input arguments added by user
    """
    RunVerifier(args).runner()


def concurrency(args):
    """
    :description: This function creates concurrency object.
    :arguments: args: input arguments added by user
    """
    RunConcurrency(args).runner()


def stress(args):
    """
    :description: This function creates stress object.
    :arguments: args: input arguments added by user
    """
    RunStress(args).runner()


def qnn_platform_validator(args):
    """
    :description: This function creates platform Validator object.
    :arguments: args: input arguments added by user
    """
    RunQNNPlatformValidator(args).runner()

def run_all(args):
    """
    :description: This function runs all frameworks.
    :arguments: args: input arguments added by user
    """
    RunQNNPlatformValidator(args).runner()
    RunBenchmark(args).runner()
    #RunNetrun(args).runner()
    RunVerifier(args).runner()
    RunConcurrency(args).runner()
    RunStress(args).runner()

def print_models():
    """
        Desc: This function will print list of model.
        Output: list of models
    """
    logger.info("#######################################################")
    logger.info("Test Package contains following models...")
    logger.info("#######################################################")
    for i, model in enumerate(MODELS):
        logger.info("\t {}. {}".format(i + 1, model))
    logger.info("#######################################################")
