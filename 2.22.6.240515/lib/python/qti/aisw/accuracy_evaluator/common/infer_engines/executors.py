##############################################################################
#
# Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################

import logging
import subprocess
import time
import os
from abc import ABC

import qti.aisw.accuracy_evaluator.common.defaults as df
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.adb import Adb
import qti.aisw.accuracy_evaluator.common.exceptions as ce

defaults = df.Defaults.getInstance()


def deleteFile(pathToFileToDelete):
    result = 0
    if os.path.exists(pathToFileToDelete):
        logging.debug('Deleting file: ' + pathToFileToDelete)
        try:
            os.remove(pathToFileToDelete)
        except:
            return -1
    else:
        logging.debug('Attempt to delete directory failed, path does not exist: ' +
                      pathToFileToDelete + '\n')
        result = -1
    return result


class Executors(ABC):

    def __init__(self, server=None, username=None, password=None):
        self.server = server
        self.username = username
        self.password = password

    def run(self):
        pass

    def close(self):
        pass


class LocalExecutor(Executors):
    """This class performs execution on local device."""

    def run(self, cmd, env=None, log_file=''):
        """This method runs the given command on local device and returns error
        status."""
        if env is None:
            subprocess_env = os.environ.copy()
        else:
            subprocess_env = env
        subprocess_env["GLOG_minloglevel"] = ""

        log_redirect = " > " + log_file + " 2>&1" if log_file else ''
        process = subprocess.Popen(cmd + log_redirect, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=subprocess_env)
        stdout, stderr = process.communicate()
        return process.returncode

    def close(self):
        pass


class AdbExecutor(Executors):
    """This class performs the execution on device connected through adb."""

    def __init__(self, pathToAdbBinary, deviceSerialNumber, inputList, inputDataPath, graphDir,
                 sdkDir, outputDir, configFilename, settingsFilename, dspVersion=None, backend=None,
                 model_lib=None, netrun_param_list=None):
        self.__adb = Adb(pathToAdbBinary, deviceSerialNumber)
        self.__inputDataPath = inputDataPath
        self.__settingsFile = settingsFilename
        self.inputList = inputList
        self.graphDir = graphDir
        self.sdkDir = sdkDir
        self.outputDir = outputDir
        self.configFile = configFilename
        self.targetBasePath = '/data/local/tmp/qnn_acc_eval'
        self.dspVersion = dspVersion
        self.backend = backend
        self.model_lib = model_lib
        self.netrun_param_list = netrun_param_list

    def buildQnnNetRunArgs(self):
        if self.backend == 'cpu':
            backend_lib = os.path.basename(qcc.CPU_BACKEND_LIB)
        elif self.backend == 'gpu':
            backend_lib = os.path.basename(qcc.GPU_BACKEND_LIB)
        elif "dspv" in self.backend:
            backend_lib = os.path.basename(qcc.DSP_BACKEND_LIBRARIES[0])
        else:
            qacc_file_logger.error(f"Invalid backend {self.backend}")
            raise ce.QnnNetRunException(f"Invalid backend {self.backend}")

        htpArgs = '--perf_profile HIGH_PERFORMANCE'

        self.inputList = self.processInputList(self.inputList)

        qnnNetRunArgs = '--backend ' + os.path.join(
            self.targetBasePath, backend_lib) + ' --output_dir ' + os.path.join(
                self.targetBasePath, "output") + ' --input_list ' + os.path.join(
                    self.targetBasePath, os.path.basename(self.inputList))
        if self.backend == "cpu" or self.backend == "gpu":
            qnnNetRunArgs += ' --model ' + os.path.join(self.targetBasePath,
                                                        os.path.basename(self.model_lib))
        if "dspv" in self.backend:
            qnnNetRunArgs += ' --retrieve_context ' + os.path.join(
                self.targetBasePath, f"{qcc.CONTEXT_BINARY_FILE}.bin")
        if self.configFile:
            qnnNetRunArgs += ' --config_file ' + self.configFile
        qnnNetRunCommand = os.path.join(self.targetBasePath, 'qnn-net-run')

        self.qnnNetRunCmdAndArgs = [qnnNetRunCommand]
        self.qnnNetRunCmdAndArgs.extend(qnnNetRunArgs.split(' '))
        self.qnnNetRunCmdAndArgs.extend(htpArgs.split(' '))
        if self.netrun_param_list:
            self.qnnNetRunCmdAndArgs.extend(self.netrun_param_list)

    def cleanup(self):
        # delete old files so that following runs do not pull incorrect results
        qacc_logger.info('Cleaning up directory on target...\n')
        return self.adbShell(['rm', '-rf', self.targetBasePath])

    def processInputList(self, input_list):

        with open(input_list, "r") as F:
            paths = F.readlines()

        modified_paths = []
        for path_per_line in paths:
            # create a list of paths if there are multiple inputs in each line
            input_paths = path_per_line.split()
            modified_input_paths = []
            for path in input_paths:
                file_name = os.path.basename(path)
                input_name = path.split(":=")[0] if ":=" in path else ""
                full_input_path = path.split(":=")[1] if ":=" in path else path
                full_dir_path = os.path.dirname(full_input_path)
                # extract the directory name one level above the inputs
                inputs_dir = full_dir_path.split('/')[-1]
                modified_input = input_name + ":=" + os.path.join(self.targetBasePath, inputs_dir,
                                                                  file_name)
                modified_input_paths.append(modified_input)
            modified_inputs = ' '.join(modified_input_paths)
            modified_inputs += '\n'
            modified_paths.append(modified_inputs)

        self.__inputDataPath = full_dir_path
        adb_input_list = os.path.join(self.outputDir, "adb_input_list.txt")
        with open(adb_input_list, "w") as F:
            F.writelines(modified_paths)

        return adb_input_list

    def runModel(self):
        qacc_logger.info('Executing model on target...')
        qnnNetRunScript = [os.path.join(self.targetBasePath, 'qnn-net-run-target.sh'), '']
        result = self.adbShell(qnnNetRunScript)
        #self.qacc_file_logger.flush()
        return result

    def pullOutput(self):
        qacc_logger.info('Downloading results for  from target device to ' + self.outputDir + '...')
        result = self.adbPull([os.path.join(self.targetBasePath, "output/"), self.outputDir])
        #self.qacc_file_logger.flush()
        return result

        #TODO: maybe should add a catch block or return an error code if any of the push step failed
        #      then automatically moves onto the next variation otherwise just from the log it will look it it still progressed.
    def pushArtifacts(self):
        qacc_logger.info('Creating necessary directories on target device...')
        result = self.adbShell(['mkdir', '-p', self.targetBasePath])
        if result == -1:
            return result

        qacc_logger.info('Pushing SDK libraries to target device...')
        backend_libraries_to_push = []
        if self.backend == 'cpu':
            backend_libraries_to_push.append(qcc.CPU_BACKEND_LIB)
        elif self.backend == 'gpu':
            backend_libraries_to_push.append(qcc.GPU_BACKEND_LIB)
        elif self.dspVersion:
            skel_lib = "lib/hexagon-" + self.dspVersion + "/unsigned/libQnnHtp" + self.dspVersion.upper(
            ) + "Skel.so"
            stub_lib = "lib/aarch64-android/libQnnHtp" + self.dspVersion.upper() + "Stub.so"
            backend_libraries_to_push.extend(qcc.DSP_BACKEND_LIBRARIES + [skel_lib, stub_lib])

        for lib_location in backend_libraries_to_push:
            result = self.adbPush([os.path.join(self.sdkDir, lib_location), self.targetBasePath])
            if result == -1:
                return result

        qacc_logger.info('Pushing SDK binaries to target device...')
        result = self.adbPush(
            [os.path.join(self.sdkDir, 'bin/aarch64-android/qnn-net-run'), self.targetBasePath])
        if result == -1:
            return result

        qacc_logger.info('Pushing model files to target device...')
        result = self.adbPush([self.__inputDataPath, self.targetBasePath])
        if result == -1:
            return result
        result = self.adbPush([self.inputList, self.targetBasePath])
        if result == -1:
            return result
        if self.backend == "cpu" or self.backend == "gpu":
            result = self.adbPush([self.model_lib, self.targetBasePath])
            if result == -1:
                return result
        elif "dspv" in self.backend:
            result = self.adbPush([
                os.path.join(self.outputDir, f"{qcc.CONTEXT_BINARY_FILE}.bin"), self.targetBasePath
            ])
            if result == -1:
                return result
        #self.qacc_file_logger.flush()

        qacc_logger.info('Creating and pushing run script to target device...')
        try:
            targetScript = open('qnn-net-run-target.sh', 'w')
            qnnNetRunShellScript = '#!/bin/sh\nexport LD_LIBRARY_PATH=' + self.targetBasePath + ':/vendor/dsp/cdsp:/vendor/lib64;export PATH=$PATH:' + self.targetBasePath + ';export ADSP_LIBRARY_PATH=\"' + self.targetBasePath + ';/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\";cd ' + self.targetBasePath + ';' + ' '.join(
                self.qnnNetRunCmdAndArgs) + '\n'
            qacc_file_logger.info(f"Net run command for adb: {qnnNetRunShellScript}")
            targetScript.write(qnnNetRunShellScript)
        except OSError:
            qacc_logger.info(
                'Failure opening or writing target script file. Please check the console or logs for details.'
            )
            targetScript.close()
            return -1
        targetScript.close()

        result = self.adbPush([os.path.join(os.getcwd(), targetScript.name), self.targetBasePath])
        if result == -1:
            return result
        result = self.adbShell(
            ['chmod', '775', os.path.join(self.targetBasePath, targetScript.name)])
        if result == -1:
            return result

        if self.configFile:
            result = self.adbPush(
                [os.path.join(self.outputDir, self.configFile), self.targetBasePath])
            if result == -1:
                return result

        if self.__settingsFile:
            result = self.adbPush(
                [os.path.join(self.outputDir, self.__settingsFile), self.targetBasePath])
            if result == -1:
                return result

        #TODO: remove creating the script file, maybe could echo directly into a file in adb.
        deleteFile(targetScript.name)
        #self.qacc_file_logger.flush()
        return result

    def adbShell(self, commandAndArgs):
        if len(commandAndArgs) == 0:
            qacc_logger.info(
                'Incorrect number of arguments provided to adb shell command. Requires at least 1, received 0.'
            )
            return -1
        if len(commandAndArgs) == 1:
            commandAndArgs.append('')
        code, out, error = self.__adb.shell(commandAndArgs[0], commandAndArgs[1:])
        if code != 10:
            qacc_logger.info('Error issuing shell command: ' + str(out) + ' ' + str(error) +
                             ', Return code = ' + str(code))
            return -1
        return 0

    def adbPush(self, commandAndArgs):
        if len(commandAndArgs) != 2:
            qacc_logger.info(
                'Incorrect number of arguments provided to adb push command. Requires 2, received '
                + len(commandAndArgs))
            return -1
        code, out, error = self.__adb.push(commandAndArgs[0], commandAndArgs[1])
        if code != 0:
            qacc_logger.info('Error issuing push command: ' + str(out) + ' ' + str(error) +
                             ', Return code = ' + str(code))
            return -1
        return 0

    def adbPull(self, commandAndArgs):
        if len(commandAndArgs) != 2:
            qacc_logger.info(
                'Incorrect number of arguments provided to adb push command. Requires 2, got ' +
                len(commandAndArgs))
            return -1
        code, out, error = self.__adb.pull(commandAndArgs[0], commandAndArgs[1])
        if code != 0:
            qacc_logger.info('Error issuing pull command: ' + str(out) + ' ' + str(error) +
                             ', Return code = ' + str(code))
            return -1
        return 0
