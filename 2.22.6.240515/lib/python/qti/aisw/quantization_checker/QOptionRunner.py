#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import platform
import qti.aisw.quantization_checker.utils.Constants as Constants
from qti.aisw.quantization_checker.utils import utils
from qti.aisw.quantization_checker.utils.FileUtils import FileUtils, ScopedFileUtils
import qti.aisw.quantization_checker.utils.target as target
import qti.aisw.quantization_checker.EnvironmentManager as em
from qti.aisw.quantization_checker.utils.ConfigParser import extractEnvironmentConfigParams
from qti.aisw.quantization_checker.utils.Progress import Progress, ProgressStage

class QOptionRunner:
    def __init__(self, sdkType, inputNetwork, inputList, sdkDir, outputDir, configFile, logger, skipBuild=False):
        self.sdkType = sdkType
        self.quantizationOption = Constants.UNQUANTIZED
        self.logger = logger
        self.fileHelper = FileUtils(self.logger)
        self.inputNetwork = inputNetwork
        self.inputList = inputList
        self.sdkDir = sdkDir
        self.outputDir = outputDir
        self.skipBuild = skipBuild
        self.configFile = configFile
        #we switch to config file inside tool's location if user-provided file is not found
        if not os.path.exists(os.path.abspath(self.configFile)):
            self.configFile = os.path.join(self.sdkDir, Constants.PYTHONPATH, Constants.CONFIG_PATH, os.path.basename(self.configFile))
        self.configParams = extractEnvironmentConfigParams(os.path.abspath(self.configFile))

    def run(self):
        if not os.path.isdir(self.outputDir):
            self.logger.print('Please enter a valid directory containing the model file and output from the QOptionGenerator.py script. Graph directory: ' + self.outputDir)
            return -1

        buildResult = -1
        runResult = -1
        outputPathName = os.path.join(self.outputDir, self.quantizationOption)
        if not os.path.exists(outputPathName):
            self.logger.print('The generated output files cannot be found in the model directory! If the generator output files were stored in a different location, please specify the location using the --output_dir option')
            return -1
        # for SNPE this is always false since we set self.skipBuild to True in the constructor call for SNPE
        if not self.skipBuild:
            if not os.path.isfile(os.path.join(outputPathName, self.quantizationOption + '.cpp')):
                self.logger.print('The generated output files cannot be found in the model directory! If the generator output files were stored in a different location, please specify the location using the --output_dir option')
                return -1
            buildResult = self.__buildModel()
            if buildResult == -1:
                self.logger.print('Error encountered during building of ' + self.quantizationOption + ' quantization option. Please consult console/log output.')
                return buildResult
            Progress.updateProgress(Progress.getStepSize(ProgressStage.BUILDER))

        runResult = self.runModel()

        if runResult == -1:
            self.logger.print('Error encountered during running of ' + self.quantizationOption + ' quantization option. Please consult console/log output.')
            return runResult
        self.logger.print('Build and execute complete for ' + self.quantizationOption + '. Please refer to output files for accuracy comparision.')

        return runResult

    def __buildModel(self):
        outputPathName = os.path.join(self.outputDir, self.quantizationOption)
        self.logger.print('Model cpp directory: ' + outputPathName + '\n')
        if platform.system() == Constants.WINDOWS:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
            modelLibArgs = ' -o ' + os.path.join(self.outputDir, Constants.MODEL_LIB_PATH) + ' -t ' + Constants.WINDOWS_X86
            python,shell=True,False
        elif platform.system() == Constants.LINUX:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
            modelLibArgs = ' -l lib' + self.quantizationOption + '.so -o ' + self.outputDir + ' -t ' + Constants.X86_LINUX_CLANG
            python,shell=False,True
        modelLibGeneratorCmd = os.path.join(self.sdkDir, sdkBinPath, Constants.QNN_MODEL_LIB_GENERATOR_BIN_NAME)
        modelLibGeneratorArgs = ' -c ' + os.path.join(outputPathName, self.quantizationOption + '.cpp') + ' -b ' + os.path.join(outputPathName, self.quantizationOption + '.bin') + modelLibArgs
        modelLibGeneratorCmdAndArgs = modelLibGeneratorCmd + modelLibGeneratorArgs
        self.logger.print('Building model...')
        result = utils.issueCommandAndWait(modelLibGeneratorCmdAndArgs, self.logger, em.getEnvironment(self.configParams, self.sdkDir, Constants.QNN), python, shell)
        if result != 0:
            self.logger.print('qnn-model-lib-generator failed. Please check the console or logs for details.')
        return result

    def runModel(self):
        workingDir = os.path.dirname(self.inputNetwork)
        with ScopedFileUtils(workingDir, self.fileHelper):
            try:
                self.logger.print('outputDir: ' + self.outputDir)
                x86_64 = target.x86_64(self.quantizationOption, self.inputList, self.sdkDir, self.sdkType, self.outputDir, self.configParams, self.logger)
                x86_64.buildNetRunArgs()
                result = x86_64.runModel()
                if result != 0:
                    self.logger.print('net-run failed for unquantized. Please check the console or logs for details.')
                    return -1
                self.logger.print('net-run complete, output saved in ' + os.path.join(self.outputDir, Constants.NET_RUN_OUTPUT_DIR))
            except Exception as e:
                self.logger.print("ERROR! running model failed: " + str(e))
                return -1
        return 0
