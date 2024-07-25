#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import itertools
import os
import sys
import platform
import subprocess, shlex
import argparse
import traceback
from qti.aisw.quantization_checker.utils.Logger import Logger, PrintOptions
import qti.aisw.quantization_checker.utils.Constants as Constants

def getArguments():
    argumentParser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Script to extract the activations from all quantization options and inspect them.")
    allArgs = argumentParser.add_argument_group('all arguments')
    allArgs.add_argument('--model', type=str, required=False, help='Path to model graph file, model directory or a directory of models.')
    allArgs.add_argument('--input_list', type=str, required=False, help='Path to a text file containing a list of input files.')
    allArgs.add_argument('--activation_width', type=str, required=False, help='Bit-width to use for activations. E.g., 8, 16. Default is 8.')
    allArgs.add_argument('--bias_width', type=str, required=False, help='Bit-width to use for biases. E.g., 8, 32. Default is 8.')
    allArgs.add_argument('--weight_width', type=str, required=False, help='Bit-width to use for weights. E.g., 8. Default is 8.')
    allArgs.add_argument('--output_dir', type=str, required=False, help='Path to store the output files created by the generator script')
    allArgs.add_argument('--skip_building_model', action='store_true', required=False, help='Stop the script from building the model. It is assumed the model.so is pre-built.')
    allArgs.add_argument('--skip_generator', action='store_true', required=False, help='Stop the script from running the converter. It is assumed that the necessary files are already available.')
    allArgs.add_argument('--skip_runner', action='store_true', required=False, help='Stop the script from running the model. It is assumed that the necessary files are already available.')
    allArgs.add_argument('--output_csv', action='store_true', required=False, help='Output analysis data to a csv file in the output directory.')
    allArgs.add_argument('--generate_histogram', action='store_true', required=False, help='Generate histogram analysis for weights/biases. Default is to skip histgoram generation.')
    allArgs.add_argument('--per_channel_histogram', action='store_true', required=False, help='Generate per channel histogram analysis for weights/biases. Default is to skip histgoram generation.')
    allArgs.add_argument('--config_file', required=True, type=str, help='Config file specifying all possible options required for execution. E.g., [MODEL_PATH, INPUT_LIST_PATH, QNN_SDK_ROOT, ACTIVATION_WIDTH, BIAS_WIDTH, OUTPUT_DIR_PATH].')
    return argumentParser.parse_args()

def issueCommandAndWait(cmdAndArgsStr, logger: Logger, environment=None, python=True, shell=False):
    logger.print('=========================================================================\n', PrintOptions.LOGFILE)
    logger.print('COMMAND ISSUED\n', PrintOptions.LOGFILE)
    logger.print('=========================================================================\n', PrintOptions.LOGFILE)
    logger.print('Issuing the following command: ' + cmdAndArgsStr + '\n', PrintOptions.LOGFILE)
    logger.print('=========================================================================\n', PrintOptions.LOGFILE)
    logger.print('COMMAND OUTPUT\n', PrintOptions.LOGFILE)
    logger.print('=========================================================================\n', PrintOptions.LOGFILE)
    logger.print('Issuing the following command: ' + cmdAndArgsStr + '\n')
    cmdAndArgs = cmdAndArgsStr
    if python:
        cmdAndArgsStr = sys.executable + ' ' + cmdAndArgsStr
    if not shell:
        if platform.system() == Constants.WINDOWS: cmdAndArgs = shlex.split(cmdAndArgsStr,posix=False)
        elif platform.system() == Constants.LINUX: cmdAndArgs = shlex.split(cmdAndArgsStr)
    result = 0
    try:
        proc = subprocess.run(cmdAndArgs, env=environment, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logger.print(str(proc.stdout, 'utf-8'), PrintOptions.CONSOLE_LOGFILE)
        result = proc.returncode
        logger.print('Return code: ' + str(result) + ' from command: ' + cmdAndArgsStr, PrintOptions.LOGFILE)
        logger.print('\n', PrintOptions.LOGFILE)
        logger.flush(PrintOptions.LOGFILE)
        logger.print('Command completed. Result: ' + str(result))
    except Exception as e:
        logger.print("Utils - ERROR! " + str(e) + '\n')
        logger.print(traceback.format_exc(), PrintOptions.LOGFILE)
        result = -1
    return result

def buildModelDict(modelsDir):
    modelBundle = {}
    count = 0

    for file in os.listdir(modelsDir):
        modelPath = os.path.join(modelsDir, file)
        if os.path.isdir(modelPath):
            currentModel = buildModelInfo(modelPath)
            if currentModel != {}:
                count = count + 1
                modelBundle[count] = currentModel

    return modelBundle

def buildModelInfo(modelDir):
    modelInfo = {}
    for file in os.listdir(modelDir):
        if file.endswith(".pb") or file.endswith(".onnx"):
            modelInfo['modelFile'] = os.path.join(modelDir, file)
        if file.endswith(".txt"):
            if os.path.isfile(os.path.join(modelDir, file)):
                modelInfo["inputList"] = os.path.join(modelDir, file)
        if os.path.isdir(os.path.join(modelDir, file)):
            if file.lower() == "data" or file.lower() == "inputs":
                if os.path.isfile(os.path.join(modelDir, file, "input_list.txt")):
                    modelInfo["inputList"] = os.path.join(modelDir, file, "input_list.txt")
                elif os.path.isfile(os.path.join(modelDir, file, "image_list.txt")):
                    modelInfo["inputList"] = os.path.join(modelDir, file, "image_list.txt")
    return modelInfo

def mergeQuantOptionsAndAlgorithms(quantOptions, quantAlgorithms):
    algorithmTuples = []
    algorithmCombinations = []
    combinedOptionsAndAlgorithms = quantOptions.copy()
    for i in range(len(quantAlgorithms)):
        algorithmTuples.extend(list(itertools.combinations(quantAlgorithms, i+1)))
    for algorithmTuple in algorithmTuples:
        algorithmCombinations.append('_'.join(algorithmTuple))
    for quantOption in quantOptions[1:]:
        for combination in algorithmCombinations:
            combinedOptionsAndAlgorithms.append(quantOption + "_" + combination)
    return combinedOptionsAndAlgorithms
