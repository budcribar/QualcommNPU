#!/usr/bin/env python3
#=============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
from qti.aisw.quantization_checker.utils.ConfigParser import extractEnvironmentConfigParams
import qti.aisw.quantization_checker.EnvironmentManager as em
import qti.aisw.quantization_checker.utils.Constants as Constants

class QOptionGenerator:
    def __init__(self, quantizationOptions, quantizationAlgorithms, modelFile, inputList, sdkDir, sdkType, activationWidth, biasWidth, weightWidth, outputDir, quantOverridesPath, configFile, inputLayout, logger):
        self.quantizationOptions = quantizationOptions
        self.quantizationAlgorithms = quantizationAlgorithms
        self.modelFile = modelFile
        self.inputList = inputList
        self.sdkDir = sdkDir
        self.sdkType = sdkType
        self.activationWidth = activationWidth
        self.biasWidth = biasWidth
        self.weightWidth = weightWidth
        self.outputDir = outputDir
        self.quantOverridesPath = quantOverridesPath
        self.configFile = configFile
        self.inputLayout = inputLayout
        self.logger = logger

    def generate(self):
        configParams = extractEnvironmentConfigParams(os.path.abspath(self.configFile))
        mlFramework = ''
        if self.modelFile.endswith(".pb"):
            mlFramework = Constants.TENSORFLOW
        elif self.modelFile.endswith(".tflite"):
            mlFramework = Constants.TFLITE
        elif self.modelFile.endswith(".onnx"):
            mlFramework = Constants.ONNX
        else:
            self.logger.print("ERROR! Input model_file not recognizeable. Please use a model file with a .pb or .onnx extension.")
            return -1, {}
        em.setEnvironment(configParams, self.sdkDir, mlFramework)
        converter = None
        if mlFramework == Constants.TENSORFLOW:
            from qti.aisw.quantization_checker.utils.ConverterTools import TensorflowConverter
            converter = TensorflowConverter(self.logger, self.sdkDir, self.sdkType, self.modelFile, self.inputList, self.quantizationOptions, self.quantizationAlgorithms)
        elif mlFramework == Constants.TFLITE:
            from qti.aisw.quantization_checker.utils.ConverterTools import TfliteConverter
            converter = TfliteConverter(self.logger, self.sdkDir, self.sdkType, self.modelFile, self.inputList, self.quantizationOptions, self.quantizationAlgorithms)
        elif mlFramework == Constants.ONNX:
            from qti.aisw.quantization_checker.utils.ConverterTools import OnnxConverter
            converter = OnnxConverter(self.logger, self.sdkDir, self.sdkType, self.modelFile, self.inputList, self.quantizationOptions, self.quantizationAlgorithms)
        else:
            self.logger.print("ERROR! Input model_file not recognizeable. Please use a model file with a .pb or .onnx extension.")
            return -1, {}
        try:
            if converter != None:
                environment = em.getEnvironment(configParams, self.sdkDir, self.sdkType, mlFramework)
                resultsMap, quantizationVariationsWithCommand = converter.convert(env=environment, activationWidth=self.activationWidth, biasWidth=self.biasWidth, weightWidth=self.weightWidth, outputDir=self.outputDir, quantOverrides=self.quantOverridesPath, inputLayout=self.inputLayout)
                for key, result in resultsMap.items():
                    if result == -1:
                        self.logger.print('Error encountered during conversion of ' + key + ' quantization option. Please consult console/log output.')
        except Exception as e:
            self.logger.print("QOptionGenerator - ERROR! Conversion failed " + str(e))
            return -1, {}
        return 0, quantizationVariationsWithCommand
