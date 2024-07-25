#=============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from qti.aisw.quantization_checker.Results.ResultSet import ResultSet
import qti.aisw.quantization_checker.Results.ResultUtils as ResultUtils
from qti.aisw.quantization_checker.utils.Logger import Logger
from qti.aisw.quantization_checker.utils.Comparator import dequantizeBiases, compareBiases
from qti.aisw.quantization_checker.utils.Comparator import dequantizeWeights, compareWeights
from qti.aisw.quantization_checker.utils.Comparator import compareActivations, analyzeInputData

class Processor:
    def __init__(self, quantizationVariations, comparisonAlgorithms, logger : Logger) -> None:
        self.logger = logger
        self.quantizationVariations = quantizationVariations
        self.comparisonAlgorithms = comparisonAlgorithms
        self.biasResults = ResultSet()
        self.biasFailurePercentageByVerifier = {}
        self.maxBiasFailureVerifier = None
        self.maxBiasFailureQuantOption = None
        self.minBiasFailureVerifier = None
        self.minBiasFailureQuantOption = None
        self.weightResults = ResultSet()
        self.weightFailurePercentageByVerifier = {}
        self.maxWeightFailureVerifier = None
        self.maxWeightFailureQuantOption = None
        self.minWeightFailureVerifier = None
        self.minWeightFailureQuantOption = None
        self.activationResults = ResultSet()
        self.activationFailurePercentageByVerifier = {}
        self.maxActivationFailureVerifier = None
        self.maxActivationFailureQuantOption = None
        self.minActivationFailureVerifier = None
        self.minActivationFailureQuantOption = None
        self.inputResults = ResultSet()

    def processWeightResults(self, opsMap) -> ResultSet:
        self.logger.print('Dequantizing weights...')
        dequantizeWeights(self.logger, self.quantizationVariations, opsMap)
        self.logger.print('Comparing weights...')
        self.weightResults = compareWeights(self.quantizationVariations, opsMap, self.comparisonAlgorithms['weight_comparison_algorithms'], self.logger)

    def getWeightResults(self) -> ResultSet:
        return self.weightResults

    def processBiasResults(self, opsMap) -> ResultSet:
        self.logger.print('Dequantizing biases...')
        dequantizeBiases(self.quantizationVariations, opsMap)
        self.logger.print('Comparing biases...')
        self.biasResults = compareBiases(self.quantizationVariations, opsMap, self.comparisonAlgorithms['bias_comparison_algorithms'], self.logger)

    def getBiasResults(self) -> ResultSet:
        return self.biasResults

    def processActivationResults(self, opsMap) -> ResultSet:
        self.logger.print('Comparing activations...')
        self.activationResults = compareActivations(self.quantizationVariations, opsMap, self.comparisonAlgorithms['act_comparison_algorithms'])

    def getActivationResults(self) -> ResultSet:
        return self.activationResults

    def processInputData(self, inputData) -> ResultSet:
        self.logger.print('Analyzing input data...')
        self.inputResults = analyzeInputData(inputData, self.comparisonAlgorithms['input_data_analysis_algorithms'])

    def getInputResults(self) -> ResultSet:
        return self.inputResults

    def getWeightFailurePercentage(self):
        self.weightFailurePercentageByVerifier = ResultUtils.calculateFailurePercentageForResults(self.quantizationVariations, self.weightResults)
        return self.weightFailurePercentageByVerifier

    def getBiasFailurePercentage(self):
        self.biasFailurePercentageByVerifier = ResultUtils.calculateFailurePercentageForResults(self.quantizationVariations, self.biasResults)
        return self.biasFailurePercentageByVerifier

    def getActivationFailurePercentage(self):
        self.activationFailurePercentageByVerifier = ResultUtils.calculateFailurePercentageForResults(self.quantizationVariations, self.activationResults)
        return self.activationFailurePercentageByVerifier

    def getMaxWeightFailurePercentage(self):
        maxFailurePercentage = 0.0
        for verifier in self.weightFailurePercentageByVerifier.keys():
            for quantOption in self.quantizationVariations:
                weightFailurePercentage = self.weightFailurePercentageByVerifier[verifier][quantOption]
                if weightFailurePercentage != 'N/A' and weightFailurePercentage > maxFailurePercentage:
                    maxFailurePercentage = weightFailurePercentage
                    self.maxWeightFailureVerifier = verifier
                    self.maxWeightFailureQuantOption = quantOption
        return maxFailurePercentage

    def getMaxBiasFailurePercentage(self):
        maxFailurePercentage = 0.0
        for verifier in self.biasFailurePercentageByVerifier.keys():
            for quantOption in self.quantizationVariations:
                biasFailurePercentage = self.biasFailurePercentageByVerifier[verifier][quantOption]
                if biasFailurePercentage != 'N/A' and biasFailurePercentage > maxFailurePercentage:
                    maxFailurePercentage = biasFailurePercentage
                    self.maxBiasFailureVerifier = verifier
                    self.maxBiasFailureQuantOption = quantOption
        return maxFailurePercentage

    def getMaxActivationFailurePercentage(self):
        maxFailurePercentage = 0.0
        for verifier in self.activationFailurePercentageByVerifier.keys():
            for quantOption in self.quantizationVariations:
                activationFailurePercentage = self.activationFailurePercentageByVerifier[verifier][quantOption]
                if activationFailurePercentage != 'N/A' and activationFailurePercentage > maxFailurePercentage:
                    maxFailurePercentage = activationFailurePercentage
                    self.maxActivationFailureVerifier = verifier
                    self.maxActivationFailureQuantOption = quantOption
        return maxFailurePercentage

    def getMinWeightFailurePercentage(self):
        return self.__getMinFailurePercentage(self.weightFailurePercentageByVerifier)

    def getMinBiasFailurePercentage(self):
        return self.__getMinFailurePercentage(self.biasFailurePercentageByVerifier)

    def getMinActivationFailurePercentage(self):
        return self.__getMinFailurePercentage(self.activationFailurePercentageByVerifier)

    # for each verifier get the best result, best is defined as the lowest failure value
    def __getMinFailurePercentage(self, failurePercentageByVerifier):
        minFailurePercentageByVerifier = {}
        for verifier in failurePercentageByVerifier.keys():
            minFailurePercentage = 100.0
            minFailureQuantOption = 'N/A'
            isNoFailures = True
            for quantOption in self.quantizationVariations:
                failurePercentage = failurePercentageByVerifier[verifier][quantOption]
                if failurePercentage != 'N/A' and failurePercentage <= minFailurePercentage:
                    minFailurePercentage = failurePercentage
                    minFailureQuantOption = quantOption
                    isNoFailures = False
            if isNoFailures:
                minFailurePercentage = 0.0
                minFailureQuantOption = 'N/A'
            minFailurePercentageByVerifier[verifier] = (minFailurePercentage, minFailureQuantOption)
        return minFailurePercentageByVerifier
