# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from .result.nd_result_set import ResultSet
from .result.nd_result_utils import calculateFailurePercentageForResults
from .nd_comparator import (dequantizeBiases, compareBiases, dequantizeWeights, compareWeights,
                            compareActivations, analyzeInputData)
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message


class Processor:

    def __init__(self, quant_schemes, comparison_algorithms, logger) -> None:
        self._logger = logger
        self._quant_schemes = quant_schemes
        self._comparison_algorithms = comparison_algorithms
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
        self._logger.info(get_progress_message('Dequantizing weights...'))
        dequantizeWeights(self._logger, self._quant_schemes, opsMap)
        self._logger.info(get_progress_message('Comparing weights...'))
        self.weightResults = compareWeights(
            self._quant_schemes, opsMap,
            self._comparison_algorithms['weight_comparison_algorithms'], self._logger)

    def getWeightResults(self) -> ResultSet:
        return self.weightResults

    def processBiasResults(self, opsMap) -> ResultSet:
        self._logger.info(get_progress_message('Dequantizing biases...'))
        dequantizeBiases(self._quant_schemes, opsMap)
        self._logger.info(get_progress_message('Comparing biases...'))
        self.biasResults = compareBiases(self._quant_schemes, opsMap,
                                         self._comparison_algorithms['bias_comparison_algorithms'],
                                         self._logger)

    def getBiasResults(self) -> ResultSet:
        return self.biasResults

    def processActivationResults(self, opsMap) -> ResultSet:
        self._logger.info(get_progress_message('Comparing activations...'))
        self.activationResults = compareActivations(
            self._quant_schemes, opsMap, self._comparison_algorithms['act_comparison_algorithms'])

    def getActivationResults(self) -> ResultSet:
        return self.activationResults

    def processInputData(self, inputData) -> ResultSet:
        self._logger.info(get_progress_message('Analyzing input data...'))
        self.inputResults = analyzeInputData(
            inputData, self._comparison_algorithms['input_data_analysis_algorithms'])

    def getInputResults(self) -> ResultSet:
        return self.inputResults

    def getWeightFailurePercentage(self):
        self.weightFailurePercentageByVerifier = calculateFailurePercentageForResults(
            self._quant_schemes, self.weightResults)
        return self.weightFailurePercentageByVerifier

    def getBiasFailurePercentage(self):
        self.biasFailurePercentageByVerifier = calculateFailurePercentageForResults(
            self._quant_schemes, self.biasResults)
        return self.biasFailurePercentageByVerifier

    def getActivationFailurePercentage(self):
        self.activationFailurePercentageByVerifier = calculateFailurePercentageForResults(
            self._quant_schemes, self.activationResults)
        return self.activationFailurePercentageByVerifier

    def getMaxWeightFailurePercentage(self):
        maxFailurePercentage = 0.0
        for verifier in self.weightFailurePercentageByVerifier.keys():
            for quantOption in self._quant_schemes:
                weightFailurePercentage = self.weightFailurePercentageByVerifier[verifier][
                    quantOption]
                if weightFailurePercentage != 'N/A' and weightFailurePercentage > maxFailurePercentage:
                    maxFailurePercentage = weightFailurePercentage
                    self.maxWeightFailureVerifier = verifier
                    self.maxWeightFailureQuantOption = quantOption
        return maxFailurePercentage

    def getMaxBiasFailurePercentage(self):
        maxFailurePercentage = 0.0
        for verifier in self.biasFailurePercentageByVerifier.keys():
            for quantOption in self._quant_schemes:
                biasFailurePercentage = self.biasFailurePercentageByVerifier[verifier][quantOption]
                if biasFailurePercentage != 'N/A' and biasFailurePercentage > maxFailurePercentage:
                    maxFailurePercentage = biasFailurePercentage
                    self.maxBiasFailureVerifier = verifier
                    self.maxBiasFailureQuantOption = quantOption
        return maxFailurePercentage

    def getMaxActivationFailurePercentage(self):
        maxFailurePercentage = 0.0
        for verifier in self.activationFailurePercentageByVerifier.keys():
            for quantOption in self._quant_schemes:
                activationFailurePercentage = self.activationFailurePercentageByVerifier[verifier][
                    quantOption]
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
            for quantOption in self._quant_schemes:
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

    def processResults(self, extractor):
        self.processInputData(extractor.getInputData())
        self.processWeightResults(extractor.getAllOps())
        self.processBiasResults(extractor.getAllOps())
        self.processActivationResults(extractor.getAllOps())
        input_data_result = self.getInputResults()
        weight_result = self.getWeightResults()
        bias_result = self.getBiasResults()
        activation_result = self.getActivationResults()
        getWeightFailurePercentage = self.getWeightFailurePercentage()
        getBiasFailurePercentage = self.getBiasFailurePercentage()
        getActivationFailurePercentage = self.getActivationFailurePercentage()

        return input_data_result, weight_result, bias_result, activation_result, getWeightFailurePercentage, getBiasFailurePercentage, getActivationFailurePercentage
