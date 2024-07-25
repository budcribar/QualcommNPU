#=============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

from typing import Dict, List, Tuple
import re
from qti.aisw.quantization_checker.Results.ResultNode import ResultNode
from qti.aisw.quantization_checker.utils import Verifiers

# Collection of methods useful for formatting the results for eventual display to the user
PASSDIV = '<div class=pass>Pass</div>'
FAILDIV = '<div class=fail>Fail</div>'

def getResultsForHtml(quantizationVariation, results) -> List[ResultNode]:
    resultList = []
    if quantizationVariation in results.keys():
        for resultNode in results[quantizationVariation]:
            comparisonPassFails = {}
            for comparisonAlgorithmName, comparisonResult in resultNode.getResults().items():
                if comparisonResult != dict():
                    comparisonPassFails[comparisonAlgorithmName] = translateTrueFalseToPassFailForHtml(comparisonResult['pass'])
            resultList.append(ResultNode(resultNode.getNodeName(), resultNode.getTensorName(), resultNode.getTensorType(), comparisonPassFails, str(resultNode.getScale()), str(resultNode.getOffset())))
    return resultList

def getActivationResultsForHtml(quantizationVariation, inputFile, results) -> List[ResultNode]:
    resultList = []
    if quantizationVariation != 'unquantized' and quantizationVariation in results.keys():
        perInputResults = getActivationsForInput(inputFile, results[quantizationVariation])
        for activationResultNode in perInputResults:
            comparisonPassFails = {}
            for comparisonAlgorithmName, comparisonResult in activationResultNode.getResults().items():
                if comparisonResult != dict():
                    comparisonPassFails[comparisonAlgorithmName] = translateTrueFalseToPassFailForHtml(comparisonResult['pass'])
            resultList.append(ResultNode(activationResultNode.getNodeName(), activationResultNode.getTensorName(), activationResultNode.getTensorType(), comparisonPassFails, str(activationResultNode.getScale()), str(activationResultNode.getOffset())))
    return resultList

def getActivationsForInput(inputFilename, results):
    perInputResults = []
    for activationResultNode in results:
        if activationResultNode.getInputFilename() == inputFilename:
            perInputResults.append(activationResultNode)
    return perInputResults

def getInputResultsForHtml(results) -> List[ResultNode]:
    resultList = []
    for resultNode in results.values():
        comparisonPassFails = {}
        for comparisonAlgorithmName, comparisonResult in resultNode.getResults().items():
            if comparisonResult != dict():
                comparisonPassFails[comparisonAlgorithmName] = translateTrueFalseToPassFailForHtml(comparisonResult['pass'])
        resultList.append(ResultNode(resultNode.getNodeName(), resultNode.getTensorName(), resultNode.getTensorName(), comparisonPassFails))
    return resultList

def translateTrueFalseToPassFailForHtml(result) -> str:
    if result:
        return PASSDIV
    else:
        return FAILDIV

def getListOfAlgorithmNamesFromResults(quantizationVariations, results) -> List[str]:
    algorithmsWithDups = []
    for quantizationVariation in quantizationVariations:
        if quantizationVariation not in results:
            continue
        for comparisonResults in results[quantizationVariation].values():
            for comparisonResult in comparisonResults:
                algorithmsWithDups.extend(comparisonResult.getResults().keys())
    return list(dict.fromkeys(algorithmsWithDups))

def getListOfAlgorithmNamesFromInputResults(results) -> List[str]:
    algorithmNames = []
    for resultNode in results.values():
        for comparisonAlgorithmName in resultNode.getResults().keys():
            if comparisonAlgorithmName not in algorithmNames:
                algorithmNames.append(comparisonAlgorithmName)
    return algorithmNames

def translateAlgorithmNamesToDescriptiveNames(algorithms) -> List[str]:
    readableAlgoNames = []
    for algorithm in algorithms:
        if algorithm == Verifiers.STATS:
            readableAlgoNames.append(Verifiers.STATS_DESCRIPTIVE_NAME)
        elif algorithm == Verifiers.DATA_RANGE:
            readableAlgoNames.append(Verifiers.DATA_RANGE_DESCRIPTIVE_NAME)
        elif algorithm == Verifiers.SQNR:
            readableAlgoNames.append(Verifiers.SQNR_DESCRIPTIVE_NAME)
        elif algorithm == Verifiers.MIN_MAX:
            readableAlgoNames.append(Verifiers.MIN_MAX_DESCRIPTIVE_NAME)
        elif algorithm == Verifiers.MAX_DIFF:
            readableAlgoNames.append(Verifiers.MAX_DIFF_DESCRIPTIVE_NAME)
        elif algorithm == Verifiers.DATA_DISTRIBUTION:
            readableAlgoNames.append(Verifiers.DATA_DISTRIBUTION_DESCRIPTIVE_NAME)
    return readableAlgoNames

def getAnalysisDescription(results) -> str:
    return Verifiers.getFailureAnalysisDescription(getListOfAlgorithmNamesThatFailedFromHtmlResults(results))

def getListOfAlgorithmNamesThatFailedFromHtmlResults(htmlResults) -> List[str]:
    listOfAlgorithmNamesThatFailed = []
    if FAILDIV in htmlResults.getResults().values():
        for algoName, result in htmlResults.getResults().items():
            if result == FAILDIV:
                listOfAlgorithmNamesThatFailed.append(algoName)
    return listOfAlgorithmNamesThatFailed

def getListOfAlgorithmsFromResults(results, algorithmKeys) -> List[List[str]]:
    onlyAlgosList = []
    for result in results:
        valuesList = []
        for key in algorithmKeys:
            if key in result.getResults().keys():
                valuesList.append(result.getResults()[key])
            else:
                valuesList.append('N/A')
        valuesList.append(getAnalysisDescription(result))
        onlyAlgosList.append(valuesList)
    return onlyAlgosList

def getListOfResultsWithoutAlgorithms(results) -> List:
    noVerifiers = []
    for result in results:
        noVerifiers.append([result.getNodeName(), result.getTensorName(), result.getTensorType(), str(result.getScale()), str(result.getOffset())])
    return noVerifiers

def translateAlgorithmNameToAlgorithmDescription(readableAlgorithmNames) -> Dict[str, str]:
    descriptions = {}
    for readableAlgorithmName in readableAlgorithmNames:
        if readableAlgorithmName == Verifiers.STATS_DESCRIPTIVE_NAME:
            descriptions[readableAlgorithmName] = Verifiers.STATS_DESCRIPTION
        elif readableAlgorithmName == Verifiers.DATA_RANGE_DESCRIPTIVE_NAME:
            descriptions[readableAlgorithmName] = Verifiers.DATA_RANGE_DESCRIPTION
        elif readableAlgorithmName == Verifiers.SQNR_DESCRIPTIVE_NAME:
            descriptions[readableAlgorithmName] = Verifiers.SQNR_DESCRIPTION
        elif readableAlgorithmName == Verifiers.MIN_MAX_DESCRIPTIVE_NAME:
            descriptions[readableAlgorithmName] = Verifiers.MIN_MAX_DESCRIPTION
        elif readableAlgorithmName == Verifiers.MAX_DIFF_DESCRIPTIVE_NAME:
            descriptions[readableAlgorithmName] = Verifiers.MAX_DIFF_DESCRIPTION
        elif readableAlgorithmName == Verifiers.DATA_DISTRIBUTION_DESCRIPTIVE_NAME:
            descriptions[readableAlgorithmName] = Verifiers.DATA_DISTRIBUTION_DESCRIPTION
    return descriptions

def getFailedNodes(results) -> List:
    failedNodes = []
    for result in results:
        # Exclude failed cases for Clustering of Unquantized Data from the summary, used only as additional information for now.
        if Verifiers.DATA_DISTRIBUTION in result.getResults():
            if FAILDIV in list(result.getResults().values())[:-1]:
                failedNodes.append(result)
        elif FAILDIV in result.getResults().values():
            failedNodes.append(result)
    return failedNodes

def formatResultsForLogConsole(quantizationVariations, results, showOnlyFailedResults = False) -> Dict[str, List]:
    formattedResults = {}
    for quantizationVariation in quantizationVariations:
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in results.keys():
            continue
        quantOptionResults = []
        resultList = results[quantizationVariation]
        for resultNode in resultList:
            for comparisonAlgorithmName, comparisonResult in resultNode.getResults().items():
                if comparisonResult != dict():
                    if showOnlyFailedResults and comparisonResult['pass'] == True:
                        # skip if only displaying failed nodes
                        continue
                    # ['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Scale', 'Offset']
                    quantOptionResults.append([resultNode.getNodeName(), resultNode.getTensorName(), comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName, str(resultNode.getScale()), str(resultNode.getOffset())])
        if len(quantOptionResults) > 0:
            quantOptionResults.sort(key=lambda x: x[2])
        else:
            quantOptionResults.append(['No Failures Found', '-', '-', '-', '-', '-', '-', '-'])
        formattedResults[quantizationVariation] = quantOptionResults
    return formattedResults

def formatActivationResultsForLogConsole(quantizationVariations, results, showOnlyFailedResults = False) -> Dict[str, List]:
    logFormattedResults = {}
    for quantizationVariation in quantizationVariations[1:]:
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in results.keys():
            continue
        quantOptionResults = []
        resultList = results[quantizationVariation]
        for activationResultNode in resultList:
            for comparisonAlgorithmName, comparisonResult in activationResultNode.getResults().items():
                if comparisonResult != dict():
                    if showOnlyFailedResults and comparisonResult['pass'] == True:
                        # skip if only displaying failed nodes
                        continue
                    # ['Op Name', nodeType, 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename', 'Scale', 'Offset']
                    quantOptionResults.append([activationResultNode.getNodeName(), activationResultNode.getTensorName(), comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName, activationResultNode.getInputFilename(), str(activationResultNode.getScale()), str(activationResultNode.getOffset())])
        if len(quantOptionResults) != 0:
            quantOptionResults.sort(key=lambda x: x[2])
        else:
            quantOptionResults.append(['No Failures Found', '-', '-', '-', '-', '-', '-', '-', '-'])
        logFormattedResults[quantizationVariation] = quantOptionResults
    return logFormattedResults

def formatActivationsForCsv(resultList) -> Tuple:
    quantOptionResults = {}
    quantOptionResultsHeader = {}
    for activationResultNode in resultList:
        formatDataForCsv(activationResultNode.getNodeName(), activationResultNode.getResults(), quantOptionResults, quantOptionResultsHeader, activationResultNode.getTensorName(), None, None, activationResultNode.getInputFilename())
    return (quantOptionResults, quantOptionResultsHeader)

def formatWeightsAndBiasesForCsv(resultList) -> Tuple:
    quantOptionResults = {}
    quantOptionResultsHeader = {}
    for resultNode in resultList:
        formatDataForCsv(resultNode.getNodeName(), resultNode.getResults(), quantOptionResults, quantOptionResultsHeader, resultNode.getTensorName(), str(resultNode.getScale()), str(resultNode.getOffset()))
    return (quantOptionResults, quantOptionResultsHeader)

def formatInputsForCsv(results) -> Tuple:
    quantOptionResults = {}
    quantOptionResultsHeader = {}
    for filename, resultNode in results.items():
        formatDataForCsv(filename, resultNode.getResults(), quantOptionResults, quantOptionResultsHeader, 'input')
    return (quantOptionResults, quantOptionResultsHeader)

def formatDataForCsv(opName, comparisonResults, quantOptionResults, quantOptionResultsHeader, nodeName=None, scale=None, offset=None, inputFilename=None):
    for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
        if comparisonResult != dict():
            perComparisonAlgorithmResults = []
            if comparisonAlgorithmName in quantOptionResults.keys():
                perComparisonAlgorithmResults = quantOptionResults[comparisonAlgorithmName]
            result = [opName, nodeName, comparisonResult['pass']]
            if scale and offset:
                result.append(scale)
                result.append(offset)
            result.append(comparisonResult['threshold'])
            if inputFilename:
                result.append(inputFilename)
            resultData = re.findall(r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?', str(comparisonResult['data']))
            resultHeader = re.findall(r'[a-zA-Z ]+[a-zA-Z]+[^ 0-9Ee]', str(comparisonResult['data']))
            if scale and offset:
                resultHeader.append('scale')
                resultHeader.append('offset')
            result[3:3] = resultData
            perComparisonAlgorithmResults.append(result)
            quantOptionResults[comparisonAlgorithmName] = perComparisonAlgorithmResults
            quantOptionResultsHeader[comparisonAlgorithmName] = resultHeader

def calculateFailurePercentageForResults(quantizationVariations, results):
    verifierQuantizationFailurePercentage = {}
    for verifierName in Verifiers.VERIFIER_NAMES.values():
        verifierQuantizationFailurePercentage[verifierName] = getFailurePercentage(
            quantizationVariations,
            getFailureNodeCount(verifierName, quantizationVariations, results),
            getTotalNodeCount(verifierName, quantizationVariations, results)
        )
    return verifierQuantizationFailurePercentage

def getFailurePercentage(quantizationVariations, failureNodeCount, totalNodeCount):
    quantizationFailurePercentage = {}
    for quantizationVariation in quantizationVariations:
        totalNodeCountForQuantization = totalNodeCount[quantizationVariation]
        if totalNodeCountForQuantization == 'N/A':
            quantizationFailurePercentage[quantizationVariation] = 'N/A'
        else:
            failureNodeCountForQuantization = 0
            if quantizationVariation in failureNodeCount:
                failureNodeCountForQuantization = failureNodeCount[quantizationVariation]
            quantizationFailurePercentage[quantizationVariation] = (float(failureNodeCountForQuantization) / float(totalNodeCountForQuantization)) * 100.0
    return quantizationFailurePercentage

def getFailureNodeCount(verifierName, quantizationVariations, results):
    verifierFailureCount = {}
    for quantizationVariation in quantizationVariations:
        resultList = results[quantizationVariation]
        if resultList:
            verifierFailureCount[quantizationVariation] = resultList.getFailureCountForVerifier(verifierName)
    return verifierFailureCount

def getTotalNodeCount(verifierName, quantizationVariations, results):
    verifierTotalCount = {}
    for quantizationVariation in quantizationVariations:
        resultList = results[quantizationVariation]
        totalCount = 0
        if resultList:
            totalCount = resultList.getTotalCountForVerifier(verifierName)
        if totalCount != 0:
            verifierTotalCount[quantizationVariation] = totalCount
        else:
            verifierTotalCount[quantizationVariation] = 'N/A'
    return verifierTotalCount
