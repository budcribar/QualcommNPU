# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from cmath import inf
from typing import Dict, Any
import numpy as np
from .result.nd_result_node import ActivationResultNode, BiasResultNode, InputResultNode, WeightResultNode
from .result.nd_result_list import ResultList
from .result.nd_result_set import ResultSet
from .nd_op import Op
import qti.aisw.accuracy_debugger.lib.quant_checker.nd_verifier as Verifiers

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message


class QuantizationComparisonAlgorithm:

    def __init__(self, algorithm: Verifiers.QcAlgorithm) -> None:
        self.algorithm = algorithm

    def compare(self, lhs, rhs, threshold) -> Dict[str, Any]:
        '''
        Given the comparison algorithm, it creates the object of
        corresponsing comparator and returns the result
        '''
        if self.algorithm == Verifiers.QcAlgorithm.MAX_ABS_DIFFERENCE:
            return MaxAbsDifference(threshold)(lhs, rhs)
        elif self.algorithm == Verifiers.QcAlgorithm.MIN_MAX_COMPARE:
            return MinMaxComparisonAlgorithm(threshold)(lhs, rhs)
        elif self.algorithm == Verifiers.QcAlgorithm.SQNR:
            return SqnrCalculation(threshold)(lhs, rhs)
        elif self.algorithm == Verifiers.QcAlgorithm.STATS:
            return CollectStats(threshold)(lhs)
        elif self.algorithm == Verifiers.QcAlgorithm.DATA_RANGE_CHECK:
            return BitWidthComparisonAlgorithm(threshold)(lhs)
        elif self.algorithm == Verifiers.QcAlgorithm.DATA_DISTRIBUTION_CHECK:
            return DistributionComparisonAlgorithm(threshold)(lhs)


class CollectStats:
    '''
    Implements the STATS comparator
    '''

    def __init__(self, threshold) -> None:
        self.threshold = float(threshold)

    def __call__(self, unquantized_data) -> Dict:
        median = np.median(unquantized_data)
        mean = np.mean(unquantized_data)
        variance = np.var(unquantized_data)
        stdDev = np.std(unquantized_data)
        min = np.amin(unquantized_data)
        max = np.amax(unquantized_data)
        skew = 0.0
        if stdDev != 0.0:
            skew = np.mean(np.power(((np.array(unquantized_data) - mean) / stdDev), 3))
        vals, counts = np.unique(unquantized_data, return_counts=True)
        index = np.argmax(counts)
        mode = vals[index]
        passes = True
        if abs(skew) > self.threshold:
            passes = False
        return {
            'pass':
            passes,
            'data':
            'skew: ' + str(skew) + ' min: ' + str(min) + ' max: ' + str(max) + ' median: ' +
            str(median) + ' variance: ' + str(variance) + ' stdDev: ' + str(stdDev) + ' mode: ' +
            str(mode),
            'threshold':
            self.threshold
        }


class SqnrCalculation:
    '''
    Implements the SQNR comparator
    '''

    def __init__(self, threshold) -> None:
        self.threshold = float(threshold)

    def __call__(self, unquantized_data, dequantized_data) -> Dict[str, Any]:
        difference = np.subtract(dequantized_data, unquantized_data)
        squaredDiff = np.square(difference)
        meanSquareError = np.mean(squaredDiff)

        if meanSquareError == 0:
            return {'pass': True, 'data': 'sqnr: ' + str(inf), 'threshold': self.threshold}
        else:
            squaredUnquantized = np.square(unquantized_data)
            meanSquareUnquantized = np.mean(squaredUnquantized)

            ratio = meanSquareUnquantized / meanSquareError
            logRatio = 10 * np.log10(ratio)
            passes = True
            if logRatio < self.threshold:
                passes = False
            return {'pass': passes, 'data': 'sqnr: ' + str(logRatio), 'threshold': self.threshold}


class MaxAbsDifference:
    '''
    Implements the MAXDIFF comparator
    '''

    def __init__(self, threshold) -> None:
        self.threshold = float(threshold)

    def __call__(self, lhs, rhs) -> Dict[str, Any]:
        result = np.absolute(np.subtract(lhs, rhs)).tolist()
        if result:
            maxResult = max(result)
            passes = True
            if maxResult > self.threshold:
                passes = False
            return {
                'pass': passes,
                'data': 'largest absolute difference: ' + str(maxResult),
                'threshold': self.threshold
            }
        else:
            return {
                'pass': True,
                'data': 'largest absolute difference: ' + str(0.0),
                'threshold': self.threshold
            }


class MinMaxComparisonAlgorithm:
    '''
    Implements the MINMAX comparator
    '''

    def __init__(self, threshold) -> None:
        self.threshold = float(threshold)

    def __call__(self, lhs, rhs) -> Dict[str, Any]:
        unquantizedMin = np.amin(lhs)
        unquantizedMax = np.amax(lhs)
        dequantizedMin = np.amin(rhs)
        dequantizedMax = np.amax(rhs)
        passes = True
        if abs(unquantizedMin -
               dequantizedMin) > self.threshold or abs(unquantizedMax -
                                                       dequantizedMax) > self.threshold:
            passes = False
        return {
            'pass':
            passes,
            'data':
            'min: ' + str(abs(unquantizedMin - dequantizedMin)) + ' max: ' +
            str(abs(unquantizedMax - dequantizedMax)),
            'threshold':
            self.threshold
        }


class BitWidthComparisonAlgorithm:
    '''
    Implements the DATA_RANGE comparator
    '''

    def __init__(self, threshold) -> None:
        self.threshold = float(threshold)

    def __call__(self, data) -> Dict:
        uniques = np.unique(data.astype(int)).shape[0]
        dataRange = np.amax(data) - np.amin(data)
        passes = True
        if uniques > self.threshold or dataRange > self.threshold:
            passes = False
        return {
            'pass': passes,
            'data': 'unique dec places: ' + str(uniques) + ' data range: ' + str(dataRange),
            'threshold': self.threshold
        }


class DistributionComparisonAlgorithm:
    '''
    Implements the DATA_DISTRIBUTION comparator
    '''

    def __init__(self, threshold) -> None:
        self.thresholdPerBin = float(threshold)

    def __call__(self, data) -> Dict[str, Any]:
        dataWithPrecision = []
        for val in data:
            if abs(val - int(val)) > 0:
                dataWithPrecision.append(val)
        maxRatio = 0
        if len(dataWithPrecision) > 0:
            dataWithPrecision = np.array(dataWithPrecision)
            dataRange = int(max(dataWithPrecision) - min(dataWithPrecision) + 1)
            uint8_max = np.iinfo(np.uint8).max + 1
            if (dataRange < uint8_max):
                dataRange = uint8_max
            hist, _ = np.histogram(dataWithPrecision, bins=dataRange)
            hist = np.where(hist > 0, hist - 1, hist)
            distRatio = (hist / dataWithPrecision.shape)
            maxRatio = np.amax(distRatio)
        passes = True
        if maxRatio > self.thresholdPerBin:
            passes = False
        return {
            'pass': passes,
            'data': 'Distribution of pixels above threshold: ' + str(maxRatio),
            'threshold': self.thresholdPerBin
        }


class Comparator:

    def __init__(self, algorithm: Verifiers.QcAlgorithm) -> None:
        self.algorithm = algorithm

    def compare(self, lhs, rhs, threshold) -> Dict[str, Any]:
        qca = QuantizationComparisonAlgorithm(self.algorithm)
        return qca.compare(lhs, rhs, threshold)


def dequantizeWeights(logger, quant_schemes, opsMap) -> None:
    '''
    :param logger: logger
    :param quant_schemes: List of quantization schemes
    :opsMap: Dictionary of quantization scheme to all ops in the model

    for each quant_scheme in all quantization schemes:
        for each op in model:
            if op has weight node:
                dequant the quantized weights and save it
    '''
    for quant_scheme in quant_schemes:
        # skip quantized models which are failed to get converted correctly
        if quant_scheme not in opsMap or quant_scheme == 'unquantized':
            continue
        for op_name, op in opsMap[quant_scheme].items():
            if op.getWeightName() not in (None, ''):
                dequantized_weight = dequantizeOpWeights(logger, op.getWeights(),
                                                         op.getWeightsDims(),
                                                         op.getIsQuantizedPerChannel(),
                                                         op.getWeightsScaleOffset())
                op.setDequantizedWeights(dequantized_weight)
                opsMap[quant_scheme][op_name] = op


def compareWeights(quant_schemes, opsMap, comparison_algorithms, logger) -> ResultSet:
    '''
    :param logger: logger
    :param quant_schemes: List of quantization schemes
    :param opsMap: Dictionary of quantization scheme to all ops in the model 
    :param comparison_algorithms: List of all comparators for which sensitivity has to be calculated

    returns object of ResultSet which contains the weight sensitivity result for each op in each quant_scheme
    '''
    # Set the threshold for DATA_RANGE_ANALYZER compator based on the bitwidths
    setDataRangeAnalyzerThreshold(comparison_algorithms, Op.getWeightWidth())

    # Create an object of ResultSet
    results = ResultSet()

    # Save the fp32 weights for each op
    unquantized_weight = {}
    for op_name, op in opsMap['unquantized'].items():
        unquantized_weight[op_name] = op.getWeights()

    # for each quant_scheme, for each op, compute all comparator results
    for quant_scheme in quant_schemes:
        # skip quantized models which fail to convert correctly
        if quant_scheme not in opsMap:
            continue
        perOpResults = ResultList()
        for op_name, op in opsMap[quant_scheme].items():
            weightName = op.getWeightName()
            if weightName not in (None, '') and len(op.getDequantizedWeights())>0:
                # grab the unquantized weights
                scale = None
                offset = None
                if quant_scheme != 'unquantized':
                    weight_scale_offsets = op.getWeightsScaleOffset()
                    if not op.getIsQuantizedPerChannel():
                        scale = weight_scale_offsets['scale']
                        offset = weight_scale_offsets['offset']
                resultNode = WeightResultNode(
                    op_name, weightName,
                    runLintingRules(quant_scheme, unquantized_weight[op_name],
                                    op.getDequantizedWeights(), comparison_algorithms, logger),
                    scale, offset)
                perOpResults.append(resultNode)
            if quant_scheme == "unquantized":
                if weightName not in (None, '') and len(unquantized_weight[op_name]) > 0:
                    resultNode = WeightResultNode(
                        op_name, weightName,
                        runLintingRules(quant_scheme, unquantized_weight[op_name],
                                        None, comparison_algorithms, logger),
                        None, None)
                    perOpResults.append(resultNode)
        results.add(quant_scheme, perOpResults)
    return results


def comparePerOp(unquantized_data, dequantized_data, comparison_algorithms) -> Dict:
    '''
    :param unquantized_data: fp32 tensor
    :param dequantized_data: dequantized tensor
    :comparison_algorithms: List of all comparators for which results has to be calculated

    returns the comparison resuts for each comparator
    '''
    results = {}
    if comparison_algorithms is not None:
        for comparisonAlgorithm in comparison_algorithms:
            results[comparisonAlgorithm['algo_name']] = doCompare(unquantized_data,
                                                                  dequantized_data,
                                                                  comparisonAlgorithm)
    else:
        results[Verifiers.MAX_DIFF] = doCompare(unquantized_data, dequantized_data,
                                                (Verifiers.MAX_DIFF, "0.5"))
    return results


def doCompare(unquantized_data, dequantized_data, comparisonAlgorithm) -> Dict:
    '''
    :param unquantized_data: fp32 tensor
    :param dequantized_data: dequantized tensor
    :comparison_algorithm: comparator for which results has to be calculated
        
    return the comparison result for one comparator
    '''
    threshold = '0.0'
    comparator = None
    if dequantized_data is not None:
        if comparisonAlgorithm['algo_name'] == Verifiers.SQNR:
            comparator = Comparator(Verifiers.QcAlgorithm.SQNR)
            threshold = Verifiers.DEFAULT_THRESHOLDS.SQNR
        elif comparisonAlgorithm['algo_name'] == Verifiers.MAX_DIFF:
            comparator = Comparator(Verifiers.QcAlgorithm.MAX_ABS_DIFFERENCE)
            threshold = Verifiers.DEFAULT_THRESHOLDS.MAX_DIFF
        elif comparisonAlgorithm['algo_name'] == Verifiers.MIN_MAX:
            comparator = Comparator(Verifiers.QcAlgorithm.MIN_MAX_COMPARE)
            threshold = Verifiers.DEFAULT_THRESHOLDS.MIN_MAX
    else:
        if comparisonAlgorithm['algo_name'] == Verifiers.STATS:
            comparator = Comparator(Verifiers.QcAlgorithm.STATS)
            threshold = Verifiers.DEFAULT_THRESHOLDS.STATS
        elif comparisonAlgorithm['algo_name'] == Verifiers.DATA_DISTRIBUTION:
            comparator = Comparator(Verifiers.QcAlgorithm.DATA_DISTRIBUTION_CHECK)
            threshold = Verifiers.DEFAULT_THRESHOLDS.DATA_DISTRIBUTION
        elif comparisonAlgorithm['algo_name'] == Verifiers.DATA_RANGE:
            comparator = Comparator(Verifiers.QcAlgorithm.DATA_RANGE_CHECK)
    if 'threshold' in comparisonAlgorithm:
        threshold = comparisonAlgorithm['threshold']
    if comparator is not None:
        return comparator.compare(unquantized_data, dequantized_data, threshold)
    else:
        return {}


def dequantizeOpWeights(logger, quantized_weight, weight_dims, is_quantized_per_channel,
                        weight_scale_offset):
    '''
    :param quantized_weight: quantized weight tensor
    :param weight_dims: shape of the weight tensor
    :param is_quantized_per_channel: True of pcq
    :weight_scale_offset: scale and offset encodings

    return dequant weight
    '''
    if is_quantized_per_channel:
        np_quantized_weight = np.array(quantized_weight, dtype=float)
        reshapequantized_weight = np.reshape(np_quantized_weight, tuple(weight_dims))
        axis = weight_scale_offset['axis']
        scaleOffsets = weight_scale_offset['scale_offsets']
        deQuantizedPcqWeights = reshapequantized_weight
        for i in range(reshapequantized_weight.shape[axis]):
            scale = scaleOffsets[i]['scale']
            offset = scaleOffsets[i]['offset']
            deQuantizedPcqWeights[:, :, :,
                                  i] = (reshapequantized_weight[:, :, :, i] + offset) * scale
        return np.reshape(deQuantizedPcqWeights,
                          weight_dims[0] * weight_dims[1] * weight_dims[2] * weight_dims[3])
    elif not is_quantized_per_channel:
        scale = weight_scale_offset['scale']
        offset = weight_scale_offset['offset']
        np_quantized_weight = np.array(quantized_weight, dtype=float)
        return np.multiply(np.add(np_quantized_weight, offset), scale)
    else:
        logger.info(get_message('returning quantized weights without dequantizing'))
        return quantized_weight


def dequantizeBiases(quant_schemes, opsMap) -> None:
    '''
    :param quant_schemes: List of quant schemes
    :param opsMap: Dictionary of quantization scheme to all ops in the model

    for each quant_scheme in all quantization schemes:
        for each op in model:
            if op has bias node:
                dequant the quantized weights and save it
    '''
    for quant_scheme in quant_schemes:
        # skip quantized models which failed to convert properly
        if quant_scheme not in opsMap or quant_scheme == 'unquantized':
            continue
        for op_name, op in opsMap[quant_scheme].items():
            if op.getBiasName() not in (None, ''):
                # Nodes from PCQ models can have scale 0, we consider scale from weight/input nodes to dequantize those
                if op.getBiasScale() == 0 and op.getIsQuantizedPerChannel():
                    dequantizedBiases = dequantizePCQOpBiases(op.getBiases(),
                                                              op.getWeightsScaleOffset(),
                                                              op.getInputNodeScale())
                else:
                    dequantizedBiases = dequantizeOpBiases(op.getBiases(), op.getBiasScale(),
                                                           op.getBiasOffset())
                op.setDequantizedBiases(dequantizedBiases)
                opsMap[quant_scheme][op_name] = op


def dequantizeOpBiases(quantizedBiases, biasScale, biasOffset):
    '''
    returns the dequant bias
    '''
    return np.multiply(np.add(quantizedBiases, biasOffset), biasScale)


def dequantizePCQOpBiases(biases, weight_scale_offset, inputNodeScale):
    scaleOffsets = weight_scale_offset['scale_offsets']
    deQuantizedPcqBiases = np.zeros(shape=biases.shape)
    for i in range(biases.shape[0]):
        weightScale = scaleOffsets[i]['scale']
        deQuantizedPcqBiases[i] = biases[i] * inputNodeScale * weightScale
    return deQuantizedPcqBiases


def compareBiases(quant_schemes, opsMap, comparison_algorithms, logger) -> ResultSet:
    setDataRangeAnalyzerThreshold(comparison_algorithms, Op.getBiasWidth())
    results = ResultSet()
    unquantizedBiases = {}
    for op_name, op in opsMap['unquantized'].items():
        unquantizedBiases[op_name] = op.getBiases()
    for quant_scheme in quant_schemes:
        # skip quantized models which fail to convert correctly
        if quant_scheme not in opsMap:
            continue
        perOpResults = ResultList()
        for op_name, op in opsMap[quant_scheme].items():
            biasName = op.getBiasName()
            if biasName not in (None, ''):
                scale = None
                offset = None
                # grab the unquantized biases
                if quant_scheme != 'unquantized':
                    if not op.getIsQuantizedPerChannel():
                        scale = op.getBiasScale()
                        offset = op.getBiasOffset()
                resultNode = BiasResultNode(
                    op_name, biasName,
                    runLintingRules(quant_scheme, unquantizedBiases[op_name],
                                    op.getDequantizedBiases(), comparison_algorithms, logger),
                    scale, offset)
                perOpResults.append(resultNode)
        results.add(quant_scheme, perOpResults)
    return results


def runLintingRules(quant_scheme, unquantized_data, dequantized_data, comparison_algorithms,
                    logger) -> Dict:
    dequantizedArray = None
    unquantizedArray = np.nan_to_num(np.array(unquantized_data))
    if quant_scheme != 'unquantized':
        dequantizedArray = np.nan_to_num(np.array(dequantized_data))
        dequantShape = list(dequantizedArray.shape)
        unquantShape = list(unquantizedArray.shape)
        if unquantShape != dequantShape:
            logger.info(
                get_message("WARNING! two data files have different shapes, " + str(unquantShape) +
                            " vs " + str(dequantShape) +
                            " please check manually! returning empty results"))
            return {}
        return comparePerOp(unquantizedArray, dequantizedArray, comparison_algorithms)
    else:
        return comparePerOp(unquantizedArray, dequantizedArray, comparison_algorithms)


def comparePerOpActivations(unquantizedActivations, quantizedScale, quantizedOffset,
                            comparison_algorithms):
    setDataRangeAnalyzerThreshold(comparison_algorithms, Op.getActivationWidth())
    results = {}
    quantizedMin = 0
    quantizedMax = 0
    if quantizedScale is not None:
        quantizedMin = (0 + quantizedOffset) * quantizedScale
        quantizedMax = (int(Verifiers.getMaxValueBasedOnBitWidth(Op.getActivationWidth())) - 1 +
                        quantizedOffset) * quantizedScale
    if comparison_algorithms is not None:
        for comparisonAlgorithm in comparison_algorithms:
            if quantizedScale is not None and comparisonAlgorithm['algo_name'] == "minmax":
                results[comparisonAlgorithm['algo_name']] = doActivationCompare(
                    unquantizedActivations, (quantizedMin, quantizedMax), comparisonAlgorithm)
            elif comparisonAlgorithm['algo_name'] != "minmax":
                results[comparisonAlgorithm['algo_name']] = doActivationCompare(
                    unquantizedActivations, None, comparisonAlgorithm)
    else:
        results['minmax'] = doActivationCompare(unquantizedActivations,
                                                (quantizedMin, quantizedMax), ("minmax", "0.5"))
    return results


def doActivationCompare(unquantizedActivations, minMax, comparisonAlgorithm) -> Dict[str, Any]:
    '''
    Given unquant_activation and comparison algo, return the analysis result
    '''
    threshold = '0.0'
    if comparisonAlgorithm['algo_name'] == Verifiers.MIN_MAX:
        comparator = Comparator(Verifiers.QcAlgorithm.MIN_MAX_COMPARE)
        threshold = Verifiers.DEFAULT_THRESHOLDS.MIN_MAX
    elif comparisonAlgorithm['algo_name'] == Verifiers.STATS:
        comparator = Comparator(Verifiers.QcAlgorithm.STATS)
        threshold = Verifiers.DEFAULT_THRESHOLDS.STATS
    elif comparisonAlgorithm['algo_name'] == Verifiers.DATA_RANGE:
        comparator = Comparator(Verifiers.QcAlgorithm.DATA_RANGE_CHECK)
    else:
        comparator = Comparator(Verifiers.QcAlgorithm.MIN_MAX_COMPARE)
        threshold = Verifiers.DEFAULT_THRESHOLDS.MIN_MAX
    if 'threshold' in comparisonAlgorithm:
        threshold = comparisonAlgorithm['threshold']
    return comparator.compare(unquantizedActivations, minMax, threshold)


def compareActivations(quant_schemes, opsMap, comparison_algorithms) -> ResultSet:
    '''
    :param quant_schemes: List of quantization schemes
    :param opsMap: Dictionary of quantization scheme to all ops in the model 
    :param comparison_algorithms: List of all comparators for which sensitivity has to be calculated
    '''
    # Create an object of ResultSet to hold the results for each quant_scheme
    results = ResultSet()

    unquantizedOps = opsMap['unquantized']

    # for each quant_scheme, for each op in model, perform activation data analysis
    for quant_scheme in quant_schemes:
        # skip quantized models which fail to convert
        if quant_scheme not in opsMap:
            continue
        perOpResults = ResultList()
        for op_name in opsMap[quant_scheme]:
            if op_name in unquantizedOps:
                quantizedOp = opsMap[quant_scheme][op_name]
                unquantizedOp = unquantizedOps[op_name]
                unquantop_name = unquantizedOp.getActivationNodeName()
                if quantizedOp.getActivationNodeName() not in (
                        None, '') and quantizedOp.getActivationNodeName() == unquantop_name:
                    activationList = unquantizedOp.getActivations()
                    for activationPerInput in activationList:
                        # analyze the unquantized activations but do not compare them to anything
                        if quant_scheme == 'unquantized':

                            perOpResults.append(
                                ActivationResultNode(
                                    unquantop_name, unquantizedOp.getActivationNodeName(),
                                    activationPerInput[0],
                                    comparePerOpActivations(activationPerInput[1], None, None,
                                                            comparison_algorithms), None, None))
                        else:
                            perOpResults.append(
                                ActivationResultNode(
                                    op_name, quantizedOp.getActivationNodeName(), activationPerInput[0],
                                    comparePerOpActivations(activationPerInput[1],
                                                            quantizedOp.getActivationScale(),
                                                            quantizedOp.getActivationOffset(),
                                                            comparison_algorithms),
                                    quantizedOp.getActivationScale(),
                                    quantizedOp.getActivationOffset()))

        results.add(quant_scheme, perOpResults)
    return results


def analyzeInputData(inputData, comparison_algorithms) -> ResultSet:
    '''
    Perform analysis for input data
    '''
    results = ResultSet()
    for filename in inputData:
        results.add(
            filename,
            InputResultNode(filename, analyzeInput(inputData[filename], comparison_algorithms)))
    return results


def analyzeInput(inputTensor, comparison_algorithms):
    '''
    Perform analysis for input data
    '''
    results = {}
    if comparison_algorithms is not None:
        for comparisonAlgorithm in comparison_algorithms:
            results[comparisonAlgorithm['algo_name']] = doInputAnalysis(
                inputTensor, comparisonAlgorithm)
    else:
        results[Verifiers.STATS] = doInputAnalysis(
            inputTensor, (Verifiers.STATS, Verifiers.DEFAULT_THRESHOLDS.STATS))
    return results


def doInputAnalysis(inputTensor, comparisonAlgorithm) -> Dict[str, Any]:
    '''
    perorm STATS comparator for input data
    '''
    if comparisonAlgorithm['algo_name'] == Verifiers.STATS:
        comparator = Comparator(Verifiers.QcAlgorithm.STATS)
    return comparator.compare(inputTensor, None, threshold=Verifiers.DEFAULT_THRESHOLDS.STATS)


def setDataRangeAnalyzerThreshold(comparison_algorithms, bitWidth) -> None:
    '''
    :param comparison algorithms:
    :param bitWidth:
    Sets the DATA_RANGE comprator threshold
    '''
    for comparisonAlgorithm in comparison_algorithms:
        if comparisonAlgorithm['algo_name'] == Verifiers.DATA_RANGE:
            comparisonAlgorithm['threshold'] = Verifiers.getMaxValueBasedOnBitWidth(bitWidth)
