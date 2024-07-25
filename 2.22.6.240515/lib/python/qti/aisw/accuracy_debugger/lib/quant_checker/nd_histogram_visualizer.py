# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import  get_progress_message
from qti.aisw.accuracy_debugger.lib.visualizer.nd_visualizers import Visualizers
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_utils import verify_path

import os
import numpy as np


def _init_directories(quant_schemes, plot_analysis_dir, requested_plots):
    for quant_scheme in quant_schemes:
        if quant_scheme == "unquantized":
            continue
        plot_analysis_per_quant_variation_dir = verify_path(plot_analysis_dir, quant_scheme)
        for plot in requested_plots:
            verify_path(plot_analysis_per_quant_variation_dir, plot + '_plots')

def visualizeWeightTensors(quantizationVariations, opsMap, plot_analysis_dir, requested_plots, logger):
    unquantizedOps = opsMap['unquantized']
    for quantizationVariation in quantizationVariations:
        if quantizationVariation == 'unquantized':
            continue
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap and quantizationVariation != 'unquantized':
            continue
        plot_analysis_per_quant_variation_dir = os.path.join(plot_analysis_dir, quantizationVariation)
        hist_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'histogram_plots')
        diff_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'diff_plots')
        cdf_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'cdf_plots')
        min_max_distribution_path = os.path.join(plot_analysis_per_quant_variation_dir, 'min_max_distribution_plots')
        for opName in opsMap[quantizationVariation]:
            quantizedOp = opsMap[quantizationVariation][opName]
            if quantizedOp.getWeightName() not in (None, '') and len(quantizedOp.getDequantizedWeights())>0:
                logger.info(
                    get_progress_message(
                        ('Processing Weight Plots for weight - ' + quantizedOp.getWeightName() + "_" + opName +
                        ' from Quant option - ' + quantizationVariation)))
                unquantizedWeights = unquantizedOps[opName].getWeights()
                quantizedWeights = quantizedOp.getDequantizedWeights()
                weightsDims = quantizedOp.getWeightsDims()
                unquantizedWeights = unquantizedWeights.reshape(weightsDims)
                quantizedWeights = quantizedWeights.reshape(weightsDims)
                sanitizedOpName = quantizedOp.getWeightName().replace('/', '_').strip() + "_" + opName
                if "histogram" in requested_plots:
                    hist_file_path = os.path.join(hist_dir, sanitizedOpName)
                    Visualizers.histogram_visualizer(unquantizedWeights[:].flatten(),
                                                quantizedWeights[:].flatten(), hist_file_path + ".png")
                if "diff" in requested_plots:
                    diff_file_path = os.path.join(diff_dir, sanitizedOpName)
                    Visualizers.diff_visualizer(unquantizedWeights[:].flatten(),
                                                quantizedWeights[:].flatten(), diff_file_path + ".png")
                if "cdf" in requested_plots:
                    cdf_file_path = os.path.join(cdf_dir, sanitizedOpName)
                    Visualizers.cdf_visualizer(unquantizedWeights[:].flatten(),
                                                quantizedWeights[:].flatten(), cdf_file_path + ".png")
                if "min_max_distribution" in requested_plots:
                    min_max_distribution_file_path = os.path.join(min_max_distribution_path, sanitizedOpName)
                    calib_min = np.amin(quantizedWeights)
                    calib_max = np.amax(quantizedWeights)
                    target_min = np.amin(unquantizedWeights)
                    target_max = np.amax(unquantizedWeights)
                    Visualizers.distribution_visualizer(
                            unquantizedWeights.flatten(), min_max_distribution_file_path + ".png", target_min, target_max,
                            calib_min, calib_max)

def visualizePerChannelWeightTensors(quantizationVariations, opsMap, plot_analysis_dir, requested_plots, logger):
    unquantizedOps = opsMap['unquantized']
    for quantizationVariation in quantizationVariations:
        if quantizationVariation == 'unquantized':
            continue
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap and quantizationVariation != 'unquantized':
            continue
        plot_analysis_per_quant_variation_dir = os.path.join(plot_analysis_dir, quantizationVariation)
        hist_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'histogram_plots')
        diff_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'diff_plots')
        cdf_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'cdf_plots')
        min_max_distribution_path = os.path.join(plot_analysis_per_quant_variation_dir, 'min_max_distribution_plots')
        for opName in opsMap[quantizationVariation]:
            quantizedOp = opsMap[quantizationVariation][opName]
            if quantizedOp.getWeightName() not in (None, '') and len(quantizedOp.getDequantizedWeights())>0:
                logger.info(
                    get_progress_message(
                        ('Processing Weight Plots for weight - ' + quantizedOp.getWeightName() + "_" + opName +
                        ' from Quant option - ' + quantizationVariation)))
                unquantizedWeights = unquantizedOps[opName].getWeights()
                quantizedWeights = quantizedOp.getDequantizedWeights()
                weightsDims = quantizedOp.getWeightsDims()
                unquantizedWeights = unquantizedWeights.reshape(weightsDims)
                quantizedWeights = quantizedWeights.reshape(weightsDims)
                sanitizedOpName = quantizedOp.getWeightName().replace('/', '_').strip() + "_" + opName
                if len(weightsDims) == 4:
                    hist_op_path = verify_path(hist_dir, sanitizedOpName)
                    diff_op_path = verify_path(diff_dir, sanitizedOpName)
                    cdf_op_path = verify_path(cdf_dir, sanitizedOpName)
                    min_max_distribution_op_path = verify_path(min_max_distribution_path, sanitizedOpName)
                    for depth in range(unquantizedWeights.shape[2]):
                        unquantizedWeightsPerChannel = unquantizedWeights[:, :, depth, :].flatten()
                        quantizedWeightsPerChannel = quantizedWeights[:, :, depth, :].flatten()
                        if "histogram" in requested_plots:
                            hist_file_path = os.path.join(hist_op_path, "channel_" + str(depth))
                            Visualizers.histogram_visualizer(unquantizedWeightsPerChannel[:].flatten(),
                                                        quantizedWeightsPerChannel[:].flatten(), hist_file_path + ".png")
                        if "diff" in requested_plots:
                            diff_file_path = os.path.join(diff_op_path, "channel_" + str(depth))
                            Visualizers.diff_visualizer(unquantizedWeightsPerChannel[:].flatten(),
                                                        quantizedWeightsPerChannel[:].flatten(), diff_file_path + ".png")
                        if "cdf" in requested_plots:
                            cdf_file_path = os.path.join(cdf_op_path, "channel_" + str(depth))
                            Visualizers.cdf_visualizer(unquantizedWeightsPerChannel[:].flatten(),
                                                    quantizedWeightsPerChannel[:].flatten(), cdf_file_path + ".png")
                        if "min_max_distribution" in requested_plots:
                            min_max_distribution_file_path = os.path.join(min_max_distribution_op_path, "channel_" + str(depth))
                            calib_min = np.amin(quantizedWeights)
                            calib_max = np.amax(quantizedWeights)
                            target_min = np.amin(unquantizedWeights)
                            target_max = np.amax(unquantizedWeights)
                            Visualizers.distribution_visualizer(
                                    unquantizedWeights.flatten(), min_max_distribution_file_path + ".png", target_min, target_max,
                                    calib_min, calib_max)
                else:
                    if "histogram" in requested_plots:
                        hist_file_path = os.path.join(hist_dir, sanitizedOpName)
                        Visualizers.histogram_visualizer(unquantizedWeights[:].flatten(),
                                                    quantizedWeights[:].flatten(), hist_file_path + ".png")
                    if "diff" in requested_plots:
                        diff_file_path = os.path.join(diff_dir, sanitizedOpName)
                        Visualizers.diff_visualizer(unquantizedWeights[:].flatten(),
                                                    quantizedWeights[:].flatten(), diff_file_path + ".png")
                    if "cdf" in requested_plots:
                        cdf_file_path = os.path.join(cdf_dir, sanitizedOpName)
                        Visualizers.cdf_visualizer(unquantizedWeights[:].flatten(),
                                                    quantizedWeights[:].flatten(), cdf_file_path + ".png")
                    if "min_max_distribution" in requested_plots:
                        min_max_distribution_file_path = os.path.join(min_max_distribution_path, sanitizedOpName)
                        calib_min = np.amin(quantizedWeights)
                        calib_max = np.amax(quantizedWeights)
                        target_min = np.amin(unquantizedWeights)
                        target_max = np.amax(unquantizedWeights)
                        Visualizers.distribution_visualizer(
                                unquantizedWeights.flatten(), min_max_distribution_file_path + ".png", target_min, target_max,
                                calib_min, calib_max)


def visualizeBiasTensors(quantizationVariations, opsMap, plot_analysis_dir, requested_plots, logger):
    unquantizedOps = opsMap['unquantized']
    for quantizationVariation in quantizationVariations:
        if quantizationVariation == 'unquantized':
            continue
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap and quantizationVariation != 'unquantized':
            continue
        plot_analysis_per_quant_variation_dir = os.path.join(plot_analysis_dir, quantizationVariation)
        hist_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'histogram_plots')
        diff_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'diff_plots')
        cdf_dir = os.path.join(plot_analysis_per_quant_variation_dir, 'cdf_plots')
        min_max_distribution_path = os.path.join(plot_analysis_per_quant_variation_dir, 'min_max_distribution_plots')
        for opName in opsMap[quantizationVariation]:
            quantizedOp = opsMap[quantizationVariation][opName]
            if quantizedOp.getBiasName() not in (None, ''):
                logger.info(
                    get_progress_message(
                        ('Processing Bias Plots for Bias - ' + quantizedOp.getBiasName() + "_" + opName +
                        ' from Quant option - ' + quantizationVariation)))
                unquantizedBiases = unquantizedOps[opName].getBiases()
                quantizedBiases = quantizedOp.getDequantizedBiases()
                sanitizedOpName = quantizedOp.getBiasName().replace('/', '_').strip() + "_" + opName
                if "histogram" in requested_plots:
                    hist_file_path = os.path.join(hist_dir, sanitizedOpName)
                    Visualizers.histogram_visualizer(unquantizedBiases[:].flatten(),
                                                quantizedBiases[:].flatten(), hist_file_path + ".png")
                if "diff" in requested_plots:
                    diff_file_path = os.path.join(diff_dir, sanitizedOpName)
                    Visualizers.diff_visualizer(unquantizedBiases[:].flatten(),
                                                quantizedBiases[:].flatten(), diff_file_path + ".png")
                if "cdf" in requested_plots:
                    cdf_file_path = os.path.join(cdf_dir, sanitizedOpName)
                    Visualizers.cdf_visualizer(unquantizedBiases[:].flatten(),
                                                quantizedBiases[:].flatten(), cdf_file_path + ".png")
                if "min_max_distribution" in requested_plots:
                    min_max_distribution_file_path = os.path.join(min_max_distribution_path, sanitizedOpName)
                    calib_min = np.amin(quantizedBiases)
                    calib_max = np.amax(quantizedBiases)
                    target_min = np.amin(unquantizedBiases)
                    target_max = np.amax(unquantizedBiases)
                    try:
                        #Sometimes Bias tensors could be singular
                        Visualizers.distribution_visualizer(
                                unquantizedBiases.flatten(), min_max_distribution_file_path + ".png", calib_min,
                                calib_max, target_min, target_max)
                    except Exception:
                        continue

def visualize(quant_schemes, opMap, dir, requested_plots: set, per_channel=False, logger=None):
    base_dir = os.path.dirname(os.path.dirname(dir))
    if len(requested_plots) > 0:
        plot_analysis_dir = os.path.join(base_dir, 'results', 'plot_analysis')
        _init_directories(quant_schemes, plot_analysis_dir, requested_plots)
        if not os.path.exists(plot_analysis_dir):
            os.makedirs(plot_analysis_dir)
        if per_channel:
            visualizePerChannelWeightTensors(quant_schemes, opMap, plot_analysis_dir, requested_plots, logger)
        else:
            visualizeWeightTensors(quant_schemes, opMap, plot_analysis_dir, requested_plots, logger)
        visualizeBiasTensors(quant_schemes, opMap, plot_analysis_dir, requested_plots, logger)
