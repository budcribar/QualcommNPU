##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
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
import math
import numpy as np
from abc import ABC
import importlib
import inspect
import os
import sys
import glob
import logging
import json
from scipy.stats import entropy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
import qti.aisw.accuracy_evaluator.common.defaults as df


class ComparatorUtils:

    @staticmethod
    def loadCustomComparators():
        """This method loads the user supplied custom comparators."""
        qacc_file_logger.info('Searching for custom comparators....')
        registered_comparators = {}
        defaults = df.Defaults.getInstance()
        custom_path = defaults.get_value("common.custom_comparators_path")
        abs_path = os.path.abspath(custom_path)
        sys.path.append(abs_path)
        if not os.path.exists(custom_path):
            qacc_file_logger.warning(
                'Custom comparator path {} - does not exist. No custom comparators loaded'.format(
                    custom_path))
            return registered_comparators

        # add to sys path
        dirs = []
        for dir in os.walk(abs_path):
            if dir[0] not in dirs:
                sys.path.append(dir[0])
                dirs.append(dir[0])

        # search comparators recursively
        files = glob.glob(custom_path + '/**/' + '*.py', recursive=True)
        files = [file.rsplit('/', 1)[-1] for file in files]
        for file in files:
            if file.endswith(".py"):
                if file.startswith('__'):
                    continue
                file = os.path.splitext(file)[0]
                _plugin = importlib.import_module(file)
                classes = inspect.getmembers(_plugin, predicate=inspect.isclass)
                for cl in classes:
                    if cl[1].__module__ == file:
                        class_hier = inspect.getmro(cl[1])
                        for class_h in class_hier:
                            if class_h.__name__ != 'ABC' and class_h.__name__ != 'object' and \
                                    class_h.__name__ != 'Comparators':
                                _name = cl[1]().name()
                                if len(_name) == 0:
                                    _name = cl[0]
                                registered_comparators[_name] = cl[1]
                                qacc_file_logger.info('Loaded custom comparator : {}'.format(_name))
        return registered_comparators


class FileComparator:
    """
    FileComparator class compares two user supplied raw files using a specific comparator
    To use:
    >>> cmpr = RMEComparator()
    >>> FileComparator.compare('file1.raw', 'file2.raw', cmpr, np_dtype)
    """

    @classmethod
    def compare(cls, op1, op2, cmp, np_dtype, save_dir=None):
        """
        This method compares two supplied files(.raw or .bin)
        Args:
            op1 : path to .raw/.bin file
            op2 : path to .raw/.bin file
            cmp : object of comparator class, eg:L1Comparator() or RMEComparator()
            np_dtype : numpy datatype of inputs op1 and op2
            save_file: Only used for PixelByPixel comparator
        Return :
            ismatch : gives whether it is a match or not
            match_percent : percentage of match between op1 and op2
        """
        qacc_file_logger.debug('FileComparator::compare({},{}), dtype {}'.format(
            op1, op2, np_dtype))
        ismatch = True
        if cmp.name() == "box":
            # checks the existence of given .raw /.bin files
            for path in op1 + op2:
                if not os.path.exists(path):
                    qacc_file_logger.error('Input file : {} does not exist '.format(path))
                    raise ce.FileComparatorException('Input file : {} does not exist '.format(path))

            match_value, match_info = cmp.compare(op1, op2)
            return ismatch, match_value, match_info

        # checks the existence of given .raw /.bin files
        for path in [op1, op2]:
            if not os.path.exists(path):
                qacc_file_logger.error('Input file : {} does not exist '.format(path))
                raise ce.FileComparatorException('Input file : {} does not exist '.format(path))

        # convert the .raw /.bin files to numpy arrays
        op2_np = np.fromfile(op2, dtype=np_dtype)
        op1_np = np.fromfile(op1, dtype=np_dtype)

        # flattening the input arrays
        op1_np = op1_np.flatten()
        op2_np = op2_np.flatten()

        # Casting the boolean arrays to int for comparison
        if (np_dtype == bool):
            op1_np = op1_np.astype(np.int8)
            op2_np = op2_np.astype(np.int8)

        # handles the shape mismatch between op1 and op2
        if op2_np.shape != op1_np.shape:
            qacc_file_logger.debug(
                'Handling shape mismatch between two tensors {}: {} vs {}: {}'.format(
                    op1, op1_np.shape, op2, op2_np.shape))
            try:
                if op2_np.shape[0] > op1_np.shape[0]:
                    L = tuple(range(op1_np.shape[0], op2_np.shape[0]))
                    op2_np = np.delete(op2_np, L, axis=0)
                else:
                    L = tuple(range(op2_np.shape[0], op1_np.shape[0]))
                    op1_np = np.delete(op1_np, L, axis=0)
            except Exception as e:
                qacc_file_logger.error('Shape Mismatch {}:{} vs {}:{}'.format(
                    op1, op1_np.shape, op2, op2_np.shape))
                qacc_file_logger.exception(e)
                raise ce.FileComparatorException(
                    '******** ERROR: Shape Mismatch------- :{} vs {}'.format(
                        op2_np.shape, op1_np.shape))
        # compare two numpy arrays and get the percentage of match
        if cmp.name() == "pixelbypixel":
            match_value, match_info = cmp.compare(op2_np, op1_np, save_dir)
        else:
            match_value, match_info = cmp.compare(op2_np, op1_np)
        if cmp.name() == "kld":
            if match_value != 0:
                ismatch = False
        else:
            if match_value < 100:
                ismatch = False

        return ismatch, match_value, match_info


class Comparators(ABC):
    """Comparators is an abstract class that compares two values using specific
    comparator type."""

    def compare(self, op1, op2):
        """This method compares two numpy arrays with specific comparator type.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        pass

    def name(self):
        """Return a unique name of the comparator (without spaces)"""
        return ''

    def display_name(self):
        """Returns the name to be displayed on the comparison table."""
        return ''


class NormComparator(Comparators):
    """
    NormComparator class contains compare method using L1_norm/L2_norm as comparator
    To use:
    >>> obj = NormComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, order=1, tol=1e-5):
        self._ord = order
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "l" + str(self._ord) + "norm"

    def display_name(self):
        return "L" + str(self._ord) + "norm(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with L1_norm/L2_norm.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ""
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        # applies l1 and l2 norms on op1 and op2 to find the mismatch percentage
        # This gives the percentage of match that is proportional to the extent of difference in
        # values of op1 and op2
        diff = np.linalg.norm(op2 - op1, ord=self._ord)
        ref_norm = np.linalg.norm(op1, ord=self._ord)
        match_percent = 100 - (diff * 100 / ref_norm)

        # check if match_percent results to NaN
        if math.isnan(match_percent):
            qacc_file_logger.warning('{} match returned NaN, marked as 0% match'.format(self.name))
            match_info = "Match percent is NaN, marking it as 0%."
            match_percent = 0.0

        return match_percent, match_info


class RMEComparator(Comparators):
    """
    RmeComparator class contains compare method using RME as comparator
    To use:
    >>> obj = RmeComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "rme"

    def display_name(self):
        return "Rme(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with RME.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ""
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        # finds the root mean square error between op1 and op2
        rmse = np.linalg.norm(op2 - op1) / np.sqrt(len(op1))
        match_percent = 100 - rmse
        if match_percent < 0:
            qacc_file_logger.warning(
                'Match percent resulted negative for RMEComparator. Hence rounding match percent '
                'to 0.0')
            match_info = "Match percent is negative, marking it as 0%."
            match_percent = 0.0

        return match_percent, match_info


class TolComparator(Comparators):
    """
    TolComparator class contains compare method using RME as comparator
    To use:
    >>> obj = TolComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=0):
        self._tol = tol

    def name(self):
        return "abs"

    def display_name(self):
        return "Abs(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with tolerance.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ""
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        # finds the percentage match between op1 and op2 based on given tolerance
        # this comparator gives match percentage that is proportional to number of unmatched
        # elements between op1 and op2
        match = np.allclose(op1, op2, rtol=self._tol, equal_nan=True)
        tdiff = np.isclose(op1, op2, rtol=self._tol, equal_nan=True)
        common = op1[tdiff]
        match_percent = (common.shape[0] * 100) / op2.shape[0]
        return round(match_percent, 3), match_info


class AvgComparator(Comparators):
    """
    AvgComparator class contains compare method using Average error
    To use:
    >>> obj = AvgComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "avg"

    def display_name(self):
        return "Avg(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with RME.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ''
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        # finds the average error between op1 and op2
        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)
        avg_value = max(np.average(np.absolute(op1)), np.average(np.absolute(op2)))
        if (avg_value != 0):
            avg_error = np.average(abs(op1 - op2)) / avg_value
        else:
            avg_error = 0.0
        avg_error *= 100.0
        match_percent = 100 - avg_error

        # check if match_percent results to NaN
        if math.isnan(match_percent):
            qacc_file_logger.warning('Average match returned NaN, marked as 0% match')
            match_info = 'Average match is NaN, marked it as 0%.'
            match_percent = 0.0

        if match_percent < 0.0:
            match_info = 'Average match is less than 0, marked it as 0%'
            match_percent = 0.0

        return match_percent, match_info


class CosComparator(Comparators):
    """
    CosComparator class contains compare method using Cosine Similarity as comparator
    To use:
    >>> obj = CosComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "cos"

    def display_name(self):
        return "CosineSimilarity(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with Cosine Similarity.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ''
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        op1_l2norm = np.linalg.norm(op1)
        op2_l2norm = np.linalg.norm(op2)

        if op1_l2norm == 0 or op2_l2norm == 0:
            return 0, "One or both of the Tensors is zero!"

        # finds the cosine similarity between op1 and op2 range is [-1, 1]
        cos_sim = np.dot(op1, op2) / (op1_l2norm * op2_l2norm)
        match_percent = (1 + cos_sim) / 2 * 100  #Mapping cos similarity to percentage.

        return match_percent, match_info


class KLDComparator(Comparators):
    """
    CosComparator class contains compare method using KL Divergence as comparator
    To use:
    >>> obj = KLDComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "kld"

    def display_name(self):
        return "KLDivergence(0,inf)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with RME.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ''
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        #Normalizing the arrays
        op1 = (op1 + 1e-5) / np.linalg.norm(op1)
        op2 = (op2 + 1e-5) / np.linalg.norm(op1)

        # finds the KL divergence between op1 and op2
        kld = entropy(op1, op2, base=2)

        if math.isinf(kld):
            match_info = "KL Divergence is inf, one or more of the elements are zero."

        return kld, match_info


class StdComparator(Comparators):
    """
    CosComparator class contains compare method using the % difference in standard deviation
    To use:
    >>> obj = StdComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "std"

    def display_name(self):
        return "RelativeStdDev(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with RME.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            match_percent : percentage of match between op1 and op2
        """
        match_info = ''
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        # finds the std deviation for op1 and op2
        std1 = np.std(op1)
        std2 = np.std(op2)
        if std1 != 0:
            pct_change = (abs(std2 - std1) / std1) * 100
        else:
            pct_change = 100.0
            match_info = "StdDev is zero, marked the match as zero"

        match_percent = 100 - pct_change
        # check if match_percent results to NaN
        if math.isnan(match_percent):
            qacc_file_logger.warning('Relative StdDev match returned NaN, marked as 0% match')
            match_info = "Relative StdDev match is NaN, marked it as 0% match."
            match_percent = 0.0

        if match_percent < 0.0:
            match_info = 'Relative StdDev match is less than 0, marked it as 0% match'
            match_percent = 0.0

        return match_percent, match_info


class SnrComparator(Comparators):
    """
    SnrComparator class contains compare method using snr
    To use:
    >>> obj = SnrComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "snr"

    def display_name(self):
        return "SNR"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with snr.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            snr : SNR between op1 and op2
        """
        match_info = ''
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        diff = op1 - op2
        sqrd_diff = np.square(diff)
        mse = np.mean(sqrd_diff)

        sqrd_op1 = np.square(op1)
        mse_op1 = np.mean(sqrd_op1)

        if mse == 0:
            return 100, "MSE is zero"

        log_ratio = 10 * np.log10(mse_op1 / mse)

        return log_ratio, match_info


class MaxErrorComparator(Comparators):
    """
    MaxErrorComparator class contains compare method using Max error
    To use:
    >>> obj = MaxErrorComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "maxerror"

    def display_name(self):
        return "Max Error"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with Max Error.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            max_error : Max error between op1 and op2
        """
        match_info = ''
        # flattening the input arrays
        op1 = op1.flatten()
        op2 = op2.flatten()

        # finds the max error between op1 and op2
        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        abs_diff = abs(op1 - op2)
        max_error = np.max(abs_diff)

        return max_error, match_info


class TopKComparator(Comparators):
    """
    TopKComparator class contains compare method using Top K error
    To use:
    >>> obj = TopKComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5, kValue=5, ordered=False):
        self._decimals = int(abs(round(math.log(tol, 10))))
        self._kValue = kValue
        self._ordered = ordered

    def name(self):
        return "topk"

    def display_name(self):
        return "topk(%)"

    def compare(self, op1, op2):
        """This method compares two numpy arrays with TopK match.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            topk match : topk match between op1 and op2
        """
        match_info = ''

        op1 = np.nan_to_num(op1)
        op2 = np.nan_to_num(op2)

        op1 = np.around(op1, decimals=self._decimals)
        op2 = np.around(op2, decimals=self._decimals)

        op1_shape = list(op1.shape)
        op2_shape = list(op2.shape)

        op1 = op1.reshape(op1_shape)
        op2 = op2.reshape(op2_shape)

        top_k_indices_from_op1 = np.flip(op1.argsort()[-self._kValue:])
        top_k_indices_from_op2 = np.flip(op2.argsort()[-self._kValue:])

        if not self._ordered:
            top_k_indices_from_op1.sort()
            top_k_indices_from_op2.sort()

        number_of_diff_indices = 0
        for index_op1, index_op2 in zip(top_k_indices_from_op1, top_k_indices_from_op2):
            if self._ordered:
                if index_op1 != index_op2:
                    number_of_diff_indices += 1
            else:
                if index_op2 not in top_k_indices_from_op1:
                    number_of_diff_indices += 1

        percent_diff = number_of_diff_indices / min(self._kValue, len(op1))
        percent_diff = (1 - percent_diff) * 100
        return percent_diff, match_info


class PixelByPixelComparator(Comparators):
    """
    PixelByPixel class outputs pixel by pixel difference plot
    To use:
    >>> obj = PixelByPixelComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))

    def name(self):
        return "pixelbypixel"

    def display_name(self):
        return "pixelbypixel(%)"

    def compare(self, op1, op2, save_dir):
        """This method compares two numpy arrays with pixelbypixel diff.

        Args:
            op1 : numpy array 1
            op2 : numpy array 2
        Returns:
            percentage match : pixelbypixel match between op1 and op2
        """
        match_info = ''
        shape = list(op1.shape)
        op1_data = op1.reshape(shape)
        op2_data = op2.reshape(shape)
        data_diff = op1_data - op2_data
        x = np.arange(0, np.prod(shape))

        plt.plot(x, op1_data, 'r', label='output')
        plt.plot(x, op2_data, 'g', label='Reference')
        plt.title('Pixel By Pixel differences between output and reference')
        plt.plot(x, data_diff, 'k', label='difference')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "plot.png"))
        plt.clf()

        avg_diff = np.sum(np.absolute(data_diff)) / np.sum(np.absolute(op1_data))
        avg_match = (1 - avg_diff) * 100
        return avg_match, match_info


class BoxComparator(Comparators):
    """
    BoxComparator class
    To use:
    >>> obj = BoxComparator()
    >>> obj.compare(op1,op2)
    """

    def __init__(self, box_input_json, tol=1e-5):
        self._decimals = int(abs(round(math.log(tol, 10))))
        self._box_input_json = box_input_json

    def name(self):
        return "box"

    def display_name(self):
        return "box"

    def _get_box_inputs(self, outputs):
        """Returns the box inputs in order based on box_input_json."""
        with open(self._box_input_json, "r") as box_json:
            tag_to_fnames = json.load(box_json)
        fnames_to_tag = {v: k for k, v in tag_to_fnames.items()}
        files = {}
        for op_fname in outputs:
            op_basename = os.path.basename(op_fname)
            files[fnames_to_tag[op_basename]] = op_fname

        return files["box"], files["class"], files["score"]

    def compare(self, op1, op2):
        """
        Args:
            op1 : list of numpy array 1
            op2 : list of numpy array 2
        Returns:
            percentage match : Box match between op1 and op2

        This algorithm only works for cases that have coordinate systems start
        with from top left as the first coordinate and bottom right as the
        second coordinate. Also assumes all inputs are float32 results.
        """
        match_info = ''

        g_box, g_cls, g_score = self._get_box_inputs(op1)
        raw_box, raw_cls, raw_score = self._get_box_inputs(op2)

        golden_box = np.fromfile(g_box, dtype=np.float32)
        golden_cls = np.fromfile(g_cls, dtype=np.float32)
        golden_score = np.fromfile(g_score, dtype=np.float32)
        dets_box = np.fromfile(raw_box, dtype=np.float32)
        dets_cls = np.fromfile(raw_cls, dtype=np.float32)
        dets_score = np.fromfile(raw_score, dtype=np.float32)

        iou_list = []

        g_box = golden_box.reshape(-1, 4)
        g_cls = golden_cls.reshape(-1)
        g_score = golden_score.reshape(-1)

        d_box = dets_box.reshape(-1, 4)
        d_cls = dets_cls.reshape(-1)
        d_score = dets_score.reshape(-1)

        #sort g_cls, g_box and d_scores by g_score
        s_g_cls = np.array([item for _, item in sorted(zip(g_score, g_cls), reverse=True)])
        s_g_box = np.array(
            [item for _, item in sorted(zip(g_score, g_box), key=lambda x: x[0], reverse=True)])
        s_score = np.array([item for _, item in sorted(zip(g_score, d_score), reverse=True)])
        s_g_score = np.array(sorted(g_score, reverse=True))
        s_cls = np.array([item for _, item in sorted(zip(g_score, d_cls), reverse=True)])

        #go for item in gt class
        #this should be O(n*n) because
        #of multiple detections
        for i_gt in range(len(s_g_cls)):

            found = False
            max_iou = -1

            #check detection one by one
            #until found the most collapsed
            #detections
            for i_dt in range(len(d_cls)):
                if d_cls[i_dt] == s_g_cls[i_gt]:
                    found = True
                    #compute iou here
                    if self.is_intersect(s_g_box[i_gt], d_box[i_dt]) is False:  #not intersected
                        iou = 0.0
                    else:  #intersected
                        int_area = self.intersect(s_g_box[i_gt], d_box[i_dt])
                        union_area = self.union(s_g_box[i_gt], d_box[i_dt], int_area)
                        iou = int_area / union_area

                    #update only higher iou is found
                    if iou > max_iou:
                        max_iou = iou

            if found:
                iou_list.append(max_iou)
            else:
                iou_list.append(0.0)

        #print out iou scores for each detections
        l_print = min(len(s_g_cls), len(s_score))
        for i in range(l_print):
            qacc_file_logger.info("Class " + str(s_g_cls[i]) + " has IoU: " + \
                             str(iou_list[i]) + " and Score: " + str(s_score[i]) + "\n")

        avg_score = sum(s_score) / len(s_score)

        return avg_score, match_info

    def is_intersect(self, gt_box, dt_box):
        """Helper function to detection intersection."""
        if gt_box[0] > dt_box[2]:
            return False
        if gt_box[1] > dt_box[3]:
            return False
        if gt_box[2] < dt_box[0]:
            return False
        if gt_box[3] < dt_box[1]:
            return False
        return True

    def intersect(self, gt_box, dt_box):
        """Helper function to calculate intersected area."""
        bb = [
            max(dt_box[0], gt_box[0]),
            max(dt_box[1], gt_box[1]),
            min(dt_box[2], gt_box[2]),
            min(dt_box[3], gt_box[3])
        ]
        int_area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
        return abs(int_area)

    def union(self, gt_box, dt_box, int_area):
        """Helper function to calculate union area."""
        gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
        dt_area = (dt_box[2] - dt_box[0] + 1) * (dt_box[3] - dt_box[1] + 1)
        union_area = (gt_area + dt_area) - int_area
        return abs(union_area)
