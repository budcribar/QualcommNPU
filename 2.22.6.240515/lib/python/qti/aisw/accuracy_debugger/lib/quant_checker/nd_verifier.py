# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum
from typing import List
import numpy as np


class QcAlgorithm(Enum):
    MAX_ABS_DIFFERENCE = 1
    MIN_MAX_COMPARE = 2
    SQNR = 3
    STATS = 4
    DATA_RANGE_CHECK = 5
    DATA_DISTRIBUTION_CHECK = 6


SQNR = 'sqnr'
MAX_DIFF = 'maxdiff'
MIN_MAX = 'minmax'
STATS = 'stats'
DATA_DISTRIBUTION = 'data_distribution_analyzer'
DATA_RANGE = 'data_range_analyzer'

VERIFIER_NAMES = {}
VERIFIER_NAMES[QcAlgorithm.MAX_ABS_DIFFERENCE] = MAX_DIFF
VERIFIER_NAMES[QcAlgorithm.MIN_MAX_COMPARE] = MIN_MAX
VERIFIER_NAMES[QcAlgorithm.SQNR] = SQNR
VERIFIER_NAMES[QcAlgorithm.STATS] = STATS
VERIFIER_NAMES[QcAlgorithm.DATA_RANGE_CHECK] = DATA_RANGE
VERIFIER_NAMES[QcAlgorithm.DATA_DISTRIBUTION_CHECK] = DATA_DISTRIBUTION


class DEFAULT_THRESHOLDS():
    SQNR = '26.0'
    MAX_DIFF = '10.0'
    MIN_MAX = '10.0'
    STATS = '2.0'
    DATA_DISTRIBUTION = '0.6'
    DATA_RANGE_8 = str(np.iinfo(np.uint8).max + 1)
    DATA_RANGE_16 = str(np.iinfo(np.uint16).max + 1)
    DATA_RANGE_32 = str(np.iinfo(np.uint32).max + 1)


def getMaxValueBasedOnBitWidth(bits):
    maxValue = DEFAULT_THRESHOLDS.DATA_RANGE_8
    if bits == str(np.iinfo(np.uint32).bits):
        maxValue = DEFAULT_THRESHOLDS.DATA_RANGE_32
    elif bits == str(np.iinfo(np.uint16).bits):
        maxValue = DEFAULT_THRESHOLDS.DATA_RANGE_16
    return maxValue


STATS_DESCRIPTIVE_NAME = 'Symmetricity'
DATA_RANGE_DESCRIPTIVE_NAME = 'Quantization Bit-Width Range'
SQNR_DESCRIPTIVE_NAME = 'Signal to Quantization Noise Ratio'
MIN_MAX_DESCRIPTIVE_NAME = 'Minimum/Maximum Difference Value'
MAX_DIFF_DESCRIPTIVE_NAME = 'Maximum Absolute Difference'
DATA_DISTRIBUTION_DESCRIPTIVE_NAME = 'Clustering of Unquantized Data'

STATS_DESCRIPTION = 'Indicates whether the data is symmetric or not.'
DATA_RANGE_DESCRIPTION = 'Indicates whether the data can be reasonably quantized within the given bit-width range.'
SQNR_DESCRIPTION = 'Indicates whether the signal to quantized noise ratio of the data is too low or not.'
MIN_MAX_DESCRIPTION = 'Indicates how different the min/max values are between the unquantized and dequantized data.'
MAX_DIFF_DESCRIPTION = 'Indicates whether the maximum discrepency between the unquantized and dequantized data is too large or not.'
DATA_DISTRIBUTION_DESCRIPTION = 'Indicates whether a large number of unique unquantized values are quantized to the same value or not. Currently we do not fail an entire node depending on this result since the influence of the checker is not conclusive.'


class QcFailedVerifier(Enum):
    STATS_FAIL = 1
    DATA_RANGE_FAIL = 2
    SQNR_FAIL = 3
    MIN_MAX_FAIL = 4
    MAX_DIFF_FAIL = 5
    DATA_DISTRIBUTION_FAIL = 6
    STATS_DATA_RANGE_FAIL = 7
    STATS_SQNR_FAIL = 8
    STATS_MIN_MAX_FAIL = 9
    STATS_MAX_DIFF_FAIL = 10
    STATS_DATA_DISTRIBUTION_FAIL = 11
    STATS_DATA_RANGE_SQNR_FAIL = 12
    STATS_DATA_RANGE_MIN_MAX_FAIL = 13
    STATS_DATA_RANGE_MAX_DIFF_FAIL = 14
    STATS_DATA_RANGE_DATA_DISTRIBUTION_FAIL = 15
    STATS_DATA_RANGE_SQNR_MIN_MAX_FAIL = 16
    STATS_DATA_RANGE_SQNR_MAX_DIFF_FAIL = 17
    STATS_DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL = 18
    STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL = 19
    STATS_DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL = 20
    STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 21
    STATS_DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 22
    STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL = 23
    STATS_DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL = 24
    STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 25
    STATS_DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 26
    STATS_SQNR_MIN_MAX_FAIL = 27
    STATS_SQNR_MAX_DIFF_FAIL = 28
    STATS_SQNR_DATA_DISTRIBUTION_FAIL = 29
    STATS_SQNR_MIN_MAX_MAX_DIFF_FAIL = 30
    STATS_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL = 31
    STATS_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 32
    STATS_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 33
    STATS_MIN_MAX_MAX_DIFF_FAIL = 34
    STATS_MIN_MAX_DATA_DISTRIBUTION_FAIL = 35
    STATS_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 36
    STATS_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 37
    DATA_RANGE_SQNR_FAIL = 38
    DATA_RANGE_MIN_MAX_FAIL = 39
    DATA_RANGE_MAX_DIFF_FAIL = 40
    DATA_RANGE_DATA_DISTRIBUTION_FAIL = 41
    DATA_RANGE_SQNR_MIN_MAX_FAIL = 42
    DATA_RANGE_SQNR_MAX_DIFF_FAIL = 43
    DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL = 44
    DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL = 45
    DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL = 46
    DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 47
    DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 48
    DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL = 49
    DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL = 50
    DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 51
    DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 52
    SQNR_MIN_MAX_FAIL = 53
    SQNR_MAX_DIFF_FAIL = 54
    SQNR_DATA_DISTRIBUTION_FAIL = 55
    SQNR_MIN_MAX_MAX_DIFF_FAIL = 56
    SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL = 57
    SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 58
    SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 59
    MIN_MAX_MAX_DIFF_FAIL = 60
    MIN_MAX_DATA_DISTRIBUTION_FAIL = 61
    MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL = 62
    MAX_DIFF_DATA_DISTRIBUTION_FAIL = 63


STATS_FAIL_DESCRIPTION = 'The data analyzed is asymmetric. Please consider using symmetric values.'
DATA_RANGE_FAIL_DESCRIPTION = 'The quantization bit-width used does not accurately capture the data. Please consider using a larger bit-width, e.g., 16-bit quantization.'
SQNR_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low. Please consider retraining the model using QAT.'
MIN_MAX_FAIL_DESCRIPTION = 'The minimum and/or the maximum values differ widely from the unquantized version. Please consider using a larger bit-width for quantization, e.g., 16-bit quantization.'
MAX_DIFF_FAIL_DESCRIPTION = 'The maximum absolute difference between the quantized and unquantized data is too large. Please consider retraining the model using QAT.'
DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'Many values have been quantized to the same value. Please consider using a larger bit-width, e.g., 16-bit quantization.'
STATS_DATA_RANGE_FAIL_DESCRIPTION = 'The data is asymmetric and is not quantized accurately. Please consider using a larger bit-width, e.g., 16-bit quantization with symmetric values.'
STATS_SQNR_FAIL_DESCRIPTION = 'The data is asymmetric and has a low signal to quantization noise ratio. Please consider retraining the model using QAT with symmetric values.'
STATS_MIN_MAX_FAIL_DESCRTIPTION = 'The data is asymmetric and has minimum/maximum values that differ from the unquantized values. Please consider using symmetric data and a larger bit-width for quantization.'
STATS_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data and/or retraining the model using QAT.'
STATS_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric and has many values close together. Please consider using symmetric data and a larger bit-width for quantization.'
STATS_DATA_RANGE_SQNR_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used and suffers from a low signal to quantization noise ratio. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_DATA_RANGE_MIN_MAX_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used and the minimum and/or the maximum values differ widely from the unquantized version. Please consider using symmetric data and a larger bit-width for quantization.'
STATS_DATA_RANGE_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_DATA_RANGE_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used and has many values close together. Please consider using a larger bit-width, e.g., 16-bit quantization with symmetric values.'
STATS_DATA_RANGE_SQNR_MIN_MAX_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio and the minimum and/or the maximum values differ widely from the unquantized version. Please consider using a larger bit-width for quantization, symmetric data and/or retraining the model using QAT.'
STATS_DATA_RANGE_SQNR_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the data using QAT.'
STATS_DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using symmetric data and a larger bit-width for quantization.'
STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, the minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger bit-width for quantization, symmetric data and/or retraining the model using QAT.'
STATS_DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, does not fit within the quantization bit-width used, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_SQNR_MIN_MAX_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio and the minimum and/or the maximum values differ widely from the unquantized version. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_SQNR_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data and/or retraining the model using QAT.'
STATS_SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and/or retraining the model using QAT.'
STATS_SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio, The minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger bit-width for quantization, symmetric data and/or retraining the model using QAT.'
STATS_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, suffers from a low signal to quantization noise ratio, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The data is asymmetric, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, the minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using symmetric data, a larger bit-width for quantization and possibly retraining the model using QAT.'
STATS_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data is asymmetric, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using symmetric data and/or a larger bit-width for quantization'
DATA_RANGE_SQNR_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used and suffers from a low signal to quantization noise ratio. Please consider using a larger bit-width and/or retraining the model using QAT.'
DATA_RANGE_MIN_MAX_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used and has minimum/maximum values that differ from the unquantized values. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_MAX_DIFF_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used and many values have been quantized to the same value. Please consider using a larger quantization bit-width.'
DATA_RANGE_SQNR_MIN_MAX_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio and has minimum/maximum values that differ from the unquantized values. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_SQNR_MAX_DIFF_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the maximum absolute difference between the quantized and unquantized data is too large. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, suffers from a low signal to quantization noise ratio, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, the minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The data does not fit within the quantization bit-width used, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger quantization bit-width and/or retraining the model using QAT.'
SQNR_MIN_MAX_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low and has minimum/maximum values that differ from the unquantized values. Please consider retraining the model using QAT.'
SQNR_MAX_DIFF_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low, The maximum absolute difference between the quantized and unquantized data is too large. Please consider retraining the model using QAT.'
SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low and many values have been quantized to the same value. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'
SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low, the minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'
SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low, the minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'
SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low, the maximum absolute difference between the quantized and unquantized data is too large, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger bit-width for quantization and/or retraining the model using QAT'
SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The signal to quantization noise ratio is low, the maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'
MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION = 'The minimum and/or the maximum values differ widely from the unquantized version and the maximum absolute difference between the quantized and unquantized data is too large. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'
MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The minimum and/or the maximum values differ widely from the unquantized version and many values have been quantized to the same value. Please consider using a larger bit-width for quantization, e.g., 16-bit quantization.'
MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The minimum and/or the maximum values differ widely from the unquantized version, the maximum absolute difference between the quantized and unquantized data is too large and DATA_DISTRIBUTION. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'
MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION = 'The maximum absolute difference between the quantized and unquantized data is too large and many values have been quantized to the same value. Please consider using a larger bit-width for quantization and/or retraining the model using QAT.'

FAILED_VERIFIERS_DESC = dict()
FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_FAIL] = STATS_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_FAIL] = DATA_RANGE_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_FAIL] = SQNR_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.MIN_MAX_FAIL] = MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.MAX_DIFF_FAIL] = MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_DISTRIBUTION_FAIL] = DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_FAIL] = STATS_DATA_RANGE_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_FAIL] = STATS_SQNR_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MIN_MAX_FAIL] = STATS_MIN_MAX_FAIL_DESCRTIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MAX_DIFF_FAIL] = STATS_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_DATA_DISTRIBUTION_FAIL] = STATS_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_DATA_RANGE_SQNR_FAIL] = STATS_DATA_RANGE_SQNR_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_DATA_RANGE_MIN_MAX_FAIL] = STATS_DATA_RANGE_MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_DATA_RANGE_MAX_DIFF_FAIL] = STATS_DATA_RANGE_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_MIN_MAX_FAIL] = STATS_DATA_RANGE_SQNR_MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_MAX_DIFF_FAIL] = STATS_DATA_RANGE_SQNR_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL] = STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL] = STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_SQNR_MIN_MAX_FAIL] = STATS_SQNR_MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_SQNR_MAX_DIFF_FAIL] = STATS_SQNR_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_SQNR_DATA_DISTRIBUTION_FAIL] = STATS_SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_SQNR_MIN_MAX_MAX_DIFF_FAIL] = STATS_SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL] = STATS_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.STATS_MIN_MAX_MAX_DIFF_FAIL] = STATS_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_MIN_MAX_DATA_DISTRIBUTION_FAIL] = STATS_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    STATS_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = STATS_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_FAIL] = DATA_RANGE_SQNR_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.DATA_RANGE_MIN_MAX_FAIL] = DATA_RANGE_MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.DATA_RANGE_MAX_DIFF_FAIL] = DATA_RANGE_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.DATA_RANGE_SQNR_MIN_MAX_FAIL] = DATA_RANGE_SQNR_MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.DATA_RANGE_SQNR_MAX_DIFF_FAIL] = DATA_RANGE_SQNR_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL] = DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL] = DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MIN_MAX_FAIL] = SQNR_MIN_MAX_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MAX_DIFF_FAIL] = SQNR_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.SQNR_DATA_DISTRIBUTION_FAIL] = SQNR_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.SQNR_MIN_MAX_MAX_DIFF_FAIL] = SQNR_MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL] = SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[QcFailedVerifier.MIN_MAX_MAX_DIFF_FAIL] = MIN_MAX_MAX_DIFF_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.MIN_MAX_DATA_DISTRIBUTION_FAIL] = MIN_MAX_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.
    MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL] = MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION
FAILED_VERIFIERS_DESC[
    QcFailedVerifier.MAX_DIFF_DATA_DISTRIBUTION_FAIL] = MAX_DIFF_DATA_DISTRIBUTION_FAIL_DESCRIPTION


def getFailureAnalysisDescription(failures: List) -> str:
    failureSet = set(failures)
    if failureSet == {STATS}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_FAIL]
    elif failureSet == {DATA_RANGE}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_FAIL]
    elif failureSet == {SQNR}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_FAIL]
    elif failureSet == {MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.MIN_MAX_FAIL]
    elif failureSet == {MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.MAX_DIFF_FAIL]
    elif failureSet == {DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_FAIL]
    elif failureSet == {STATS, SQNR}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_FAIL]
    elif failureSet == {STATS, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MIN_MAX_FAIL]
    elif failureSet == {STATS, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MAX_DIFF_FAIL]
    elif failureSet == {STATS, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_SQNR_FAIL]
    elif failureSet == {STATS, DATA_RANGE, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_MIN_MAX_FAIL]
    elif failureSet == {STATS, DATA_RANGE, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_MAX_DIFF_FAIL]
    elif failureSet == {STATS, DATA_RANGE, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_SQNR_MIN_MAX_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_SQNR_MAX_DIFF_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.STATS_DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, SQNR, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.STATS_DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {STATS, DATA_RANGE, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.STATS_DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.STATS_DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, DATA_RANGE, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.STATS_DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, SQNR, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_MIN_MAX_FAIL]
    elif failureSet == {STATS, SQNR, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_MAX_DIFF_FAIL]
    elif failureSet == {STATS, SQNR, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, SQNR, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {STATS, SQNR, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, SQNR, MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.STATS_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, SQNR, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {STATS, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {STATS, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.STATS_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, SQNR}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_FAIL]
    elif failureSet == {DATA_RANGE, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_FAIL]
    elif failureSet == {DATA_RANGE, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_MAX_DIFF_FAIL]
    elif failureSet == {DATA_RANGE, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_MIN_MAX_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_MAX_DIFF_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.DATA_RANGE_SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.DATA_RANGE_SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, SQNR, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.DATA_RANGE_SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {DATA_RANGE, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[
            QcFailedVerifier.DATA_RANGE_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {DATA_RANGE, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.DATA_RANGE_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {SQNR, MIN_MAX}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MIN_MAX_FAIL]
    elif failureSet == {SQNR, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MAX_DIFF_FAIL]
    elif failureSet == {SQNR, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {SQNR, MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {SQNR, MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {SQNR, MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {SQNR, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.SQNR_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {MIN_MAX, MAX_DIFF}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.MIN_MAX_MAX_DIFF_FAIL]
    elif failureSet == {MIN_MAX, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.MIN_MAX_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {MIN_MAX, MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.MIN_MAX_MAX_DIFF_DATA_DISTRIBUTION_FAIL]
    elif failureSet == {MAX_DIFF, DATA_DISTRIBUTION}:
        return FAILED_VERIFIERS_DESC[QcFailedVerifier.MAX_DIFF_DATA_DISTRIBUTION_FAIL]


STATS_FAIL_SOLN_DESC = 'Data is asymmetric, attempting to normalize data.'
DATA_RANGE_FAIL_SOLN_DESC = 'Available quantization bit-width is not wide enough, attempting to increase bit-width used for quantization.'
SQNR_FAIL_SOLN_DESC = 'SQNR is low, attempting to increase bit-width and adjust quantization encodings.'
MIN_MAX_FAIL_SOLN_DESC = 'Available quantization bit-width is not wide enough, attempting to increase bit-width used for quantization.'
MAX_DIFF_FAIL_SOLN_DESC = 'The maximum difference between the unquantized and de-quantized data is large, attempting to increase bit-width and adjust quantization encodings.'
DATA_DISTRIBUTION_FAIL_SOLN_DESC = 'Available quantization bit-width is not wide enough, attempting to increase bit-width used for quantization.'

FAILED_VERIFIERS_SOLN_DESC = dict()
FAILED_VERIFIERS_SOLN_DESC[QcFailedVerifier.STATS_FAIL] = STATS_FAIL_SOLN_DESC
FAILED_VERIFIERS_SOLN_DESC[QcFailedVerifier.DATA_RANGE_FAIL] = DATA_RANGE_FAIL_SOLN_DESC
FAILED_VERIFIERS_SOLN_DESC[QcFailedVerifier.SQNR_FAIL] = SQNR_FAIL_SOLN_DESC
FAILED_VERIFIERS_SOLN_DESC[QcFailedVerifier.MIN_MAX_FAIL] = MIN_MAX_FAIL_SOLN_DESC
FAILED_VERIFIERS_SOLN_DESC[QcFailedVerifier.MAX_DIFF_FAIL] = MAX_DIFF_FAIL_SOLN_DESC
FAILED_VERIFIERS_SOLN_DESC[
    QcFailedVerifier.DATA_DISTRIBUTION_FAIL] = DATA_DISTRIBUTION_FAIL_SOLN_DESC
