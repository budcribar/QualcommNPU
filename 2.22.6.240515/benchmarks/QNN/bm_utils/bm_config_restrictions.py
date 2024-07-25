# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from __future__ import absolute_import
from .bm_constants import BmConstants

BENCH_NAME = 'qnn_bench'
CONFIG_VALID_DEVICEOSTYPES = [
    BmConstants.CONFIG_DEVICEOSTYPES_ANDROID_AARCH64,
    BmConstants.CONFIG_DEVICEOSTYPES_WINDOWS_AARCH64,
    BmConstants.CONFIG_DEVICEOSTYPES_QNX_AARCH64,
    BmConstants.CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC112,
    BmConstants.CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC93,
    BmConstants.CONFIG_DEVICEOSTYPES_AARCH64_LINUX_OE_GCC82,
    BmConstants.CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_OE_GCC94,
    BmConstants.CONFIG_DEVICEOSTYPES_AARCH64_UBUNTU_OE_GCC75
]

CONFIG_VALID_MEASURMENTS = [
    BmConstants.MEASURE_TIMING,
    BmConstants.MEASURE_MEM
]
