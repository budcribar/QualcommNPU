# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from collections import namedtuple

VerifyResult = namedtuple('VerifyResult', ['LayerType', 'Size', 'Tensor_dims', 'Verifier', 'Metric', 'Units'])
