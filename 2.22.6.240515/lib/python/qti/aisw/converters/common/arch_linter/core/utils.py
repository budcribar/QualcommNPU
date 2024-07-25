# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import math
import qti.aisw.converters.common.arch_linter.core.constants as const

def back_fill_shape(shape):
    if len(shape) < const.MAX_RANK:
        return [1]*(const.MAX_RANK - len(shape)) + shape
    else:
        return shape