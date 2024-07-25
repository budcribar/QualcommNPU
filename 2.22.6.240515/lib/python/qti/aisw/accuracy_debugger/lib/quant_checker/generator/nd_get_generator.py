# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from .nd_qnn_generator import QnnGenerator
from .nd_snpe_generator import SnpeGenerator


def get_generator_cls(engine):
    "Return the Generator calss"
    if engine == 'QNN':
        return QnnGenerator
    else:
        return SnpeGenerator
