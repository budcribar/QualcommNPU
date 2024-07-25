# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from .nd_qnn_extractor import QnnExtractor
from .nd_snpe_extractor import SnpeExtractor


def get_extractor_cls(engine):
    if engine == 'QNN':
        return QnnExtractor
    else:
        return SnpeExtractor
