# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import abstractmethod


class Converter:
    @abstractmethod
    def __init__(self, context):
        pass
