# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.framework_diagnosis.frameworks.nd_tensorflow_framework_1_6_0 import TensorFlowFramework_1_6_0


class TensorFlowFramework_1_11_0(TensorFlowFramework_1_6_0):
    __VERSION__ = '1.11.0'

    def __init__(self, logger):
        super(TensorFlowFramework_1_11_0, self).__init__(logger)
        self._graph = None

