# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_tensorflow_framework_2_3_0 import TensorFlowFramework_2_3_0


class TensorFlowFramework_2_10_1(TensorFlowFramework_2_3_0):
    __VERSION__ = '2.10.1'

    def __init__(self, logger):
        super(TensorFlowFramework_2_10_1, self).__init__(logger)
        self._graph = None
