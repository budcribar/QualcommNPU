# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.framework_diagnosis.frameworks.nd_onnx_framework_1_3_0 import OnnxFramework_1_3_0
from lib.utils.nd_exceptions import FrameworkError
from lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message
import logging


class OnnxFramework_1_8_0(OnnxFramework_1_3_0):
    __VERSION__ = '1.8.0'
    def __init__(self, logger):
        super(OnnxFramework_1_8_0, self).__init__(logger)
