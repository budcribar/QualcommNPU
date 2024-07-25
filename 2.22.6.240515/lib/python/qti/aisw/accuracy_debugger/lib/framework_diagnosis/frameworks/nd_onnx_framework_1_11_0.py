# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_onnx_framework_1_8_0 import OnnxFramework_1_8_0
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message
from onnx import helper, shape_inference, version_converter
from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_onnx_extract import extract_model
import logging
import numpy as np
import os
import onnx


class OnnxFramework_1_11_0(OnnxFramework_1_8_0):
    __VERSION__ = '1.11.0'

    def __init__(self, logger, custom_op_lib=None):
        super(OnnxFramework_1_11_0, self).__init__(logger, custom_op_lib)
