# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_onnx_framework_1_3_0 import OnnxFramework_1_3_0
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message
import logging


class OnnxFramework_1_6_0(OnnxFramework_1_3_0):
    __VERSION__ = '1.6.0'

    def __init__(self, logger, custom_op_lib=None):
        super(OnnxFramework_1_6_0, self).__init__(logger, custom_op_lib)

    def extract(self, start_layer_output_name, end_layer_output_name=None, out_model_path=None):
        raise NotImplementedError('Method extract is not implemented for onnx version < 1.8.0')
