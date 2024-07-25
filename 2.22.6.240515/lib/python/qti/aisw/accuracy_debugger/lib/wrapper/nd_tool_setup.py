# =============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from qti.aisw.accuracy_debugger.lib.runner.component_runner import *
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine


class ToolConfig(object):

    def __init__(self):
        pass

    def run_framework_diagnosis(self, args):
        """Runs the framework diagnosis tool in the specified environment."""
        try:
            exec_framework_diagnosis(args)
        except:
            return -1
        else:
            return 0

    def run_qnn_inference_engine(self, args, need_std_out=False):
        try:
            exec_inference_engine(args, Engine.QNN.value)
        except:
            return -1
        else:
            return 0

    def run_snpe_inference_engine(self, args):
        try:
            exec_inference_engine(args, Engine.SNPE.value)
        except:
            return -1
        else:
            return 0

    def run_verifier(self, args):
        try:
            exec_verification(args)
        except:
            return -1
        else:
            return 0

    def run_qnn_compare_encodings(self, args):
        """This feature will extract encodings from given QNN net json and compares it with AIMET
        encodings and also dumps an excel sheet highlighting any mismatches."""
        try:
            exec_compare_encodings(args, Engine.QNN.value)
        except:
            return -1
        else:
            return 0

    def run_snpe_compare_encodings(self, args):
        """This feature will extract encodings from given SNPE dlc and compares it with AIMET
        encodings and also dumps an excel sheet highlighting any mismatches."""
        try:
            exec_compare_encodings(args, Engine.SNPE.value)
        except:
            return -1
        else:
            return 0

    def run_accuracy_deep_analyzer(self, args):
        """Runs the deep_analyzer tool in the specified environment."""
        try:
            exec_deep_analyzer(args)
        except:
            return -1
        else:
            return 0

    def run_tensor_inspection(self, args):
        """This feature compares given reference output and target output tensors
            and dumps various statistics to represent differences between them."""
        try:
            exec_tensor_inspection(args)
        except:
            return -1
        else:
            return 0
