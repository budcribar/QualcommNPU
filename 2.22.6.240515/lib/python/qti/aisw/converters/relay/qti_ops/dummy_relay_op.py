# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm
import tvm.relay.op.op as _op
from tvm import relay
from tvm.relay.transform.infer_layout_utils import InferCorrectLayoutOutput

class DummyRelayOp:
    _op_name = "qti.aisw.dummy"
    _instance = None

    def __new__(cls):
        # Multiple calls of constructor will only have relay op registration once for this singleton pattern
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # Bypass this line if register_convert_op_layout or register_op_attr is set through the decorator
            _op.register(cls._op_name)
            # TODO: Define the op property here. See other files for more details
        return cls._instance

    @staticmethod
    def relation_func(arg_types, attrs):
        # TODO: Define the customized output shape here if needed. See TfliteDetectionPostprocessRelayOp if multiple outputs
        raise NotImplementedError

    @staticmethod
    @_op.register_convert_op_layout(_op_name)
    def convert_layout(attrs, inputs, tinfos, desired_layouts):
        # TODO: Define the ConvertOpLayout here if needed
        raise NotImplementedError

    @staticmethod
    @tvm.ir.register_op_attr(_op_name, "FInferCorrectLayout")
    def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):
        # TODO: Define the InferCorrectLayout here if needed
        raise NotImplementedError

# Instantiate and only register once
DummyRelayOp()
