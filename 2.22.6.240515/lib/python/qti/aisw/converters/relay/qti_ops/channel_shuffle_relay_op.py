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

class ChannelShuffleRelayOp:
    _op_name = "qti.aisw.channel_shuffle"
    _instance = None

    def __new__(cls):
        # Singleton pattern
        if not cls._instance:
            cls._instance = super().__new__(cls)
            custom_op = _op.get(cls._op_name)
            custom_op.set_num_inputs(1)
            custom_op.add_type_rel("Identity")
            _op.register_pattern(cls._op_name, _op.OpPattern.INJECTIVE)
        return cls._instance

    @staticmethod
    @_op.register_convert_op_layout(_op_name)
    def convert_layout(attrs, inputs, tinfos, desired_layouts):
        assert len(desired_layouts) == 1, "Only one desired layout is expected."
        assert len(inputs) == 1, "Number of inputs mismatched."
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = desired_layouts[0]
        call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        return relay.Call(_op.get(ChannelShuffleRelayOp._op_name), inputs, call_attrs)

    @staticmethod
    @tvm.ir.register_op_attr(_op_name, "FInferCorrectLayout")
    def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):
        layout = tvm.tir.data_layout.layout(attrs["data_layout"])
        return InferCorrectLayoutOutput([layout], [layout], attrs)

ChannelShuffleRelayOp()
