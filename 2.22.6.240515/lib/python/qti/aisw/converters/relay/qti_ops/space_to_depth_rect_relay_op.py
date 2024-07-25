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

class SpaceToDepthRectRelayOp:
    _op_name = "qti.aisw.space_to_depth_rect"
    _instance = None

    def __new__(cls):
        # Singleton pattern
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # Custom S2D op with mode attributes and rectangular block_size
            custom_op = _op.get(cls._op_name)
            custom_op.set_num_inputs(1)
            custom_op.add_type_rel(cls._op_name, SpaceToDepthRectRelayOp.relation_func)
            _op.register_pattern(cls._op_name, _op.OpPattern.INJECTIVE)
        return cls._instance

    @staticmethod
    def relation_func(arg_types, attrs):
        assert len(arg_types) == 1, "type relation arg number mismatch!"
        input_type = arg_types[0]
        input_shape = input_type.shape
        blk_h = int(attrs["block_size"][0])
        blk_w = int(attrs["block_size"][1])
        if attrs["layout"] == "NCHW":
            output_shape = [input_shape[0], input_shape[1]*blk_h*blk_w, input_shape[2]//blk_h, input_shape[3]//blk_w]
        elif attrs["layout"] == "NHWC":
            output_shape = [input_shape[0], input_shape[1]//blk_h, input_shape[2]//blk_w, input_shape[3]*blk_h*blk_w]

        return relay.TensorType(output_shape, input_type.dtype)

    @staticmethod
    @_op.register_convert_op_layout(_op_name)
    def convert_layout(attrs, inputs, tinfos, desired_layouts):
        # Update the layout for new S2D op
        new_attrs = {
            "block_size": attrs["block_size"],
            "layout": desired_layouts[0][:],
            "mode": attrs["mode"]
        }
        call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        return relay.Call(_op.get(SpaceToDepthRectRelayOp._op_name), inputs, call_attrs)

    @staticmethod
    @tvm.ir.register_op_attr(_op_name, "FInferCorrectLayout")
    def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):
        layout = tvm.tir.data_layout.layout(attrs["layout"])
        return InferCorrectLayoutOutput([layout], [layout], attrs)

SpaceToDepthRectRelayOp()
