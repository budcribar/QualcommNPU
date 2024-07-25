# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm

from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_infer_type
from tvm.relay.frontend.common import set_span
import numpy as np

from qti.aisw.converters.relay.qti_ops.space_to_depth_rect_relay_op import SpaceToDepthRectRelayOp
from qti.aisw.converters.common.utils.converter_utils import log_debug3


@tvm.ir.transform.module_pass(opt_level=3)
class IdentifySpaceToDepthRect:
    def __init__(self, data_layout):
        assert data_layout == "NCHW", "Unsupported data layout {}".format(data_layout)
        self.data_layout = data_layout

    def transform_module(self, mod, ctx):
        data_layout = self.data_layout

        class MatchAndRewrite(DFPatternCallback):
            def __init__(self):
                super(MatchAndRewrite, self).__init__(require_type=True)

                self._space_to_depth_op = tvm.relay.op.op.get(SpaceToDepthRectRelayOp._op_name)

                # Match following pattern to space_to_depth op:
                # Before:
                #   %0 = reshape(%x, newshape=[1, 3, 128, 4, 128, 4]) /* C.graph: aten::view, warning: no trace info 19 */ /* ty=Tensor[(1, 3, 128, 4, 128, 4), float32] */;
                #   %1 = transpose(%0, axes=[0, 1, 3, 5, 2, 4]) /* C.graph: aten::permute, warning: no trace info 20 */ /* ty=Tensor[(1, 3, 4, 4, 128, 128), float32] */;
                #   %2 = reshape(%1, newshape=[1, 48, 128, 128]) /* C.graph: aten::reshape, warning: no trace info 21 */ /* ty=Tensor[(1, 48, 128, 128), float32] */;
                # After:
                #   %0 = space_to_depth_rect(%x, block_size=[4, 4], layout="NCHW", mode="CRD") /* ty=Tensor[(1, 48, 128, 128), float32] span=aten__reshape_0:0:0, output_names:[] */;

                self._data = wildcard()
                self._reshape1 = is_op("reshape")(self._data)
                self._transpose = is_op("transpose")(self._reshape1).has_attr({"axes": [0, 1, 3, 5, 2, 4]})| \
                                  is_op("transpose")(self._reshape1).has_attr({"axes": [0, 3 ,5, 1, 2, 4]})| \
                                  is_op("transpose")(self._reshape1).has_attr({"axes": [0, 1, 4, 2, 3]})
                self._reshape2 = is_op("reshape")(self._transpose)

                self.pattern = self._reshape2

            def callback(self, pre, post, node_map):
                def get_shape(key):
                    return run_infer_type(node_map[key][0]).checked_type.shape

                def check_reshape6d(reshape6d_input_shape, reshape6d_output_shape):
                    if len(reshape6d_input_shape) == 4 and len(reshape6d_output_shape) == 6:
                        # Check H and W is split into blocks
                        if reshape6d_input_shape[2] == np.prod(reshape6d_output_shape[2:4]) and \
                           reshape6d_input_shape[3] == np.prod(reshape6d_output_shape[4:6]) :
                            return True
                    elif len(reshape6d_input_shape) == 4 and len(reshape6d_output_shape) == 5:
                        # Check H and W is split into blocks
                        if reshape6d_input_shape[2] == np.prod(reshape6d_output_shape[2]) and \
                           reshape6d_input_shape[3] == np.prod(reshape6d_output_shape[3:]) :
                            return True
                    return False

                def check_reshape4d(reshape4d_input_shape, reshape4d_output_shape):
                    if len(reshape4d_input_shape) == 6 and len(reshape4d_output_shape) == 4:
                        # Check that the block_size is reshaped into Channel
                        return reshape4d_output_shape[1] == np.prod(reshape4d_input_shape[1:4])
                    elif len(reshape4d_input_shape) == 5 and len(reshape4d_output_shape) == 4:
                        # Check that the block_size is reshaped into Channel
                        return reshape4d_output_shape[1] == np.prod(reshape4d_input_shape[1:3])
                    return False

                reshape_1_input_shape = get_shape(self._data)
                reshape_1_output_shape = get_shape(self._reshape1)
                reshape_2_input_shape = get_shape(self._transpose)
                reshape_2_output_shape = get_shape(self._reshape2)

                if not check_reshape6d(reshape_1_input_shape, reshape_1_output_shape):
                    log_debug3(
                        "Reshape6d input shape {} mismatch with output shape {}.".format(
                            reshape_1_input_shape, reshape_1_output_shape
                        )
                    )
                    return post

                if not check_reshape4d(reshape_2_input_shape, reshape_2_output_shape):
                    log_debug3(
                        "Reshape4d input shape {} mismatch with output shape {}.".format(
                            reshape_2_input_shape, reshape_2_output_shape
                        )
                    )
                    return post

                block_size_h = 0
                block_size_w = 0
                if len(reshape_2_input_shape) == 5:
                    block_size_h = 1
                    block_size_w = reshape_1_output_shape[4]
                elif len(reshape_2_input_shape) == 6:
                    block_size_h = reshape_1_output_shape[3]
                    block_size_w = reshape_1_output_shape[5]
                else:
                    log_debug3(
                        "Invalid block_size_h {} and block_size_w {}.".format(
                            block_size_h, block_size_w
                        )
                    )
                    return post

                data = node_map[self._data][0]
                transpose_6d = node_map[self._transpose][0]
                axes = transpose_6d.attrs['axes'][:]

                if axes == [0, 1, 3, 5, 2, 4]:
                    mode = "CRD"
                elif axes == [0, 1, 4, 2, 3]:
                    mode = "CRD"
                else:
                    mode = "DCR"

                data = node_map[self._data][0]
                new_attrs = {
                    "block_size": [int(block_size_h), int(block_size_w)],
                    "layout": data_layout,
                    "mode": mode
                }


                call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)

                spans = tvm.relay.SequentialSpan([node_map[node][0].span for node in [self._reshape1, self._transpose, self._reshape2] if node_map[node][0].span is not None])
                call_space_to_depth = set_span(tvm.relay.Call(self._space_to_depth_op, [data], call_attrs), spans)
                return call_space_to_depth


        new_expr = rewrite(MatchAndRewrite(), mod["main"])
        mod.update_func(mod.get_global_var("main"), new_expr)

        return mod
