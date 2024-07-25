# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm

from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_infer_type
from tvm.relay.frontend.common import set_span
import numpy as np

from qti.aisw.converters.relay.qti_ops.depth_to_space_rect_relay_op import DepthToSpaceRectRelayOp
from qti.aisw.converters.common.utils.converter_utils import log_debug3


@tvm.ir.transform.module_pass(opt_level=3)
class IdentifyDepthToSpaceRect:
    def __init__(self, data_layout):
        assert data_layout == "NCHW", "Unsupported data layout {}".format(data_layout)
        self.data_layout = data_layout

    def transform_module(self, mod, ctx):
        # CRD mode: elements along the depth dimension are rearranged in the order of column, row, and then depth.
        #   Cases 1 (6D reshape):
        #     input: [n, c, h, w]
        #     reshape: [n, c/(blk_h*blk_w), blk_h, blk_w, h, w]
        #     transpose: [n, c/(blk_h*blk_w), h, blk_h, w, blk_w] with [0, 1, 4, 2, 5, 3]
        #     reshape: [n, c/(blk_h*blk_w), h*blk_h, w*blk_w]
        #   Cases 2 (5D reshape, blk_h = 1):
        #     input: [n, c, h, w]
        #     reshape: [n, c/blk_w, blk_w, h, w]
        #     transpose: [n, c/blk_w, h, w, blk_w] with [0, 1, 3, 4, 2]
        #     reshape: [n, c/blk_w, h, w*blk_w]
        #   Cases 3 (5D reshape, blk_w = 1):
        #     input: [n, c, h, w]
        #     reshape: [n, c/blk_h, blk_h, h, w]
        #     transpose: [n, c/blk_h, h, blk_h, w] with [0, 1, 3, 2, 4]
        #     reshape: [n, c/blk_h, h*blk_h, w]
        #
        # DCR mode: elements along the depth dimension are rearranged in the order of depth, column, and then row.
        #   Cases 1 (6D reshape):
        #     input: [n, c, h, w]
        #     reshape: [n, blk_h, blk_w, c/(blk_h*blk_w), h, w]
        #     transpose: [n, c/(blk_h*blk_w), h, blk_h, w, blk_w] with [0, 3, 4, 1, 5, 2]
        #     reshape: [n, c/(blk_h*blk_w), h*blk_h, w*blk_w]
        #   Cases 2 (5D reshape, blk_h = 1):
        #     input: [n, c, h, w]
        #     reshape: [n, blk_w, c/blk_w, h, w]
        #     transpose: [n, c/blk_w, h, w, blk_w] with [0, 2, 3, 4, 1]
        #     reshape: [n, c/blk_w, h, w*blk_w]
        #   Cases 3 (5D reshape, blk_w = 1):
        #     input: [n, c, h, w]
        #     reshape: [n, blk_h, c/blk_h, h, w]
        #     transpose: [n, c/blk_h, h, blk_h, w] with [0, 2, 3, 1, 4]
        #     reshape: [n, c/blk_h, h*blk_h, w]

        data_layout = self.data_layout
        transpose_axes_CRD = [[0, 1, 4, 2, 5, 3], [0, 1, 3, 4, 2], [0, 1, 3, 2, 4]]
        transpose_axes_DCR = [[0, 3, 4, 1, 5, 2], [0, 2, 3, 4, 1], [0, 2, 3, 1, 4]]

        class MatchAndRewrite(DFPatternCallback):
            def __init__(self):
                super(MatchAndRewrite, self).__init__(require_type=True)

                self._depth_to_space_rect_op = tvm.relay.op.op.get(DepthToSpaceRectRelayOp._op_name)

                # Match following pattern to depth_to_space op:
                # %25 = reshape(%24, newshape=[1, 3, 2, 2, 256, 256]) /* ty=Tensor[(1, 3, 2, 2, 256, 256), float32] */;
                # %26 = transpose(%25, axes=[0, 1, 4, 2, 5, 3]) /* ty=Tensor[(1, 3, 256, 2, 256, 2), float32] */;
                # %27 = reshape(%26, newshape=[1, 3, 512, 512]) /* ty=Tensor[(1, 3, 512, 512), float32] */;
                self._data = wildcard()
                self._reshape1 = is_op("reshape")(self._data)
                self._transpose = is_op("transpose")(self._reshape1)
                self._reshape2 = is_op("reshape")(self._transpose)

                self.pattern = self._reshape2

            def callback(self, pre, post, node_map):
                def get_shape(key):
                    return run_infer_type(node_map[key][0]).checked_type.shape

                def get_transpose_axes(key):
                    return run_infer_type(node_map[key][0]).attrs.axes

                def check_reshape6d(reshape6d_input_shape, reshape6d_output_shape):
                    if len(reshape6d_input_shape) == 4 and len(reshape6d_output_shape) == 6:
                        # Check the Channel dimension is split into blocks
                        if reshape6d_input_shape[1] == np.prod(reshape6d_output_shape[1:4]):
                            return True
                    elif len(reshape6d_input_shape) == 4 and len(reshape6d_output_shape) == 5:
                        # Check the Channel dimension is split into blocks
                        if reshape6d_input_shape[1] == np.prod(reshape6d_output_shape[1:3]):
                            return True
                    return False

                def check_reshape4d(reshape4d_input_shape, reshape4d_output_shape):
                    if len(reshape4d_input_shape) == 6 and len(reshape4d_output_shape) == 4:
                        # Check the block_size is reshaped into H and W
                        if np.prod(reshape4d_input_shape[2:4]) == reshape4d_output_shape[2] and \
                                np.prod(reshape4d_input_shape[4:]) == reshape4d_output_shape[3]:
                            return True
                    elif len(reshape4d_input_shape) == 5 and len(reshape4d_output_shape) == 4:
                        # Check the block_size is reshaped into H or W
                        if np.prod(reshape4d_input_shape[2:4]) == reshape4d_output_shape[2] and \
                                np.prod(reshape4d_input_shape[4:]) == reshape4d_output_shape[3]:
                            return True
                        elif np.prod(reshape4d_input_shape[2:3]) == reshape4d_output_shape[2] and \
                                np.prod(reshape4d_input_shape[3:]) == reshape4d_output_shape[3]:
                            return True
                    return False

                def check_transpose(transpose_axes, block_size_h, block_size_w):
                    # Check the block_size and transpose_axes are fit for some cases, e.g.,
                    #     input: [n, c, h, w]
                    #     reshape: [n, c/blk_w, blk_w, h, w]
                    #     transpose: [n, c/blk_w, h, blk_w, w] with [0, 1, 3, 2, 4]
                    #     reshape: [n, c/blk_w, h, blk_w*w]
                    if transpose_axes in transpose_axes_CRD:
                        if transpose_axes == transpose_axes_CRD[0]:
                            return True
                        elif transpose_axes == transpose_axes_CRD[1] and block_size_h == 1:
                            return True
                        elif transpose_axes == transpose_axes_CRD[2] and block_size_w == 1:
                            return True
                    elif transpose_axes in transpose_axes_DCR:
                        if transpose_axes == transpose_axes_DCR[0]:
                            return True
                        elif transpose_axes == transpose_axes_DCR[1] and block_size_h == 1:
                            return True
                        elif transpose_axes == transpose_axes_DCR[2] and block_size_w == 1:
                            return True
                    return False

                reshape_1_input_shape = get_shape(self._data)
                reshape_1_output_shape = get_shape(self._reshape1)
                reshape_2_input_shape = get_shape(self._transpose)
                reshape_2_output_shape = get_shape(self._reshape2)
                transpose_axes = get_transpose_axes(self._transpose)

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

                block_size_h = reshape_2_output_shape[2] // reshape_1_input_shape[2]
                block_size_w = reshape_2_output_shape[3] // reshape_1_input_shape[3]

                if not check_transpose(list(transpose_axes), block_size_h, block_size_w):
                    log_debug3(
                        "Transpose axes {} mismatch with D2S axes.".format(
                            list(transpose_axes)
                        )
                    )
                    return post

                mode = "CRD" if list(transpose_axes) in transpose_axes_CRD else "DCR"
                data = node_map[self._data][0]
                new_attrs = {
                    "block_size": [int(block_size_h), int(block_size_w)],
                    "layout": data_layout,
                    "mode": mode,
                }

                call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
                spans = tvm.relay.SequentialSpan(
                    [node_map[node][0].span for node in [self._reshape1, self._transpose, self._reshape2] if node_map[node][0].span is not None])
                call_depth_to_space_rect = set_span(tvm.relay.Call(self._depth_to_space_rect_op, [data], call_attrs), spans)
                return call_depth_to_space_rect

        new_expr = rewrite(MatchAndRewrite(), mod["main"])
        mod.update_func(mod.get_global_var("main"), new_expr)

        return mod
