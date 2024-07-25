# ==============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm
import numpy as np

from tvm.relay.dataflow_pattern import *
from tvm.relay.frontend.common import set_span
from tvm import relay

from qti.aisw.converters.relay.qti_ops.tflite_detection_postprocess_relay_op import TfliteDetectionPostprocessRelayOp

@tvm.ir.transform.module_pass(opt_level=3)
class IdentifyTFLiteDetectionPostProcess:
    def transform_module(self, mod, ctx):
        class MatchAndRewrite(DFPatternCallback):
            def __init__(self):
                super(MatchAndRewrite, self).__init__(require_type=True)
                self.detection_postprocess = tvm.relay.op.op.get(TfliteDetectionPostprocessRelayOp._op_name)
                # Match following patterns to Detection output post processing

                # box prob (loc_prob in tvm) preprocess from yxhw to xyhw
                #%533 = split(%532, indices_or_sections=4, axis=2);
                #%534 = %533.1;
                #%535 = %533.0;
                #%536 = %533.3;
                #%537 = %533.2;
                #%538 = (%534, %535, %536, %537);
                #%539 = concatenate(%538, axis=2);

                # Anchor preprocess from yxhw to ltrb
                #%540 = split(%v_param_365, indices_or_sections=4, axis=1);
                #%541 = %540.3;
                #%542 = %540.1;
                #%543 = multiply(%541, -0.5f);
                #%544 = %540.2;
                #%545 = %540.0;
                #%546 = multiply(%544, -0.5f);
                #%547 = multiply(%541, 0.5f);
                #%548 = multiply(%544, 0.5f);
                #%549 = add(%542, %543);
                #%550 = add(%545, %546);
                #%551 = add(%542, %547);
                #%552 = add(%545, %548);
                #%553 = (%549, %550, %551, %552);
                #%554 = concatenate(%553, axis=1);

                # NMS part
                #%555 = transpose(%430, axes=[0, 2, 1]);
                #%556 = reshape(%539, newshape=[1, 76824]);
                #%557 = expand_dims(%554, axis=0);
                #%558 = vision.multibox_transform_loc(%555, %556, %557, clip=False, threshold=-inff, variances=[1f, 1f, 1f, 1f]);
                #%559 = %558.0;
                #%560 = %558.1;
                #%561 = %558.1;
                #%562 = vision.non_max_suppression(%559, %560, %561, 100, 0.5f, meta[relay.attrs.NonMaximumSuppressionAttrs][0]);
                #%563 = vision.get_valid_counts(%562, 0f, meta[relay.attrs.GetValidCountsAttrs][0]);
                #%564 = %563.1;

                # Format processing
                #%565 = strided_slice(%564, begin=[0, 0, 0], end=[1, 100, 6], strides=[1], axes=None);
                #%566 = split(%565, indices_or_sections=6, axis=2);
                #%567 = %566.3;
                #%568 = %566.2;
                #%569 = %566.5;
                #%570 = %566.4;
                #%571 = (%567, %568, %569, %570);
                #%572 = %566.0;
                #%573 = %566.1;
                #%574 = concatenate(%571, axis=2);
                #%575 = reshape(%572, newshape=[1, -1]);
                #%576 = reshape(%573, newshape=[1, -1]);
                #%577 = %563.0;
                #%578 = (%574, %575, %576, %577);

                self._anchors_yxhw = wildcard()
                self._anchors_split = is_op('split')(self._anchors_yxhw).has_attr({'indices_or_sections': 4, 'axis': 1})

                self._anchors_hight_pos = is_op('multiply')(is_tuple_get_item(self._anchors_split, 2), is_expr((relay.const(0.5))))
                self._anchors_hight_neg = is_op('multiply')(is_tuple_get_item(self._anchors_split, 2), is_expr((relay.const(-0.5))))
                self._anchors_width_pos = is_op('multiply')(is_tuple_get_item(self._anchors_split, 3), is_expr((relay.const(0.5))))
                self._anchors_width_neg = is_op('multiply')(is_tuple_get_item(self._anchors_split, 3), is_expr((relay.const(-0.5))))

                self._anchors_left = is_op('add')(is_tuple_get_item(self._anchors_split, 1), self._anchors_width_neg)
                self._anchors_right = is_op('add')(is_tuple_get_item(self._anchors_split, 1), self._anchors_width_pos)
                self._anchors_top = is_op('add')(is_tuple_get_item(self._anchors_split, 0), self._anchors_hight_neg)
                self._anchors_bottom = is_op('add')(is_tuple_get_item(self._anchors_split, 0), self._anchors_hight_pos)
                self._anchors_ltrb = is_tuple((self._anchors_left,
                                     self._anchors_top,
                                     self._anchors_right,
                                     self._anchors_bottom))
                self._anchors_concat = is_op('concatenate')(self._anchors_ltrb).has_attr({'axis': 1})


                self._box_prob_yxhw = wildcard()
                self._box_prob_yxhw_split = is_op('split')(self._box_prob_yxhw).has_attr({'indices_or_sections': 4})
                self._box_prob_xywh_tuple = is_tuple((is_tuple_get_item(self._box_prob_yxhw_split, 1),
                                                      is_tuple_get_item(self._box_prob_yxhw_split, 0),
                                                      is_tuple_get_item(self._box_prob_yxhw_split, 3),
                                                      is_tuple_get_item(self._box_prob_yxhw_split, 2)))
                self._box_prob_xywh_concate = is_op('concatenate')(self._box_prob_xywh_tuple).has_attr({'axis': 2})
                self._box_prob_xywh_reshape = is_op("reshape")(self._box_prob_xywh_concate)
                self._class_prob = wildcard()
                self._class_prob_transpose = is_op("transpose")(self._class_prob).has_attr({"axes": [0, 2, 1]})
                self._anchors_concat_expand = is_op('expand_dims')(self._anchors_concat).has_attr({'axis': 0})

                self._vision_multibox_transform_loc = is_op("vision.multibox_transform_loc")(self._class_prob_transpose,
                                                                                             self._box_prob_xywh_reshape,
                                                                                             self._anchors_concat_expand)

                self._vision_non_max_suppression = is_op("vision.non_max_suppression")(is_tuple_get_item(self._vision_multibox_transform_loc, 0),
                                                                                       is_tuple_get_item(self._vision_multibox_transform_loc, 1),
                                                                                       is_tuple_get_item(self._vision_multibox_transform_loc, 1),
                                                                                       wildcard(),
                                                                                       wildcard())
                self._vision_get_valid_counts = is_op("vision.get_valid_counts")(self._vision_non_max_suppression, wildcard())
                self._get_valid_counts_tuple_get_item_0 = is_tuple_get_item(self._vision_get_valid_counts, 0)
                self._get_valid_counts_tuple_get_item_1 = is_tuple_get_item(self._vision_get_valid_counts, 1)

                self._strided_slice = is_op('strided_slice')(self._get_valid_counts_tuple_get_item_1).has_attr({'begin': [0, 0, 0]})
                self._strided_slice_split = is_op('split')(self._strided_slice).has_attr({'indices_or_sections': 6})
                self._boxes = is_tuple((is_tuple_get_item(self._strided_slice_split, 3),
                                        is_tuple_get_item(self._strided_slice_split, 2),
                                        is_tuple_get_item(self._strided_slice_split, 5),
                                        is_tuple_get_item(self._strided_slice_split, 4)))
                self._boxes_concat = is_op('concatenate')(self._boxes).has_attr({'axis': 2})
                self._class_ids = is_op('reshape')(is_tuple_get_item(self._strided_slice_split, 0))
                self._scores = is_op('reshape')(is_tuple_get_item(self._strided_slice_split, 1))

                self._output = is_tuple((self._boxes_concat, self._class_ids, self._scores, self._get_valid_counts_tuple_get_item_0))

                self.pattern = self._output


            def callback(self, pre, post, node_map):
                # scaling factors in form of (dy, dx, dh, dw) to match QNN definition
                delta_scaling_factors = [
                    int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[1].value)),
                    int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[0].value)),
                    int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[2].value)),
                    int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[3].value))
                ]
                # Return the shape of in sequence [scores, boxes, cls_ids and num_detections] to match IRModule
                output_dims = [
                    node_map[self._scores][0].checked_type.shape,
                    node_map[self._boxes_concat][0].checked_type.shape,
                    node_map[self._class_ids][0].checked_type.shape,
                    node_map[self._get_valid_counts_tuple_get_item_0][0].checked_type.shape
                ]
                new_attrs = {
                    'delta_scaling_factors': delta_scaling_factors,
                    'confidence_threshold': node_map[self._vision_multibox_transform_loc][0].attrs.threshold,
                    'iou_threshold': node_map[self._vision_non_max_suppression][0].args[4].data.numpy().tolist(),
                    # TBD:
                    # since we can't pass "use_regular_nms" from tflite frontend to here
                    # and backend validator currently only support regular mode,
                    # we leave "nms_type" 1 here
                    'nms_type': 1,
                    'background_class_idx': node_map[self._vision_non_max_suppression][0].attrs.id_index,
                    'use_bg_in_nms': False,
                    'output_background': True,
                    'share_location': True,
                    'nms_eta': 1.0,
                    'detection_limit': node_map[self._vision_non_max_suppression][0].args[3].data.numpy().tolist(),
                    'output_dims': output_dims
                }

                class_prob = node_map[self._class_prob][0]
                box_prob = node_map[self._box_prob_yxhw][0]
                anchors = node_map[self._anchors_yxhw][0]
                call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
                relay_detection_postprocess = tvm.relay.Call(
                    self.detection_postprocess,
                    [class_prob, box_prob, anchors],
                    call_attrs
                )
                # tflite TFLite_detection_postprocess output order is (boxes, classes, scores, num_detection),
                # our relay detection output order is (scores, boxes, classes, num_detection),
                old_span = node_map[self._output][0].span
                output_names = [
                    old_span.output_names[2],
                    old_span.output_names[0],
                    old_span.output_names[1],
                    old_span.output_names[3]
                ]
                relay_detection_postprocess = set_span(relay_detection_postprocess, span=old_span.source_name.name, op_type=old_span.op_type, output_names=output_names)
                # to pass tvm type checker, we need to switch relay detection output order to align with
                # TFLite_detection_postprocess output order
                relay_detection_postprocess = relay.Tuple((
                    relay.TupleGetItem(relay_detection_postprocess, 1),
                    relay.TupleGetItem(relay_detection_postprocess, 2),
                    relay.TupleGetItem(relay_detection_postprocess, 0),
                    relay.TupleGetItem(relay_detection_postprocess, 3)
                ))
                return relay_detection_postprocess

        new_expr = rewrite(MatchAndRewrite(), mod["main"])
        mod.update_func(mod.get_global_var("main"), new_expr)

        return mod
