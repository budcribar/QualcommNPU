# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from .onnx_translations import *
from .util import *

from qti.aisw.converters.common import ir_graph

# ------------------------------------------------------------------------------
#   BatchPermutation
# ------------------------------------------------------------------------------
class OnnxBatchPermutationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.BatchPermutationOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxBatchPermutationTranslation(),
                                      converter_type('BatchPermutation', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   BboxTransform
# ------------------------------------------------------------------------------
class OnnxBboxTransformTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph

        # Caffe2 and QNN input orders are different
        #     Caffe2: [rois, deltas, im_info]
        #     QNN: [locations, deltas, batch_idx, im_info]
        input_names = self.extract_input_names(src_op, converter_context)
        for input_name in input_names:
            constant_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if constant_op and not graph.has_buffer(input_name):
                constant_node = graph.add(constant_op, [], [input_name])
                graph.add_src_op_info(constant_node.op.name, [], [input_name])

        # raise warning that QNN only support scale 1 in im_info
        log_warning("BboxTransform only supports scale 1 in im_info.")

        # remove scale in im_info
        im_info_name = input_names[2]
        im_info_buf = graph.get_buffer(im_info_name)
        batch, im_info_rank = im_info_buf.shape
        im_info_const_name = im_info_name + '_const'
        if not graph.has_buffer(im_info_const_name):
            im_info_const_node = graph.add(op_adapter.ConstantOp(im_info_const_name, np.array([0, 1], dtype=np.int32)), [], [im_info_const_name])
            graph.add_src_op_info(im_info_const_node.op.name, [], [im_info_const_name])
        im_info_gather_name = im_info_name + '_gather'
        if not graph.has_buffer(im_info_gather_name):
            im_info_gather_node = graph.add(op_adapter.GatherOp(im_info_gather_name, axis=1), [im_info_name, im_info_const_name], [im_info_gather_name])
            graph.add_src_op_info(im_info_gather_node.op.name, [im_info_name, im_info_const_name], [im_info_gather_name])
        input_names[2] = im_info_gather_name

        rois_name = input_names[0]
        rois_buf = graph.get_buffer(rois_name)
        num_rois, roi_rank = rois_buf.shape

        # 5-tuple roi position of [batch, x1, y1, x2, y2]
        # batch will be extracted as third input
        if roi_rank == 5:
            # split rois into batch indices and roi positions
            # example: rois (10, 5) -> batch indices (10)
            #                       -> roi positions (10, 4)

            # prepare batch indices and roi positions
            batch_idx_name = rois_name + '_batch_indices'
            roi_pos_name = rois_name + '_roi_positions'
            rois_split_name = rois_name + '_split'
            rois_split_node = graph.add(op_adapter.SplitOp(rois_split_name, axis=1, split_index=[1]), [rois_name], [batch_idx_name, roi_pos_name])
            graph.add_src_op_info(rois_split_node.op.name, [rois_name], [batch_idx_name, roi_pos_name])

            # prepare batch indices reshape and cast
            batch_idx_reshape_name = batch_idx_name + '_reshape'
            batch_idx_reshape_node = graph.add(op_adapter.ReshapeOp(batch_idx_reshape_name, shape=[num_rois]), [batch_idx_name], [batch_idx_reshape_name])
            graph.add_src_op_info(batch_idx_reshape_node.op.name, [batch_idx_name], [batch_idx_reshape_name])
            batch_idx_cast_name = batch_idx_name + '_cast'
            batch_idx_cast_node = graph.add(op_adapter.CastOp(batch_idx_cast_name, from_type='float32', to_type='int32'), [batch_idx_reshape_name], [batch_idx_cast_name])
            graph.add_src_op_info(batch_idx_cast_node.op.name, [batch_idx_reshape_name], [batch_idx_cast_name])

            # update op input names
            input_names[0] = roi_pos_name
            input_names.insert(2, batch_idx_cast_name)

        # 4-tuple roi position of [x1, y1, x2, y2]
        # only batch == 1 is supported, so add zero tensor as third input
        else:
            log_assert(batch == 1,
                       "Input batch to BboxTransform is expected to be 1, but got {}".format(batch))
            batch_idx_const_name = rois_name + '_batch_indices'
            batch_idx_const_op = op_adapter.ConstantOp(batch_idx_const_name, np.zeros(num_rois, dtype=np.int32))
            batch_idx_const_node = graph.add(batch_idx_const_op, [], [batch_idx_const_name])
            graph.add_src_op_info(batch_idx_const_node.op.name, [], [batch_idx_const_name])
            input_names.insert(2, batch_idx_const_name)

        params = extract_attributes(src_op,
                                    attr_infos=[('weights', 'lf', []),
                                                ('apply_scale', 'i', 1),
                                                ('rotated', 'i'),
                                                ('angle_bound_on', 'i', 1),
                                                ('angle_bound_lo', 'i', -90),
                                                ('angle_bound_hi', 'i', 90),
                                                ('clip_angle_thresh', 'f', 1.0),
                                                ('legacy_plus_one', 'i')])

        log_assert(params.legacy_plus_one == 0,
                   "BboxTransform does not support legacy_plus_one.")

        # TODO: add support for rotated box in the future
        log_assert(params.rotated == 0,
                   "BboxTransform does not support rotated box.")

        op = op_adapter.AxisAlignedBboxTransformOp(str(src_op.name),
                                                   weights=params.weights)

        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, input_names, output_names)
        graph.add_src_op_info(node.op.name, input_names, output_names)
        return None


OnnxTranslations.register_translation(OnnxBboxTransformTranslation(),
                                      converter_type('BboxTransform', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   BoxWithNmsLimit
# ------------------------------------------------------------------------------
class OnnxBoxWithNmsLimitTranslation(OnnxTranslationBase):
    SUPPORTED_NMS_KERNEL_METHODS = ['hard', 'linear', 'gaussian']
    caffe2_to_ir_nms_kernel_method = {
        "hard": ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_HARD,
        "linear": ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_LINEAR,
        "gaussian": ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_GAUSSIAN,
    }

    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph

        # Caffe2 and QNN inputs are different
        #     Caffe2: [scores, boxes, batch_splits]
        #     QNN: [boxes, scores, batch_indices, batch_splits]
        input_names = self.extract_input_names(src_op, converter_context)
        input_names[0], input_names[1] = input_names[1], input_names[0]
        for input_name in input_names:
            constant_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if constant_op and not graph.has_buffer(input_name):
                constant_node = graph.add(constant_op, [], [input_name])
                graph.add_src_op_info(constant_node.op.name, [], [input_name])

        # get batch from batch_splits
        batch = 1
        if len(input_names) > 2:
            batch_splits_name = input_names[2]
            if not graph.has_buffer(batch_splits_name):
                constant_node = graph.add(op_adapter.ConstantOp(batch_splits_name, np.zeros(1, dtype=np.int32)), [], [batch_splits_name])
                graph.add_src_op_info(constant_node.op.name, [], [batch_splits_name])
            else:
                batch_splits_cast_name = batch_splits_name + '_cast'
                # QNN batch_splits requires datatype int32
                batch_splits_cast_node = graph.add(op_adapter.CastOp(batch_splits_cast_name, from_type='float32', to_type='int32'), [batch_splits_name], [batch_splits_cast_name])
                graph.add_src_op_info(batch_splits_cast_node.op.name, [batch_splits_name], [batch_splits_cast_name])
                input_names[2] = batch_splits_cast_name
            batch_splits_buf = graph.get_buffer(batch_splits_name)
            batch = batch_splits_buf.shape[0]

        # add batch_indices
        log_assert(batch == 1,
                   "Input batch to BoxWithNmsLimit is expected to be 1, but got {}".format(batch))
        boxes_name = input_names[0]
        boxes_buf = graph.get_buffer(boxes_name)
        num_rois, _ = boxes_buf.shape
        batch_idx_const_name = boxes_name + '_batch_indices'
        batch_idx_const_op = op_adapter.ConstantOp(batch_idx_const_name, np.zeros(num_rois, dtype=np.int32))
        batch_idx_const_node = graph.add(batch_idx_const_op, [], [batch_idx_const_name])
        graph.add_src_op_info(batch_idx_const_node.op.name, [], [batch_idx_const_name])
        input_names.insert(2, batch_idx_const_name)

        params = extract_attributes(src_op,
                                    attr_infos=[('score_thresh', 'f', 0.05),
                                                ('nms', 'f', 0.3),
                                                ('detections_per_im', 'i', 100),
                                                ('soft_nms_enabled', 'i', 0),
                                                ('soft_nms_method', 's', 'linear'),
                                                ('soft_nms_sigma', 'f', 0.5),
                                                ('soft_nms_min_score_thres', 'f', 0.001),
                                                ('rotated', 'i'),
                                                ('cls_agnostic_bbox_reg', 'i', 0),
                                                ('input_boxes_include_bg_cls', 'i', 1),
                                                ('output_classes_include_bg_cls', 'i'),
                                                ('legacy_plus_one', 'i'),
                                                ('input_scores_fg_cls_starting_id', 'i', 1)])

        if params.input_boxes_include_bg_cls == 0:
            # last class stands for background in Caffe2, which is not supported by QNN
            # so use Gather to remove background
            # prepare scores without background
            scores_name = input_names[1]
            scores_buf = graph.get_buffer(scores_name)
            _, num_classes = scores_buf.shape
            scores_const_name = scores_name + '_scores_const'
            scores_const_node = graph.add(op_adapter.ConstantOp(scores_const_name, np.arange(num_classes-1, dtype=np.int32)), [], [scores_const_name])
            graph.add_src_op_info(scores_const_node.op.name, [], [scores_const_name])
            scores_gather_name = scores_name + '_scores_gather'
            scores_gather_node = graph.add(op_adapter.GatherOp(scores_gather_name, axis=1), [scores_name, scores_const_name], [scores_gather_name])
            graph.add_src_op_info(scores_gather_node.op.name, [scores_name, scores_const_name], [scores_gather_name])

            # update op input names
            input_names[1] = scores_gather_name

        log_assert(params.output_classes_include_bg_cls == 0,
                   "BoxWithNmsLimit does not support output_classes_include_bg_cls.")

        log_assert(params.legacy_plus_one == 0,
                   "BoxWithNmsLimit does not support legacy_plus_one.")

        # rotated BoxWithNmsLimit is not supported
        log_assert(params.rotated == 0,
                   "BoxWithNmsLimit does not support rotated box.")

        if params.soft_nms_method not in self.SUPPORTED_NMS_KERNEL_METHODS:
            raise ValueError(
               "nms kernel method {} was not supported for BoxWithNmsLimit Op. Please choose from methods: {}".format(params.soft_nms_method, self.SUPPORTED_NMS_KERNEL_METHODS))
        nms_kernel_method = self.caffe2_to_ir_nms_kernel_method[params.soft_nms_method]
        if params.soft_nms_enabled == 0:
            nms_kernel_method = ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_HARD

        op =  op_adapter.BoxWithNmsLimitOp(str(src_op.name),
                                           nms_kernel_method=nms_kernel_method,
                                           nms_score_threshold=params.soft_nms_min_score_thres,
                                           score_threshold=params.score_thresh,
                                           pre_nms_limit=params.detections_per_im,
                                           iou_threshold=params.nms,
                                           sigma=params.soft_nms_sigma)

        # Caffe2 and QNN outputs are different
        #     Caffe2: [scores, boxes, classes, batch_splits, keeps, keeps_size]
        #     QNN: [boxes, scores, classes, batch_indices, batch_splits, keeps, keeps_size]
        caffe2_output_names = self.extract_output_names(src_op, converter_context)
        qnn_output_names = [caffe2_output_names[1], caffe2_output_names[0], caffe2_output_names[2], caffe2_output_names[1] + '_batch_indices'] + caffe2_output_names[3:]

        # handle keeps_size
        if len(caffe2_output_names) >= 6 and params.input_boxes_include_bg_cls == 0:
            keeps_size_name = caffe2_output_names[5]
            # add BoxWithNmsLimit node at first
            qnn_output_names[6] = keeps_size_name + '_without_background'
            node = graph.add(op, input_names, qnn_output_names)
            graph.add_src_op_info(node.op.name, input_names, qnn_output_names)

            # since input background is removed by QNN
            # so use Concat to add background back
            # prepare keeps_size with background
            keeps_size_const_name = keeps_size_name + '_const'
            keeps_size_const_node = graph.add(op_adapter.ConstantOp(keeps_size_const_name, np.array([0], dtype=np.int32)), [], [keeps_size_const_name])
            graph.add_src_op_info(keeps_size_const_node.op.name, [], [keeps_size_const_name])
            keeps_size_concat_node = graph.add(op_adapter.ConcatOp(keeps_size_name, axis=0), [keeps_size_const_name, qnn_output_names[6]], [keeps_size_name])
            graph.add_src_op_info(keeps_size_concat_node.op.name, [keeps_size_const_name, qnn_output_names[6]], [keeps_size_name])
        else:
            node = graph.add(op, input_names, qnn_output_names)
            graph.add_src_op_info(node.op.name, input_names, qnn_output_names)
        return None


OnnxTranslations.register_translation(OnnxBoxWithNmsLimitTranslation(),
                                      converter_type('BoxWithNmsLimit', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   CollectRpnProposals
# ------------------------------------------------------------------------------
class OnnxCollectRpnProposalsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('rpn_max_level', 'i', 6),
                                                ('rpn_min_level', 'i', 2),
                                                ('rpn_post_nms_topN', 'i', 2000)])

        if params.rpn_min_level < 2 or params.rpn_min_level > 6:
            raise ValueError("CollectRpnProposals Op {} only support parameter rpn_min_level in range [2, 6], got {}".format(
                    src_op.name, params.rpn_min_level))

        if params.rpn_max_level < 2 or params.rpn_max_level > 6:
            raise ValueError("CollectRpnProposals Op {} only support parameter rpn_max_level in range [2, 6], got {}".format(
                    src_op.name, params.rpn_max_level))

        if params.rpn_max_level < params.rpn_min_level:
            raise ValueError("CollectRpnProposals Op {} expected parameter rpn_max_level >= rpn_min_level, got rpn_max_level {} and rpn_min_level {}".format(
                    src_op.name, params.rpn_max_level, params.rpn_min_level))

        return op_adapter.CollectRpnProposalsOp(str(src_op.name),
                                                rpn_min_level=params.rpn_min_level,
                                                rpn_max_level=params.rpn_max_level,
                                                post_nms_top=params.rpn_post_nms_topN)


OnnxTranslations.register_translation(OnnxCollectRpnProposalsTranslation(),
                                      converter_type('CollectRpnProposals', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   DistributeFpnProposals
# ------------------------------------------------------------------------------
class OnnxDistributeFpnProposalsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('legacy_plus_one', 'i'),
                                                ('roi_canonical_level', 'i', 4),
                                                ('roi_canonical_scale', 'i', 224),
                                                ('roi_max_level', 'i', 5),
                                                ('roi_min_level', 'i', 2)])

        if params.legacy_plus_one != 0:
            raise ValueError(
                "The parameter 'legacy_plus_one' in DistributeFpnProposals op do not support non-zero values")

        if params.roi_min_level < 2 or params.roi_min_level > 5:
            raise ValueError(
                "The parameter 'roi_min_level' in DistributeFpnProposals op must be in range [2,5]")

        if params.roi_max_level < 2 or params.roi_max_level > 5:
            raise ValueError(
                "The parameter 'roi_max_level' in DistributeFpnProposals op must be in range [2,5]")

        if params.roi_max_level < params.roi_min_level:
            raise ValueError("The parameter 'roi_max_level' must be >= 'roi_min_level' in DistributeFpnProposals op")

        return op_adapter.DistributeFpnProposalsOp(str(src_op.name),
                                                   roi_canonical_level=params.roi_canonical_level,
                                                   roi_canonical_scale=params.roi_canonical_scale,
                                                   roi_max_level=params.roi_max_level,
                                                   roi_min_level=params.roi_min_level)

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        # change output order of Caffe2 to align with QNN opdef.
        output_names.insert(0, output_names.pop())
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node


OnnxTranslations.register_translation(OnnxDistributeFpnProposalsTranslation(),
                                      converter_type('DistributeFpnProposals', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   GenerateProposals
# ------------------------------------------------------------------------------
class OnnxGenerateProposalsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph

        # Caffe2 and QNN inputs are different
        #     Caffe2: [scores, deltas, im_info, anchors]
        #     QNN: [scores, deltas, anchors, im_info]
        input_names = self.extract_input_names(src_op, converter_context)
        input_names[2], input_names[3] = input_names[3], input_names[2]
        for input_name in input_names:
            constant_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if constant_op and not graph.has_buffer(input_name):
                constant_node = graph.add(constant_op, [], [input_name])
                graph.add_src_op_info(constant_node.op.name, [], [input_name])

        # check anchor box dimension
        # 4 means that box format is [x1, y1, x2, y2]
        # 5 means that box format is [ctr_x, ctr_y, w, h, angle], which is not supported by QNN
        anchor_name = input_names[2]
        anchor_buf = graph.get_buffer(anchor_name)
        _, box_dim = anchor_buf.shape
        log_assert(box_dim == 4,
                   "Rotated GenerateProposals is not supported.")

        # raise warning that QNN only support scale 1 in im_info
        log_warning("GenerateProposals only supports scale 1 in im_info.")

        # remove scale in im_info
        im_info_name = input_names[3]
        im_info_const_name = im_info_name + '_const'
        if not graph.has_buffer(im_info_const_name):
            im_info_const_node = graph.add(op_adapter.ConstantOp(im_info_const_name, np.array([0, 1], dtype=np.int32)), [], [im_info_const_name])
            graph.add_src_op_info(im_info_const_node.op.name, [], [im_info_const_name])
        im_info_gather_name = im_info_name + '_gather'
        if not graph.has_buffer(im_info_gather_name):
            im_info_gather_node = graph.add(op_adapter.GatherOp(im_info_gather_name, axis=1), [im_info_name, im_info_const_name], [im_info_gather_name])
            graph.add_src_op_info(im_info_gather_node.op.name, [im_info_name, im_info_const_name], [im_info_gather_name])
        input_names[3] = im_info_gather_name

        params = extract_attributes(src_op,
                                    attr_infos=[('spatial_scale', 'f', 0.0625),
                                                ('pre_nms_topN', 'i', 6000),
                                                ('post_nms_topN', 'i', 300),
                                                ('nms_thresh', 'f', 0.7),
                                                ('min_size', 'f', 16.0),
                                                ('angle_bound_on', 'i', 1),
                                                ('angle_bound_lo', 'i', -90),
                                                ('angle_bound_hi', 'i', 90),
                                                ('clip_angle_thresh', 'f', 1.0),
                                                ('legacy_plus_one', 'i')])

        log_assert(params.legacy_plus_one == 0,
                   "GenerateProposals does not support legacy_plus_one.")

        op =  op_adapter.GenerateProposalsOp(str(src_op.name),
                                             img_size_ratio=[1.0/params.spatial_scale, 1.0/params.spatial_scale],
                                             min_size=params.min_size,
                                             pre_nms_limit=params.pre_nms_topN,
                                             post_nms_limit=params.post_nms_topN,
                                             iou_threshold=params.nms_thresh,
                                             bbox_xform_clip=True)

        # Caffe2 and QNN outputs are different
        #     Caffe2: [rois, rois_probs]
        #     QNN: [scores, boxes, batch_indices]
        caffe2_output_names = self.extract_output_names(src_op, converter_context)
        qnn_output_names = [caffe2_output_names[1], caffe2_output_names[0] + '_boxes', caffe2_output_names[0] + '_batch_indices']

        node = graph.add(op, input_names, qnn_output_names)
        graph.add_src_op_info(node.op.name, input_names, qnn_output_names)

        # concat boxes and batch indices together to align with Caffe2 rois
        # example: batch indices (1000) -> rois (1000, 5)
        #          boxes (1000, 4)      ->
        num_boxes = min(params.pre_nms_topN, params.post_nms_topN)
        batch_indices_name = qnn_output_names[2]
        batch_indices_reshape_name = batch_indices_name + '_reshape'
        batch_indices_reshape_node = graph.add(op_adapter.ReshapeOp(batch_indices_reshape_name, shape=[num_boxes, 1]), [batch_indices_name], [batch_indices_reshape_name])
        graph.add_src_op_info(batch_indices_reshape_node.op.name, [batch_indices_name], [batch_indices_reshape_name])
        batch_indices_cast_name = batch_indices_name + '_cast'
        batch_indices_cast_node = graph.add(op_adapter.CastOp(batch_indices_cast_name, from_type='int32', to_type='float32'), [batch_indices_reshape_name], [batch_indices_cast_name])
        graph.add_src_op_info(batch_indices_cast_node.op.name, [batch_indices_reshape_name], [batch_indices_cast_name])
        rois_node = graph.add(op_adapter.ConcatOp(caffe2_output_names[0], axis=1), [batch_indices_cast_name, qnn_output_names[1]], [caffe2_output_names[0]])
        graph.add_src_op_info(rois_node.op.name, [batch_indices_cast_name, qnn_output_names[1]], [caffe2_output_names[0]])
        return None


OnnxTranslations.register_translation(OnnxGenerateProposalsTranslation(),
                                      converter_type('GenerateProposals', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   AliasWithName, CopyCPUToGPU, CopyGPUToCPU
# ------------------------------------------------------------------------------
class OnnxIdentityTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        if len(input_names) != len(output_names):
            raise RuntimeError("Identity only supports same number of inputs and outputs")
        return op_adapter.IdentityOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxIdentityTranslation(),
                                      converter_type('AliasWithName', 'onnx_caffe2'),
                                      converter_type('CopyCPUToGPU', 'onnx_caffe2'),
                                      converter_type('CopyGPUToCPU', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   ResizeNearest
# ------------------------------------------------------------------------------
class OnnxResizeNearestTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('height_scale', 'f', 1.0),
                                                ('width_scale', 'f', 1.0),
                                                ('order', 's', 'NCHW')])

        return op_adapter.ResizeOp(str(src_op.name),
                                   interpolation_mode=ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
                                   scale_height=params.height_scale,
                                   scale_width=params.width_scale)


OnnxTranslations.register_translation(OnnxResizeNearestTranslation(),
                                      converter_type('ResizeNearest', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   RoIAlign
# ------------------------------------------------------------------------------
class OnnxRoIAlignTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph

        input_names = self.extract_input_names(src_op, converter_context)
        for input_name in input_names:
            constant_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if constant_op and not graph.has_buffer(input_name):
                constant_node = graph.add(constant_op, [], [input_name])
                graph.add_src_op_info(constant_node.op.name, [], [input_name])

        rois_name = input_names[1]
        rois_buf = graph.get_buffer(rois_name)
        num_rois, roi_rank = rois_buf.shape
        # 5-tuple roi position of [batch, x1, y1, x2, y2]
        # batch will be extracted as third input
        if roi_rank == 5:
            # split rois into batch indices and roi positions
            # example: rois (10, 5) -> batch indices (10)
            #                       -> roi positions (10, 4)

            # prepare batch indices and roi positions
            batch_idx_name = rois_name + '_batch_indices'
            roi_pos_name = rois_name + '_roi_positions'
            rois_split_name = rois_name + '_split'
            rois_split_node = graph.add(op_adapter.SplitOp(rois_split_name, axis=1, split_index=[1]), [rois_name], [batch_idx_name, roi_pos_name])
            graph.add_src_op_info(rois_split_node.op.name, [rois_name], [batch_idx_name, roi_pos_name])

            # prepare batch indices reshape and cast
            batch_idx_reshape_name = batch_idx_name + '_reshape'
            batch_idx_reshape_node = graph.add(op_adapter.ReshapeOp(batch_idx_reshape_name, shape=[num_rois]), [batch_idx_name], [batch_idx_reshape_name])
            graph.add_src_op_info(batch_idx_reshape_node.op.name, [batch_idx_name], [batch_idx_reshape_name])
            batch_idx_cast_name = batch_idx_name + '_cast'
            batch_idx_cast_node = graph.add(op_adapter.CastOp(batch_idx_cast_name, from_type='float32', to_type='int32'), [batch_idx_reshape_name], [batch_idx_cast_name])
            graph.add_src_op_info(batch_idx_cast_node.op.name, [batch_idx_reshape_name], [batch_idx_cast_name])

            # update op input names
            input_names[1] = roi_pos_name
            input_names.append(batch_idx_cast_name)

        # 4-tuple roi position of [x1, y1, x2, y2]
        # only batch == 1 is supported, so add zero tensor as third input
        else:
            feature_name = input_names[0]
            feature_buf = graph.get_buffer(feature_name)
            batch = feature_buf.shape[0]
            log_assert(batch == 1,
                       "Input batch to RoIAlign is expected to be 1, but got {}".format(batch))
            batch_idx_const_name = rois_name + '_batch_indices'
            batch_idx_const_op = op_adapter.ConstantOp(batch_idx_const_name, np.zeros(num_rois, dtype=np.int32))
            batch_idx_const_node = graph.add(batch_idx_const_op, [], [batch_idx_const_name])
            graph.add_src_op_info(batch_idx_const_node.op.name, [], [batch_idx_const_name])
            input_names.append(batch_idx_const_name)

        params = extract_attributes(src_op,
                                    attr_infos=[('order', 's', 'NCHW'),
                                                ('spatial_scale', 'f', 1.0),
                                                ('pooled_h', 'i', 1),
                                                ('pooled_w', 'i', 1),
                                                ('sampling_ratio', 'i', -1),
                                                ('aligned', 'i', 0)])

        # zero spatial_scale is not supported
        log_assert(params.spatial_scale != 0.0,
                   "RoiAlign does not support zero spatial_scale.")

        op =  op_adapter.RoiAlignOp(str(src_op.name),
                                    spatial_scale=1.0/params.spatial_scale,
                                    pooled_size_h=params.pooled_h,
                                    pooled_size_w=params.pooled_w,
                                    sampling_ratio=params.sampling_ratio,
                                    aligned=bool(params.aligned),
                                    allow_invalid_roi=False)

        output_names = self.extract_output_names(src_op, converter_context)
        node = graph.add(op, input_names, output_names)
        graph.add_src_op_info(node.op.name, input_names, output_names)
        return None


OnnxTranslations.register_translation(OnnxRoIAlignTranslation(),
                                      converter_type('RoIAlign', 'onnx_caffe2'))

