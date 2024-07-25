##############################################################################
#
# Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################

# BSD 3-Clause License
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Source: https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd
# License: https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/LICENSE

##############################################################################

from PIL import Image
import numpy as np
import os
import math
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_postprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class mlcommons_retinanet_postproc(qacc_memory_postprocessor):

    def __init__(self, dims, prior_filepath, score_threshold, nms_threshold, max_detections,
                 num_classes, fpn_features):
        torch = Helper.safe_import_package("torch")
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        dims = dims.split(',')
        self.dims = [int(dims[0]), int(dims[1])]  # width, height
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes
        self.fpn_features = fpn_features

        # Load prior boxes from the file.
        self.anchors = np.fromfile(prior_filepath, dtype=np.float32).reshape(-1, 4)
        self.anchors = torch.from_numpy(self.anchors)

        self.dataset = self.extra_params['dataset']
        self.input_file_list = []
        with open(self.dataset.get_input_list_file(), "r") as f:
            f = f.readlines()
            for line in f:
                self.input_file_list.append(line.strip())

    def execute(self, data, meta, input_idx, *args, **kwargs):
        torch = Helper.safe_import_package("torch")
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        orig_image = os.path.join(self.dataset.base_path, self.input_file_list[input_idx])
        image_name = os.path.basename(orig_image)
        image_src = Image.open(orig_image)
        image_shape = list(image_src.size)

        image_boxes = []
        image_scores = []
        image_labels = []
        num_anchors = 0
        for i in range(len(self.fpn_features)):
            boxes = data[i][0]
            scores = data[i + 5][0]
            topk_idxs = data[i + 10][0]

            boxes = torch.from_numpy(boxes)
            scores = torch.from_numpy(scores)
            topk_idxs = torch.from_numpy(topk_idxs)

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')

            keep_idxs = scores >= self.score_threshold

            boxes_per_level = boxes[keep_idxs]
            scores_per_level = scores[keep_idxs]
            labels_per_level = topk_idxs[keep_idxs] % self.num_classes
            anchor_idxs_per_level = anchor_idxs[keep_idxs]

            anchors_per_level = self.anchors[num_anchors:num_anchors + self.fpn_features[i]]
            num_anchors += self.fpn_features[i]
            boxes_per_level = self.bbox_transform_inv(boxes_per_level,
                                                      anchors_per_level[anchor_idxs_per_level.long()])
            boxes_per_level = self.clip_boxes(boxes_per_level, self.dims)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        keep_mask = torch.zeros_like(image_scores, dtype=torch.bool)
        for class_id in torch.unique(image_labels):
            curr_indices = torch.where(image_labels == class_id)[0]
            curr_keep_indices = torchvision.ops.nms(image_boxes[curr_indices],
                                                    image_scores[curr_indices], self.nms_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True

        keep_indices = torch.where(keep_mask)[0]
        keep = keep_indices[image_scores[keep_indices].sort(descending=True)[1]]
        keep = keep_indices[:self.max_detections]

        dets = image_boxes[keep]
        scores = image_scores[keep]
        scores = scores.numpy()

        labels = image_labels[keep]
        labels = labels.numpy()

        dets[:, 0:4:2], dets[:, 1:4:2] = (dets[:, 0:4:2] * image_shape[0]) / self.dims[0], (
            dets[:, 1:4:2] * image_shape[1]) / self.dims[1]
        final_output = [[
            float(labels[ind]),
            float(scores[ind]),
            float(j[0]),
            float(j[1]),
            float(j[2] - j[0]),
            float(j[3] - j[1])
        ] for ind, j in enumerate(dets)]

        entry = image_name
        entry += ',' + str(len(final_output))

        if (len(final_output) > 0):
            for i in final_output:
                entry += ',' + str(int(i[0]))
                entry += ',' + str(i[1])
                entry += ',' + str(i[2])
                entry += ',' + str(i[3])
                entry += ',' + str(i[4])
                entry += ',' + str(i[5])
        entry += '\n'

        out_data = []
        out_data.append(entry)

        return out_data, meta

    # boxes -- anchors, dets -- model output
    def bbox_transform_inv(self, dets, boxes):
        torch = Helper.safe_import_package("torch")
        bbox_xform_clip = math.log(1000. / 16)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = 1, 1, 1, 1
        dx = dets[:, 0::4] / wx
        dy = dets[:, 1::4] / wy
        dw = dets[:, 2::4] / ww
        dh = dets[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=bbox_xform_clip)
        dh = torch.clamp(dh, max=bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]

        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5) * pred_h
        c_to_c_w = torch.tensor(0.5) * pred_w

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4),
                                 dim=2).flatten(1)

        return pred_boxes

    ######################################################################
    # Clip the bounding boxes which exceed the image boundaries.
    ######################################################################
    def clip_boxes(self, boxes, im_shape):
        """Clip boxes to image boundaries.

        :param boxes: [N, 4* num_classes]
        :param im_shape: tuple of 2
        :return: [N, 4* num_classes]
        """
        torch = Helper.safe_import_package("torch")
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)

        boxes[:, 2] = torch.clamp(boxes[:, 2], max=im_shape[0])
        boxes[:, 3] = torch.clamp(boxes[:, 3], max=im_shape[1])

        return boxes
