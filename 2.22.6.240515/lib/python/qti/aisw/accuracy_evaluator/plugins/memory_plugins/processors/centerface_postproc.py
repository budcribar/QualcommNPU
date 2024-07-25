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

# MIT License

# Copyright (c) 2019 StarClouds

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Source: https://github.com/Star-Clouds/CenterFace/blob/master/prj-python/
# License: https://github.com/Star-Clouds/CenterFace

##############################################################################

import numpy as np
from PIL import Image
import os
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_postprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger

######################################################################
###  Filter the bounding boxes whose overlap is greater than the
###  thresold. Select only the bounding box with high score.
######################################################################
def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class centerface_postproc(qacc_memory_postprocessor):

    def __init__(self, dims, heatmap_threshold, nms_threshold, **kwargs):
        self.dims = dims.split(',')
        self.heatmap_threshold = heatmap_threshold
        self.nms_threshold = nms_threshold
        self.dataset = self.extra_params['dataset']
        self.input_file_list = []

        with open(self.dataset.get_input_list_file(), "r") as f:
            f = f.readlines()
            for line in f:
                self.input_file_list.append(line.strip())

    def execute(self, data, meta, input_idx, *args, **kwargs):
        """Processing the inference outputs.

        Function expects the outputs in the
        order: heatmap, scale, offset, lms
        """
        #Read and reshape all model outputs
        heatmap = data[0]
        scale = data[1]
        offset = data[2]
        lms = data[3]

        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > self.heatmap_threshold)

        orig_image = os.path.join(self.dataset.base_path, self.input_file_list[input_idx])
        image_name = os.path.basename(orig_image)
        image_src = Image.open(orig_image)
        image_shape = list(image_src.size)

        # Get input dimensions
        size = [int(self.dims[1]), int(self.dims[0])]  # height, width

        #Extract Pad-size, pad_h or pad_w
        #Larger dimension will have 0 padding
        #Other Axis will have size difference / 2
        #In case size is odd, extra pad line is added on the right and at the bottom
        #Pad-size calculated based on padding algo in image_operations.py
        img_scale = min(size[0] / image_shape[0], size[1] / image_shape[1])
        pad = [
            round(((size[0] - int(image_shape[0] * img_scale)) / 2) - 0.1),
            round(((size[1] - int(image_shape[1] * img_scale)) / 2) - 0.1)
        ]
        pad_w = pad[0]
        pad_h = pad[1]

        landmark = lms
        boxes, lms = [], []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0,
                             (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0,
                                                                   (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])

                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms.append(lm)

            boxes = np.asarray(boxes, dtype=np.float32)
            # keep = self.nms(boxes[:, :4], boxes[:, 4], self.nms_threshold)
            keep = nms(boxes, self.nms_threshold)
            boxes = boxes[keep, :]
            lms = np.asarray(lms, dtype=np.float32)
            lms = lms[keep, :]

        dets = boxes
        if len(dets) > 0:

            dets[:, 0:4:2], dets[:,
                                 1:4:2] = (dets[:, 0:4:2] - pad_w) / img_scale, (dets[:, 1:4:2] -
                                                                                 pad_h) / img_scale
            lms[:, 0:10:2], lms[:,
                                1:10:2] = (lms[:, 0:10:2] - pad_w) / img_scale, (lms[:, 1:10:2] -
                                                                                 pad_h) / img_scale

        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)

        bboxes = [[int(i[0]),
                   int(i[1]),
                   int(i[2]) - int(i[0]),
                   int(i[3]) - int(i[1]), i[4]] for i in boxes]

        name = orig_image[(orig_image[0:len(orig_image) - len(image_name) - 1]).rfind('/') + 1:]
        out_data = []
        out_data.append(name.split('/')[0])
        out_data.append(name.split('/')[1].split('.')[0])
        out_data.append(np.array(bboxes))
        return out_data, meta
