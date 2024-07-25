##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
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

from typing import List
from PIL import Image
import os
import numpy as np
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class centerface_postproc(qacc_plugin):

    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_PATH,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_PATH,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        """Processing the inference outputs.

        Function expects the outputs in the
        order:
        """
        if not pin_list[0].is_path_input():
            print('centerface_postproc is a path based plugin!')
            return

        # Fetch dtypes from outputs info of config YAML
        infer_dtypes = [
            info[0] for info in list(pin_list[0].read_pipeline_cache_val(
                qcc.PIPELINE_INFER_OUTPUT_INFO).values())
        ]
        # Param for user to provide a list of dtypes
        inp_dtypes = pin_list[0].get_param('dtypes', infer_dtypes)
        heatmap_threshold = pin_list[0].get_param('heatmap_threshold', 0.05)
        nms_threshold = pin_list[0].get_param('nms_threshold', 0.3)
        dims = pin_list[0].get_param('dims').split(',')

        #Read and reshape all model outputs
        heatmap = np.fromfile(pin_list[0].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[0]))
        heatmap = heatmap.reshape(1, 1, 200, 200)
        scale = np.fromfile(pin_list[1].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[1]))
        scale = scale.reshape(1, 2, 200, 200)
        offset = np.fromfile(pin_list[2].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[2]))
        offset = offset.reshape(1, 2, 200, 200)
        lms = np.fromfile(pin_list[3].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[3]))
        lms = lms.reshape(1, 10, 200, 200)

        heatmap = np.squeeze(heatmap)

        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > heatmap_threshold)

        input_list = pin_list[0].get_orig_path_list(1)
        # Assuming the 0th index to be the image path in the input record.
        orig_image = input_list[0][0]
        image_name = os.path.basename(orig_image)
        image_src = Image.open(orig_image)
        image_shape = list(image_src.size)

        # Get input dimensions

        size = [int(dims[1]), int(dims[0])]  # height, width

        #Extract Pad-size, pad_h or pad_w
        #Larger dimension will have 0 padding
        #Other Axis will have size differnece / 2

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
            keep = self.nms(boxes[:, :4], boxes[:, 4], nms_threshold)
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

        pout_list[0].set_output_extn('.txt')
        out_dir = pout_list[0].get_out_dir()
        in_dir_name = os.path.basename(os.path.dirname(pin_list[0].get_input()))
        # Update output file name only when inputs are in 'Result_id' folders
        if in_dir_name.startswith('Result'):
            out_inx = in_dir_name.split('_')[1]
            out_file = os.path.basename(
                pin_list[0].get_input()).split('.')[0] + '_' + out_inx + '.txt'
            updated_path = os.path.join(out_dir, out_file)
            pout_list[0].set_path_output(updated_path)
        det_file = pout_list[0].get_output_path()
        with open(det_file, 'w') as f:
            f.write(orig_image[(orig_image[0:len(orig_image) - len(image_name) - 1]).rfind('/') +
                               1:])
            f.write('\n')
            f.write(str(len(bboxes)))
            f.write('\n')
            for i in bboxes:
                f.write(' '.join(str(e) for e in i))
                f.write('\n')

        pout_list[0].set_status(qcc.STATUS_SUCCESS)
        # Disable the other two outputs.
        for pi in range(1, len(pout_list)):
            pout_list[pi].set_status(qcc.STATUS_REMOVE)

    #NMS Supress bboxes if overlap > 30%, keep bboxes with higher confidence score
    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections, ), dtype=bool)
        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)
            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]
            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue
                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True
        return keep
