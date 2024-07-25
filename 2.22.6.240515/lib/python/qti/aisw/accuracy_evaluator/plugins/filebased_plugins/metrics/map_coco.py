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
#
# Copyright (c) 2018-2021 cTuning foundation
#
# Copyright (c) cTuning foundation <admin@cTuning.org>
# All rights reserved

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the cTuning foundation
#       nor the names of its contributors may be used to endorse
#       or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##############################################################################
##############################################################################
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# License: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/LICENSE
# Source: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/ba50a82dce412df97f088c572d86d7977753bf74/lib/dataset/coco.py
###############################################################################

import builtins
import numpy as np
import os
import csv
import json
import pickle
from collections import defaultdict
from qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.keypoint_utils import oks_nms

orig_prin_fn = builtins.print
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc.plugin import pl_print

class_map = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
    28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    84, 85, 86, 87, 88, 89, 90
]


class file_row:

    def __init__(self, fpath, row):
        self.fpath = fpath
        self.row = row


###################################################################################################
# Bounding box rectangle specified by top-left point (x, y), width(w) and height (h) in pixels.
###################################################################################################


class box:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


###################################################################################################
# Ground truth element contains a class id, file id and a bounding box
###################################################################################################


class gt_box:

    def __init__(self, fid, cid, bbox):
        self.fid = fid
        self.cid = cid
        self.bbox = bbox


###################################################################################################
# detection result element contains a class id, score and a bounding box
###################################################################################################


class det_box:

    def __init__(self, fid, cid, score, bbox):
        self.fid = fid
        self.cid = cid
        self.score = score
        self.bbox = bbox


class map_coco(qacc_metric):

    def __init__(self):
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.6")

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):

        results_file = m_in.get_result_file()
        annotations_file = m_in.get_groundtruth()

        results_json_file = os.path.join(m_out.get_out_dir(), "coco_results.json")

        res_array = []
        processed_image_ids = []
        map_list = ['bbox']

        coco_mapping = m_in.get_param('map_80_to_90', False)
        seg_map = m_in.get_param('segm', default=False)
        keypoint_map = m_in.get_param('keypoint_map', default=False)
        if seg_map:
            map_list.append('segm')
        if keypoint_map:
            map_list.append('keypoints')
        mslice_indx = -1 if seg_map else None
        with open(results_file, 'r') as f:
            for line in f:
                frow = line.strip().split(',')
                row = frow[1:mslice_indx]
                filepath = os.path.basename(frow[0])
                short_name, ext = os.path.splitext(filepath)
                fid = int(short_name.split('_')[-1])
                processed_image_ids.append(fid)
                t = 0
                n = int(row[t])
                t += 1
                if seg_map:
                    maskfile_path = frow[-1]
                    image_mask = self.load_mask(maskfile_path)
                    assert len(image_mask) == n, "length of masks and detections should match"

                for i in range(n):
                    cid = int(row[t])
                    if coco_mapping:
                        cid = class_map[cid + 1]
                    t += 1
                    score = float(row[t])
                    t += 1
                    bbox = box(float(row[t]), float(row[t + 1]), float(row[t + 2]),
                               float(row[t + 3]))
                    t += 4
                    if seg_map:
                        mask = image_mask[i]
                        res = self.detection_to_coco_object(det_box(fid, cid, score, bbox),
                                                            mask=mask)
                    elif (keypoint_map):
                        res = np.concatenate([
                            np.array(row[t:t + 34], dtype=np.float32).reshape(-1, 2),
                            np.ones((17, 1), dtype=np.float32)
                        ], axis=1).reshape(51).tolist()
                        t += 34
                        res = self.detection_to_coco_object(det_box(fid, cid, score, bbox),
                                                            keypoints=res)
                    else:
                        res = self.detection_to_coco_object(det_box(fid, cid, score, bbox))
                    res_array.append(res)

        with open(results_json_file, 'w') as json_file:
            json_file.write(json.dumps(res_array, indent=6, sort_keys=False))

        results = {}
        for map_type in map_list:
            mAP, mAP_50, recall, all_metrics = self.evaluate(processed_image_ids, results_json_file,
                                                             annotations_file, map_type)
            results[map_type + '_mAP'] = mAP
            results[map_type + '_mAP_50'] = mAP_50

        m_out.set_result(results)
        if len(results) > 2:
            if (seg_map):
                m_out.set_result_str(
                    "bbox_map(50): {:.12f} \nbbox_map(50:05:95): {:.12f} \nseg_map(50): {:.12f} \nseg_map(50:05:95): {:.12f}"
                    .format(results['bbox_mAP_50'], results['bbox_mAP'], results['segm_mAP_50'],
                            results['segm_mAP']))
            elif (keypoint_map):
                m_out.set_result_str(
                    "bbox_map(50): {:.12f} \nbbox_map(50:05:95): {:.12f} \nkeypoints_map(50): {:.12f} \nkeypoints_map(50:05:95): {:.12f}"
                    .format(results['bbox_mAP_50'], results['bbox_mAP'],
                            results['keypoints_mAP_50'], results['keypoints_mAP']))
        else:
            m_out.set_result_str('mAP(50): {:.12f}\nmAP(50:05:95): {:.12f}'.format(
                results['bbox_mAP_50'], results['bbox_mAP']))

        m_out.set_status(qcc.STATUS_SUCCESS)

    def read_file(self, results_file):
        rows = []
        with open(results_file, 'r') as f:
            in_rows = csv.reader(f)
            for row in in_rows:
                fpath = row.pop(0).replace('\\', '/')
                rows.append(file_row(fpath, row))
        return (rows)

    def detection_to_coco_object(self, det, mask=None, keypoints=None):
        """Returns result object in COCO format."""
        det_box = det.bbox
        results = {
            "image_id": det.fid,
            "category_id": det.cid,
            "bbox": [det_box.x, det_box.y, det_box.w, det_box.h],
            "score": det.score,
        }
        if mask is not None:
            results["segmentation"] = mask
        if keypoints is not None:
            results["keypoints"] = keypoints

        return results

    @staticmethod
    def evaluate(image_ids_list, results_dir, annotations_file, map_type):
        '''
        Calculate COCO metrics via evaluator from pycocotool package.
        MSCOCO evaluation protocol: http://cocodataset.org/#detections-eval

        This method uses original COCO json-file annotations
        and results of detection converted into json file too.
        '''
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.6")
        builtins.print = pl_print
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        cocoGt = COCO(annotations_file)
        cocoDt = cocoGt.loadRes(results_dir)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=map_type)
        cocoEval.params.imgIds = image_ids_list
        cocoEval.params.recThrs = np.linspace(.0, 1.00,
                                              int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        if map_type == "keypoints":
            cocoEval.params.useSegm = None

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        builtins.print = orig_prin_fn

        if map_type == 'bbox' or map_type == 'segm':
            # These are the same names as object returned by CocoDetectionEvaluator has
            all_metrics = {
                "DetectionBoxes_Precision/mAP": cocoEval.stats[0],
                "DetectionBoxes_Precision/mAP@.50IOU": cocoEval.stats[1],
                "DetectionBoxes_Precision/mAP@.75IOU": cocoEval.stats[2],
                "DetectionBoxes_Precision/mAP (small)": cocoEval.stats[3],
                "DetectionBoxes_Precision/mAP (medium)": cocoEval.stats[4],
                "DetectionBoxes_Precision/mAP (large)": cocoEval.stats[5],
                "DetectionBoxes_Recall/AR@1": cocoEval.stats[6],
                "DetectionBoxes_Recall/AR@10": cocoEval.stats[7],
                "DetectionBoxes_Recall/AR@100": cocoEval.stats[8],
                "DetectionBoxes_Recall/AR@100 (small)": cocoEval.stats[9],
                "DetectionBoxes_Recall/AR@100 (medium)": cocoEval.stats[10],
                "DetectionBoxes_Recall/AR@100 (large)": cocoEval.stats[11]
            }

            mAP = all_metrics['DetectionBoxes_Precision/mAP']
            mAP_50 = all_metrics['DetectionBoxes_Precision/mAP@.50IOU']
            recall = all_metrics['DetectionBoxes_Recall/AR@100']

            return mAP, mAP_50, recall, all_metrics

        elif map_type == 'keypoints':

            metrics_list = [
                'AP', 'Ap(.5)', 'AP(.75)', 'AP(M)', 'AP(L)', 'AR', 'AR(.5)', 'AR(.75)', 'AR(M)',
                'AR(L)'
            ]
            res_info_str = ''
            for ind, name in enumerate(metrics_list):
                res_info_str += "{}: {:.12f}\n".format(name, cocoEval.stats[ind])

            return cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[5], res_info_str
        else:
            raise ValueError("Invalid maptype:{}".format(map_type))

    def load_mask(self, mask_file):
        with open(mask_file, 'rb') as mask_f:
            image_masks = pickle.load(mask_f)
            for mask in image_masks:
                mask['counts'] = mask['counts'].decode('utf-8')

        return image_masks
