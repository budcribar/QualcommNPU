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

# Copyright (c) 2019

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

##############################################################################
# WiderFace evaluation code
# author: wondervictor
# mail: tianhengcheng@gmail.com
# copyright@wondervictor
#
# License : https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT
# Source : https://github.com/biubug6/Pytorch_Retinaface/blob/master/widerface_evaluate/evaluation.py
#
##############################################################################

import numpy as np
import os
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class widerface_AP(qacc_metric):

    ## The plugin expects ground truth mat files, Below are the mat files the metric plugin expects
    ## wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat

    def __init__(self):
        scipy = Helper.safe_import_package("scipy")

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):

        results_file = m_in.get_result_file()
        gt_path = m_in.get_groundtruth()
        iou_thresh = m_in.get_param('IoU_threshold', 0.4)

        pred = get_preds(results_file)
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(
            gt_path)
        event_num = len(event_list)
        thresh_num = 1000
        setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        aps = []
        for _idx, gt_list in enumerate(setting_gts):
            # different setting
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype('float')
            # [hard, medium, easy]

            for i in range(event_num):
                event_name = str(event_list[i][0][0])
                if event_name not in pred:
                    continue
                img_list = file_list[i][0]
                pred_list = pred[event_name]
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = facebox_list[i][0]

                for j in range(len(img_list)):
                    if str(img_list[j][0][0]) not in pred_list:
                        continue
                    pred_info = pred_list[str(img_list[j][0][0])]
                    gt_boxes = gt_bbx_list[j][0].astype('float')
                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)

                    if len(gt_boxes) == 0 or len(pred_info) == 0:
                        continue
                    ignore = np.zeros(gt_boxes.shape[0])
                    if len(keep_index) != 0:
                        ignore[keep_index - 1] = 1
                    pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                    _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                    pr_curve += _img_pr_info

            pr_curve = dataset_pr_info(
                thresh_num, pr_curve,
                count_face)  # compute the pr curve for easy, medium and hard cases.

            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]
            ap = voc_ap(recall, propose)  # compute the avrage precision under the curve.
            aps.append(ap)

        print("==================== Results ====================")
        print("Easy   Val AP: {}".format(aps[0]))
        print("Medium Val AP: {}".format(aps[1]))
        print("Hard   Val AP: {}".format(aps[2]))
        print("=================================================")

        m_out.set_result({
            'Easy   Val AP': aps[0],
            'Medium Val AP': aps[1],
            'Hard   Val AP': aps[2]
        })

        m_out.set_status(qcc.STATUS_SUCCESS)


######################################################################
###  Reading the events, hard, medium and easy ground truth boxes
######################################################################
def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""
    scipy = Helper.safe_import_package("scipy")
    gt_mat = scipy.io.loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = scipy.io.loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = scipy.io.loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = scipy.io.loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


#########################################################################
### Reading the predicted bounding boxes from the model,
### organizing the predictions in dictionary event wise aas shown below
###
### boxes dict --| eve0  -- image predicitons
###            --| eve1  -- image predictions
###            --| ...
###            --| eve61 -- image predictions
#########################################################################
def get_preds(pred_file):

    boxes = dict()
    with open(pred_file, 'r') as fp:
        prev_event = None
        curr_event_dict = dict()
        while (1):
            imgname = fp.readline().strip()  # read image name
            if not imgname:
                boxes[prev_event] = curr_event_dict
                break  # end of file
            curr_event = imgname.split("/")[0]  # get the event name
            if (prev_event is None):
                prev_event = curr_event
            elif (prev_event
                  != curr_event):  # if current event is different the create new dict element.
                boxes[prev_event] = curr_event_dict
                prev_event = curr_event
                curr_event_dict = dict()

            nboxes = int(fp.readline().strip())  # no.of bboxes for the image
            box = np.zeros((nboxes, 5), dtype=float)
            for i in range(nboxes):
                line = fp.readline().strip().split(" ")
                for j in range(len(line)):
                    box[i][j] = float(line[j])

            curr_event_dict[imgname.split("/")[1].split(".")[0]] = box
    return boxes


######################################################################
###  Compute the precision and recall for an image.
######################################################################
def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


######################################################################
###  Compute the pr curve for single image
######################################################################
def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            continue
        r_index = r_index[-1]
        p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
        pr_info[t, 0] = len(p_index)
        pr_info[t, 1] = pred_recall[r_index]
    return pr_info


######################################################################
###  Compute the precision-recall curve for easy/medium/hard cases
######################################################################
def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


######################################################################
###  Compute the box overlap between groundtruth and predicted boxes.
######################################################################
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)
            if iw <= 0 or ih <= 0:
                continue
            ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) +
                       box_area - iw * ih)
            overlaps[n, k] = iw * ih / ua
    return overlaps


######################################################################
###  Compute the average precision under the curve.
######################################################################
def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
