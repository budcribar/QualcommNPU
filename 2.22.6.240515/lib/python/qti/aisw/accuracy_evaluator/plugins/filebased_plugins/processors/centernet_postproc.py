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

# Copyright (c) 2019 Xingyi Zhou
# All rights reserved.

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

# Source: https://github.com/xingyizhou/CenterNet#object-detection-on-coco-validation
# License: https://github.com/xingyizhou/CenterNet/blob/master/LICENSE

##############################################################################

import os
from typing import List
import numpy as np
import cv2

from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class centernet_postprocess(qacc_plugin):

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

    def __init__(self):
        torch = Helper.safe_import_package("torch")

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        """Processing the inference outputs.

        Function expects the outputs in the
        order:
        """
        torch = Helper.safe_import_package("torch")
        if not pin_list[0].is_path_input():
            print('centernet_postprocess is a path based plugin!')
            return

        # Fetch dtypes from outputs info of config YAML
        infer_dtypes = [
            info[0] for info in list(pin_list[0].read_pipeline_cache_val(
                qcc.PIPELINE_INFER_OUTPUT_INFO).values())
        ]

        # Param for user to provide a list of dtypes
        inp_dtypes = pin_list[0].get_param('dtypes', infer_dtypes)
        if len(inp_dtypes) < 3:
            qacc_logger.error("centernet_postprocess plugin expects atleast 3 entries of dtypes")
            pout_list[0].set_status(qcc.STATUS_ERROR)
            return

        ########## Get the fpn scales from the config yaml file discuss with the team ##########
        top_k = pin_list[0].get_param('top_k', 100)
        output_dims = pin_list[0].get_param('output_dims').split(',')
        num_classes = int(pin_list[0].get_param('num_classes', 1))
        score = float(pin_list[0].get_param('score', 1))

        out_height, out_width = int(output_dims[0]), int(
            output_dims[1])  # output height, output width

        input_list = pin_list[0].get_orig_path_list(1)
        # Assuming the 0th index to be the image path in the input record.
        orig_image = input_list[0][0]
        image_name = os.path.basename(orig_image)
        image = cv2.imread(orig_image)
        height, width = image.shape[0:2]
        new_height = int(height)
        new_width = int(width)

        c = np.array([new_width / 2.0, new_height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0
        scale = 1.0

        output = {}
        node_names = ["hm", "wh", "hps", "reg", "hm_hp", "hp_offset"]

        for idx, node_name in enumerate(node_names):
            output[node_name] = np.fromfile(pin_list[idx].get_input(),
                                            dtype=Helper.get_np_dtype(inp_dtypes[0]))

        output[node_names[0]] = torch.from_numpy(output[node_names[0]].reshape(
            1, 1, out_height, out_width))
        output[node_names[1]] = torch.from_numpy(output[node_names[1]].reshape(
            1, 2, out_height, out_width))
        output[node_names[2]] = torch.from_numpy(output[node_names[2]].reshape(
            1, 34, out_height, out_width))
        output[node_names[3]] = torch.from_numpy(output[node_names[3]].reshape(
            1, 2, out_height, out_width))
        output[node_names[4]] = torch.from_numpy(output[node_names[4]].reshape(
            1, 17, out_height, out_width))
        output[node_names[5]] = torch.from_numpy(output[node_names[5]].reshape(
            1, 2, out_height, out_width))

        output["hm"] = output["hm"].sigmoid_()
        output["hm_hp"] = output["hm_hp"].sigmoid_()

        reg = output["reg"]
        hm_hp = output["hm_hp"]
        hp_offset = output["hp_offset"]

        dets = self.multi_pose_decode(
            output["hm"],
            output["wh"],
            output["hps"],
            reg=reg,
            hm_hp=hm_hp,
            hp_offset=hp_offset,
            K=top_k,
        )

        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = self.multi_pose_post_process(dets.copy(), c, s, out_height, out_width)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale

        category_id = 1
        dets = dets[0][category_id]

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

        # write the detection to the output file

        detections = []
        for det in dets:
            if det[4] > score:
                detections.append(det)

        with open(det_file, 'w') as f:
            f.write(image_name.split(".")[0])
            f.write(',')
            f.write(str(len(detections)))

            for det in detections:
                f.write(',')
                bbox = det[:4]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                det_str = str(category_id)+","+str(det[4])+","+str(bbox[0]) + \
                    ","+str(bbox[1])+","+str(bbox[2])+","+str(bbox[3])
                f.write(det_str)
                for i in range(len(det[5:len(det)])):
                    f.write("," + str(det[i + 5]))

            f.write('\n')

        pout_list[0].set_status(qcc.STATUS_SUCCESS)
        # Disable the other two outputs.
        for i in range(1, len(pout_list)):
            pout_list[i].set_status(qcc.STATUS_REMOVE)

    def multi_pose_decode(self, heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
        torch = Helper.safe_import_package("torch")
        batch, cat, height, width = heat.size()
        num_joints = kps.shape[1] // 2
        # perform nms on heatmaps
        heat = self.nms(heat)
        scores, inds, clses, ys, xs = self.topk(heat, K=K)

        kps = self.transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        if reg is not None:
            reg = self.transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = self.transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        bboxes = torch.cat(
            [
                xs - wh[..., 0:1] / 2,
                ys - wh[..., 1:2] / 2,
                xs + wh[..., 0:1] / 2,
                ys + wh[..., 1:2] / 2,
            ],
            dim=2,
        )
        if hm_hp is not None:
            hm_hp = self.nms(hm_hp)
            thresh = 0.1
            kps = (kps.view(batch, K, num_joints, 2).permute(0, 2, 1,
                                                             3).contiguous())  # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
            hm_score, hm_inds, hm_ys, hm_xs = self.topk_channel(hm_hp, K=K)  # b x J x K
            if hp_offset is not None:
                hp_offset = self.transpose_and_gather_feat(hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = (torch.stack([hm_xs, hm_ys],
                                  dim=-1).unsqueeze(2).expand(batch, num_joints, K, K, 2))
            dist = ((reg_kps - hm_kps)**2).sum(dim=4)**0.5
            min_dist, min_ind = dist.min(dim=3)  # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = ((hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + (hm_kps[..., 1:2] < t) +
                    (hm_kps[..., 1:2] > b) + (hm_score < thresh) +
                    (min_dist > (torch.max(b - t, r - l) * 0.3)))
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, K, num_joints * 2)
        detections = torch.cat([bboxes, scores, kps, clses], dim=2)

        return detections

    def multi_pose_post_process(self, dets, c, s, h, w):
        # dets: batch x max_dets x 40
        # return list of 39 in image coord
        ret = []
        for i in range(dets.shape[0]):
            bbox = self.transform_preds(dets[i, :, :4].reshape(-1, 2), c, s, (w, h))
            pts = self.transform_preds(dets[i, :, 5:39].reshape(-1, 2), c, s, (w, h))
            top_preds = (np.concatenate([bbox.reshape(-1, 4), dets[i, :, 4:5],
                                         pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist())
            ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
        return ret

    def transform_preds(self, coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = self.get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self.affine_transform(coords[p, 0:2], trans)
        return target_coords

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def get_affine_transform(self, center, scale, rot, output_size,
                             shift=np.array([0, 0], dtype=np.float32), inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def nms(self, heat, kernel=3):
        torch = Helper.safe_import_package("torch")
        pad = (kernel - 1) // 2

        hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()

        return heat * keep

    def topk(self, scores, K=40):
        torch = Helper.safe_import_package("torch")
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self.gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self.gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self.gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self.gather_feat(feat, ind)
        return feat

    def topk_channel(self, scores, K=40):
        torch = Helper.safe_import_package("torch")
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs
