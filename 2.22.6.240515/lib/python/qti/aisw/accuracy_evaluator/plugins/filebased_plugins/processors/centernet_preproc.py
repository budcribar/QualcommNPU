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

from typing import List
import cv2
import numpy as np
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class centernet_preproc(qacc_plugin):

    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_PATH,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_CV2
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        image = cv2.imread(pin.get_input())
        scale = pin.get_param('scale', 1.0)
        fix_res = pin.get_param('fix_res', True)
        pad = pin.get_param('pad', 0)

        dims = pin.get_param('dims').split(',')

        input_h, input_w = int(dims[0]), int(dims[1])  # height, width

        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        if fix_res:
            inp_height, inp_width = input_h, input_w
            c = np.array([new_width / 2.0, new_height / 2.0], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | pad) + 1
            inp_width = (new_width | pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = self.get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        img = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height),
                             flags=cv2.INTER_LINEAR)

        if not img is None:
            if pout.is_memory_output():
                pout.set_mem_output(img)
            else:
                img.tofile(pout.get_output_path())
            pout.set_status(qcc.STATUS_SUCCESS)

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
