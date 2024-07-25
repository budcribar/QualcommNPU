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
##############################################################################################################
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# LICENSE: https://github.com/facebookresearch/detectron2/blob/7801ac3d595a0663d16c0c9a6a339c77b2ddbdfb/LICENSE
# SOURCE: https://github.com/facebookresearch/detectron2/blob/7801ac3d595a0663d16c0c9a6a339c77b2ddbdfb/detectron2/layers/mask_ops.py#L17
###############################################################################################################

from typing import List
import os
from PIL import Image
import numpy as np
import pickle

from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class object_detection(qacc_plugin):
    """Used for Object-detection models like Yolo to process the raw files and
    generate the bounding boxes and scores."""
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
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.6")

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        """Processing the inference outputs.

        Function expects the outputs in the
        order: bboxes, scores, labels and mask. Order to be controlled via config.yaml
        """
        if not pin_list[0].is_path_input():
            print('object_detection is a path based plugin!')
            return
        torch = Helper.safe_import_package("torch")
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.6")
        bs = pin_list[0].read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)  #batchsize
        # Fetch dtypes from outputs info of config YAML
        infer_dtypes = [
            info[0] for info in list(pin_list[0].read_pipeline_cache_val(
                qcc.PIPELINE_INFER_OUTPUT_INFO).values())
        ]

        # Param for user to provide a list of dtypes
        inp_dtypes = pin_list[0].get_param('dtypes', infer_dtypes)
        if len(inp_dtypes) < 3:
            qacc_logger.error("Object_detection plugin expects atleast 3 entries of dtypes")
            pout_list[0].set_status(qcc.STATUS_ERROR)
            return

        bboxes = np.fromfile(pin_list[0].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[0]))

        scores = np.fromfile(pin_list[1].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[1]))

        labels = np.fromfile(pin_list[2].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[2]))

        resize_mask = pin_list[0].get_param('mask', default=False)
        padded_outputs = pin_list[0].get_param('padded_outputs', default=False)

        if len(pin_list) == 3:
            bboxes_t = bboxes.reshape(bs, -1, 4)
            scores_t = scores.reshape(bs, -1)
            labels_t = labels.reshape(bs, -1)

        else:
            counts = np.fromfile(pin_list[3].get_input(), dtype=Helper.get_np_dtype(inp_dtypes[3]))
            if resize_mask:
                mask_output = np.fromfile(pin_list[4].get_input(),
                                          dtype=Helper.get_np_dtype(inp_dtypes[4]))

                mdims = pin_list[0].get_param('mask_dims').split(',')
                mask_output = mask_output.reshape(bs, int(mdims[0]), int(mdims[1]), int(mdims[2]),
                                                  int(mdims[3]))
                mask_indices = np.arange(mask_output.shape[1])
                mask_prob = mask_output[:, mask_indices, labels][:, :, None]
                mask_prob = mask_prob.reshape(-1, 1, int(mdims[2]), int(mdims[3]))

            bboxes_t = []
            scores_t = []
            labels_t = []
            mask_prob_t = []
            if padded_outputs:
                bboxes = bboxes.reshape(bs, -1, 4)
                scores = scores.reshape(bs, -1)
                labels = labels.reshape(bs, -1)
                for i in range(bs):
                    bboxes_t.append(bboxes[i][:counts[i]])
                    scores_t.append(scores[i][:counts[i]])
                    labels_t.append(labels[i][:counts[i]])
            else:
                bboxes = bboxes.reshape(-1, 4)
                count = 0
                for i in range(bs):
                    bboxes_t.append(bboxes[count:count + int(counts[i])])
                    scores_t.append(scores[count:count + int(counts[i])])
                    labels_t.append(labels[count:count + int(counts[i])])
                    if resize_mask:
                        mask_prob_t.append(mask_prob[count:count + int(counts[i])])
                    count += int(counts[i])

        # Get detection results
        pout_list[0].set_output_extn('.txt')

        input_list = pin_list[0].get_orig_path_list(bs)
        num_inputs = len(input_list)
        for i in range(num_inputs):
            bboxes = bboxes_t[i]
            scores = scores_t[i]
            labels = labels_t[i]

            # Get input dimensions
            dims = pin_list[0].get_param('dims').split(',')
            height = int(dims[0])
            width = int(dims[1])

            # Assuming the 0th index to be the image path in the input record.
            orig_image = input_list[i][0]

            image_name = os.path.basename(orig_image)
            image_src = Image.open(orig_image)
            w, h = image_src.size

            # Convert bbox format from (x,y,width,height) to (x1,y1,x2,y2)
            xywh_to_xyxy = pin_list[0].get_param('xywh_to_xyxy', False)
            if xywh_to_xyxy:
                bboxes = self.xywh_to_xyxy(bboxes)

            # Swap XY coordinates of bbox
            xy_swap = pin_list[0].get_param('xy_swap', False)
            if xy_swap:
                bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

            label_offset = pin_list[0].get_param('label_offset', 0)
            labels -= label_offset

            # Score threshold
            score_threshold = pin_list[0].get_param('score_threshold', 0.001)

            # Scale x,y coordinates
            scale = str(pin_list[0].get_param('scale', default='1')).split(',')

            # skip padding while rescaling to original image shape
            skip_pad = pin_list[0].get_param('skip_padding', False)

            # Process  based on type.
            process_type = pin_list[0].get_param('type')

            if process_type == 'letterbox':
                bboxes = torch.from_numpy(np.array(bboxes))
                bboxes = self.postprocess_letterbox((height, width), bboxes, (h, w),
                                                    skip_pad=skip_pad)
            elif process_type == 'aspect_ratio':
                bboxes = torch.from_numpy(np.array(bboxes))
                bboxes = self.postprocess_letterbox((height, width), bboxes, (h, w), skip_pad=True)
            elif process_type == 'stretch':
                bboxes = self.postprocess_stretch((h, w), scale, bboxes)

            maskfile = None
            if resize_mask:
                pred_masks = self.resize_mask(mask_prob_t[i], bboxes, (h, w))
                #stores binary masks in RLE format
                #for more info link: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
                mask_util = Helper.safe_import_package("pycocotools.mask")
                mask_rles = [
                    mask_util.encode(
                        np.array(pmask[:, :, None], order="F", dtype="uint8"))[0]
                    for pmask in pred_masks
                ]

                maskfile = os.path.join(pout_list[0].get_out_dir(),
                                        str(int(image_name.split(".")[0])) + "_mask.pkl")
                with open(maskfile, "wb") as mf:
                    pickle.dump(mask_rles, mf)

            out_path = os.path.dirname(pout_list[0].get_output_path())
            updated_path = os.path.join(out_path, f'{str(int(image_name.split(".")[0]))}.txt')
            pout_list[0].set_path_output(updated_path)
            num_entry = self.det_result(bboxes, scores, labels, image_name, score_threshold,
                                        det_file=pout_list[0].get_output_path(), mask_file=maskfile)

        pout_list[0].set_status(qcc.STATUS_SUCCESS)
        # Disable the other two outputs.
        for pi in range(1, len(pout_list)):
            pout_list[pi].set_status(qcc.STATUS_REMOVE)

    def det_result(self, boxes, confs, labels, image_name, score_threshold, det_file,
                   mask_file=None):
        with open(det_file, 'w', newline='') as detf:
            prev_box_entry = ""
            num_entry = 0
            temp_line = ""
            for i, box in enumerate(boxes):
                if (confs[i] < score_threshold):
                    continue
                x1, y1, x2, y2 = map(np.float32, box)
                box_entry = ""
                box_entry += ',' + str(int(labels[i]))
                box_entry += ',' + str(confs[i].item())
                box_entry += ',' + str(x1.round(3)) + ',' + str(y1.round(3))
                box_entry += ',' + str((x2 - x1).round(3)) + ',' + str((y2 - y1).round(3))
                if box_entry != prev_box_entry:
                    temp_line += box_entry
                    num_entry += 1
                    prev_box_entry = box_entry
            if mask_file is not None:
                curr_line = "{},{}{},{} \n".format(image_name, num_entry, temp_line, mask_file)
            else:
                curr_line = "{},{}{} \n".format(image_name, num_entry, temp_line)
            detf.write(curr_line)
            detf.close()
        return num_entry

    def xywh_to_xyxy(self, bbox):
        torch = Helper.safe_import_package("torch")
        # Convert nx4 boxes format from [x,y,w,h] to [x1,y1,x2,y2] where xy1=top-left, xy2=bottom-right
        y = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
        y[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
        y[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
        y[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
        y[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
        return y

    def postprocess_letterbox(self, input_shape, coords, image_shape, skip_pad=False):
        # Rescale coords (xyxy) from input_shape to image_shape
        scale = min(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1])
        if not skip_pad:
            pad = [(input_shape[1] - int(image_shape[1] * scale)) / 2,
                   (input_shape[0] - int(image_shape[0] * scale)) / 2]
            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= scale

        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, 0].clamp_(0, image_shape[1])  # x1
        coords[:, 1].clamp_(0, image_shape[0])  # y1
        coords[:, 2].clamp_(0, image_shape[1])  # x2
        coords[:, 3].clamp_(0, image_shape[0])  # y2
        return coords

    def postprocess_stretch(self, image_shape, scale, coords):
        # Rescale coords based on image_shape (h, w)
        scale_x = scale_y = int(scale[0])
        if len(scale) == 2:
            scale_y = int(scale[1])
        coords[:, [0, 2]] *= image_shape[1]
        coords[:, [1, 3]] *= image_shape[0]
        coords[:, [0, 2]] /= scale_x
        coords[:, [1, 3]] /= scale_y

        return coords

    def resize_mask(self, mask_prob, bboxes, image_size, threshold=0.5):
        torch = Helper.safe_import_package("torch")
        #resize predicted masks to image_size(h,w)
        nc = bboxes.shape[0]
        image_mask = torch.zeros((nc, image_size[0], image_size[1]),
                                 dtype=torch.bool if threshold >= 0 else torch.uint8)
        for mid in range(nc):
            masks_patch, sptl_inds = object_detection.paste_masks_in_image(
                mask_prob[mid, None, :, :], bboxes[mid, None, :], image_size)
            if threshold >= 0:
                masks_patch = (masks_patch >= threshold).to(dtype=torch.bool)
            else:
                masks_patch = (masks_patch * 255).to(dtype=torch.uint8)
            image_mask[(mid, ) + sptl_inds] = masks_patch

        return image_mask

    @staticmethod
    def paste_masks_in_image(masks, bboxes, image_size):
        """
        Paste mask of a fixed resolution (e.g., 28 x 28) into an image.
        The location, height, and width for pasting each mask is determined by their corresponding bounding boxes in bboxes
        """
        torch = Helper.safe_import_package("torch")
        bboxes = torch.from_numpy(np.array(bboxes))
        masks = torch.from_numpy(masks)
        box_x0, box_y0, box_x1, box_y1 = torch.split(bboxes, 1, dim=1)
        coord_y = torch.arange(0, image_size[0], device='cpu', dtype=torch.float32) + 0.5
        coord_x = torch.arange(0, image_size[1], device='cpu', dtype=torch.float32) + 0.5
        coord_y = (coord_y - box_y0) / (
            box_y1 -
            box_y0) * 2 - 1  #normalize coordinates and shift it into [-1 , 1], shape (N, y)
        coord_x = (coord_x - box_x0) / (
            box_x1 -
            box_x0) * 2 - 1  #normalize coordinates and shift it into [-1 , 1], shape (N, x)

        gx = coord_x[:, None, :].expand(masks.shape[0], coord_y.size(1),
                                        coord_x.size(1))  #shape (N, y, w)
        gy = coord_y[:, :, None].expand(masks.shape[0], coord_y.size(1),
                                        coord_x.size(1))  #shape (N, y, w)
        grid_xy = torch.stack([gx, gy], dim=3)  #grid of xy coordinates
        image_mask = torch.nn.functional.grid_sample(masks, grid_xy.to(
            masks.dtype), align_corners=False)  #resize mask to image shape

        return image_mask[:, 0], (slice(0, image_size[0]), slice(0, image_size[1]))
