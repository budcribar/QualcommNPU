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
##############################################################################################################
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# LICENSE: https://github.com/facebookresearch/detectron2/blob/7801ac3d595a0663d16c0c9a6a339c77b2ddbdfb/LICENSE
# SOURCE: https://github.com/facebookresearch/detectron2/blob/7801ac3d595a0663d16c0c9a6a339c77b2ddbdfb/detectron2/layers/mask_ops.py#L17
###############################################################################################################
import os
from PIL import Image
import numpy as np
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_postprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class object_detection(qacc_memory_postprocessor):
    """Used for Object-detection models like Yolo to process the raw files and
    generate the bounding boxes and scores."""

    def __init__(self, dims=None, dtypes=None, xywh_to_xyxy=False, type='stretch', mask_dims=None,
                 xy_swap=False, label_offset=0, score_threshold=0.001, skip_padding=False,
                 scale='1', mask=False, padded_outputs=False):
        torch = Helper.safe_import_package("torch")
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.6")
        self.dataset = self.extra_params['dataset']
        self.input_file_list = []

        with open(self.dataset.get_input_list_file(), "r") as f:
            f = f.readlines()
            for line in f:
                self.input_file_list.append(line.strip())

        self.dims = dims.split(',')
        self.rect_to_dims = xywh_to_xyxy
        self.xy_swap = xy_swap
        self.label_offset = label_offset
        self.score_threshold = score_threshold
        # Process  based on type (letterbox, aspect-ratio, stretch)
        self.process_type = type
        # Scale x,y coordinates
        self.scale = str(scale).split(',')
        # skip padding while rescaling to original image shape
        self.skip_pad = skip_padding
        # self.inp_dtypes = dtypes # this can come from user in case of partitioned models and from the model in case of full models
        self.inp_dtypes = ['float32', 'float32', 'int32', 'int32']

        if mask_dims:
            self.mdims = mask_dims.split(',')

        self.mask = mask
        self.padded_outputs = padded_outputs

    def execute(self, data, meta, input_idx, *args, **kwargs):
        """Processing the inference outputs.

        Function expects the outputs in the
        order: bboxes, scores, labels and mask. Order to be controlled via config.yaml
        """
        torch = Helper.safe_import_package("torch")
        pycocotools = Helper.safe_import_package("pycocotools", "2.0.6")
        bboxes = data[0]
        scores = data[1]
        labels = data[2]

        if len(data) == 3:
            bboxes = bboxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

        else:
            counts = data[3][0]
            if self.mask:
                mask_output = data[4]  #data[3] corresponds to count
                mask_output = mask_output.reshape(1, int(self.mdims[0]), int(self.mdims[1]),
                                                  int(self.mdims[2]), int(self.mdims[3]))
                mask_indices = np.arange(mask_output.shape[1])
                mask_prob = mask_output[:, mask_indices, labels][:, :, None]
                mask_prob = mask_prob.reshape(-1, 1, int(self.mdims[2]), int(self.mdims[3]))

            bboxes = bboxes.reshape(-1, 4)[:counts]
            scores = scores.reshape(-1)[:counts]
            labels = labels.reshape(-1)[:counts]

        height = int(self.dims[0])
        width = int(self.dims[1])

        orig_image = os.path.join(self.dataset.base_path, self.input_file_list[input_idx])
        image_name = os.path.basename(orig_image)
        image_src = Image.open(orig_image)
        w, h = image_src.size

        # Convert bbox format from (x,y,width,height) to (x1,y1,x2,y2)
        if self.rect_to_dims:
            bboxes = self.xywh_to_xyxy(bboxes)

        # Swap XY coordinates of bbox
        if self.xy_swap:
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

        labels -= self.label_offset

        if self.process_type == 'letterbox':
            bboxes = torch.from_numpy(np.array(bboxes))
            bboxes = self.postprocess_letterbox((height, width), bboxes, (h, w),
                                                skip_pad=self.skip_pad)
        elif self.process_type == 'aspect_ratio':
            bboxes = torch.from_numpy(np.array(bboxes))
            bboxes = self.postprocess_letterbox((height, width), bboxes, (h, w), skip_pad=True)
        elif self.process_type == 'stretch':
            bboxes = self.postprocess_stretch((h, w), self.scale, bboxes)
        elif self.process_type == 'orgimage':
            bboxes = torch.from_numpy(np.array(bboxes))
            bboxes = self.postprocess_orgimage((height, width), bboxes, (h, w),
                                               skip_pad=self.skip_pad)
        out_data = []
        entry = self.det_result(bboxes, scores, labels, image_name, self.score_threshold)
        out_data.append(entry)

        if self.mask:
            pred_masks = self.resize_mask(mask_prob, bboxes, (h, w))
            #stores binary masks in RLE format
            #for more info link: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
            mask_util = Helper.safe_import_package("pycocotools.mask")
            mask_rles = [
                mask_util.encode(
                    np.array(pmask[:, :, None], order="F", dtype="uint8"))[0]
                for pmask in pred_masks
            ]
            out_data.append(mask_rles)
        return out_data, meta

    def det_result(self, boxes, confs, labels, image_name, score_threshold):
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

        curr_line = "{},{}{} \n".format(image_name, num_entry, temp_line)

        return curr_line

    def xywh_to_xyxy(self, bbox):
        # Convert nx4 boxes format from [x,y,w,h] to [x1,y1,x2,y2] where xy1=top-left, xy2=bottom-right
        torch = Helper.safe_import_package("torch")
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

    def postprocess_orgimage(self, input_shape, coords, image_shape, skip_pad=False):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, 0].clamp_(0, input_shape[1])  # x1
        coords[:, 1].clamp_(0, input_shape[0])  # y1
        coords[:, 2].clamp_(0, input_shape[1])  # x2
        coords[:, 3].clamp_(0, input_shape[0])  # y2

        coords[:, 0:4:2], coords[:, 1:4:2] = (coords[:, 0:4:2] * image_shape[1]) / input_shape[1], (
            coords[:, 1:4:2] * image_shape[0]) / input_shape[0]

        return coords

    def resize_mask(self, mask_prob, bboxes, image_size, threshold=0.5):
        #resize predicted masks to image_size(h,w)
        torch = Helper.safe_import_package("torch")
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
        """Paste mask of a fixed resolution (e.g., 28 x 28) into an image.

        The location, height, and width for pasting each mask is
        determined by their corresponding bounding boxes in bboxes
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
