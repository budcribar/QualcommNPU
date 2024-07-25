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

import cv2
from PIL import Image
import numpy as np

from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_preprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class mlcommons_retinanet_preproc(qacc_memory_preprocessor):

    def __init__(self):
        torch = Helper.safe_import_package("torch")
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")

    def execute(self, data, meta, input_idx, *args, **kwargs):
        torch = Helper.safe_import_package("torch")
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        image = Image.open(data[0]).convert('RGB')
        image = torchvision.transforms.functional.to_tensor(image)
        image = self.normalize(image)
        image = torch.nn.functional.interpolate(image[None], size=(800, 800), scale_factor=None,
                                                mode='bilinear', recompute_scale_factor=None,
                                                align_corners=False)[0]
        out_data = [image.numpy()]
        return out_data, meta

    def normalize(self, image):
        torch = Helper.safe_import_package("torch")
        if not image.is_floating_point():
            raise TypeError(f"Expected input images to be of floating type (in range [0, 1]), "
                            f"but found type {image.dtype} instead")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        return (image - mean[:, None, None]) / std[:, None, None]
