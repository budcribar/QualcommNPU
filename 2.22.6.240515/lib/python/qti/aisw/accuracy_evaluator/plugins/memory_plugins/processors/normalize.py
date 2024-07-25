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

import numpy as np
from PIL import Image
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_preprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class normalize(qacc_memory_preprocessor):
    """Normalize the given image data with the given configuration and returns
    the normalized image."""
    supported_libraries = ['numpy', 'torchvision']

    def execute(self, data, meta, input_idx, library='numpy', means={
        "R": 0,
        "G": 0,
        "B": 0
    }, std={
        "R": 1,
        "G": 1,
        "B": 1
    }, channel_order='RGB', norm=255.0, normalize_first=True, pil_to_tensor_input=True,
                typecasting_required=True, **kwargs):
        if library not in normalize.supported_libraries:
            raise ValueError('normalize plugin does not support library ' + library)
        out_data = []
        for inp in data:
            if channel_order == "BGR":
                mean = [means["B"], means["G"], means["R"]]
                std = [std["B"], std["G"], std["R"]]
            else:
                mean = [means["R"], means["G"], means["B"]]
                std = [std["R"], std["G"], std["B"]]

            if library == 'numpy':
                img = normalize.norm_numpy(inp, mean, std, norm, normalize_first)
            elif library == 'torchvision':
                img = normalize.norm_tv(inp, mean, std, pil_to_tensor_input, typecasting_required)
            out_data.append(img.astype(np.float32))
        return out_data, meta

    def norm_numpy(inp, means, std, norm=255.0, normalize_first=True):
        if not (list(np.shape(inp))[-1] == 3):
            qacc_file_logger.error(
                'Normalization must be applied with the data in NHWC format. Check config file.')
            return

        means = np.array(means, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        if normalize_first:
            inp = np.true_divide(inp, norm)
        # Subtract means and Scale by STD
        inp = (inp - means) / std
        if not normalize_first:
            inp = np.true_divide(inp, norm)
        return inp

    @staticmethod
    def norm_tv(inp, means, std, pil_to_tensor_input=True, typecasting_required=True):
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        if not isinstance(inp, Image.Image):  # To check if the input is valid PIL image or not
            qacc_file_logger.error(
                'This version(0.7.0) of torchvision supports only valid PIL images as input')
            return
        if pil_to_tensor_input:
            inp = torchvision.transforms.functional.to_tensor(inp)
        inp = torchvision.transforms.functional.normalize(inp, mean=means, std=std)
        if typecasting_required:
            inp = inp.numpy()
        return inp
