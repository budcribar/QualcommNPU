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
import numpy as np
from PIL import Image
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class normalize(qacc_plugin):
    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    supported_libraries = ['numpy', 'torchvision']

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        library_name = pin.get_param('library', 'numpy')
        if library_name not in self.supported_libraries:
            print('normalize plugin does not support library ' + library_name)
            pout.set_status(qcc.STATUS_ERROR)
            return
        if not pin.is_memory_input():
            print('Only in memory input supported for normalize plugin.')
            pout.set_status(qcc.STATUS_ERROR)
            return

        inp = pin.get_input()
        means = pin.get_param('means', {"R": 0, "G": 0, "B": 0})
        std = pin.get_param('std', {"R": 1, "G": 1, "B": 1})
        channel_order = pin.get_param('channel_order', "RGB")
        if channel_order == "BGR":
            means = [means["B"], means["G"], means["R"]]
            std = [std["B"], std["G"], std["R"]]
        else:
            means = [means["R"], means["G"], means["B"]]
            std = [std["R"], std["G"], std["B"]]

        if library_name == 'numpy':
            img = normalize.norm_numpy(pin, pout, inp, means, std)
        elif library_name == 'torchvision':
            img = normalize.norm_tv(pin, pout, inp, means, std)

        if not img is None:
            pout.set_mem_output(img)
            pout.set_status(qcc.STATUS_SUCCESS)

    @staticmethod
    def norm_numpy(pin: PluginInputInfo, pout: PluginOutputInfo, inp, means, std):
        if not (list(np.shape(inp))[-1] == 3):
            print('Normalization must be applied with the data in NHWC format. Check config file.')
            pout.set_status(qcc.STATUS_ERROR)
            return

        norm = np.float32(pin.get_param('norm', 255.0))
        normalize_first = pin.get_param('normalize_first', True)
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
    def norm_tv(pin: PluginInputInfo, pout: PluginOutputInfo, inp, means, std):
        if not isinstance(inp, Image.Image):  #To check if the input is valid PIL image or not
            print('This version(0.7.0) of torchvision supports only valid PIL images as input')
            pout.set_status(qcc.STATUS_ERROR)
            return
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        if pin.get_param('pil_to_tensor_input', True):
            inp = torchvision.transforms.functional.to_tensor(inp)

        inp = torchvision.transforms.functional.normalize(inp, mean=means, std=std)

        typecast_required = pin.get_param('typecasting_required', True)
        if typecast_required:
            inp = inp.numpy()
        return inp
