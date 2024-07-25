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
import os
from itertools import groupby
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class lprnet_predict(qacc_plugin):
    """Used for LPRNET license plate prediction."""
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
        super().__init__()
        self.vocab = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
            9: '9', 10: '<Anhui>', 11: '<Beijing>', 12: '<Chongqing>', 13: '<Fujian>',
            14: '<Gansu>', 15: '<Guangdong>', 16: '<Guangxi>', 17: '<Guizhou>', 18: '<Hainan>',
            19: '<Hebei>', 20: '<Heilongjiang>', 21: '<Henan>', 22: '<HongKong>', 23: '<Hubei>',
            24: '<Hunan>', 25: '<InnerMongolia>', 26: '<Jiangsu>', 27: '<Jiangxi>', 28: '<Jilin>',
            29: '<Liaoning>', 30: '<Macau>', 31: '<Ningxia>', 32: '<Qinghai>', 33: '<Shaanxi>',
            34: '<Shandong>', 35: '<Shanghai>', 36: '<Shanxi>', 37: '<Sichuan>', 38: '<Tianjin>',
            39: '<Tibet>', 40: '<Xinjiang>', 41: '<Yunnan>', 42: '<Zhejiang>', 43: '<police>',
            44: 'A', 45: 'B', 46: 'C', 47: 'D', 48: 'E', 49: 'F', 50: 'G', 51: 'H', 52: 'I',
            53: 'J', 54: 'K', 55: 'L', 56: 'M', 57: 'N', 58: 'O', 59: 'P', 60: 'Q', 61: 'R',
            62: 'S', 63: 'T', 64: 'U', 65: 'V', 66: 'W', 67: 'X', 68: 'Y', 69: 'Z', 70: '_', -1: ''
        } # yapf: disable

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        """Processes LPRNET output and predicts the License Plate number."""
        plugin_input = pin_list[0]
        if not plugin_input.is_path_input():
            print(f'{self.__class__.__name__} is a path based plugin!')
            return

        bs = plugin_input.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)  #batchsize

        data = np.fromfile(plugin_input.get_input(), dtype=np.float32).reshape((bs, 88, 71))
        outputs = [self.postprocess(data[i]) for i in range(bs)]
        # Write the post-processed outputs into files with Result_{input_idx}.txt.
        out_file = os.path.basename(os.path.dirname(pin_list[0].get_input()))
        out_dir = os.path.dirname(pout_list[0].get_output_path())
        updated_path = os.path.join(out_dir, f'{out_file}.txt')
        pout_list[0].set_path_output(updated_path)

        with open(pout_list[0].get_output_path(), 'w') as f:
            for output in outputs:
                f.write(output)
                f.write('\n')

        pout_list[0].set_status(qcc.STATUS_SUCCESS)

    def postprocess(self, vec) -> str:
        # vec: single image model output
        # vec is logits matrix, with num_classes=71 (as in vocab)
        # do greedy detection
        classes = np.argmax(vec, axis=1)
        # ignore consecutive repetitions
        # ignore class 70:'_'
        classes = [x[0] for x in groupby(classes) if x[0] != 70]
        return ''.join([self.vocab[x] for x in classes])
