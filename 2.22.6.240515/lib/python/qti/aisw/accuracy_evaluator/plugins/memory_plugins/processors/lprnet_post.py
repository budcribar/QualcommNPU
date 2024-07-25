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
from itertools import groupby
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_postprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class lprnet_predict(qacc_memory_postprocessor):
    """Used for LPRNET license plate prediction."""

    def __init__(self):
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

    def execute(self, data, meta, input_idx, *args, **kwargs):
        """Processes LPRNET output and predicts the License Plate number."""

        for i in range(len(data)):
            data[i] = self.postprocess(data[i][0])

        return data, meta

    def postprocess(self, vec) -> str:

        # vec: single image model output
        # vec is logits matrix, with num_classes=71 (as in vocab)
        # do greedy detection

        classes = np.argmax(vec, axis=1)
        # ignore consecutive repetitions
        # ignore class 70:'_'
        classes = [x[0] for x in groupby(classes) if x[0] != 70]
        return ''.join([self.vocab[x] for x in classes])
