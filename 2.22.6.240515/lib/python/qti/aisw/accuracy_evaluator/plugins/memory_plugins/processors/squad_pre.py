################################################################################
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
################################################################################

import numpy as np
import os
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_preprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class squad_preprocess(qacc_memory_preprocessor):

    def __init__(self, mask_type=None, **kwargs):
        self.boolean_mask = mask_type and mask_type.lower().startswith('bool')

    def execute(self, data, meta, input_idx, *args, **kwargs):
        out_data = []
        for idx, item in enumerate(data):
            datatype = np.int64
            if idx == 1 and self.boolean_mask is True:
                datatype = bool
            out_data.append(np.fromfile(item, dtype=datatype))
        return out_data, meta
