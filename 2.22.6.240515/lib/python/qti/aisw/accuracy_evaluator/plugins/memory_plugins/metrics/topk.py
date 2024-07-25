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
import os
import numpy as np
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_metric
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class topk(qacc_memory_metric):
    """Metric plugin to calculate the number of times where the correct label
    is among the top k predicted labels."""

    def __init__(self, dataset, kval=[1, 5], softmax_index=0, label_offset=0, round=7,
                 input_image_index=0):
        # Metric Specific items : Parsed from config
        if isinstance(kval, list):
            self.kvals = kval
        elif isinstance(kval, str):
            self.kvals = [int(k) for k in kval.split(',')]
        else:
            self.kvals = [kval]
        self.k_val_dict = {k: [] for k in self.kvals}

        self.round_to = round
        self.input_image_index = input_image_index
        self.softmax_index = softmax_index
        self.label_offset = label_offset

        annotations_file = dataset.get_dataset_annotation_file()
        input_file_list = dataset.get_input_list_file()

        # ground truth
        annotation_data = [x.strip().split(' ') for x in open(annotations_file, 'r').readlines()]
        self.gt = {x[0]: int(x[1]) for x in annotation_data}

        input_files = [
            x.strip().split(',')[input_image_index] for x in open(input_file_list).readlines()
        ]
        self.input_files = [os.path.basename(x) for x in input_files]

    def calculate(self, data, meta, input_idx):
        raw_data = data[self.softmax_index]
        # TODO: following is valid for BS=1
        # raw_data.shape: [BS, num_classes + offset]
        top_data = np.argsort(raw_data[0, self.label_offset:])[::-1].tolist()

        fname = self.input_files[input_idx]
        for k in self.kvals:
            self.k_val_dict[k].append(self.gt[fname] in top_data[:k])

        return

    def finalize(self):
        result = {}
        total = len(self.k_val_dict[self.kvals[0]])
        for k in self.kvals:
            assert len(self.k_val_dict[k]) == total
            score = sum(self.k_val_dict[k]) / (total or 1)
            result[f'top{k}'] = score

        result['count'] = total
        return result
