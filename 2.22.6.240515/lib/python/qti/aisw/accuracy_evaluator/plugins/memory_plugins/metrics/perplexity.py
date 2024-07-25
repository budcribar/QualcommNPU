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
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_metric
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class perplexity(qacc_memory_metric):

    def __init__(self, dataset, logits_index=0):
        torch = Helper.safe_import_package("torch")
        self.logits_index = logits_index
        self.annotations_file = dataset.get_dataset_annotation_file()
        self.all_labels_paths = [
            x.strip().split(',')[0] for x in open(self.annotations_file).readlines()
        ]
        self.losses = []

    def calculate(self, data, meta, input_idx):
        torch = Helper.safe_import_package("torch")
        for input_data in data[self.logits_index]:
            labels_file = self.all_labels_paths[input_idx]
            labels = np.fromfile(labels_file, dtype=np.int64).reshape(input_data.shape[:-1])
            shift_logits = torch.tensor(input_data[..., :-1, :])
            shift_labels = torch.tensor(labels[..., 1:])
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            self.losses.append(loss.item())

        return

    def finalize(self):
        loss = np.mean(self.losses)
        ppl = np.exp(loss)
        result = {'perplexity': ppl}
        return result
