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
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_metric
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class precision(qacc_memory_metric):

    def __init__(self, dataset, input_image_index=0, round=7):

        self.inp_img_idx = input_image_index
        self.round = round
        self.annotations_file = dataset.get_dataset_annotation_file()
        self.input_file_list = dataset.get_orig_input_list_file()
        self.total = dataset.total_entries
        self.all_results = [None] * dataset.total_entries

    def calculate(self, data, meta, input_idx):
        try:
            for inp_data in data:
                self.all_results[input_idx] = inp_data
        except:
            logging.info("Metric plugin failed in calculate")

        return

    def finalize(self):

        max_inputs = self.total

        # read filenames from inputfilelist.txt
        # for multi input models, take the filename at index "inp_img_idx" for ground truth mapping
        input_files = [
            x.strip().split(',')[self.inp_img_idx] for x in open(self.input_file_list).readlines()
        ]
        input_files = [os.path.basename(x) for x in input_files]

        # ground truth, expected as "filename <space> correct_text"
        annotations_data = [x.strip().split(' ') for x in open(self.annotations_file).readlines()]
        annotations_data = dict(annotations_data)

        total = min(max_inputs, len(input_files), len(self.all_results))
        predictions = self.all_results[:total]  # skip the repeated elements in the last batch

        # compute precision
        correct = sum(pred == annotations_data[filename]
                      for filename, pred in zip(input_files, predictions))
        precision = round(correct / total, self.round)

        res_str = f'precision: {precision:0.0{self.round}f}'
        save_results = {}
        save_results['output'] = res_str
        return save_results
