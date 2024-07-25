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
import os
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class precision(qacc_metric):

    def __init__(self):
        pass

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):

        results_file = m_in.get_result_file()  # model output paths
        annotations_file = m_in.get_groundtruth()  # ground truth filepath
        input_file_list = m_in.get_orig_inp_file()  # inputlist.txt as provided by user

        inp_img_idx = m_in.get_param('input_image_index', 0)
        round_to = m_in.get_param('round', 7)  # fp precision for final score
        batch_size = m_in.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)
        max_inputs = m_in.read_pipeline_cache_val(qcc.PIPELINE_MAX_INPUTS)

        # read filenames from inputfilelist.txt
        # for multi input models, take the filename at index "inp_img_idx" for ground truth mapping
        input_files = [x.strip().split(',')[inp_img_idx] for x in open(input_file_list).readlines()]
        input_files = [os.path.basename(x) for x in input_files]

        # ground truth, expected as "filename <space> correct_text"
        annotations_data = [x.strip().split(' ') for x in open(annotations_file).readlines()]
        annotations_data = dict(annotations_data)

        # paths to model outputs (batched)
        model_output_files = [rec.strip() for rec in open(results_file).readlines()]

        predictions = []
        for fpath in model_output_files:
            # for bs>1, the output file is expected to have multiple lines
            predictions += [x.strip() for x in open(fpath).readlines()]

        total = min(max_inputs, len(input_files), len(predictions))
        predictions = predictions[:total]  # skip the repeated elements in the last batch

        # compute precision
        correct = sum(pred == annotations_data[filename]
                      for filename, pred in zip(input_files, predictions))
        precision = round(correct / total, round_to)

        res_str = f'precision: {precision:0.0{round_to}f}'

        m_out.set_result({'precision': precision})
        m_out.set_result_str(res_str)
        m_out.set_status(qcc.STATUS_SUCCESS)
