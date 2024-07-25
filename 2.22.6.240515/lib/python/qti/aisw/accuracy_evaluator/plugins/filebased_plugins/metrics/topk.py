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
import numpy as np
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class topk(qacc_metric):

    def __init__(self):
        pass

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):

        results_file = m_in.get_result_file()  # model output paths
        annotations_file = m_in.get_groundtruth()  # ground truth filepath
        input_file_list = m_in.get_orig_inp_file()  # inputlist.txt as provided by user

        k_vals = list(map(int, m_in.get_param('kval', '5').split(',')))
        k_val_dict = {x: 0 for x in k_vals}
        softmax_index = m_in.get_param(
            'softmax_index', 0)  # for multiple output models, the index for softmax ouptut
        offset = m_in.get_param('label_offset', 0)
        inp_img_idx = m_in.get_param('input_image_index', 0)
        round_to = m_in.get_param('round', 3)  # fp precision for final score
        batch_size = m_in.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)
        max_inputs = m_in.read_pipeline_cache_val(qcc.PIPELINE_MAX_INPUTS)

        # read filenames from inputfilelist.txt
        # for multi input models, take the filename at index "inp_img_idx" for ground truth mapping
        input_files = [x.strip().split(',')[inp_img_idx] for x in open(input_file_list).readlines()]
        input_files = [os.path.basename(x) for x in input_files]

        # ground truth
        annotations_data = [x.strip().split(' ') for x in open(annotations_file, 'r').readlines()]
        annotations_data = {x[0]: int(x[1]) for x in annotations_data}

        # paths to model outputs (batched)
        raw_files = [
            rec.split(',')[softmax_index].strip() for rec in open(results_file, 'r').readlines()
        ]

        total_count = 0  # index in input_files
        rf_index = 0  # index in raw_files
        max_count = min(max_inputs, len(input_files), batch_size * len(raw_files))
        while total_count < max_count:
            # raw data should be of dimension : [batch_size x classes]
            raw_data = np.fromfile(raw_files[rf_index], np.float32).reshape(batch_size, -1)

            for b_idx in range(batch_size):
                # skip first "offset" classes
                _raw_data = raw_data[b_idx, offset:]

                ground_truth = annotations_data[input_files[total_count]]
                _top_data = np.argsort(_raw_data)[::-1].tolist()

                # update topk
                for k in k_vals:
                    if ground_truth in _top_data[0:k]:
                        k_val_dict[k] += 1

                total_count += 1
                if total_count >= max_count:
                    break
            rf_index += 1

        print(f'TOPK - Number of inputs: {total_count}')
        res_str = ''
        result = {}
        for k in k_vals:
            score = k_val_dict[k] / total_count  # division is float by default
            res_str += f'top{k} score: {score:0.0{round_to}f}\n'
            result[f'top{k}'] = score
        m_out.set_result(result)
        m_out.set_result_str(res_str)
        m_out.set_status(qcc.STATUS_SUCCESS)
