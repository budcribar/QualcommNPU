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
import numpy as np
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class perplexity(qacc_metric):

    def __init__(self):
        torch = Helper.safe_import_package("torch")

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):
        torch = Helper.safe_import_package("torch")
        results_file = m_in.get_result_file()  # model output paths
        annotations_file = m_in.get_groundtruth()  # ground truth filepath
        batch_size = m_in.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)
        max_inputs = m_in.read_pipeline_cache_val(qcc.PIPELINE_MAX_INPUTS)

        output_node_info = m_in.read_pipeline_cache_val(qcc.PIPELINE_INFER_OUTPUT_INFO)
        logits_idx = m_in.get_param('logits_index', 0)
        logits_shape = output_node_info[list(output_node_info.keys())[logits_idx]][1]
        assert batch_size == logits_shape[0]
        logits_shape_bs1 = [1, *logits_shape[1:]]

        all_labels_paths = [x.strip().split(',')[0] for x in open(annotations_file).readlines()]
        all_logits_paths = [
            x.strip().split(',')[logits_idx] for x in open(results_file).readlines()
        ]

        current = 0
        logits_file_idx = 0
        losses = np.zeros(max_inputs)
        while current < max_inputs:
            logits_path = all_logits_paths[logits_file_idx]
            logits_batch = np.fromfile(logits_path, dtype=np.float32).reshape(logits_shape)

            for bs_i in range(batch_size):
                logits = logits_batch[[bs_i]]

                labels_path = all_labels_paths[current]
                labels = np.fromfile(labels_path, dtype=np.int64).reshape(logits_shape_bs1[:-1])

                shift_logits = torch.tensor(logits[..., :-1, :])
                shift_labels = torch.tensor(labels[..., 1:])

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                losses[current] = loss.item()

                current += 1
                if current >= max_inputs:
                    break

            logits_file_idx += 1

        loss = np.mean(losses)
        ppl = np.exp(loss)

        result = {'perplexity': ppl}
        res_str = f'perplexity: {ppl:0.7f}'

        m_out.set_result(result)
        m_out.set_result_str(res_str)
        m_out.set_status(qcc.STATUS_SUCCESS)
