################################################################################
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
################################################################################
##############################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SOURCE: https://github.com/mjpost/sacrebleu
# LICENCE: https://github.com/mjpost/sacrebleu/blob/master/LICENSE.txt
###############################################################################

from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class bleu(qacc_metric):
    """Class for bleu metric using sacrebleu library."""

    def __init__(self):
        sacrebleu = Helper.safe_import_package("sacrebleu", "2.3.1")

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):
        sacrebleu = Helper.safe_import_package("sacrebleu", "2.3.1")
        results_file = m_in.get_result_file()
        annotation_file = m_in.gt_file

        round_value = m_in.get_param('round', 1)

        with open(results_file) as f:
            prediction_filepath = f.readlines()[0].strip()

        sys = [line.strip() for line in open(annotation_file).readlines()]
        refs = [[line.strip() for line in open(prediction_filepath).readlines()]]

        bleu = sacrebleu.metrics.BLEU()
        score = bleu.corpus_score(sys, refs).score

        save_results = {}
        save_results['bleu'] = round(score, round_value)
        res_str = f'bleu: {save_results["bleu"]}'
        m_out.set_result(save_results)
        m_out.set_result_str(res_str)
        m_out.set_status(qcc.STATUS_SUCCESS)
