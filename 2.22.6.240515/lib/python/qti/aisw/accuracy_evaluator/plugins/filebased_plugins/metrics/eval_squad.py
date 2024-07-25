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
# Source: https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
################################################################################

import json
import re
import string
from collections import Counter
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_metric, MetricInputInfo, MetricResult
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class squad_eval_base(qacc_metric):
    """Base Class for metrics for SQuAD v1.1 dataset."""

    def __init__(self, res_dict_key='metric'):
        self.res_dict_key = res_dict_key

    def _metric(self, prediction, ground_truth):
        return 0

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):
        results_file = m_in.get_result_file()
        dev_file = m_in.get_groundtruth()
        expected_version = '1.1'

        with open(results_file) as f:
            prediction_filepath = f.readlines()[0].strip()

        with open(dev_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            if (dataset_json['version'] != expected_version):
                print(
                    f'Evaluation expects v-{expected_version}, but got dataset with v-{dataset_json["version"]}'
                )
            dataset = dataset_json['data']

        with open(prediction_filepath) as prediction_file:
            predictions = json.load(prediction_file)

        res = self._evaluate(dataset, predictions)
        res_str = f'{self.res_dict_key}: {res[self.res_dict_key]}, count: {res["count"]}'
        m_out.set_result(res)
        m_out.set_result_str(res_str)
        m_out.set_status(qcc.STATUS_SUCCESS)

    def _metric_max_over_ground_truths(self, prediction, ground_truths):
        scores_for_ground_truths = [
            self._metric(prediction, ground_truth) for ground_truth in ground_truths
        ]
        return max(scores_for_ground_truths)

    def _evaluate(self, dataset, predictions):
        score = total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    if qa['id'] not in predictions:
                        # SCORE ONLY PREDICTED OUTPUT - NOT ENTIRE VAL DATA
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    total += 1
                    score += self._metric_max_over_ground_truths(prediction, ground_truths)

        if total == 0:
            score = 0
        else:
            score = 100.0 * score / total

        return {self.res_dict_key: score, 'count': total}


class squad_f1(squad_eval_base):
    """F1 for SQuAD v1.1 dataset."""

    def __init__(self):
        super().__init__(res_dict_key='f1')

    def _metric(self, prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


class squad_em(squad_eval_base):
    """Exact Match for SQuAD v1.1 dataset."""

    def __init__(self):
        super().__init__(res_dict_key='exact_match')

    def _metric(self, prediction, ground_truth):
        return (normalize_answer(prediction) == normalize_answer(ground_truth))
