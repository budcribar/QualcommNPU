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
# Source: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
################################################################################
import os
import pickle
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_metric
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class squad_eval(qacc_memory_metric):
    """Class for metrics(f1 and exact) for SQuAD dataset."""

    def __init__(self, dataset, vocabulary=None, max_answer_length=30, n_best_size=20,
                 do_lower_case=True, squad_version=1, round=6, cached_vocab_path=None, **kwargs):

        transformers = Helper.safe_import_package("transformers", "4.31.0")
        self.round_value = round
        self.vocabulary = vocabulary
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.do_lower_case = do_lower_case
        self.squad_version = squad_version
        self.all_results = []

        vocab_info = '''vocabulary param can be any one of the below:
        1) vocabulary from huggingface.co and cache (e.g. "bert-base-uncased")
        2) vocabulary from huggingface.co (user-uploaded) and cache (e.g. "deepset/roberta-base-squad2")
        3) path for local directory containing vocabulary files(tokenizer was saved using _save_pretrained('./test/saved_model/')_)
        '''

        if not (self.squad_version == 1 or self.squad_version == 2):
            qacc_file_logger.error('Only squad versions 1 and 2 are supported')
            return

        if self.vocabulary is None:
            qacc_file_logger.error("'vocabulary' parameter is mandatory for loading tokenizer")
            qacc_file_logger.info(vocab_info)
            return

        if cached_vocab_path != None and os.path.isdir(cached_vocab_path):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                cached_vocab_path, do_lower_case=do_lower_case, use_fast=False)
            qacc_file_logger.info(f"Loading from local cached path at {str(cached_vocab_path)}")
        else:
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.vocabulary, do_lower_case=self.do_lower_case, use_fast=False)
            except Exception:
                qacc_logger.error(
                    "Failed to download tokenizer. Provide path for cached vocabulary instead.")
                return

        cached_features_file = os.path.join(self.extra_params['work_dir'], "cached_features.pickle")
        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'rb') as data:
                data = pickle.load(data)
            self.features, self.examples = data["features"], data["examples"]
            qacc_file_logger.info(f'Loaded features from {str(cached_features_file)}')
        else:
            qacc_file_logger.error("cached features file is not present")
            return

        # create the platform specific metric directory to store 'predictions.json'
        self.path_to_metric = os.path.join(self.extra_params['work_dir'], 'metric',
                                           self.extra_params['inference_schema_name'])
        if not os.path.exists(self.path_to_metric):
            os.makedirs(self.path_to_metric)

    def calculate(self, data, meta, input_idx):
        for result in data:
            self.all_results.append(result)

    def finalize(self):
        transformers = Helper.safe_import_package("transformers", "4.31.0")
        from transformers.data.metrics.squad_metrics import compute_predictions_logits
        try:
            output_prediction_file = os.path.join(self.path_to_metric, 'predictions.json')
            qacc_file_logger.info("all results: {}, examples: {}, features: {}".format(
                len(self.all_results), len(self.examples), len(self.features)))
            output_nbest_file = None
            output_null_log_odds_file = None
            verbose_logging = False
            version_2_with_negative = True if self.squad_version == 2 else False  #True for squad2 and False for squad1
            null_score_diff_threshold = 0.0
            predictions = compute_predictions_logits(
                self.examples,
                self.features,
                self.all_results,
                self.n_best_size,
                self.max_answer_length,
                self.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                verbose_logging,
                version_2_with_negative,
                null_score_diff_threshold,
                self.tokenizer,
            )
        except Exception as e:
            qacc_file_logger.error(f"Failed to process compute_predictions_logits. Error {e}")

        results = transformers.data.metrics.squad_metrics.squad_evaluate(self.examples, predictions)
        save_results = {}
        save_results['f1'] = round(results['f1'], self.round_value)
        save_results['exact'] = round(results['exact'], self.round_value)
        save_results['total'] = results['total']
        return save_results
