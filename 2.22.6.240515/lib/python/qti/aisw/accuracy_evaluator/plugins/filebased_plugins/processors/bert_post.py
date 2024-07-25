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
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
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
################################################################################

import collections
import json
import math
import os
import pickle
from typing import List
import numpy as np
import six
import qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.tokenization as tokenization
from qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.bert_pre import get_features_from_cache, get_features, squad_read, read_squad_examples
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class bert_predict(qacc_plugin):
    """Predict Answers for SQuAD dataset given start and end logits.

    Used for BERT-Large model.
    """

    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_DIR,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_DIR,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        # Dir plugins always have a single entry.
        self.execute_index(pin_list[0], pout_list[0])

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        if not pin.is_directory_input():
            print('Only directory based input supported for', self.__class__.__name__)
            return

        inputs = pin.get_input()
        work_dir = pin.read_pipeline_cache_val(qcc.PIPELINE_WORK_DIR)

        # path to read pickle files created during dataset loading
        dev_path = pin.get_orig_path_list()

        # parameters for feature generation
        batch_size = pin.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)
        vocab_path = pin.get_param('vocab_path', None)
        max_seq_length = pin.get_param('max_seq_length', 384)
        doc_stride = pin.get_param('doc_stride', 128)
        max_query_length = pin.get_param('max_query_length', 64)
        # expect features cache from preprocessing unless preprocessing is skipped
        features_cache_path = os.path.join(work_dir, squad_read.FEATURES_CACHE_FNAME)
        # parameters for answer processing
        n_best_size = pin.get_param('n_best_size', 20)
        max_answer_length = pin.get_param('max_answer_length', 30)
        # parameters for "packing" strategy
        do_unpacking = pin.get_param('packing_strategy', False)
        packing_map_cache_path = os.path.join(work_dir, squad_read.PACKING_MAP_FNAME)
        # path to save final processed answers
        output_prediction_file = os.path.join(pout.get_out_dir(), 'post.json')

        print(f'in bert_predict, doing packing_strategy: {do_unpacking}')

        # get the examples and features
        eval_examples = read_squad_examples(input_file=dev_path, process=False)
        eval_features = get_features_from_cache(features_cache_path)
        if eval_features is None:  # cache not found
            eval_features, correct = get_features(cache_path=None, dev_path=dev_path,
                                                  vocab_path=vocab_path,
                                                  max_seq_length=max_seq_length,
                                                  doc_stride=doc_stride,
                                                  max_query_length=max_query_length)
            if not correct:
                pout.set_status(qcc.STATUS_ERROR)
                return

        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

        # read packing map cache for unpacking
        if do_unpacking:
            if not os.path.exists(packing_map_cache_path):
                print(f'invalid path for packing_map_cache: {packing_map_cache_path}')
                pout.set_status(qcc.STATUS_ERROR)
                return
            with open(packing_map_cache_path, 'rb') as cache_file:
                output_map = pickle.load(cache_file)

        # read model outputs
        results = []
        for batch_idx, input_pair in enumerate(inputs):
            start_logits_path, end_logits_path = input_pair
            start_logits = np.fromfile(start_logits_path, dtype=np.float32)
            end_logits = np.fromfile(end_logits_path, dtype=np.float32)
            # reshape to [batch_size, max_seq_len]
            start_logits = start_logits.reshape([batch_size, -1])
            end_logits = end_logits.reshape([batch_size, -1])

            for i in range(batch_size):
                idx = batch_idx * batch_size + i
                if not do_unpacking:
                    if idx >= len(eval_features):  # for last batch
                        break
                    results.append(
                        RawResult(unique_id=eval_features[idx].unique_id,
                                  start_logits=start_logits[i].tolist(),
                                  end_logits=end_logits[i].tolist()))
                else:  # require unpacking
                    if idx >= len(output_map):  # for last batch
                        break
                    # info on which features were packed together
                    feat_idx_and_sls = output_map[idx]
                    feat_idxs = [x[0] for x in feat_idx_and_sls]
                    sls = [x[1] for x in feat_idx_and_sls]

                    # unpack
                    packed_data = [start_logits[i].tolist(), end_logits[i].tolist()]
                    unpacked_data = bert_predict.unpack(packed_data, sls)

                    # add to "results" for each original feature that was packed
                    for data_idx, feat_idx in enumerate(feat_idxs):
                        results.append(
                            RawResult(unique_id=eval_features[feat_idx].unique_id,
                                      start_logits=unpacked_data[data_idx][0],
                                      end_logits=unpacked_data[data_idx][1]))

        # predict answers
        prediction_json = self._get_predictions(eval_examples, eval_features, results, n_best_size,
                                                max_answer_length, True)
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(prediction_json, indent=4) + "\n")
        print(f'writing predictions to file: {output_prediction_file}')

        pout.set_dir_outputs([[output_prediction_file]])
        pout.set_status(qcc.STATUS_SUCCESS)

    @staticmethod
    def unpack(data, sls):
        # data is single aic output [start_logits, end_logits]
        # sls is list of sequence lengths # [190, 192]
        # returns:
        #     list of [[start_logits, end_logits], ...]
        result = []
        offset = 0
        for sl in sls:
            res = [x[offset:offset + sl] for x in data]
            result.append(res)
            offset += sl
        return result

    def _get_final_text(self, pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #     pred_text = steve smith
        #     orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def _get_predictions(self, all_examples, all_features, all_results, n_best_size,
                         max_answer_length, do_lower_case):
        """Write final predictions to the json file and log-odds of null if
        needed."""
        all_predictions = collections.OrderedDict()

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        # all_nbest_json = collections.OrderedDict()
        # scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result.get(feature.unique_id, None)
                if result is None:
                    continue
                start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self._get_best_indexes(result.end_logits, n_best_size)
                # if we could have irrelevant answers, get the min score of irrelevant
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(feature_index=feature_index, start_index=start_index,
                                              end_index=end_index,
                                              start_logit=result.start_logits[start_index],
                                              end_logit=result.end_logits[end_index]))

            # if there are no prelim_predictions,
            # it means that inference was done on a subset of data
            # so, ignore this example
            if not prelim_predictions:
                continue

            prelim_predictions = sorted(prelim_predictions, key=lambda x:
                                        (x.start_logit + x.end_logit), reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = self._get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(text=final_text, start_logit=pred.start_logit,
                                     end_logit=pred.end_logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="__no_prediction__", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            all_predictions[example.qas_id] = nbest_json[0]["text"]

        print(f'Number of examples predicted for: {len(all_predictions)}')
        return all_predictions
