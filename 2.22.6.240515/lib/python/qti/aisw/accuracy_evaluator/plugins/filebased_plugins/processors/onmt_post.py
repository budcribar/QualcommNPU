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
# SOURCE: https://github.com/NVIDIA/FasterTransformer/tree/54e1b4a981f00b83bc22b2939743ec1e58164b86
# LICENCE: https://github.com/NVIDIA/FasterTransformer/blob/54e1b4a981f00b83bc22b2939743ec1e58164b86/LICENSE
###############################################################################

from typing import List
import numpy as np
import pickle
import os
import argparse
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class onmt_postprocess(qacc_plugin):
    """Post processes the WMT20 test set."""

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

    def __init__(self):
        onmt = Helper.safe_import_package("onmt", "2.3.0")
        sentencepiece = Helper.safe_import_package("sentencepiece", "0.1.98")

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        # Dir plugins always have a single entry.
        self.execute_index(pin_list[0], pout_list[0])

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        if not pin.is_directory_input():
            print('Only directory based input supported for', self.__class__.__name__)
            pout.set_status(qcc.STATUS_ERROR)
            return

        onmt = Helper.safe_import_package("onmt", "2.3.0")
        sentencepiece = Helper.safe_import_package("sentencepiece", "0.1.98")

        inputs = pin.get_input()
        work_dir = pin.read_pipeline_cache_val(qcc.PIPELINE_WORK_DIR)

        sentencepiece_model_path = pin.get_param('sentencepiece_model_path', None)
        unrolled_count = pin.get_param('unrolled_count', 26)
        vocab_path = pin.get_param('vocab_path', None)
        skip_sentencepiece = pin.get_param('skip_sentencepiece', None)

        batch_size = pin.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)
        max_inputs = pin.read_pipeline_cache_val(qcc.PIPELINE_MAX_INPUTS)

        if sentencepiece_model_path == None:
            print('sentencepiece_model_path is a mandatory parameter!')
            pout.set_status(qcc.STATUS_ERROR)
            return

        cached_data_file = os.path.join(work_dir, "cached_dataset.pickle")
        if os.path.exists(cached_data_file):
            with open(cached_data_file, 'rb') as data:
                data = pickle.load(data)
            fields, dataset = data["fields"], data["dataset"]
            print('Loaded cache data from ' + str(cached_data_file))
        else:
            qacc_logger.warning('Cached data not found at {}'.format(cached_data_file))
            qacc_logger.info('Generating data required for onmt_post plugin...')
            if vocab_path == None or skip_sentencepiece == None:
                qacc_logger.error(
                    'vocab_path and skip_sentencepiece are required to generate dataset!')
                pout.set_status(qcc.STATUS_ERROR)
                return
            data_list = pin.get_orig_path_list()
            fields, dataset = onmt_postprocess.generate_dataset(data_list, vocab_path,
                                                                skip_sentencepiece,
                                                                sentencepiece_model_path)

        if isinstance(dict(fields)["src"], onmt.inputters.text_dataset.TextMultiField):
            src_vocab = dataset.src_vocabs[inds[b]] \
                if dataset.src_vocabs else None
            try:
                src_raw = dataset.examples[inds[b]].src[0]
            except:
                src_raw = None
        else:
            src_vocab = None
            src_raw = None

        src = None
        attn = None
        sp = sentencepiece.SentencePieceProcessor(model_file=sentencepiece_model_path)

        #detokenized outputs path
        output_prediction_file = os.path.join(pout.get_out_dir(), "predictions.txt")
        txt = open(output_prediction_file, 'w')

        # read model outputs
        for batch_idx, input_pair in enumerate(inputs):
            beams_path, lengths_path = input_pair
            beams = np.fromfile(beams_path, dtype=np.int64)
            lengths = np.fromfile(lengths_path, dtype=np.int64)
            # reshape to [batch_size, max_seq_len]
            beams = beams.reshape([batch_size, -1])
            lengths = lengths.reshape([batch_size, -1])

            for i in range(batch_size):
                idx = batch_idx * batch_size + i
                if idx >= max_inputs:  # for last batch
                    break

                pred_sents = onmt_postprocess.build_target_tokens(src, src_vocab, src_raw, beams[i],
                                                                  attn, fields)
                txt.write(''.join(sp.decode(pred_sents[:unrolled_count])) + "\n")
        txt.close()

        pout.set_dir_outputs([[output_prediction_file]])
        pout.set_status(qcc.STATUS_SUCCESS)

    def build_target_tokens(src, src_vocab, src_raw, pred, attn, fields):
        tgt_field = dict(fields)["tgt"].base_field
        vocab = tgt_field.vocab
        tokens = []

        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        return tokens

    def generate_dataset(inputs, vocab_path, skip_sentencepiece, sentencepiece_model_path):
        onmt = Helper.safe_import_package("onmt", "2.3.0")
        sentencepiece = Helper.safe_import_package("sentencepiece", "0.1.98")
        data = []
        for example_path in inputs:
            text = open(example_path[0][0], 'r').read()
            data.append(text)

        if not skip_sentencepiece:
            sp = sentencepiece.SentencePieceProcessor()
            sp.Load(sentencepiece_model_path)
            for i in range(len(data)):
                pieces = sp.EncodeAsPieces(data[i].strip())
                data[i] = " ".join(pieces)

        data = np.array(data)
        opt = argparse.Namespace(data_type='text')
        src_reader = onmt.inputters.str2reader['text'].from_opt(opt)
        src_data = {'reader': src_reader, 'data': data, 'dir': None}
        _readers, _data, _dir = onmt.inputters.Dataset.config([('src', src_data)])
        with open(vocab_path, 'rb') as f:
            fields = pickle.load(f)
        fields.pop('corpus_id')
        dataset = onmt.inputters.Dataset(fields, readers=_readers, data=_data, dirs=_dir,
                                         sort_key=onmt.inputters.str2sortkey['text'])
        return fields, dataset
