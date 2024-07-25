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
import argparse
import os
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class onmt_preprocess(qacc_plugin):
    """Pre processes the WMT20 test set."""

    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_DIR,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_DIR,
        qcc.IO_DTYPE: qcc.DTYPE_INT64,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def __init__(self):
        torch = Helper.safe_import_package("torch")
        onmt = Helper.safe_import_package("onmt", "2.3.0")
        sentencepiece = Helper.safe_import_package("sentencepiece", "0.1.98")

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        pin, pout = pin_list[0], pout_list[0]
        torch = Helper.safe_import_package("torch")
        onmt = Helper.safe_import_package("onmt", "2.3.0")
        sentencepiece = Helper.safe_import_package("sentencepiece", "0.1.98")

        inputs = pin.get_input()
        outdir = pout.get_out_dir()
        work_dir = pin.read_pipeline_cache_val(qcc.PIPELINE_WORK_DIR)

        vocab_path = pin.get_param('vocab_path', None)
        src_seq_len = pin.get_param('src_seq_len', 128)
        skip_sentencepiece = pin.get_param('skip_sentencepiece', True)
        sentencepiece_model_path = pin.get_param(
            'sentencepiece_model_path', None)  #This is required when skip_sentenpiece is False

        if vocab_path == None:
            print('Vocabulary is a mandatory parameter!')
            pout.set_status(qcc.STATUS_ERROR)
            return

        if not skip_sentencepiece and not sentencepiece_model_path:
            print('Please provide valid sentencepiece model path!')
            pout.set_status(qcc.STATUS_ERROR)
            return

        data = []
        for example_path in inputs:
            text = open(example_path[0], 'r').read()
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
        _readers, _data = onmt.inputters.Dataset.config([('src', src_data)])
        with open(vocab_path, 'rb') as f:
            fields = pickle.load(f)
        fields.pop('corpus_id')
        dataset = onmt.inputters.Dataset(fields, readers=_readers, data=_data,
                                         sort_key=onmt.inputters.str2sortkey['text'])
        data_iter = onmt.inputters.OrderedIterator(dataset=dataset, batch_size=1)
        data_list = list(data_iter)

        is_calib = pin.read_pipeline_cache_val(qcc.PIPELINE_PREPROC_IS_CALIB)
        if not is_calib:
            cached_data_file = os.path.join(work_dir, "cached_dataset.pickle")
            with open(cached_data_file, 'wb') as cache_file:
                pickle.dump({"fields": fields, "dataset": dataset}, cache_file)
            print('Saved cache data at ' + str(cached_data_file))

        outdir = os.path.join(outdir, 'pre')
        os.makedirs(outdir, exist_ok=True)
        out_filepaths = [[] for i in range(len(data_list))]
        for i in range(len(data_list)):
            idx = data_list[i].indices[0].item()
            dirpath = os.path.join(outdir, str(idx))
            os.makedirs(dirpath, exist_ok=True)

            src = data_list[i].src[0]
            src_lengths = data_list[i].src[1]
            pad_length = src_seq_len - src.shape[0]
            src = torch.cat([src, torch.zeros(pad_length, src.shape[1], 1).long()], axis=0)

            src_path = os.path.join(dirpath, 'src.raw')
            src.numpy().tofile(src_path)
            src_len_path = os.path.join(dirpath, 'src_lengths.raw')
            src_lengths.numpy().tofile(src_len_path)
            out_filepaths[idx] = [src_path, src_len_path]

        pout.set_dir_outputs(out_filepaths)
        pout.set_status(qcc.STATUS_SUCCESS)
