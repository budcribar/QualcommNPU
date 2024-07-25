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

from itertools import chain
from pathlib import Path
import numpy as np
import os

from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_dataset, DatasetPluginInputInfo, DatasetPluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc.plugin import _parse_range


class tokenize_wikitext_2(qacc_dataset):
    '''Tokenize data from files using GPT2TokenizerFast or bloom tokenizer.
    '''

    def __init__(self):
        transformers = Helper.safe_import_package("transformers", "4.31.0")

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):
        self.dataset_path = Path(d_in.get_base_path())
        self.input_file = d_in.get_inputlist_file()
        self.outdir = Path(d_out.get_out_dir())
        self.seq_len = d_in.get_param('seq_length')
        self.past_seq_len = d_in.get_param('past_seq_length', 0)
        self.past_shape = d_in.get_param('past_shape', 0)
        self.num_past = d_in.get_param('num_past', 0)
        self.tokenizer_name = d_in.get_param('tokenizer_name')
        self.pos_id_required = d_in.get_param('pos_id', True)
        self.mask_dtype = d_in.get_param('mask_dtype', 'float32')
        self.cached_path = d_in.get_param('cached_path', None)
        # get tokens for all files listed in self.input_file
        #     paths are relative to self.dataset_path
        tokens = self.tokenize()
        # write "model inputs" to files under the dir self.outdir
        input_paths = self.write_files(tokens)
        input_list_path = self.outdir / 'inputlist.txt'
        annotation_path = self.outdir / 'labels.txt'
        # write the input file list and the labels path list
        with open(input_list_path, 'w') as input_f, open(annotation_path, 'w') as labels_f:
            for paths in input_paths:
                input_f.write(','.join(map(str, paths)))
                labels_f.write(str(self.outdir / paths[0]))
                input_f.write('\n')
                labels_f.write('\n')
        if d_in.get_calibration_type() == qcc.CALIBRATION_TYPE_INDEX:
            calib_list_path = self.outdir / 'calibration.txt'
            cf = open(d_in.get_calibration_file(), 'r')
            indices_str = cf.read().replace('\n', ',').strip()
            indices = sorted(set(chain.from_iterable(map(_parse_range, indices_str.split(",")))))
            cf.close()
            with open(input_list_path) as f, open(calib_list_path, 'w') as f2:
                for index, line in enumerate(f):
                    if index in indices:
                        f2.write(line)
            d_out.set_calibration_file(calib_list_path)  # relative to self.outdir
            d_out.set_calibration_type(
                qcc.CALIBRATION_TYPE_DATASET)  # No Longer a index based plugin
        # overwrite with new paths
        # d_out.base_path = self.outdir
        d_out.set_inputlist_file(input_list_path)  # relative to self.outdir
        d_out.inputlist_path_modified = True
        d_out.set_annotation_file(annotation_path)
        d_out.set_status(qcc.STATUS_SUCCESS)

    def write_files(self, tokens):
        input_ids_list = tokens['input_ids']
        attention_mask_list = tokens['attention_mask']
        pos_ids = np.arange(self.seq_len, dtype=np.int64)
        if self.num_past:
            zero_past = np.zeros(self.past_shape, dtype=np.float32)
        input_paths = []
        for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            input_ids_path = self.outdir / str(i) / 'input_ids.raw'
            pos_ids_path = self.outdir / str(i) / 'position_ids.raw'
            mask_path = self.outdir / str(i) / 'attention_mask.raw'
            input_ids = np.array(input_ids, dtype=np.int64)
            attention_mask = np.array(attention_mask, dtype=self.mask_dtype)
            (self.outdir / str(i)).mkdir(parents=True, exist_ok=False)
            input_ids.tofile(input_ids_path)
            pos_ids.tofile(pos_ids_path)
            attention_mask.tofile(mask_path)
            if self.pos_id_required:
                paths = [input_ids_path, pos_ids_path, mask_path]
            else:
                paths = [input_ids_path, mask_path]
            for past_idx in range(self.num_past):
                past_path = self.outdir / str(i) / f'zero_past_{past_idx}.raw'
                zero_past.tofile(past_path)
                paths.append(past_path)
            input_paths.append([x.relative_to(self.outdir) for x in paths])
        return input_paths

    def tokenize(self):
        '''read the text from files and tokenize
        return a dictionary {
            'input_ids': [[], ...]
            'attention_mask': [[], ...]
        }
        where each sub list is of length self.seq_len
        '''

        def read_file(filename):
            path = self.dataset_path / filename
            return open(path, encoding="utf-8").readlines()

        def grouper(tokens):
            all_tokens = {k: sum(map(lambda x: x[k], tokens), []) for k in tokens[0].keys()}
            total_length = len(all_tokens[list(all_tokens.keys())[0]])
            # if total_length >= seq_len:
            #     total_length = (total_length // seq_len) * seq_len
            result = {
                k: [t[i:i + self.seq_len] for i in range(0, total_length, self.seq_len)]
                for k, t in all_tokens.items()
            }
            return result

        transformers = Helper.safe_import_package("transformers", "4.31.0")
        # read the paths to files that contain text
        with open(self.input_file) as f:
            files = [x.strip() for x in f.readlines()]
        # read the text and tokenize
        if os.path.isdir(self.cached_path):
            qacc_file_logger.info(f"loading tokenizer from cached folder {self.cached_path}")
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.cached_path)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        data = sum(map(read_file, files), [])
        tokens = list(map(tokenizer, data))
        # split by seq_length
        grouped = grouper(tokens)
        # zero pad the last list/ group
        for k in grouped:
            pad_len = self.seq_len - len(grouped[k][-1])
            grouped[k][-1] += [0] * pad_len
        # left pad the attention_mask by past_seq_length
        for i in range(len(grouped['attention_mask'])):
            grouped['attention_mask'][i] = [0] * self.past_seq_len + grouped['attention_mask'][i]
        del tokenizer
        return grouped
