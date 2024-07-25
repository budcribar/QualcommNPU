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

import os
import pickle
from typing import DefaultDict, List, Tuple
from queue import Queue
from collections import defaultdict, namedtuple
import numpy as np
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.create_squad_data import (
    convert_examples_to_features, InputFeatures)
from qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.spfhp import pack_using_spfhp
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger

PackedInputs = namedtuple('PackedInputs',
                          ['input_ids', 'input_mask', 'segment_ids', 'input_position_ids'])


def get_features_from_cache(cache_path):
    eval_features = None
    if cache_path and os.path.exists(cache_path):
        print(f'Reading features from cache: {cache_path}')
        with open(cache_path, 'rb') as cache_file:
            eval_features = pickle.load(cache_file)
    return eval_features


def read_squad_examples(input_file, process=True):
    """Read a SQuAD pickle file into a list of SquadExample."""
    """Arg process=True is used to read the file from input_file list which
    [['path/inputfile.pkl']]."""
    """Arg process=False is used to read the file from input_file list which
    [[['path/inputfile.pkl']]]."""

    examples = []
    for i in input_file:
        if process == True:
            with open(i[0], 'rb') as fp:
                input_data = pickle.load(fp)
        else:
            with open(i[0][0], 'rb') as fp:
                input_data = pickle.load(fp)
        examples.append(input_data)
    return examples


def get_features(*, cache_path, dev_path, vocab_path, max_seq_length, doc_stride,
                 max_query_length) -> Tuple[List, bool]:
    """Read examples, preprocess, and return features, status."""

    if vocab_path is None:
        print('"vocab_path" param is required to preprocess examples.')
        return None, False

    # read "examples" from dev json
    eval_examples = read_squad_examples(input_file=dev_path)
    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)

    # convert "examples" to "features" - preprocessing
    transformers = Helper.safe_import_package("transformers", "4.31.1")
    tokenizer = transformers.BertTokenizer(vocab_path)
    convert_examples_to_features(examples=eval_examples, tokenizer=tokenizer,
                                 max_seq_length=max_seq_length, doc_stride=doc_stride,
                                 max_query_length=max_query_length, output_fn=append_feature)

    # if cache_path is provided, create features cache
    if cache_path:
        print(f'Caching features at: {cache_path}')
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(eval_features, cache_file)

    return eval_features, True


class squad_read(qacc_plugin):
    """Reads the SQuAD dataset json file.

    Pre processes the question-context pairs into features for Language
    models like BERT-Large.
    """

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

    PACKING_MAP_FNAME = 'packing_map.pickle'
    FEATURES_CACHE_FNAME = 'features.pickle'

    def __init__(self):
        transformers = Helper.safe_import_package("transformers", "4.31.1")

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):
        # Dir plugins always have a single entry.
        self.execute_index(pin_list[0], pout_list[0])

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        if not pin.is_directory_input():
            print('Only directory based input supported for', self.__class__.__name__)
            return

        inputs = pin.get_input()
        outdir = pout.get_out_dir()
        work_dir = pin.read_pipeline_cache_val(qcc.PIPELINE_WORK_DIR)

        # path to read pickle files created during dataset loading
        dev_path = inputs

        # parameters for feature generation
        vocab_path = pin.get_param('vocab_path', None)
        max_seq_length = pin.get_param('max_seq_length', 384)
        doc_stride = pin.get_param('doc_stride', 128)
        max_query_length = pin.get_param('max_query_length', 64)
        features_cache_path = os.path.join(work_dir, squad_read.FEATURES_CACHE_FNAME)

        # parameters for "packing" strategy
        do_packing = pin.get_param('packing_strategy', False)
        max_sequence_per_pack = pin.get_param('max_sequence_per_pack', 3)
        packing_map_cache_path = os.path.join(work_dir, squad_read.PACKING_MAP_FNAME)
        print(f'in squad_read, doing packing_strategy: {do_packing}')

        # is the mask boolean or int64 or compressed mask?
        mask_type = pin.get_param('mask_type', None)  # None, boolean, compressed
        compressed_mask_len = pin.get_param('compressed_mask_length', None)

        # are we processing calibration data?
        is_calib = pin.read_pipeline_cache_val(qcc.PIPELINE_PREPROC_IS_CALIB)

        # control the files written based on number of inputs to model
        num_inputs = len(pin.read_pipeline_cache_val(qcc.PIPELINE_INFER_INPUT_INFO))

        # get the features
        eval_features, correct = get_features(cache_path=None if is_calib else features_cache_path,
                                              dev_path=dev_path, vocab_path=vocab_path,
                                              max_seq_length=max_seq_length, doc_stride=doc_stride,
                                              max_query_length=max_query_length)
        if not correct:
            pout.set_status(qcc.STATUS_ERROR)
            return

        outdir = os.path.join(outdir, 'pre')
        if do_packing:
            # eval_features is of type PackedInputs and not InputFeatures after packing
            eval_features, packing_map = self.do_packing_strategy(
                features=eval_features, max_sequence_length=max_seq_length,
                max_sequence_per_pack=max_sequence_per_pack, mask_type=mask_type,
                compressed_mask_len=compressed_mask_len)
            # do not save the mapping file for calibration data
            if not is_calib:
                with open(packing_map_cache_path, 'wb') as cache_file:
                    pickle.dump(packing_map, cache_file)
        # save features to file
        # output files will be saved in following format
        #   pin.get_out_dir() / 'pre' / index / 'input_ids.raw'
        #   pin.get_out_dir() / 'pre' / index / 'input_mask.raw'
        #   pin.get_out_dir() / 'pre' / index / 'segment_ids.raw'
        boolean_mask = mask_type and mask_type.lower().startswith('bool')
        out_filepaths = squad_read.write_features(eval_features, outdir,
                                                  is_packed_inputs=do_packing,
                                                  boolean_mask=boolean_mask, num_inputs=num_inputs)
        # out_filepaths has following format
        # [ ['path/to/input_ids_0.raw', 'path/to/input_mask_0.raw', 'path/to/segment_ids_0.raw'],
        #   ['path/to/input_ids_1.raw', 'path/to/input_mask_1.raw', 'path/to/segment_ids_1.raw'],
        # ...(num_features)...
        # ]
        # in case of packing strategy, each sublist would be
        # ['path/to/input_ids_0.raw', 'path/to/input_mask_0.raw', 'path/to/segment_ids_0.raw', 'input_position_ids_0.raw']
        pout.set_dir_outputs(out_filepaths)
        pout.set_status(qcc.STATUS_SUCCESS)

    @staticmethod
    def write_features(features: List, outdir: str, is_packed_inputs: bool = False,
                       boolean_mask: bool = False, num_inputs=3) -> List[str]:
        # we will get a list of InputFeatures in case of non-packing approach
        #     and a list of PackedInputs is case of packing approach
        # if is_packed_inputs is True,
        #    then, write one extra file
        field_names = ['input_ids', 'input_mask', 'segment_ids']
        if is_packed_inputs:
            field_names.append('input_position_ids')
        field_names = field_names[:num_inputs]

        extension = '.raw'
        # datatypes, to handle special case of boolean mask
        dtypes = [np.int64] * len(field_names)
        if boolean_mask:
            dtypes[1] = bool

        out_filepaths = []
        for i, eval_feature in enumerate(features):
            dirpath = os.path.join(outdir, str(i))
            os.makedirs(dirpath, exist_ok=True)

            # use getattr instead of f.input_ids so that dynamic field_names is supported
            arrays = [getattr(eval_feature, fname) for fname in field_names]
            file_paths = [os.path.join(dirpath, fname + extension) for fname in field_names]

            # file write
            for arr, fpath, dtype in zip(arrays, file_paths, dtypes):
                arr = np.array(arr).astype(dtype)
                arr.tofile(fpath)

            # add to output list
            out_filepaths.append(file_paths)

        return out_filepaths

    def do_packing_strategy(self, *, features: List[InputFeatures], max_sequence_per_pack: int,
                            max_sequence_length: int, mask_type: str,
                            compressed_mask_len: int = None):
        print('Getting Strategy set and frequency')
        # generate histogram
        sequence_lengths = [sum(f.input_mask) for f in features]
        histogram = [0] * max_sequence_length
        for sl in sequence_lengths:
            histogram[sl - 1] += 1

        # get packing strategy
        strategy_set, strategy_repeat_count = pack_using_spfhp(np.array(histogram),
                                                               max_sequence_length,
                                                               max_sequence_per_pack)

        # pack and write packed sequences
        print('Packing the Features')
        packed_inputs, packing_map = self.do_packing(features, sequence_lengths,
                                                     strategy_set=strategy_set,
                                                     strategy_repeat_count=strategy_repeat_count,
                                                     max_sequence_length=max_sequence_length,
                                                     mask_type=mask_type,
                                                     compressed_mask_len=compressed_mask_len)

        print(f'Number of sequences before packing: {len(features)}')
        print(f'Number of sequences after packing : {len(packed_inputs)}')
        return packed_inputs, packing_map

    @staticmethod
    def gen_2d_mask(sequence_lengths, padding_length):
        # create block diagonal matrix for mask
        # mask = [0,0,0,1,1]
        # input_mask = [ ones(3,3), zeros(3,2)
        #                zeros(2,3), ones(2,2) ]
        mask = np.concatenate([[i] * sl for i, sl in enumerate(sequence_lengths)])[np.newaxis, :]
        input_mask = 1 * np.equal(mask, mask.transpose())
        input_mask = input_mask.astype(np.int64)
        input_mask = np.pad(input_mask, [0, padding_length])
        return input_mask

    @staticmethod
    def pack_features(feat_list: List[InputFeatures], sls: List[int], max_sl: int = 384,
                      mask_type: str = None, compressed_mask_len: int = None):
        """Pack together the provided features provide mask_type='compressed'
        for compressed mask along with compressed_mask_len (defaults to
        len(sls))"""

        input_ids = np.concatenate([feat.input_ids[:sl] for feat, sl in zip(feat_list, sls)])
        segment_ids = np.concatenate([feat.segment_ids[:sl] for feat, sl in zip(feat_list, sls)])

        input_ids = input_ids.astype(np.int64)
        segment_ids = segment_ids.astype(np.int64)

        # create input_position_ids
        position_ids = np.concatenate([np.arange(sl, dtype=np.int64) for sl in sls])

        # padding
        pad_len = max_sl - sum(sls)
        assert pad_len >= 0
        assert len(input_ids) == len(segment_ids) == len(position_ids)

        if mask_type and mask_type.lower() == 'compressed':  # compressed mask
            input_mask = np.array(sls)
            if compressed_mask_len is not None:
                assert len(input_mask) <= compressed_mask_len
                input_mask = np.pad(input_mask, [0, compressed_mask_len - len(input_mask)])
        else:  # 2D mask
            input_mask = squad_read.gen_2d_mask(sls, pad_len)

        input_ids = np.pad(input_ids, [0, pad_len])
        segment_ids = np.pad(segment_ids, [0, pad_len])
        position_ids = np.pad(position_ids, [0, pad_len])

        return PackedInputs(input_ids, input_mask, segment_ids, position_ids)

    def do_packing(self, features, sequence_lengths, *, strategy_set, strategy_repeat_count,
                   max_sequence_length, mask_type, compressed_mask_len=None):
        # create queues for each Sequence Length
        # and fill them up with the respective features (along with their index)
        features_by_sl = [Queue() for _ in range(max_sequence_length + 1)]
        for i, (feat, sl) in enumerate(zip(features, sequence_lengths)):
            features_by_sl[sl].put((i, feat))

        # store which features (and what sl) were packed into which directory
        # key: output dir (idx)
        # value: list of input feat (idx), and sl
        # example, {0:[(5001, 192), (5600, 191)]}
        output_map: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        output_idx = 0

        # iterate over strategy
        # and do the packing
        all_packed_inputs: List[PackedInputs] = []
        for group, group_freq in zip(strategy_set, strategy_repeat_count):
            # group (list) contains the SL that has to be packed together
            # pack it "group_freq" times
            for _ in range(group_freq):
                # get data to be packed together
                to_pack = []
                for sl in group:  # group: (192,190)
                    idx, feat = features_by_sl[sl].get_nowait()
                    to_pack.append(feat)
                    # store which directories are packed together
                    output_map[output_idx].append((idx, sl))

                packed_inputs = squad_read.pack_features(to_pack, group, max_sequence_length,
                                                         mask_type=mask_type,
                                                         compressed_mask_len=compressed_mask_len)
                all_packed_inputs.append(packed_inputs)
                output_idx += 1

        # verify if packed all
        for i in range(max_sequence_length + 1):
            assert features_by_sl[i].empty()

        return all_packed_inputs, output_map
