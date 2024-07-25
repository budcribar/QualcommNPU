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

import os
import pickle
import numpy as np
from queue import Queue
from collections import defaultdict, namedtuple
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_dataset, DatasetPluginInputInfo, DatasetPluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper

PackedInputs = namedtuple('PackedInputs',
                          ['input_ids', 'attention_mask', 'token_type_ids', 'input_position_ids'])


class create_squad_examples(qacc_dataset):
    """Used for creating examples from squad json file."""
    PACKING_MAP_FNAME = 'packing_map.pickle'
    FEATURES_CACHE_FNAME = 'cached_features.pickle'

    def __init__(self):
        transformers = Helper.safe_import_package("transformers", "4.31.0")

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):
        transformers = Helper.safe_import_package("transformers", "4.31.0")
        input_path = d_in.get_annotation_file()
        calib_path = d_in.get_calibration_file()
        outdir = d_out.get_out_dir()

        version = d_in.get_param('squad_version', 1)
        max_inputs = d_in.get_param("max_inputs", -1)
        max_calib = d_in.get_param("max_calib", -1)

        if calib_path.endswith('.txt'):
            json_path = open(calib_path, 'r').read().strip()
            calib_path = calib_path.rsplit('/', 1)[0] + '/' + json_path

        if version == 1:
            processor = transformers.data.processors.squad.SquadV1Processor()
            qacc_file_logger.info('Loaded Squadv1 processor')
        elif version == 2:
            processor = transformers.data.processors.squad.SquadV2Processor()
            qacc_file_logger.info('Loaded Squadv2 processor')
        else:
            qacc_file_logger.error('Only squad versions 1 and 2 are supported')
            d_out.set_status(qcc.STATUS_ERROR)
            return

        if input_path:
            data_file_path = self.save_examples(d_in, d_out, processor, input_path, outdir,
                                                'input_data', max_inputs)
            d_out.set_inputlist_file(data_file_path)
            d_out.inputlist_path_modified = True
        if calib_path:
            data_file_path = self.save_examples(d_in, d_out, processor, calib_path, outdir,
                                                'calib_data', max_calib)
            d_out.set_calibration_file(data_file_path)
            d_out.calibration_path_modified = True

        # Set output
        d_out.set_status(qcc.STATUS_SUCCESS)

    def save_examples(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo, processor,
                      data_path, outdir, data_type, max_count):

        vocab_info = '''vocabulary param can be any one of the below:
        1) vocabulary from huggingface.co and cache (e.g. "bert-base-uncased")
        2) vocabulary from huggingface.co (user-uploaded) and cache (e.g. "deepset/roberta-base-squad2")
        3) path for local directory containing vocabulary files(tokenizer was saved using _save_pretrained('./test/saved_model/')_)
        '''
        transformers = Helper.safe_import_package("transformers", "4.31.0")
        vocabulary = d_in.get_param('vocabulary', None)
        max_seq_length = d_in.get_param('max_seq_length', 386)
        max_query_length = d_in.get_param('max_query_length', 64)
        doc_stride = d_in.get_param('doc_stride', 128)
        threads = d_in.get_param('threads', 8)
        do_lower_case = d_in.get_param('do_lower_case', True)
        num_inputs = d_in.get_param("model_inputs_count", 2)
        cached_vocab_path = d_in.get_param('cached_vocab_path', None)

        # parameters for "packing" strategy
        do_packing = d_in.get_param('packing_strategy', False)
        max_sequence_per_pack = d_in.get_param('max_sequence_per_pack', 3)
        qacc_file_logger.info(f'in create_examples, doing packing_strategy: {do_packing}')

        # is the mask boolean or int64 or compressed mask?
        mask_type = d_in.get_param('mask_type', None)  # None, boolean, compressed
        compressed_mask_len = d_in.get_param('compressed_mask_length', None)

        if vocabulary is None:
            qacc_file_logger.error("'vocabulary' parameter is mandatory for loading tokenizer")
            qacc_file_logger.info(vocab_info)
            d_out.set_status(qcc.STATUS_ERROR)
            return

        work_dir = outdir.replace('/dataset', '')
        outdir = os.path.join(outdir, data_type)
        qacc_file_logger.info(f'Loading examples from {str(data_path)}')
        examples = processor.get_dev_examples('', filename=data_path)

        if max_count != -1:
            examples = examples[:max_count]

        os.makedirs(outdir, exist_ok=True)
        qacc_file_logger.info(f'Saving examples to {str(outdir)}')
        data_file_path = os.path.join(outdir, 'datafile.txt')
        txt = open(data_file_path, 'w')

        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        if do_packing:
            input_names.append('input_position_ids')
        model_inputs = input_names[:num_inputs]
        qacc_file_logger.info(f'Model inputs: {str(model_inputs)}')

        if cached_vocab_path != None and os.path.isdir(cached_vocab_path):
            tokenizer = transformers.AutoTokenizer.from_pretrained(cached_vocab_path,
                                                                   do_lower_case=do_lower_case,
                                                                   use_fast=False)
            qacc_file_logger.info(f"Loading from local cached path at {str(cached_vocab_path)}")
        else:
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    vocabulary, do_lower_case=do_lower_case, use_fast=False)
            except Exception:
                qacc_logger.error(
                    "Failed to download tokenizer. Provide path for cached vocabulary instead.")
                return

        features = transformers.squad_convert_examples_to_features(
            examples=examples,  #[examples[0], examples[1]], #[examples[:2]],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            threads=threads,
        )
        qacc_file_logger.info("Examples Count: {}, Features Count: {}".format(
            len(examples), len(features)))

        if data_type != 'calib_data':
            #save cached_features.pickle file in workdir path
            cached_features_file = os.path.join(work_dir,
                                                create_squad_examples.FEATURES_CACHE_FNAME)

            with open(cached_features_file, 'wb') as cache_file:
                pickle.dump({"features": features, "examples": examples}, cache_file)
            with open(cached_features_file, 'rb') as data:
                data = pickle.load(data)
            qacc_file_logger.info(f'Saved features at {str(cached_features_file)}')

        if do_packing:
            # eval_features is of type PackedInputs after packing
            features, packing_map = create_squad_examples.do_packing_strategy(
                features=features, max_sequence_length=max_seq_length,
                max_sequence_per_pack=max_sequence_per_pack, mask_type=mask_type,
                compressed_mask_len=compressed_mask_len)
            # do not save the mapping file for calibration data
            if data_type != 'calib_data':
                packing_map_cache_path = os.path.join(work_dir,
                                                      create_squad_examples.PACKING_MAP_FNAME)
                with open(packing_map_cache_path, 'wb') as cache_file:
                    pickle.dump(packing_map, cache_file)

        boolean_mask = mask_type and mask_type.lower().startswith('bool')
        for index, feature in enumerate(features):
            paths = ""
            for model_input in model_inputs:
                filename = 'feature_' + str(index) + "_" + model_input + '.raw'
                dtype = np.int64
                if model_input == 'attention_mask' and boolean_mask is True:
                    dtype = bool
                path = create_squad_examples.save_raw_file(getattr(feature, model_input), outdir,
                                                           filename, dtype)
                path = path.replace(d_out.get_out_dir() + '/', '')
                paths += path + ','
            txt.write(paths[:-1] + '\n')

        txt.close()
        return data_file_path

    @staticmethod
    def save_raw_file(data, dirpath, filename, dtype):
        data = np.asarray([data]).astype(dtype)
        save_path = os.path.join(dirpath, filename)
        data.tofile(save_path)
        return save_path

    @staticmethod
    def do_packing_strategy(*, features, max_sequence_per_pack: int, max_sequence_length: int,
                            mask_type: str, compressed_mask_len: int = None):
        qacc_file_logger.info('Getting Strategy set and frequency')
        # generate histogram
        sequence_lengths = [sum(f.attention_mask) for f in features]
        histogram = [0] * max_sequence_length
        for sl in sequence_lengths:
            histogram[sl - 1] += 1

        # get packing strategy
        from qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.spfhp import pack_using_spfhp
        strategy_set, strategy_repeat_count = pack_using_spfhp(np.array(histogram),
                                                               max_sequence_length,
                                                               max_sequence_per_pack)

        # pack and write packed sequences
        qacc_file_logger.info('Packing the Features')
        packed_inputs, packing_map = create_squad_examples.do_packing(
            features, sequence_lengths, strategy_set=strategy_set,
            strategy_repeat_count=strategy_repeat_count, max_sequence_length=max_sequence_length,
            mask_type=mask_type, compressed_mask_len=compressed_mask_len)

        qacc_file_logger.info(f'Number of sequences before packing: {len(features)}')
        qacc_file_logger.info(f'Number of sequences after packing : {len(packed_inputs)}')
        return packed_inputs, packing_map

    @staticmethod
    def do_packing(features, sequence_lengths, *, strategy_set, strategy_repeat_count,
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
        output_map = defaultdict(list)
        output_idx = 0

        # iterate over strategy
        # and do the packing
        all_packed_inputs = []
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

                packed_inputs = create_squad_examples.pack_features(
                    to_pack, group, max_sequence_length, mask_type=mask_type,
                    compressed_mask_len=compressed_mask_len)
                all_packed_inputs.append(packed_inputs)
                output_idx += 1

        # verify if packed all
        for i in range(max_sequence_length + 1):
            assert features_by_sl[i].empty()

        return all_packed_inputs, output_map

    @staticmethod
    def pack_features(feat_list, sls, max_sl=384, mask_type=None, compressed_mask_len=None):
        """Pack together the provided features provide mask_type='compressed'
        for compressed mask along with compressed_mask_len (defaults to
        len(sls))"""

        input_ids = np.concatenate([feat.input_ids[:sl] for feat, sl in zip(feat_list, sls)])
        token_type_ids = np.concatenate(
            [feat.token_type_ids[:sl] for feat, sl in zip(feat_list, sls)])

        input_ids = input_ids.astype(np.int64)
        token_type_ids = token_type_ids.astype(np.int64)

        # create input_position_ids
        position_ids = np.concatenate([np.arange(sl, dtype=np.int64) for sl in sls])

        # padding
        pad_len = max_sl - sum(sls)
        assert pad_len >= 0
        assert len(input_ids) == len(token_type_ids) == len(position_ids)

        if mask_type and mask_type.lower() == 'compressed':  # compressed mask
            attention_mask = np.array(sls)
            if compressed_mask_len is not None:
                assert len(attention_mask) <= compressed_mask_len
                attention_mask = np.pad(attention_mask,
                                        [0, compressed_mask_len - len(attention_mask)])
        else:  # 2D mask
            attention_mask = create_squad_examples.gen_2d_mask(sls, pad_len)

        input_ids = np.pad(input_ids, [0, pad_len])
        token_type_ids = np.pad(token_type_ids, [0, pad_len])
        position_ids = np.pad(position_ids, [0, pad_len])

        return PackedInputs(input_ids, attention_mask, token_type_ids, position_ids)

    @staticmethod
    def gen_2d_mask(sequence_lengths, padding_length):
        # create block diagonal matrix for mask
        # mask = [0,0,0,1,1]
        # input_mask = [ ones(3,3), zeros(3,2)
        #                zeros(2,3), ones(2,2) ]
        mask = np.concatenate([[i] * sl for i, sl in enumerate(sequence_lengths)])[np.newaxis, :]
        attention_mask = 1 * np.equal(mask, mask.transpose())
        attention_mask = attention_mask.astype(np.int64)
        attention_mask = np.pad(attention_mask, [0, padding_length])
        return attention_mask
