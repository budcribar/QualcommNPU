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

import os
import pickle
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_postprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class squad_postprocess(qacc_memory_postprocessor):
    """Predict Answers for SQuAD dataset given start and end scores.

    Used for Roberta model.
    """
    PACKING_MAP_FNAME = 'packing_map.pickle'
    FEATURES_CACHE_FNAME = 'cached_features.pickle'

    def __init__(self, packing_strategy=False, **kwargs):
        transformers = Helper.safe_import_package("transformers", "4.31.0")
        work_dir = self.extra_params['work_dir']
        cached_features_file = os.path.join(work_dir, squad_postprocess.FEATURES_CACHE_FNAME)
        # parameters for "packing" strategy
        self.do_unpacking = packing_strategy
        packing_map_cache_path = os.path.join(work_dir, squad_postprocess.PACKING_MAP_FNAME)

        if self.do_unpacking:
            if not os.path.exists(packing_map_cache_path):
                qacc_file_logger.error(
                    f'invalid path for packing_map_cache: {packing_map_cache_path}')
                return
            with open(packing_map_cache_path, 'rb') as cache_file:
                self.output_map = pickle.load(cache_file)

        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'rb') as data:
                data = pickle.load(data)
            self.features = data["features"]
            qacc_file_logger.info(f'Loaded features from {str(cached_features_file)}')
        else:
            qacc_file_logger.error("cached features file is not present")
            return

    def execute(self, data, meta, input_idx, *args, **kwargs):
        transformers = Helper.safe_import_package("transformers", "4.31.0")
        start_logits = data[0][0]
        end_logits = data[1][0]
        out_data = []
        if self.do_unpacking:
            # info on which features were packed together
            feat_idx_and_sls = self.output_map[input_idx]
            feat_idxs = [x[0] for x in feat_idx_and_sls]
            sls = [x[1] for x in feat_idx_and_sls]

            # unpack
            packed_data = [start_logits, end_logits]
            unpacked_data = self.unpack(packed_data, sls)

            # add to "results" for each original feature that was packed
            for data_idx, feat_idx in enumerate(feat_idxs):
                result = transformers.data.processors.squad.SquadResult(
                    self.features[feat_idx].unique_id, list(unpacked_data[data_idx][0]),
                    list(unpacked_data[data_idx][1]))
                out_data.append(result)
        else:
            result = transformers.data.processors.squad.SquadResult(
                self.features[input_idx].unique_id, list(start_logits), list(end_logits))
            out_data.append(result)
        return out_data, meta

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
