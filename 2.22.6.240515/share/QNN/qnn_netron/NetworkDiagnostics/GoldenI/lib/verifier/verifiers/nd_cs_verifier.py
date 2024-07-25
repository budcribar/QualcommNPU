# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path

from numpy import dot
from numpy.linalg import norm
import pandas as pd

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError
from lib.utils.nd_exceptions import VerifierInputError
from lib.verifier.models.nd_verify_result import VerifyResult
from lib.verifier.verifiers.nd_base_verifier import BaseVerifier


class CosineSimilarityVerifier(BaseVerifier):
    INF_ARRAY_SIZE = 'inference_tensor_size'
    GOLD_ARRAY_SIZE = 'golden_tensor_size'
    INF_MIN_VAL = 'inference_tensor_min_value'
    GOLD_MIN_VAL = 'golden_tensor_min_value'
    INF_MAX_VAL = 'inference_tensor_max_value'
    GOLD_MAX_VAL = 'gold_tensor_max_value'
    INF_RANGE = 'inference_tensor_range:abs(max - min)'
    GOLD_RANGE = 'golden_tensor_range:abs(max - min)'
    COSINE_SIMILARITY = 'cosine_similarity'
    NAME = 'Cosine Similarity'
    UNITS = ''

    def __init__(self, multiplier=1.0, scale=1.0):
        super(CosineSimilarityVerifier, self).__init__()
        self.multiplier = multiplier
        self.scale = scale
        self._result_dataframe = None

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):

        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_COSINE_SIMILARITY_INCORRECT_INPUT_SIZE"))

        golden_raw = golden_output[0]
        inference_raw = inference_output[0]

        if len(golden_raw) != len(inference_raw):
            raise VerifierError(get_message("ERROR_VERIFIER_COSINE_SIMILARITY_DIFFERENT_SIZE")
                                (str(len(golden_raw)), (str(len(inference_raw)))))

        cosine_similarity = dot(golden_raw, inference_raw) / (norm(golden_raw) * norm(inference_raw))
        self._setup_dataframe(cosine_similarity=cosine_similarity,
                              inf_size=len(inference_raw), gold_size=len(golden_raw),
                              inf_min=min(inference_raw), gold_min=min(golden_raw),
                              inf_max=max(inference_raw), gold_max=max(golden_raw),
                              inf_range=abs(max(inference_raw) - min(inference_raw)),
                              gold_range=abs(max(golden_raw) - min(golden_raw)))
        return VerifyResult(layer_type, len(golden_raw), tensor_dimensions, CosineSimilarityVerifier.NAME,
                            cosine_similarity, CosineSimilarityVerifier.UNITS)

    def _setup_dataframe(self, cosine_similarity, inf_size, gold_size, inf_min, gold_min, inf_max,
                         gold_max, inf_range, gold_range):
        self._result_dataframe = pd.DataFrame(columns=[CosineSimilarityVerifier.COSINE_SIMILARITY,
                                                       CosineSimilarityVerifier.INF_ARRAY_SIZE,
                                                       CosineSimilarityVerifier.GOLD_ARRAY_SIZE,
                                                       CosineSimilarityVerifier.INF_MIN_VAL,
                                                       CosineSimilarityVerifier.GOLD_MIN_VAL,
                                                       CosineSimilarityVerifier.INF_MAX_VAL,
                                                       CosineSimilarityVerifier.GOLD_MAX_VAL,
                                                       CosineSimilarityVerifier.INF_RANGE,
                                                       CosineSimilarityVerifier.GOLD_RANGE])

        self._result_dataframe[CosineSimilarityVerifier.COSINE_SIMILARITY] = [cosine_similarity]
        self._result_dataframe[CosineSimilarityVerifier.INF_ARRAY_SIZE] = [inf_size]
        self._result_dataframe[CosineSimilarityVerifier.GOLD_ARRAY_SIZE] = [gold_size]
        self._result_dataframe[CosineSimilarityVerifier.INF_MIN_VAL] = [inf_min]
        self._result_dataframe[CosineSimilarityVerifier.GOLD_MIN_VAL] = [gold_min]
        self._result_dataframe[CosineSimilarityVerifier.INF_MAX_VAL] = [inf_max]
        self._result_dataframe[CosineSimilarityVerifier.GOLD_MAX_VAL] = [gold_max]
        self._result_dataframe[CosineSimilarityVerifier.INF_RANGE] = [inf_range]
        self._result_dataframe[CosineSimilarityVerifier.GOLD_RANGE] = [gold_range]

    def _to_csv(self, file_path):
        if isinstance(self._result_dataframe, pd.DataFrame):
            self._result_dataframe.to_csv(file_path, encoding='utf-8', index=False)

    def _to_html(self, file_path):
        if isinstance(self._result_dataframe, pd.DataFrame):
            self._result_dataframe.to_html(file_path, classes='table', index=False)

    def save_data(self, path):
        filename = Path(path)
        self._to_csv(filename.with_suffix(".csv"))
        self._to_html(filename.with_suffix(".html"))

    @staticmethod
    def validate_config(configs):
        return True, {}
