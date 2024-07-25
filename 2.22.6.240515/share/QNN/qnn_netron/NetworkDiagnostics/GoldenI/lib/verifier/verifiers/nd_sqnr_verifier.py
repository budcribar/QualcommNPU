# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError
from lib.utils.nd_exceptions import VerifierInputError
from lib.verifier.models.nd_verify_result import VerifyResult
from lib.verifier.verifiers.nd_base_verifier import BaseVerifier


class SQNRVerifier(BaseVerifier):
    INF_ARRAY_SIZE = 'inference_tensor_size'
    GOLD_ARRAY_SIZE = 'golden_tensor_size'
    INF_MIN_VAL = 'inference_tensor_min_value'
    GOLD_MIN_VAL = 'golden_tensor_min_value'
    INF_MAX_VAL = 'inference_tensor_max_value'
    GOLD_MAX_VAL = 'gold_tensor_max_value'
    INF_MEAN = 'inference_tensor_mean'
    GOLD_MEAN = 'golden_tensor_mean'
    SQNR = 'SQNR: 10 * np.log10(mss / mse)'
    NAME = 'SQNR'
    UNITS = 'Signal-to-quantization-noise ratio'

    def __init__(self):
        super(SQNRVerifier, self).__init__()
        self._result_dataframe = None

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):
        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_SQNR_INCORRECT_INPUT_SIZE"))

        golden_raw = golden_output[0]
        inference_raw = inference_output[0]

        if len(golden_raw) != len(inference_raw):
            raise VerifierError(get_message("ERROR_VERIFIER_SQNR_DIFFERENT_SIZE")
                                (str(len(golden_raw)), (str(len(inference_raw)))))

        delta = golden_raw - inference_raw
        mse = np.mean(delta ** 2)
        if mse > 0:
            mss = np.mean(np.array(inference_raw) ** 2)
            sqnr = 10 * np.log10(mss / mse)
        else:
            sqnr = 'SAME'
        self._setup_dataframe(sqnr=sqnr,
                              inf_size=len(inference_raw), gold_size=len(golden_raw),
                              inf_min=min(inference_raw), gold_min=min(golden_raw),
                              inf_max=max(inference_raw), gold_max=max(golden_raw),
                              inf_mean=np.mean(inference_raw),
                              gold_mean=np.mean(golden_raw))
        return VerifyResult(layer_type, len(golden_raw), tensor_dimensions, SQNRVerifier.NAME,
                            sqnr, SQNRVerifier.UNITS)

    def _setup_dataframe(self, sqnr, inf_size, gold_size, inf_min, gold_min, inf_max,
                         gold_max, inf_mean, gold_mean):
        self._result_dataframe = pd.DataFrame(columns=[SQNRVerifier.SQNR,
                                                       SQNRVerifier.INF_ARRAY_SIZE,
                                                       SQNRVerifier.GOLD_ARRAY_SIZE,
                                                       SQNRVerifier.INF_MIN_VAL,
                                                       SQNRVerifier.GOLD_MIN_VAL,
                                                       SQNRVerifier.INF_MAX_VAL,
                                                       SQNRVerifier.GOLD_MAX_VAL,
                                                       SQNRVerifier.INF_MEAN,
                                                       SQNRVerifier.GOLD_MEAN])

        self._result_dataframe[SQNRVerifier.SQNR] = [sqnr]
        self._result_dataframe[SQNRVerifier.INF_ARRAY_SIZE] = [inf_size]
        self._result_dataframe[SQNRVerifier.GOLD_ARRAY_SIZE] = [gold_size]
        self._result_dataframe[SQNRVerifier.INF_MIN_VAL] = [inf_min]
        self._result_dataframe[SQNRVerifier.GOLD_MIN_VAL] = [gold_min]
        self._result_dataframe[SQNRVerifier.INF_MAX_VAL] = [inf_max]
        self._result_dataframe[SQNRVerifier.GOLD_MAX_VAL] = [gold_max]
        self._result_dataframe[SQNRVerifier.INF_MEAN] = [inf_mean]
        self._result_dataframe[SQNRVerifier.GOLD_MEAN] = [gold_mean]

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
