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


class MSEVerifier(BaseVerifier):
    INF_ARRAY_SIZE = 'inference_tensor_size'
    GOLD_ARRAY_SIZE = 'golden_tensor_size'
    INF_MIN_VAL = 'inference_tensor_min_value'
    GOLD_MIN_VAL = 'golden_tensor_min_value'
    INF_MAX_VAL = 'inference_tensor_max_value'
    GOLD_MAX_VAL = 'gold_tensor_max_value'
    INF_MEAN = 'inference_tensor_mean'
    GOLD_MEAN = 'golden_tensor_mean'
    MSE = 'MSE:mean((x - y)^2)'
    DELTA = 'golden_raw - inference_raw'
    NAME = 'MSE'
    UNITS = 'Mean Squared Error'

    def __init__(self):
        super(MSEVerifier, self).__init__()
        self._result_dataframe = None

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):

        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_MSE_INCORRECT_INPUT_SIZE"))

        golden_raw = golden_output[0]
        inference_raw = inference_output[0]

        if len(golden_raw) != len(inference_raw):
            raise VerifierError(get_message("ERROR_VERIFIER_MSE_DIFFERENT_SIZE")
                                (str(len(golden_raw)), (str(len(inference_raw)))))

        delta = golden_raw - inference_raw
        mse = np.mean(delta ** 2)
        self._setup_dataframe(mse=mse, delta=delta,
                              inf_size=len(inference_raw), gold_size=len(golden_raw),
                              inf_min=min(inference_raw), gold_min=min(golden_raw),
                              inf_max=max(inference_raw), gold_max=max(golden_raw),
                              inf_mean=np.mean(inference_raw),
                              gold_mean=np.mean(golden_raw))
        return VerifyResult(layer_type, len(golden_raw), tensor_dimensions, MSEVerifier.NAME, mse,
                            MSEVerifier.UNITS)

    def _setup_dataframe(self, mse, delta, inf_size, gold_size, inf_min, gold_min, inf_max,
                         gold_max, inf_mean, gold_mean):
        self._result_dataframe = pd.DataFrame(columns=[MSEVerifier.MSE,
                                                       MSEVerifier.DELTA,
                                                       MSEVerifier.INF_ARRAY_SIZE,
                                                       MSEVerifier.GOLD_ARRAY_SIZE,
                                                       MSEVerifier.INF_MIN_VAL,
                                                       MSEVerifier.GOLD_MIN_VAL,
                                                       MSEVerifier.INF_MAX_VAL,
                                                       MSEVerifier.GOLD_MAX_VAL,
                                                       MSEVerifier.INF_MEAN,
                                                       MSEVerifier.GOLD_MEAN])

        self._result_dataframe[MSEVerifier.MSE] = [mse]
        self._result_dataframe[MSEVerifier.DELTA] = [delta]
        self._result_dataframe[MSEVerifier.INF_ARRAY_SIZE] = [inf_size]
        self._result_dataframe[MSEVerifier.GOLD_ARRAY_SIZE] = [gold_size]
        self._result_dataframe[MSEVerifier.INF_MIN_VAL] = [inf_min]
        self._result_dataframe[MSEVerifier.GOLD_MIN_VAL] = [gold_min]
        self._result_dataframe[MSEVerifier.INF_MAX_VAL] = [inf_max]
        self._result_dataframe[MSEVerifier.GOLD_MAX_VAL] = [gold_max]
        self._result_dataframe[MSEVerifier.INF_MEAN] = [inf_mean]
        self._result_dataframe[MSEVerifier.GOLD_MEAN] = [gold_mean]

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