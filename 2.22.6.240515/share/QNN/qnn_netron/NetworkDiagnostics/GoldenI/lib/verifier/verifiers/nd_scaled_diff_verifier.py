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


class ScaledDiffVerifier(BaseVerifier):
    INF_ARRAY_SIZE = 'inference_tensor_size'
    GOLD_ARRAY_SIZE = 'golden_tensor_size'
    INF_MIN_VAL = 'inference_tensor_min_value'
    GOLD_MIN_VAL = 'golden_tensor_min_value'
    INF_MAX_VAL = 'inference_tensor_max_value'
    GOLD_MAX_VAL = 'gold_tensor_max_value'
    INF_MEAN = 'inference_tensor_mean'
    GOLD_MEAN = 'golden_tensor_mean'
    SCALEDDIFF = 'ScaledDiff: (np.abs(data - golden).mean()/(max_val - min_val)) * 255'
    NAME = 'ScaledDiff'
    UNITS = 'Scaled diff'

    def __init__(self):
        super(ScaledDiffVerifier, self).__init__()
        self._result_dataframe = None
        self._output_encoding = None

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):
        if not self._output_encoding:
            raise VerifierInputError(get_message("ERROR_VERIFIER_SCALED_DIFF_MISSING_OUTPUT_ENCODING"))
        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_SQNR_INCORRECT_INPUT_SIZE"))

        golden_raw = golden_output[0]
        inference_raw = inference_output[0]

        if len(golden_raw) != len(inference_raw):
            raise VerifierError(get_message("ERROR_VERIFIER_SQNR_DIFFERENT_SIZE")
                                (str(len(golden_raw)), (str(len(inference_raw)))))

        delta = np.abs(golden_raw - inference_raw)
        diff = np.mean(delta)
        scaled_diff = diff/self._output_encoding['scale'] if self._output_encoding['bw'] == 8 \
                                    else diff*255/(self._output_encoding['max'] - self._output_encoding['min'])

        self._setup_dataframe(scaled_diff=scaled_diff,
                              inf_size=len(inference_raw), gold_size=len(golden_raw),
                              inf_min=min(inference_raw), gold_min=min(golden_raw),
                              inf_max=max(inference_raw), gold_max=max(golden_raw),
                              inf_mean=np.mean(inference_raw),
                              gold_mean=np.mean(golden_raw))
        return VerifyResult(layer_type, len(golden_raw), tensor_dimensions, ScaledDiffVerifier.NAME,
                            scaled_diff, ScaledDiffVerifier.UNITS)

    def _setup_dataframe(self, scaled_diff, inf_size, gold_size, inf_min, gold_min, inf_max,
                         gold_max, inf_mean, gold_mean):
        self._result_dataframe = pd.DataFrame(columns=[ScaledDiffVerifier.SCALEDDIFF,
                                                       ScaledDiffVerifier.INF_ARRAY_SIZE,
                                                       ScaledDiffVerifier.GOLD_ARRAY_SIZE,
                                                       ScaledDiffVerifier.INF_MIN_VAL,
                                                       ScaledDiffVerifier.GOLD_MIN_VAL,
                                                       ScaledDiffVerifier.INF_MAX_VAL,
                                                       ScaledDiffVerifier.GOLD_MAX_VAL,
                                                       ScaledDiffVerifier.INF_MEAN,
                                                       ScaledDiffVerifier.GOLD_MEAN])

        self._result_dataframe[ScaledDiffVerifier.SCALEDDIFF] = [scaled_diff]
        self._result_dataframe[ScaledDiffVerifier.INF_ARRAY_SIZE] = [inf_size]
        self._result_dataframe[ScaledDiffVerifier.GOLD_ARRAY_SIZE] = [gold_size]
        self._result_dataframe[ScaledDiffVerifier.INF_MIN_VAL] = [inf_min]
        self._result_dataframe[ScaledDiffVerifier.GOLD_MIN_VAL] = [gold_min]
        self._result_dataframe[ScaledDiffVerifier.INF_MAX_VAL] = [inf_max]
        self._result_dataframe[ScaledDiffVerifier.GOLD_MAX_VAL] = [gold_max]
        self._result_dataframe[ScaledDiffVerifier.INF_MEAN] = [inf_mean]
        self._result_dataframe[ScaledDiffVerifier.GOLD_MEAN] = [gold_mean]

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

    def set_output_encoding(self, output_encoding):
        self._output_encoding = output_encoding
    @staticmethod
    def validate_config(configs):
        return True, {}