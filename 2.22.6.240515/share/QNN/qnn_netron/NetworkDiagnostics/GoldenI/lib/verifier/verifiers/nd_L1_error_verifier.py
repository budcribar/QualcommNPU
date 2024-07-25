# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
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


class L1ErrorVerifier(BaseVerifier):
    INF_ARRAY_SIZE = 'inference_tensor_size'
    GOLD_ARRAY_SIZE = 'golden_tensor_size'
    INF_MIN_VAL = 'inference_tensor_min_value'
    GOLD_MIN_VAL = 'golden_tensor_min_value'
    INF_MAX_VAL = 'inference_tensor_max_value'
    GOLD_MAX_VAL = 'gold_tensor_max_value'
    INF_RANGE = 'inference_tensor_range:abs(max - min)'
    GOLD_RANGE = 'golden_tensor_range:abs(max - min)'
    ERROR = 'error_absolute (sum_of_absolute_delta * multiplier * scale)'
    DELTA = 'golden_raw - inference_raw'
    SUM = 'sum_of_absolute_delta'
    MEAN = 'mean_of_absolute_delta'
    MULTIPLIER = 'multiplier'
    SCALE = 'scale'
    L1_NAME = 'L1'
    L1_UNITS = 'Absolute Error'
    MAE_NAME = 'MAE'
    MAE_UNITS = 'Mean Absolute Error'

    def __init__(self, mean=False, multiplier=1.0, scale=1.0):
        super(L1ErrorVerifier, self).__init__()
        self.multiplier = multiplier
        self.scale = scale
        self.mean = mean
        self._result_dataframe = None

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):

        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_L1ERROR_INCORRECT_INPUT_SIZE"))

        golden_raw = golden_output[0]
        inference_raw = inference_output[0]

        if len(golden_raw) != len(inference_raw):
            raise VerifierError(get_message("ERROR_VERIFIER_L1ERROR_DIFFERENT_SIZE")
                                (str(len(golden_raw)), (str(len(inference_raw)))))

        delta = golden_raw - inference_raw
        l1_mean = np.mean(np.absolute(delta))
        l1_sum = np.sum(np.absolute(delta))
        l1_error = l1_sum * self.multiplier * self.scale
        self._setup_dataframe(error=l1_error, sum=l1_sum, mean=l1_mean, delta=delta,
                              inf_size=len(inference_raw), gold_size=len(golden_raw),
                              inf_min=min(inference_raw), gold_min=min(golden_raw),
                              inf_max=max(inference_raw), gold_max=max(golden_raw),
                              inf_range=abs(max(inference_raw) - min(inference_raw)),
                              gold_range=abs(max(golden_raw) - min(golden_raw)))
        if self.mean:
            return VerifyResult(layer_type, len(golden_raw), tensor_dimensions, L1ErrorVerifier.MAE_NAME,
                                l1_mean, L1ErrorVerifier.MAE_UNITS)
        else:
            return VerifyResult(layer_type, len(golden_raw), tensor_dimensions, L1ErrorVerifier.L1_NAME,
                                l1_error, L1ErrorVerifier.L1_UNITS)

    def _setup_dataframe(self, error, sum, mean, delta, inf_size, gold_size, inf_min, gold_min, inf_max,
                         gold_max, inf_range, gold_range):
        self._result_dataframe = pd.DataFrame(columns=[L1ErrorVerifier.ERROR,
                                                       L1ErrorVerifier.SUM,
                                                       L1ErrorVerifier.MEAN,
                                                       L1ErrorVerifier.MULTIPLIER,
                                                       L1ErrorVerifier.SCALE,
                                                       L1ErrorVerifier.DELTA,
                                                       L1ErrorVerifier.INF_ARRAY_SIZE,
                                                       L1ErrorVerifier.GOLD_ARRAY_SIZE,
                                                       L1ErrorVerifier.INF_MIN_VAL,
                                                       L1ErrorVerifier.GOLD_MIN_VAL,
                                                       L1ErrorVerifier.INF_MAX_VAL,
                                                       L1ErrorVerifier.GOLD_MAX_VAL,
                                                       L1ErrorVerifier.INF_RANGE,
                                                       L1ErrorVerifier.GOLD_RANGE])

        self._result_dataframe[L1ErrorVerifier.ERROR] = [error]
        self._result_dataframe[L1ErrorVerifier.SUM] = [sum]
        self._result_dataframe[L1ErrorVerifier.MEAN] = [mean]
        self._result_dataframe[L1ErrorVerifier.MULTIPLIER] = [self.multiplier]
        self._result_dataframe[L1ErrorVerifier.SCALE] = [self.scale]
        self._result_dataframe[L1ErrorVerifier.DELTA] = [delta]
        self._result_dataframe[L1ErrorVerifier.INF_ARRAY_SIZE] = [inf_size]
        self._result_dataframe[L1ErrorVerifier.GOLD_ARRAY_SIZE] = [gold_size]
        self._result_dataframe[L1ErrorVerifier.INF_MIN_VAL] = [inf_min]
        self._result_dataframe[L1ErrorVerifier.GOLD_MIN_VAL] = [gold_min]
        self._result_dataframe[L1ErrorVerifier.INF_MAX_VAL] = [inf_max]
        self._result_dataframe[L1ErrorVerifier.GOLD_MAX_VAL] = [gold_max]
        self._result_dataframe[L1ErrorVerifier.INF_RANGE] = [inf_range]
        self._result_dataframe[L1ErrorVerifier.GOLD_RANGE] = [gold_range]

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
        params={}
        err_str="Unable to validate L1_Error. Expected format as multiplier <val> scale <val>"
        if len(configs) % 2 != 0:
            return False, {'error':"Cannot pair verifier parameter.{}".format(err_str)}
        if len(configs) > 1:
            for i in range(0,len(configs),2):
                if configs[i] == 'multiplier' or configs[i] == 'scale':
                    if configs[i] not in params:
                        try:
                            params[configs[i]]=float(configs[i+1])
                        except ValueError:
                            return False,  {'error':"Can't convert data:{}.{}".format(configs[i+1],err_str)}
                else:
                    return False,  {'error':"Illegal parameter: {}.{}".format(configs[i],err_str)}
        return True, params
