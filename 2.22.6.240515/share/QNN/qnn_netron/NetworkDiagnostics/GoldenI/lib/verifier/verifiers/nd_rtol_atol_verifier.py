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
import warnings

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError, VerifierInputError
from lib.verifier.models.nd_verify_result import VerifyResult
from lib.verifier.verifiers.nd_base_verifier import BaseVerifier
from lib.visualizer.nd_histogram_visualizer import HistogramVisualizer


class RtolAtolVerifier(BaseVerifier):
    INF_ARRAY_SIZE = 'inference_tensor_size'
    GOLD_ARRAY_SIZE = 'golden_tensor_size'
    INF_MIN_VAL = 'inference_tensor_min_value'
    GOLD_MIN_VAL = 'golden_tensor_min_value'
    INF_MAX_VAL = 'inference_tensor_max_value'
    GOLD_MAX_VAL = 'gold_tensor_max_value'
    INF_RANGE = 'inference_tensor_range:abs(max - min)'
    GOLD_RANGE = 'golden_tensor_range:abs(max - min)'
    ERROR = 'error_(%)'
    ELEMENTS_NOT_CLOSE = 'number_of_elements_not_close'
    ELEMENTS_5_ERROR = 'number_of_elements_with_>_5%_error'

    def __init__(self, rtolmargin=1e-2, atolmargin=1e-2):
        pd.options.mode.use_inf_as_na = True

        self.NAME = 'RTOLATOL Verifier'
        self.UNITS = 'Percent Error (%)'

        self.rtolmargin = rtolmargin
        self.atolmargin = atolmargin

        self._golden_tensor = None
        self._inference_tensor = None

        self._result_dataframe = None

    def _calculate_margins(self, expected_output):
        return self.atolmargin, self.rtolmargin

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):
        """
        Calculates the percentage of the number of elements that are not close.
        Golden_output and inference_output must be of the same length.

        :param golden_output: List containing a single 1-D array of values
        :param inference_output: List containing a single 1-D array of values
        :return: percent from 0 to 1 of how many elements did not match
        """
        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_RTOL_ATOL_INCORRECT_INPUT_SIZE"))

        golden_output = golden_output[0] if isinstance(golden_output[0], np.ndarray) else np.array(golden_output[0])
        inference_output = inference_output[0] if isinstance(inference_output[0], np.ndarray) else np.array(inference_output[0])

        if len(golden_output) != len(inference_output):
            raise VerifierError(get_message("ERROR_VERIFIER_RTOL_ATOL_DIFFERENT_SIZE")
                                (str(len(golden_output)), (str(len(inference_output)))))

        self._golden_tensor = golden_output
        self._inference_tensor = inference_output

        atolmargin, rtolmargin = self._calculate_margins(golden_output)
        match_array = np.isclose(inference_output, golden_output, atol=atolmargin, rtol=rtolmargin)
        percent_not_close = (len(match_array) - np.count_nonzero(match_array)) / len(match_array)
        percent_not_close = percent_not_close * 100

        # Calculate the number of elements with error greater than 5%
        num_elem_fivep_error = self._calculate_num_elem_fivep_error()

        self._setup_dataframe(inf_size=len(inference_output), gold_size=len(golden_output),
                              inf_min=min(inference_output), gold_min=min(golden_output),
                              inf_max=max(inference_output), gold_max=max(golden_output),
                              inf_range=abs(max(inference_output) - min(inference_output)),
                              gold_range=abs(max(golden_output) - min(golden_output)),
                              error=percent_not_close,
                              elements_not_close=len(match_array) - np.count_nonzero(match_array),
                              elements_over_5_error=num_elem_fivep_error)

        return VerifyResult(layer_type, len(golden_output), tensor_dimensions, self._get_verifier_name(), percent_not_close, self.UNITS)

    def _calculate_num_elem_fivep_error(self):
        sub = self._golden_tensor - self._inference_tensor
        denom = np.maximum(np.absolute(self._golden_tensor), np.absolute(self._inference_tensor))
        with warnings.catch_warnings(record=True) as w:
            percent_error = np.absolute(sub) / denom
            percent_error = percent_error[np.nonzero(percent_error)]
            num_elem_fivep_error = len(percent_error[np.where(percent_error > 0.05)])
        return num_elem_fivep_error

    def _get_verifier_name(self):
        return self.NAME

    def _setup_dataframe(self, inf_size, gold_size, inf_min, gold_min, inf_max, gold_max, inf_range, gold_range,
                         error, elements_not_close, elements_over_5_error):
        self._result_dataframe = pd.DataFrame(columns=[RtolAtolVerifier.ERROR,
                                                       RtolAtolVerifier.ELEMENTS_NOT_CLOSE,
                                                       RtolAtolVerifier.ELEMENTS_5_ERROR,
                                                       RtolAtolVerifier.INF_ARRAY_SIZE,
                                                       RtolAtolVerifier.GOLD_ARRAY_SIZE,
                                                       RtolAtolVerifier.INF_MIN_VAL,
                                                       RtolAtolVerifier.GOLD_MIN_VAL,
                                                       RtolAtolVerifier.INF_MAX_VAL,
                                                       RtolAtolVerifier.GOLD_MAX_VAL,
                                                       RtolAtolVerifier.INF_RANGE,
                                                       RtolAtolVerifier.GOLD_RANGE])

        self._result_dataframe[RtolAtolVerifier.ERROR] = [error]
        self._result_dataframe[RtolAtolVerifier.ELEMENTS_NOT_CLOSE] = [elements_not_close]
        self._result_dataframe[RtolAtolVerifier.ELEMENTS_5_ERROR] = [elements_over_5_error]
        self._result_dataframe[RtolAtolVerifier.INF_ARRAY_SIZE] = [inf_size]
        self._result_dataframe[RtolAtolVerifier.GOLD_ARRAY_SIZE] = [gold_size]
        self._result_dataframe[RtolAtolVerifier.INF_MIN_VAL] = [inf_min]
        self._result_dataframe[RtolAtolVerifier.GOLD_MIN_VAL] = [gold_min]
        self._result_dataframe[RtolAtolVerifier.INF_MAX_VAL] = [inf_max]
        self._result_dataframe[RtolAtolVerifier.GOLD_MAX_VAL] = [gold_max]
        self._result_dataframe[RtolAtolVerifier.INF_RANGE] = [inf_range]
        self._result_dataframe[RtolAtolVerifier.GOLD_RANGE] = [gold_range]

    def save_data(self, path):
        self._save_to_histogram(path)
        self._save_to_csv(path)
        self._save_to_html(path)

    def _save_to_histogram(self, path):
        path_to_save = Path(path).with_suffix('.png')
        HistogramVisualizer.visualize(self._inference_tensor, self._golden_tensor, path_to_save)

    def _save_to_csv(self, path):
        if isinstance(self._result_dataframe, pd.DataFrame):
            path_to_save = Path(path)
            self._result_dataframe.to_csv(path_to_save.with_suffix('.csv'), encoding='utf-8', index=False)

    def _save_to_html(self, path):
        if isinstance(self._result_dataframe, pd.DataFrame):
            self._result_dataframe.to_html(Path(path).with_suffix('.html'), classes='table', index=False)

    @staticmethod
    def validate_config(configs):
        params={}
        err_str="Unable to validate  RtolAtol. Expected format as rtolmargin <val> atolmargin <val>"
        if len(configs) % 2 != 0:
            return False, {'error':"Cannot pair verifier parameter.{}".format(err_str)}
        if len(configs) > 1:
            for i in range(0,len(configs),2):
                if configs[i] == 'rtolmargin' or configs[i] == 'atolmargin':
                    if configs[i] not in params:
                        try:
                            params[configs[i]]=float(configs[i+1])
                        except ValueError:
                            return False,  {'error':"Can't convert data:{}.{}".format(configs[i+1],err_str)}
                else:
                    return False,  {'error':"Illegal parameter: {}.{}".format(configs[i],err_str)}
        return True, params
