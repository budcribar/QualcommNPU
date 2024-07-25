# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path

import numpy as np
import pandas

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError
from lib.utils.nd_exceptions import VerifierInputError
from lib.verifier.models.nd_verify_result import VerifyResult
from lib.verifier.verifiers.nd_base_verifier import BaseVerifier


class TopKVerifier(BaseVerifier):
    def __init__(self, k=1, ordered=False):
        super(TopKVerifier, self).__init__()
        self.k = k
        self.order_matters = ordered
        self.test_result = None

    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):
        if len(golden_output) != 1 or len(inference_output) != 1:
            raise VerifierInputError(get_message("ERROR_VERIFIER_TOPK_INCORRECT_INPUT_SIZE"))

        golden_output, inference_output = golden_output[0], inference_output[0]
        if len(golden_output) != len(inference_output):
            raise VerifierError(get_message("ERROR_VERIFIER_TOPK_DIFFERENT_SIZE")
                                (len(golden_output), len(inference_output)))

        top_k_indices_from_golden_output = np.flip(golden_output.argsort()[-self.k:])
        top_k_indices_from_inference_output = np.flip(inference_output.argsort()[-self.k:])

        if not self.order_matters:
            top_k_indices_from_golden_output.sort()
            top_k_indices_from_inference_output.sort()

        df = pandas.DataFrame()
        df['golden_classification'] = top_k_indices_from_golden_output
        df['inference_classification'] = top_k_indices_from_inference_output
        if self.order_matters:
            df['is_equal'] = df['golden_classification'] == df['inference_classification']
        else:
            df['in_golden'] = df['inference_classification'].isin(df['golden_classification'])

        self.test_result = df

        number_of_diff_indices = 0
        for index_gold, index_inf in zip(top_k_indices_from_golden_output, top_k_indices_from_inference_output):
            if self.order_matters:
                if index_gold != index_inf:
                    number_of_diff_indices += 1
            else:
                if index_inf not in top_k_indices_from_golden_output:
                    number_of_diff_indices += 1

        percent_diff = number_of_diff_indices/min(self.k, len(golden_output))
        percent_diff = percent_diff * 100
        summary_result = VerifyResult(layer_type, Size=golden_output.shape, Tensor_dims=tensor_dimensions,
                                      Verifier='TopK', Metric=percent_diff, Units='Percent incorrect')

        return summary_result

    def _to_csv(self, file_path):
        if isinstance(self.test_result, pandas.DataFrame):
            self.test_result.to_csv(file_path, encoding='utf-8', index=False)

    def _to_html(self, file_path):
        if isinstance(self.test_result, pandas.DataFrame):
            self.test_result.to_html(file_path, classes='table', index=False)

    def save_data(self, path):
        filename = Path(path)
        self._to_csv(filename.with_suffix(".csv"))
        self._to_html(filename.with_suffix(".html"))

    @staticmethod
    def validate_config(configs):
        params={}
        err_str="Unable to validate Topk. Expected format as ordered <val> k <val>"
        if len(configs) % 2 != 0:
            return False, {'error':"Cannot pair verifier parameter.{}".format(err_str)}
        if len(configs) > 1:
            for i in range(0,len(configs),2):
                try:
                    if configs[i] == 'ordered':
                        if configs[i] not in params:
                            params[configs[i]]=bool(configs[i+1])
                    elif configs[i] == 'k':
                        if configs[i] not in params:
                            params[configs[i]]=int(configs[i+1])
                    else:
                        return False,  {'error':"Illegal parameter: {}.{}".format(configs[i],err_str)}
                except ValueError:
                        return False,  {'error':"Can't convert data:{}.{} ".format(configs[i+1],err_str)}
        return True, params