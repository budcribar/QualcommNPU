# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import abc


class BaseVerifier:
    @abc.abstractmethod
    def verify(self, layer_type, tensor_dimensions, golden_output, inference_output):
        raise NotImplementedError

    @abc.abstractmethod
    def save_data(self, path):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def validate_config(configs):
        raise NotImplementedError
