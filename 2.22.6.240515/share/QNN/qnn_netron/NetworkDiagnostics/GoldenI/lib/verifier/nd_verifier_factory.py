# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.verifier.verifiers.nd_adjusted_rtol_atol_verifier import AdjustedRtolAtolVerifier
from lib.verifier.verifiers.nd_mean_iou_verifier import MeanIOUVerifier
from lib.verifier.verifiers.nd_rtol_atol_verifier import RtolAtolVerifier
from lib.verifier.verifiers.nd_topk_verifier import TopKVerifier
from lib.verifier.verifiers.nd_L1_error_verifier import L1ErrorVerifier
from lib.verifier.verifiers.nd_cs_verifier import CosineSimilarityVerifier
from lib.verifier.verifiers.nd_mse_verifier import MSEVerifier
from lib.verifier.verifiers.nd_sqnr_verifier import SQNRVerifier
from lib.verifier.verifiers.nd_scaled_diff_verifier import ScaledDiffVerifier


class VerifierFactory(object):
    @staticmethod
    def factory(verifier, config):
        def get_args_if_exists(*args):
            return [config[arg] for arg in args if arg in config]

        if verifier == 'adjustedrtolatol':
            return AdjustedRtolAtolVerifier(*get_args_if_exists("levels_num"))
        elif verifier == 'rtolatol':
            return RtolAtolVerifier(*get_args_if_exists("rtolmargin", "atolmargin"))
        elif verifier == 'topk':
            return TopKVerifier(*get_args_if_exists("k", "args"))
        elif verifier == 'meaniou':
            return MeanIOUVerifier(*get_args_if_exists("background_classification"))
        elif verifier == "l1error":
            return L1ErrorVerifier(False, *get_args_if_exists("multiplier", "scale"))
        elif verifier == "cosinesimilarity":
            return CosineSimilarityVerifier(*get_args_if_exists("multiplier", "scale"))
        elif verifier == "mae":
            return L1ErrorVerifier(True, *get_args_if_exists("multiplier", "scale"))
        elif verifier == "mse":
            return MSEVerifier()
        elif verifier == "sqnr":
            return SQNRVerifier()
        elif verifier == "scaleddiff":
            return ScaledDiffVerifier()
        else:
            return None

    @staticmethod
    def validate_configs(verifier, config):
    # validate_configs(verifier : str, config : List[str]) -> (bool, dict):

        if not config:
            return True, {}
        if verifier == 'adjustedrtolatol':
            return AdjustedRtolAtolVerifier().validate_config(config)
        elif verifier == 'rtolatol':
            return RtolAtolVerifier().validate_config(config)
        elif verifier == 'topk':
            return TopKVerifier().validate_config(config)
        elif verifier == 'meaniou':
            return MeanIOUVerifier().validate_config(config)
        elif verifier == "l1error":
            return L1ErrorVerifier().validate_config(config)
        elif verifier == "cosinesimilarity":
            return CosineSimilarityVerifier().validate_config(config)
        elif verifier == "mae":
            return L1ErrorVerifier().validate_config(config)
        elif verifier == "mse":
            return MSEVerifier().validate_config(config)
        elif verifier == "sqnr":
            return SQNRVerifier().validate_config(config)
        elif verifier == "scaleddiff":
            return ScaledDiffVerifier().validate_config(config)
        else:
            return False, {'error':"Illegal verifier name: "+ verifier}