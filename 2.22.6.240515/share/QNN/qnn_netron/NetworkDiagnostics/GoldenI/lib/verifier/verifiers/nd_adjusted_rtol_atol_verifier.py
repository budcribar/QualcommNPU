# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from lib.verifier.verifiers.nd_rtol_atol_verifier import RtolAtolVerifier


class AdjustedRtolAtolVerifier(RtolAtolVerifier):
    def __init__(self, levels_num=4):
        super(AdjustedRtolAtolVerifier, self).__init__()
        self.NAME = 'AdjustedRTOLATOL Verifier'

        self.levels_num = levels_num

    def _calculate_margins(self, expected_output):
        min = np.min(expected_output)
        max = np.max(expected_output)
        step = (max - min) / 255
        # allow at least four step sizes of absolute tolerance which is only 1.5 % (4/255)
        adjustedatolmargin = self.levels_num * step
        # and add some 10% rtol, mainly for very small values
        adjustedrtolmargin = 1e-1
        return adjustedatolmargin, adjustedrtolmargin

    def _get_verifier_name(self):
        return self.NAME

    @staticmethod
    def validate_config(configs):
        params={}
        err_str="Unable to validate adjusted RtolAtol. Expected format as levels_num <val>"

        if len(configs) % 2 != 0:
            return False, {'error':"Cannot pair verifier parameter.{}".format(err_str)}
        if len(configs) > 1:
            for i in range(0,len(configs),2):
                if configs[i] == 'levels_num':
                    if configs[i] not in params:
                        try:
                            params[configs[i]]=int(configs[i+1])
                        except ValueError:
                            return False,  {'error':"Can't convert data:{}.{}".format(configs[i+1],err_str)}
                else:
                    return False,  {'error':"Illegal parameter: {}.{}".format(configs[i],err_str)}
        return True, params