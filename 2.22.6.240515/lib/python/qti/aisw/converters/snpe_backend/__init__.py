# ==============================================================================
#
#  Copyright (c) 2019-2020,2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# @deprecated
# to allow for backward compatibility adding this import also at top-level so that
# it is possible to do <from qti.aisw import modeltools>
# moving forward the signature to use will be <from qti.aisw.dlc_utils import modeltools>
try:
    import sys
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.dlc_utils import dlcontainer
except ImportError as ie:
    raise ie
