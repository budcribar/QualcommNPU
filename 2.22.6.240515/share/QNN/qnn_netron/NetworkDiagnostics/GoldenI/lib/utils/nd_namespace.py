# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from argparse import Namespace


class Namespace(Namespace):
    def __init__(self, data=None, **kwargs):
        if data is not None:
            kwargs.update(data)
        super(Namespace, self).__init__(**kwargs)
