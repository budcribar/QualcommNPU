# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os


def verify_path(*paths):
    '''
    Verifies whether directory exists or not, if not then
    creates the directory and returns the path
    '''
    path = os.path.join(*paths)
    if os.path.exists(path) and os.path.isdir(path):
        return path
    os.makedirs(path)
    return path
