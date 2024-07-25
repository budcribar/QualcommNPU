# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from lib.utils.nd_errors import get_warning_message, get_progress_message


def symlink(name, directory, logger):
    symlink_name = os.path.join(directory, '..', name)

    if not os.path.exists(directory):
        logger.warning(get_warning_message('WARNING_UTILS_CANNOT_CREATE_SYMLINK')(symlink_name, directory))
        return

    if os.path.exists(symlink_name) and not os.path.islink(symlink_name):
        logger.warning(get_warning_message('WARNING_UTILS_CANNOT_CREATE_SYMLINK')(symlink_name, directory))
        return
    elif os.path.islink(symlink_name):
        os.remove(symlink_name)

    logger.info(get_progress_message('PROGRESS_UTILS_CREATE_SYMLINK')(symlink_name, directory))
    os.symlink(directory, symlink_name)
