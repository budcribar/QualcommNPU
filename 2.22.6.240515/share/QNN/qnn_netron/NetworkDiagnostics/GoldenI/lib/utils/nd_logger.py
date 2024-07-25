# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os

def setup_logger(verbose, output_dir='.'):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    formatter = logging.Formatter(formatter)
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(lvl)
    stream_handler.setFormatter(formatter)

    log_file = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(log_file):
        os.mknod(log_file)
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    file_handler.setLevel(lvl)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(lvl)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
