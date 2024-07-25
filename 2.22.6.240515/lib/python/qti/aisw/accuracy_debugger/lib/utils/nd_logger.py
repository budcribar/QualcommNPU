# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os


def setup_logger(verbose, output_dir='.', t_stamp='', append_to_existing_log_file=False,
                 disable_console_logging=False):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    formatter = logging.Formatter(formatter)
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.name = "stream_handler"
    stream_handler.setLevel(lvl)  # stream handler log level set depending on verbose
    stream_handler.setFormatter(formatter)
    # To create empty file
    log_file = os.path.join(output_dir, 'log' + t_stamp + '.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            pass
    file_access = 'a' if append_to_existing_log_file else 'w'
    file_handler = logging.FileHandler(filename=log_file, mode=file_access)
    file_handler.name = "file_handler"
    file_handler.setLevel(
        logging.DEBUG)  # file handler log level set to DEBUG so all logs are written to file
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(
        logging.DEBUG)  # set base logger level to DEBUG so that all logs are caught by handlers
    if not disable_console_logging:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def get_logger_log_file_path(logger, handler_name=None):
    file_handlers = [
        handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
    ]
    if len(file_handlers) == 0:
        return ""
    elif len(file_handlers) == 1:
        return file_handlers[0].baseFilename
    else:  # in case of multiple file handlers
        file_handler = next((h for h in file_handlers if h.name == handler_name), None)
        if file_handler is not None:
            return file_handler.baseFilename
        else:
            return ""
