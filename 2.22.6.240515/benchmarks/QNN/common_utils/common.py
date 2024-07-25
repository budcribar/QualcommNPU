#
# Copyright (c) 2017-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import subprocess
import sys
import logging
from threading import Timer

logger = logging.getLogger(__name__)


class Timeouts:
    DEFAULT_POPEN_TIMEOUT = 5000
    ADB_DEFAULT_TIMEOUT = 5000


def __format_output(output):
    """
    Separate lines in output into a list and strip each line.
    :param output: str
    :return: []
    """
    stripped_out = []
    if output is not None and len(output) > 0:
        stripped_out = [line.strip()
                        for line in output.split('\n') if line.strip()]
    return stripped_out


def execute(command, args=[], cwd='.', shell=False,
            timeout=Timeouts.DEFAULT_POPEN_TIMEOUT,
            output_to_list=True):
    """
    Execute command in cwd.
    :param command: str
    :param args: []
    :param cwd: filepath
    :param shell: True/False
    :param timeout: float
    :output_to_list: True/False
    :return: int, [], []
    """
    try:
        logger.debug("Host Command: {} {}".format(command, args))
        process = subprocess.Popen([command] + args,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   cwd=cwd,
                                   shell=shell)
        try:
            # timer is created to kill the process after the timeout
            timer = Timer(float(timeout), process.kill)
            timer.start()
            output, error = process.communicate()
            if sys.version_info[0] == 3:
                output = output.decode('utf-8')
                error = error.decode('utf-8')
        finally:
            # If the timer is alive, that implies process exited within the timeout;
            # Hence stopping the timer task;
            if timer.is_alive():
                timer.cancel()
            else:
                logger.error("Timer expired for the process. Process didn't \
                              finish within the given timeout of %f" % (timeout))

        return_code = process.returncode
        logger.debug(
            "Result Code (%d): stdout: (%s) stderr: (%s)" %
            (return_code, output, error))
        cmd_output = __format_output(output) if output_to_list else output
        return return_code, cmd_output, __format_output(error)
    except OSError as error:
        return -1, [], __format_output(str(error))
