# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import subprocess
import sys
from threading import Timer

logger = logging.getLogger()


class Timeouts:
    DEFAULT_POPEN_TIMEOUT = 1800
    ADB_DEFAULT_TIMEOUT = 3600


def _format_output(output):
    """
    Separate lines in output into a list and strip each line.
    :param output: str
    :return: list
    """
    stripped_out = []
    if output is not None and len(output) > 0:
        stripped_out = [line.strip() for line in output.split('\n') if line.strip()]
    return stripped_out


def execute(command, args=None, cwd='.', shell=False, timeout=Timeouts.DEFAULT_POPEN_TIMEOUT):
    """
    Execute command in cwd.
    :param command: str
    :param args: list
    :param cwd: file path
    :param shell: True/False
    :param timeout: float
    :return: int, list, list
    """

    if args is None:
        args = []

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
                output = output.decode()
                error = error.decode()
        finally:
            # If the timer is alive, that implies process exited within the timeout;
            # Hence stopping the timer task;
            if timer.isAlive():
                timer.cancel()
            else:
                logger.error("Timer expired for the process. Process didn't finish within the given timeout of {}"
                             .format(timeout))

        return_code = process.returncode
        logger.debug("Result Code ({}): stdout: ({}) stderr: ({})".format(return_code, output, error))
        return return_code, _format_output(output), _format_output(error)
    except OSError as error:
        return -1, [], _format_output(str(error))
