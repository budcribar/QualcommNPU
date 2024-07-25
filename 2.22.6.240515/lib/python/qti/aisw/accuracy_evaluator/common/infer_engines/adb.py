##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################

import logging
import re
import time
import subprocess
from functools import wraps
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


def execute(command, args=[], cwd='.', shell=False, timeout=1800):
    """Execute command in cwd.

    :param command: str
    :param args: []
    :param cwd: filepath
    :param shell: True/False
    :param timeout: float
    :return: int, [], []
    """
    qacc_logger.debug("Host Command: {} {}".format(command, args))
    process = subprocess.Popen([command] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd=cwd, shell=shell)
    (output, error) = process.communicate()
    return_code = process.returncode
    qacc_logger.debug("Result Code (%d): stdout: (%s) stderr: (%s)" % (return_code, output, error))
    return return_code, output, error


class Adb(object):

    def __init__(self, adb_executable, device):
        self.__adb_executable = adb_executable
        self.__adb_device = device
        self.ADB_DEFAULT_TIMEOUT = 500

    def push(self, src, dst, cwd='.'):
        dst_dir_exists = False
        if (self.__execute('shell', ['[ -d %s ]' % dst], cwd=cwd)[0] == 0):
            dst_dir_exists = True
        code, out, err = self.__execute('push', [src, dst], cwd=cwd)
        # print(str(code) + ' ' + out + ' ' + err)
        if code == 0:
            # Check if push was successful
            if src[-1] == '/':
                src = src[:-1]
            file_name = src.split('/')[-1]
            # check if destination directory exists
            # if it exists, then append file name to dst
            # otherwise, adb will rename src dir to dst
            if dst_dir_exists:
                dst = (dst + file_name) if dst[-1] == '/' else (dst + '/' + file_name)
            code, out, err = self.__execute('shell', ['[ -e %s ]' % dst], cwd=cwd)
        return code, out, err

    def pull(self, src, dst, cwd='.'):
        return self.__execute('pull', [src, dst], cwd=cwd)

    def shell(self, command, args=[]):
        shell_args = ["{} {}; echo $?".format(command, ' '.join(args))]
        code, out, err = self.__execute('shell', shell_args)
        if code == 0:
            if len(out) > 0:
                try:
                    code = int(out[-1])
                    out = out[:-1]
                except ValueError as ex:
                    code = -1
                    out.append(ex.message)
            else:
                code = -1

            if code != 0 and len(err) == 0:
                err = out
        else:
            code = -1
        return code, out, err

    def __execute(self, command, args, cwd='.'):
        if self.__adb_device:
            adb_command_args = ["-s", self.__adb_device, command] + args
        else:
            adb_command_args = [command] + args
        (return_code, output, error) = execute(self.__adb_executable, adb_command_args, cwd=cwd,
                                               timeout=self.ADB_DEFAULT_TIMEOUT)
        # when the process gets killed, it will return -9 code; Logging this error for debug purpose
        if return_code == -9:
            qacc_logger.error(
                "adb command didn't execute within the timeout. Is device in good state?")
        return (return_code, output, error)

    def get_devices(self):
        code, out, err = self.__execute('devices', [])
        if code != 0:
            qacc_logger.error("Could not retrieve list of adb devices connected, following error "
                              "occurred: {0}".format("\n".join(err)))
            return code, out, err

        devices = []
        for line in out:
            match_obj = re.match("^([a-zA-Z0-9]+)\s+device", line, re.M)
            if match_obj:
                devices.append(match_obj.group(1))
        return code, devices, err
