# =============================================================================
#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from abc import ABC
from telnetlib import Telnet
from threading import Timer
import logging
import subprocess
import sys
import traceback
import time
import re


class ExecutionEngine(ABC):

    def __init__(self, server=None, username=None, password=None):
        self.server = server
        self.username = username
        self.password = password

    def execute(self):
        pass

    def close(self):
        pass

class TelnetExecutor(ExecutionEngine):
    """
    This class performs the execution on remote device through telnet
    """
    __instance = None

    def __init__(self, host: str, username: str, password=None, logger=None):

        if TelnetExecutor.__instance is not None:
            raise ValueError('instance of TelnetExecutor already exists')
        else:
            TelnetExecutor.__instance = self

        self.host = host.strip()
        self.username = username.strip()
        if password:
            self.password = password
        else:
            self.password = None
        self.is_connected = False
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger()
        try:
            self.telnet_dev = self.setup_telnet_connect()
            assert self.telnet_dev, "setup Telnet failed with ip %s" % self.host

            self.is_connected = True

        except Exception as e:
            raise Exception("Unable to establish connection with remote @ %s" % self.host)

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = TelnetExecutor()
        return cls.__instance

    def setup_telnet_connect(self, device_port=23, timeout=100, debug_level=0):
        device = Telnet()
        device.set_debuglevel(debug_level)
        try:
            device.open(self.host, device_port, timeout)
            device.read_until(b"login: ", timeout)
            device.write(self.username.encode("ascii") + b"\n")
            if self.password:
                device.read_until(b"Password: ")
                device.write(self.password.encode("ascii") + b"\n")
            self._logger.info('Telnet connection established successfully')
        except Exception as e:
            self._logger.info("Unable to connect {0} error code {1}".format(self.host, e))
            return None
        return device

    def execute(self, cmd, log_file='', cwd='.', shell=False, timeout=None):
        ret_code = 0
        std_out = ''
        err_msg = ''
        sync_ms = 0
        # Taking a delay of 100ms as default
        delay = 100 if sync_ms == 0 else sync_ms
        try:
            log_redirect = " > " + log_file + " 2>&1" if log_file else ''
            cmd_str = str('\n%s%s\n#<CMD DONE>\necho "<<"$?">>"\n#<REMOTE DONE>' %
                          (cmd, log_redirect))
            self.telnet_dev.write(cmd_str.encode("ascii") + b"\n")
            time.sleep(float(delay) / 1000)
            self.remote_result_str = self.telnet_dev.read_until(b"<REMOTE DONE>").decode(
                "utf-8", "ignore")
            self._logger.info("remote cmd response: " + self.remote_result_str)
            returncode = re.search(r"<<(\d+)>>", self.remote_result_str)
            errcode = int(returncode.group(1))
            if errcode:
                self._logger.info("remote execution failed with err_code " + str(errcode))
                ret_code = 1

        except Exception as e:
            self._logger.info("remote telnet command failed to execute:\n\t%s" % repr(e))
            err_msg = str(e)
            traceback.print_exc()

        return ret_code, std_out, err_msg

    def close(self):
        if self.is_connected:
            self.telnet_dev.close()
            self._logger.info('Telnet connection closed')
