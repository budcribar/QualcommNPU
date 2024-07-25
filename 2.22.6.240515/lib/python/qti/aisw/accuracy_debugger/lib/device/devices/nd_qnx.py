# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
import shutil
import stat

from qti.aisw.accuracy_debugger.lib.device.helpers import nd_device_utilities
from qti.aisw.accuracy_debugger.lib.device.devices.nd_device_interface import DeviceInterface
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.device.helpers.nd_ftp import Filesystem
from qti.aisw.accuracy_debugger.lib.device.helpers.nd_telnet import TelnetExecutor
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeviceError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message


class QnxInterface(DeviceInterface):

    def __init__(self, logger=None, device_ip=None, device_username=None, device_password=None):
        if not logger:
            logger = logging.getLogger()

        self._logger = logger
        self.device = 'qnx'
        self.device_ip = device_ip
        self.device_username = device_username
        self.device_password = device_password

        # establish telnet connection with target
        try:
            self.telnet_ins = TelnetExecutor(self.device_ip, self.device_username,
                                             self.device_password, logger=self._logger)
        except Exception as e:
            self.telnet_ins = TelnetExecutor.getInstance()
            self.telnet_ins.telnet_dev = self.telnet_ins.setup_telnet_connect()

        # establish ftp connection with target
        try:
            self.ftp_ins = Filesystem(self.device_ip, self.device_username, self.device_password,
                                      logger=self._logger)
        except Exception as e:
            self.ftp_ins = Filesystem.getInstance()
            self.ftp_ins.ftp_dev = self.ftp_ins.setup_ftp_connect()

        if not self.is_connected():
            raise DeviceError(get_message("ERROR_REMOTE_CONNECTION"))

    def is_connected(self):
        telnet_status = True
        ftp_status = True
        try:
            self.telnet_ins.telnet_dev.read_very_eager()
        except Exception:
            self._logger.error("Telnet connection is inactive")
            telnet_status = False
        try:
            self.ftp_ins.ftp_dev.retrlines('LIST')
        except Exception:
            self._logger.error("FTP connection is inactive")
            ftp_status = False
        if telnet_status and ftp_status:
            return True
        else:
            return False

    def execute(self, commands, cwd='.', env=None):
        if env is None:
            env = {}
        env_vars = ['export {}="{}"'.format(k, v) for k, v in env.items()]
        x86_shell_commands = ['cd ' + cwd] + env_vars + commands
        x86_shell_command = ' && '.join(x86_shell_commands)
        return self.telnet_ins.execute(x86_shell_command, shell=True, cwd=cwd)

    def push(self, src_path, dst_path):
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if os.path.isfile(src_path):
                self.ftp_ins.put_file(src_path, dst_path)
        except Exception as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def make_directory(self, dir_name):
        ret = 0
        stdout = ''
        stderr = ''
        try:
            self.telnet_ins.execute('mkdir -p ' + dir_name)
        except Exception as e:
            ret = -1
            stderr = str(e)
        return ret, stdout, stderr

    def pull(self, device_src_path, host_dst_dir):
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if self.ftp_ins.ftp_is_folder(device_src_path):
                self.ftp_ins.pull_folder(host_dst_dir, device_src_path)
            else:
                self.ftp_ins.pull_file(host_dst_dir, device_src_path)
        except Exception as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def remove(self, target_path):
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if self.ftp_ins.ftp_is_folder(target_path):
                self.ftp_ins.ftp_remove(target_path)
            else:
                self.ftp_dev.delete(target_path)
        except Exception as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def close(self):
        #close ftp connection
        self.ftp_ins.close()
        #close telnet connection
        self.telnet_ins.close()
