# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
from shutil import which

from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import DeviceError
from lib.device.devices.nd_device_interface import DeviceInterface
from lib.device.helpers.nd_adb import Adb


class LinuxEmbeddedInterface(DeviceInterface):
    def __init__(self, device, adb_path, logger=None):
        if not logger:
            logger = logging.getLogger()
        self.logger = logger
        self.device = 'linux_embedded'

        if not adb_path:
            adb_command_in_path = which('adb')
            if not adb_command_in_path:
                raise DeviceError(get_message("ERROR_ADB_NOT_INSTALLED"))
            adb_path = adb_command_in_path
        elif not os.path.exists(adb_path):
            raise DeviceError(get_message("ERROR_ADB_PATH_INVALID"))

        self.device = device
        self._adb = Adb(adb_path, device, logger)

        self._linux_embedded_init()

    def _linux_embedded_init(self):
        self._adb.shell('setenforce', ['0'])
        self._adb.shell('mount', ['-o', 'remount,rw,exec', '/'])

    def is_connected(self):
        return self._adb.is_device_online(self.device)

    def execute(self, commands, cwd='.', env=None):
        mount_point = self._adb.shell('df -P ' + cwd + ' | tail -1')[1][0].split(' ')[-1]

        self._adb.shell('mount', ['-o', 'remount,rw,exec', mount_point])

        if env is None:
            env = {}

        env_vars = ['export {}="{}"'.format(k, v) for k, v in env.items()]

        linux_shell_commands = ['cd ' + cwd] + env_vars + commands
        linux_shell_command = '&& '.join(linux_shell_commands)
        return self._adb.shell(linux_shell_command)

    def push(self, src_path, dst_path):
        dir_name = os.path.dirname(dst_path)

        if os.path.isdir(src_path):
            dir_name = os.path.join(dir_name, os.path.basename(src_path))
            dst_path = os.path.join(dst_path, dir_name)
            dir_name = dst_path

        self.make_directory(dir_name)
        return self._adb.push(src_path, dst_path)

    def make_directory(self, dir_name):
        return self._adb.shell('mkdir -p ' + dir_name)

    def pull(self, src_path, dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        return self._adb.pull(src_path, dst_path)

    def remove(self, target_path):
        return self._adb.shell('rm -rf ' + target_path)
