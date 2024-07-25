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

from lib.utils.nd_exceptions import DeviceError
from lib.utils.nd_errors import get_message
from lib.device.devices.nd_device_interface import DeviceInterface
from lib.device.helpers.nd_adb import Adb


class AndroidInterface(DeviceInterface):
    def __init__(self, device_id, adb_path, logger=None):
        if not logger:
            logger = logging.getLogger()

        self.device = 'android'

        if not adb_path:
            adb_command_in_path = which('adb')
            if not adb_command_in_path:
                raise DeviceError(get_message("ERROR_ADB_NOT_INSTALLED"))
            adb_path = adb_command_in_path
        elif not os.path.exists(adb_path):
            raise DeviceError(get_message("ERROR_ADB_PATH_INVALID"))

        self._adb = Adb(adb_path, device_id, logger)

    def is_connected(self):
        return self._adb.is_device_online()

    def execute(self, commands, cwd='.', env=None):
        if env is None:
            env = {}

        env_vars = ['export {}="{}"'.format(k, v) for k, v in env.items()]

        android_shell_commands = ['cd ' + cwd] + env_vars + commands
        android_shell_command = ' && '.join(android_shell_commands)
        return self._adb.shell(android_shell_command)

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
