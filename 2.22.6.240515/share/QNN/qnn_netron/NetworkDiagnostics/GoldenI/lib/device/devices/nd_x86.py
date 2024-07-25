# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
import shutil
import stat

from lib.device.helpers import nd_device_utilities
from lib.device.devices.nd_device_interface import DeviceInterface
from lib.utils.nd_errors import get_message


class X86Interface(DeviceInterface):
    def __init__(self, logger=None):
        if not logger:
            logger = logging.getLogger()

        self._logger = logger
        self.device = 'x86'

    def is_connected(self):
        return True

    def execute(self, commands, cwd='.', env=None):
        if env is None:
            env = {}

        env_vars = ['export {}="{}"'.format(k, v) for k, v in env.items()]

        x86_shell_commands = ['cd ' + cwd] + env_vars + commands
        x86_shell_command = ' && '.join(x86_shell_commands)
        return nd_device_utilities.execute(x86_shell_command, shell=True, cwd=cwd)

    def push(self, src_path, dst_path):
        ret = 0
        stdout = ''
        stderr = ''
        if not os.path.exists(src_path):
            ret = -1
            stderr = get_message("ERROR_DEVICE_MANAGER_X86_NON_EXISTENT_PATH")(src_path)

        # if src_path is a file
        if os.path.isfile(src_path):
            if not os.path.exists(os.path.dirname(dst_path)):
                os.makedirs(os.path.dirname(dst_path))
            shutil.copy(src_path, dst_path)

        # if src_path is a directory
        for root, dirs, file_lists in os.walk(src_path):
            # sets up path to root paths in destination
            dst_root_path = os.path.join(dst_path, root.replace(src_path, "").lstrip(os.sep))
            if not os.path.isdir(dst_root_path):
                os.makedirs(dst_root_path)
            for file in file_lists:
                src_rel_path = root.replace(src_path, "").lstrip(os.sep)
                file_root_dst_path = os.path.join(dst_path, src_rel_path, file)
                file_root_src_path = os.path.join(root, file)
                shutil.copyfile(file_root_src_path, file_root_dst_path)
                os.chmod(file_root_dst_path, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)

        return ret, stdout, stderr

    def make_directory(self, dir_name):
        ret = 0
        stdout = ''
        stderr = ''
        try:
            os.makedirs(dir_name)
        except OSError as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def pull(self, device_src_path, host_dst_dir):
        return self.push(device_src_path, host_dst_dir)

    def remove(self, target_path):
        if os.path.isfile(target_path):
            os.remove(target_path)
        elif os.path.isdir(target_path):
            shutil.rmtree(target_path)
