# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
import shutil
import stat
import sys

from qti.aisw.accuracy_debugger.lib.device.helpers import nd_device_utilities
from qti.aisw.accuracy_debugger.lib.device.devices.nd_device_interface import DeviceInterface
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import X86_windows_Architectures

class X86WindowsMsvcInterface(DeviceInterface):

    def __init__(self, logger=None, device_setup_path=None):
        """
        Initializes the Device class for X86Windows.

        Args:
            logger (logging.Logger, optional): A logger instance to use for
                printing debug messages. If not specified, a new logger will be created.
            device_setup_path (str, optional): The path to the device setup file to use for
                configuring the device. Defaults to None.

        """
        if not logger:
            logger = logging.getLogger()

        self._logger = logger
        self.device = X86_windows_Architectures.x86_64_windows_msvc.value
        self.device_setup_path = device_setup_path

    def is_connected(self):
        return True

    def execute(self, commands, cwd='.', env=None):

        bin_path = os.path.join(self.device_setup_path , "bin")
        cd_cmd = 'cd ' + cwd + ';'
        # Execute 'check-windows-dependency.ps1' powershell file to check windows dependencies
        # File must be present in bin_path
        dependency_check_cmd = '& {bin_path}\check-windows-dependency.ps1'.format(bin_path = bin_path)
        x86_windows_shell_command = cd_cmd + f' if($?) {{ {dependency_check_cmd} }} ;'

        # Execute 'envsetup.ps1' powershell file for the environment setup
        env_setup_cmd = '& {bin_path}\envsetup.ps1'.format(bin_path = bin_path)
        x86_windows_shell_command += f' if($?) {{ {env_setup_cmd} }} ;'

        # Setup extra environment variables which are provided by user.
        for key, value in env.items():
            env[key] = value.replace('/', '\\')
            env_setup_command = '$env:{key} += "{path_sep}{value}"'.format(key=key, value=env[key], path_sep=os.pathsep)
            x86_windows_shell_command += f' if($?) {{ {env_setup_command} }} ;'


        # binaries_path Directory must have executables
        binaries_path = os.path.join(bin_path, self.device)

        for cmd in commands:
            cmd = cmd.replace('/', '\\')
            if '.exe' in cmd.split(' ')[0]:
                cmd = '& {binaries_path}\{cmd}'.format(binaries_path=binaries_path, cmd=cmd)
            else:
                cmd = sys.executable + ' ' + '{binaries_path}\{cmd}'.format(binaries_path=binaries_path, cmd=cmd)
            x86_windows_shell_command += f' if($?) {{ {cmd} }} ;'

        return nd_device_utilities.execute(x86_windows_shell_command, shell=True, cwd=cwd, powershell=True)

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
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError as e:
                ret = -1
                stderr = str(e)
        else:
            stderr = 'Directory {} already exists'.format(dir_name)

        return ret, stdout, stderr

    def pull(self, device_src_path, host_dst_dir):
        return self.push(device_src_path, host_dst_dir)

    def remove(self, target_path):
        # Remove file or directory present at target_path
        if os.path.isfile(target_path):
            os.remove(target_path)
        elif os.path.isdir(target_path):
            shutil.rmtree(target_path)
