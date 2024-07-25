# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
from stat import *

from qti.aisw.accuracy_debugger.lib.device.devices.nd_device_interface import DeviceInterface
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.device.helpers.nd_ssh import SSHExecutor
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeviceError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message


class WosRemoteInterface(DeviceInterface):

    def __init__(self, logger=None, device_ip=None, device_username=None, device_password=None):
        if not logger:
            logger = logging.getLogger()

        self._logger = logger
        self.device = 'wos-remote'
        self.device_ip = device_ip
        self.device_username = device_username
        self.device_password = device_password

        # establish ssh connection with target
        try:
            self.ssh_ins = SSHExecutor(self.device_ip, self.device_username, self.device_password,
                                       logger=self._logger)
        except Exception as e:
            self.ssh_ins = SSHExecutor.getInstance()
            self.ssh_ins.ssh_client = self.ssh_ins.setup_ssh_client()

        # establish sftp connection with target
        try:
            self.ftp = self.ssh_ins.ssh_client.open_sftp()
        except Exception:
            self._logger.error("SFTP connection is inactive")

        if not self.is_connected():
            raise DeviceError(get_message("ERROR_REMOTE_CONNECTION"))

    def is_connected(self):
        ssh_status = True
        try:
            if self.ssh_ins.ssh_client.get_transport() is not None:
                ssh_status = self.ssh_ins.ssh_client.get_transport().is_active()
        except Exception:
            self._logger.error("SSH connection is inactive")
            ssh_status = False
        return ssh_status

    def execute(self, commands, cwd='.', env=None):
        if env:
            dependency_check_cmd = '& ./check-windows-dependency.ps1'
            env_path_cd_cmd = 'cd ' + env + ';'
            windows_shell_command = env_path_cd_cmd + f' if($?) {{ {dependency_check_cmd} }} ;'
            env_setup_cmd = '& ./envsetup.ps1'
            windows_shell_command += f' if($?) {{ {env_path_cd_cmd} }} ;'
            windows_shell_command += f' if($?) {{ {env_setup_cmd} }} ;'
            curr_dir_command = 'cd ' + cwd + ';'
            windows_shell_command += f' if($?) {{ {curr_dir_command} }} ;'
        else:
            windows_shell_command = 'cd ' + cwd + ';'
        for cmd in commands:
            windows_shell_command += f' if($?) {{ {cmd} }} ;'
        return self.ssh_ins.execute(windows_shell_command, shell=True, cwd=cwd)

    def push(self, src_path, dst_path):
        """
        Push file at src_path to dst_path in the remote device.
        src_path: Location of the host file to be pushed.
        dst_path: Location of folder where file has to be pushed.
        """
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if os.path.isfile(src_path):
                if self.is_folder(dst_path):
                    dst_path = os.path.join(dst_path, os.path.basename(src_path))
                self.ftp.put(src_path, dst_path)
        except Exception as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def make_directory(self, dir_name):
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if not self.is_folder(dir_name):
                base_dir = os.path.dirname(dir_name)
                if not self.is_folder(base_dir):
                    # Recursive call to create base directory
                    self.make_directory(base_dir)
                self.ftp.mkdir(dir_name)
        except Exception as e:
            ret = -1
            stderr = str(e)
        return ret, stdout, stderr

    def is_folder(self, remote_path):
        try:
            # Get attributes of the remote path
            attributes = self.ftp.lstat(remote_path)
            # Check if it's a directory
            return S_ISDIR(attributes.st_mode)
        except FileNotFoundError:
            # Handle the case where the path doesn't exist
            return False

    def is_path(self, remote_path):
        try:
            # Get attributes of the remote path
            if self.ftp.lstat(remote_path):
                return True
        except FileNotFoundError:
            # Handle the case where the path doesn't exist
            return False

    def pull(self, device_src_path, host_dst_dir):
        """
        Pull given file/folder from target to host
        """
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if self.is_folder(device_src_path):
                self.pull_folder(host_dst_dir, device_src_path)
            else:
                self.pull_file(os.path.join(host_dst_dir, os.path.basename(device_src_path)),
                               device_src_path)
        except Exception as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def pull_file(self, host_dest_path, device_src_path):
        """
        Pull one file from target system
        """
        host_dest_dir = os.path.dirname(host_dest_path)
        if not os.path.exists(host_dest_dir):
            os.makedirs(host_dest_dir, exist_ok=True)

        src_name = os.path.basename(device_src_path)
        host_dest_path = os.path.join(host_dest_dir, src_name)
        self._logger.debug('Pulling %s to %s from device' % (device_src_path, host_dest_path))
        self.ftp.get(device_src_path, host_dest_path)

    def pull_folder(self, host_dest_path, device_src_path):
        """
        Pull folder from target system
        """
        names = self.ftp.listdir(device_src_path)
        host_dest_path = os.path.join(host_dest_path, os.path.basename(device_src_path))
        if not os.path.exists(host_dest_path):
            os.makedirs(host_dest_path, exist_ok=True)

        for path in names:
            device_path = os.path.join(device_src_path, path)
            if self.is_folder(device_path):
                self.pull_folder(host_dest_path, device_path)
            else:
                self.pull_file(os.path.join(host_dest_path, path), device_path)

    def remove(self, target_path):
        """
        Remove given file/folder from target system
        """
        ret = 0
        stdout = ''
        stderr = ''
        try:
            if self.is_folder(target_path):
                names = self.ftp.listdir(target_path)
                for path in names:
                    self.remove(os.path.join(target_path, path))

                # Delete folder
                self.ftp.rmdir(target_path)

            else:
                # Delete file
                self.ftp.remove(target_path)
        except Exception as e:
            ret = -1
            stderr = str(e)

        return ret, stdout, stderr

    def close(self):
        #close ftp connection
        self.ftp.close()
        self._logger.info('SFTP connection closed')
        #close ssh connection
        self.ssh_ins.close()
