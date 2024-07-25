# =============================================================================
#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import ftplib
import os

from qti.aisw.accuracy_debugger.lib.device.helpers.nd_telnet import TelnetExecutor


class Filesystem:
    """
    This class handles ftp connection and file transfer functionalities
    """
    __instance = None

    def __init__(self, host: str, username: str, password=None, logger=None):
        if Filesystem.__instance is not None:
            raise ValueError('instance of TelnetExecutor already exists')
        else:
            Filesystem.__instance = self
        self.remote_dev = TelnetExecutor.getInstance()
        # self.local_dev = ExecutionHandlers.getInstance().get_compile_handler()
        self.ftp_server = host.strip()
        self.ftp_username = username.strip()
        if password:
            self.ftp_password = password
        else:
            self.ftp_password = None
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger()
        self.ftp_dev = self.setup_ftp_connect()

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = Filesystem()
        return cls.__instance

    def setup_ftp_connect(self, device_port=21, timeout=-999):
        """
        This method establishes the ftp connection
        """
        device = ftplib.FTP()
        try:
            device.connect(self.ftp_server, device_port, timeout)
            device.login(self.ftp_username, self.ftp_password)
            self._logger.debug(f'FTP connection established with {self.ftp_server}')
        except Exception as e:
            self._logger.error('setup_ftp_connect : ' + str(e))
            return None
        return device

    def put_file(self, local_file, remote_file):
        """
        This method copies file to remote ftp machine
        """
        # create dir path in remote
        dir_name = os.path.dirname(remote_file)
        file_name = os.path.basename(remote_file)
        create_remote_dir = 'mkdir -p ' + dir_name
        err_status = self.remote_dev.execute(create_remote_dir)
        fd = open(local_file, 'rb')
        push_cmd = 'STOR %s' % file_name
        self.ftp_dev.cwd(dir_name)
        self.ftp_dev.storbinary(push_cmd, fd)
        self.ftp_dev.sendcmd('SITE CHMOD 755 ' + file_name)
        fd.close()

    def ftp_is_folder(self, path):
        """
        This method checks if given path is a folder in remote machine
        """
        current = self.ftp_dev.pwd()
        try:
            self.ftp_dev.cwd(path)
        except:
            self.ftp_dev.cwd(current)
            return False
        self.ftp_dev.cwd(current)
        return True

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
        pull_cmd = 'RETR %s' % device_src_path
        # Seems only binary mode works with QNX FTP deamon
        fd = open(host_dest_path, 'wb')
        if fd is None:
            self._logger.error('Cannot open file (%s) on host' % host_dest_path)
            raise Exception('File not exist')
        else:
            self.ftp_dev.retrbinary(pull_cmd, fd.write)
            self._logger.debug('Pulled %s from device' % (host_dest_path))
            fd.close()

    def pull_folder(self, host_dest_path, device_src_path):
        """
        Pull folder from target system
        """
        try:
            names = self.ftp_dev.nlst(device_src_path)
        except ftplib.all_errors as e:
            # some FTP servers complain when you try and list non-existent paths
            self._logger.error('Could not pull {0}: {1}'.format(device_src_path, e))
            return
        host_dest_path = os.path.join(host_dest_path, os.path.basename(device_src_path))
        if not os.path.exists(host_dest_path):
            os.makedirs(host_dest_path, exist_ok=True)

        for path in names:
            if self.ftp_is_folder(path):
                self.pull_folder(host_dest_path, path)
            else:
                filename = os.path.basename(path)
                self.pull_file(os.path.join(host_dest_path, filename), path)

    def ftp_remove(self, path):
        """
        Recursively delete a directory tree on a remote server
        """
        try:
            names = self.ftp_dev.nlst(path)
        except ftplib.all_errors as e:
            # some FTP servers complain when you try and list non-existent paths
            self._logger.error('Could not remove {0}: {1}'.format(path, e))
            return

        for name in names:
            if os.path.split(name)[1] in ('.', '..'): continue
            self._logger.debug('Checking {0}'.format(name))
            try:
                self.ftp_dev.cwd(name)  # if we can cwd to it, it's a folder
                self.ftp_remove(name)
            except ftplib.all_errors:
                self.ftp_dev.delete(name)
        try:
            self.ftp_dev.rmd(path)
        except ftplib.all_errors as e:
            self._logger.debug('Could not remove {0}: {1}'.format(path, e))

    def close(self):
        self.ftp_dev.close()
        self._logger.debug(f'FTP connection closed')
