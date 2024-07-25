# =============================================================================
#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.device.helpers.nd_telnet import ExecutionEngine
import paramiko
import logging
import time
import os


class SSHExecutor(ExecutionEngine):
    """
    This class performs the execution on remote device through ssh
    """
    __instance = None

    def __init__(self, host: str, username: str, password=None, logger=None):

        if SSHExecutor.__instance is not None:
            raise ValueError('instance of SSHExecutor already exists')
        else:
            SSHExecutor.__instance = self

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
            self.ssh_client = self.setup_ssh_client()
            assert self.ssh_client, "SSH setup failed with ip %s" % self.host

            self.is_connected = True

        except Exception as e:
            raise Exception("Unable to establish connection with remote @ %s" % self.host)

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = SSHExecutor()
        return cls.__instance

    def setup_ssh_client(self, device_port=22, timeout=100, debug_level=0):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(self.host, port=device_port, username=self.username, password=self.password, allow_agent=False)
            self._logger.info('SSH connection established successfully')
            return client
        except paramiko.AuthenticationException:
            self._logger.info("Authentication failed when connecting to", self.host)
        except paramiko.SSHException as e:
            self._logger.info("Could not connect to", self.host, ":", str(e))
        except Exception as e:
            self._logger.info("Unable to connect {0} error code {1}".format(self.host, str(e)))
        return None

    def execute(self, cmd, log_file='', cwd='.', shell=False, timeout=None):
        """
        This method runs the given command on remote device through ssh and returns error status
        """

        if self.ssh_client.get_transport() is not None:
            self.is_connected = self.ssh_client.get_transport().is_active()
        if not self.is_connected:
            self.ssh_client = self.setup_ssh_client()
        log_redirect = " > " + log_file + " 2>&1" if log_file else ''
        if os.path.exists(log_file):
            cmd = f"rm {log_file}; "+ cmd
            self._logger.debug(f"Removing existing {log_file}")

        std_err = ''
        try:
            ssh_stdin, ssh_stdout, err_code = self.ssh_client.exec_command(cmd + log_redirect)
            self._logger.debug(f"Executing command : '{cmd + log_redirect}' on target")
        except Exception as e:
            err_code = 0
            std_err = str(e)
            self._logger.error(f"Failed to execute command {cmd + log_redirect} on host target. Reason : {e}")

        # Log command output
        for line in iter(lambda: ssh_stdout.readline(2048), ""):
            print(line, end="")
        err_code = ssh_stdout.channel.recv_exit_status()

        # Taking a delay of 100ms as default
        delay = 100
        time.sleep(float(delay) / 1000)
        return err_code, ssh_stdout, std_err

    def close(self):
        if self.is_connected:
            self.ssh_client.close()
            self._logger.info('SSH connection closed')
