# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from __future__ import absolute_import
from multiprocessing import Process
from .bm_constants import *
import os
import time
import re
from common_utils.adb import Adb
from common_utils.tshell_server import TShellRunner
from common_utils.exceptions import AdbShellCmdFailedException
import logging

logger = logging.getLogger(__name__)

REGX_SPACES = re.compile(r'[\s]+')
ONE_HOUR_IN_SEC = 1 * 60 * 60.0


class DeviceFactory(object):
    @staticmethod
    def make_device(device_id, config, product):
        assert device_id, "device id is required"
        assert config, "config is required"
        return BenchmarkDevice(product, device_id, device_id, config.device_path,
                               config.platform, config.host_rootpath, config.hostname)


class BenchmarkDevice(object):
    def __init__(self, product, device_name, serial_no, device_root_dir,
                 platform, host_output_dir, host_name='localhost'):
        assert device_root_dir, "device root directory is required"
        self._device_name = device_name
        self._comm_id = serial_no
        self._device_root_dir = device_root_dir
        self._host_output_dir = host_output_dir
        self._host_name = host_name
        self._mem_proc = None
        self._power_proc = None
        self._platform = platform
        self.adb = Adb('adb', serial_no, hostname=host_name)
        self.product = product

        if self._platform == product.PLATFORM_OS_ANDROID:
            self._device_type = product.DEVICE_TYPE_ARM_ANDROID
        elif self._platform == product.PLATFORM_OS_QNX:
            self._device_type = product.DEVICE_TYPE_ARM_QNX
        elif self._platform == product.PLATFORM_OS_WINDOWS:
            self._device_type = product.DEVICE_TYPE_WINDOWS
        else:
            raise Exception("device: Invalid platform !!!", platform)

        return

    def __str__(self):
        return (('[Device Name:%s ' % self._device_name) +
                ('Device ID:%s ' % self._comm_id) +
                ('HOST NAME:%s ' % self.host_name) +
                ('Device DIR:%s]' % self._device_root_dir))

    @property
    def device_name(self):
        return self._device_name

    @property
    def host_name(self):
        return self._host_name

    @property
    def comm_id(self):
        return self._comm_id

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_root_dir(self):
        return self._device_root_dir

    @property
    def host_output_dir(self):
        return self._host_output_dir

    def __mem_log_file(self):
        return os.path.join(self._device_root_dir,
                            self.product.MEM_LOG_FILE_NAME)

    def push_win_artifacts(self, artifacts):
        self.tshell = TShellRunner()
        self.tshell.start()
        self.tshell.run('open-device ' + self._comm_id , no_exception=True)
        for _host_path, _dev_dir in artifacts:
            if os.path.isfile(_host_path):
                dev_file = os.path.join(_dev_dir, os.path.basename(_host_path)).replace('/', '\\')
                self.push_to_win_device(_host_path, dev_file)
            elif os.path.isdir(_host_path):
                for _root, _dirs, _files in os.walk(_host_path):
                    for _file in _files:
                        rel_dir = os.path.relpath(_root, _host_path)
                        dev_file = os.path.join(_dev_dir, rel_dir, _file).replace('/', '\\')
                        _host_file = os.path.join(_root, _file)
                        self.push_to_win_device(_host_file, dev_file)

    def __capture_mem_droid(self, exe_name):
        time_out = ONE_HOUR_IN_SEC
        t0 = time.time()
        ps_name = exe_name

        # Find the Process ID
        ps_pid = None
        while time_out > (time.time() - t0):
            ret, version_output, err = self.adb.shell(
                'getprop', ['ro.build.version.release'])
            android_version = version_output[0].strip().split()[0]
            if android_version >= "8.0.0":
                ret, ps_output, err = self.adb.shell(
                    'ps', ['-A', '|', 'grep', ps_name])
            else:
                ret, ps_output, err = self.adb.shell(
                    'ps', ['|', 'grep', ps_name])
            if ps_output:
                ps_pid = REGX_SPACES.split(ps_output[0].strip())[1]
                logger.debug(ps_output)
                logger.debug("Found PID ({0}) of the Process".format(ps_pid))
                break
            if ps_pid is not None:
                break

        assert ps_pid, "ERROR: Could not find the Process ID of {0}".format(
            exe_name)

        num_of_samples = 0
        mem_log_file = self.__mem_log_file()
        logger.debug(
            "Capturing memory usage of {0} with PID {1}".format(
                exe_name, ps_pid))
        logger.debug(
            "Time required to determine the PID:{0}".format(
                (time.time() - t0)))
        while time_out > (time.time() - t0):
            if num_of_samples == 0:
                logger.debug(
                    "Memory Log Capture available at: {0}".format(mem_log_file))
                create_or_append = ">"
            else:
                create_or_append = "| cat >>"
            self.adb.shell(
                'dumpsys', [
                    'meminfo', ps_pid, create_or_append, mem_log_file])
            num_of_samples += 1
        return

    def execute(self, commands):
        functions = {
            'shell': self.adb.shell,
            'push': self.adb.push,
            'pull': self.adb.pull
        }
        for b_cmd in commands:
            ret, out, err = functions[b_cmd.function](*b_cmd.params)
            if ret != 0:
                logger.error(out)
                logger.error(err)
                raise AdbShellCmdFailedException
        return

    def execute_win(self, commands):
        for b_cmd in commands:
            if b_cmd[0] == 'push':
                dev_file = os.path.join(b_cmd[2], os.path.basename(b_cmd[1])).replace('/', '\\')
                self.push_to_win_device(b_cmd[1], dev_file)
            elif b_cmd[0] == 'pull':
                self.pull_from_win_device(b_cmd[1], b_cmd[2])
            elif b_cmd[0] == 'shell':
                result = self.tshell.run("cmdd {}".format(b_cmd[1]))
            elif b_cmd[0] == 'dird':
                result = self.tshell.run("dird {}".format(b_cmd[1].replace('/', '\\')),
                                         no_exception = True)
                if result.return_code:
                    logger.error("Test Execution failed on Windows Device !!!")
                    raise Exception("Test Execution failed on Windows Device !!!")
        return

    def push_to_win_device(self, src_file, dest_file):
        result = self.tshell.run("putd -Recurse {0} {1}".format(src_file, dest_file))
        if result.return_code:
            logger.error("Data Push failed on Windows Device !!!")
            raise Exception("Data Push failed on Windows Device !!!")

    def pull_from_win_device(self, src_file, dest_file):
        result = self.tshell.run("getd -Recurse {0} {1}".format(src_file, dest_file))
        if result.return_code:
            logger.error("Data Pull failed on Windows Device !!!")
            raise Exception("Data Pull failed on Windows Device !!!")

    def start_measurement(self, benchmark):
        if benchmark.measurement.type == self.product.MEASURE_MEM:
            if self._mem_proc is None:
                logger.info("starting memory capture in a parallel process")
                if self._platform == self.product.PLATFORM_OS_ANDROID:
                    logger.info("Android platform")
                    self._mem_proc = Process(
                        target=self.__capture_mem_droid, args=(
                            benchmark.exe_name,))
                    self._mem_proc.start()
                else:
                    raise Exception(
                        "start_measurement: Invalid platform !!!",
                        self._platform)
            else:
                logger.info("memory capture is already started")
        return

    def stop_measurement(self, benchmark):
        if benchmark.measurement.type == self.product.MEASURE_MEM:
            if self._mem_proc is not None:
                self._mem_proc.terminate()
                self._mem_proc = None
                logger.info("memory capture is terminated")
                self.adb.pull(self.__mem_log_file(), benchmark.host_result_dir)
        return
