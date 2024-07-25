# =============================================================================
#
#  Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.device.devices.nd_android import AndroidInterface
from qti.aisw.accuracy_debugger.lib.device.devices.nd_linux_embedded import LinuxEmbeddedInterface
from qti.aisw.accuracy_debugger.lib.device.devices.nd_wos import WosInterface
from qti.aisw.accuracy_debugger.lib.device.devices.nd_x86 import X86Interface
from qti.aisw.accuracy_debugger.lib.device.devices.nd_x86_windows_msvc import X86WindowsMsvcInterface
from qti.aisw.accuracy_debugger.lib.device.devices.nd_qnx import QnxInterface
from qti.aisw.accuracy_debugger.lib.device.devices.nd_wos_remote import WosRemoteInterface
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import X86_windows_Architectures


class DeviceFactory(object):

    @staticmethod
    def factory(device, deviceId, logger=None, device_ip=None, device_username=None, device_password=None, device_setup_path=None):
        if not deviceId:
            deviceId = ''
        if device == "android":
            return AndroidInterface(deviceId, None, logger)
        elif device == "linux-embedded":
            return LinuxEmbeddedInterface(deviceId, None, logger)
        elif device == "x86":
            return X86Interface(logger)
        elif device == "x86_64-windows-msvc":
            return X86WindowsMsvcInterface(logger, device_setup_path=device_setup_path)
        elif device == "wos":
            return WosInterface(logger, device_setup_path=device_setup_path)
        elif device == "qnx":
            return QnxInterface(logger, device_ip, device_username, device_password)
        elif device == "wos-remote":
            return WosRemoteInterface(logger, device_ip, device_username, device_password)
