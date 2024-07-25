# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.device.devices.nd_android import AndroidInterface
from lib.device.devices.nd_linux_embedded import LinuxEmbeddedInterface
from lib.device.devices.nd_x86 import X86Interface


class DeviceFactory(object):
    @staticmethod
    def factory(device, deviceId, logger=None):
        if not deviceId:
            deviceId=''
        if device == "android":
            return AndroidInterface(deviceId, None, logger)
        elif device == "linux-embedded":
            return LinuxEmbeddedInterface(deviceId, None, logger)
        elif device == "x86":
            return X86Interface(logger)
