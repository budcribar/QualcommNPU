# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import abc


class DeviceInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_connected(self):
        raise NotImplementedError('Method is_connected must be implemented to use this base class')

    @abc.abstractmethod
    def execute(self, args, cwd='.', env=None):
        raise NotImplementedError('Method execute must be implemented to use this base class')

    @abc.abstractmethod
    def pull(self, src_path, dst_path):
        raise NotImplementedError('Method pull must be implemented to use this base class')

    @abc.abstractmethod
    def push(self, src_path, dst_path):
        raise NotImplementedError('Method push must be implemented to use this base class')

    @abc.abstractmethod
    def make_directory(self, dir_name):
        raise NotImplementedError('Method make_directory must be implemented to use this base class')

    @abc.abstractmethod
    def remove(self, target_path):
        raise NotImplementedError('Method remove must be implemented to use this base class')
