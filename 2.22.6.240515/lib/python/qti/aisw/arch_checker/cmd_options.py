# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABC, abstractmethod

class CmdOptions(ABC):
    def __init__(self, command, args):
        self.args = args
        self.command = command
        self.initialized = False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def parse(self):
        pass