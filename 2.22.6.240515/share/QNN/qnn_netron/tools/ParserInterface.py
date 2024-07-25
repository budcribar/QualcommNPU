# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import abc


class ParserInterface:
    @abc.abstractmethod
    def parse(self, file):
        """
        Parse the specified file.
        :param file: the file to parse
        """
        pass
