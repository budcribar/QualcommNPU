# ==============================================================================
# Copyright (c) 2020 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================

import logging
logger = logging.getLogger(__name__)

from .bm_constants import BmConstants
from .bm_jsonkeys import BmJsonKeys


class QNN(BmConstants, BmJsonKeys):

    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if QNN.__instance is None:
            QNN()
        return QNN.__instance

    def __init__(self):

        BmConstants.__init__(self)
        BmJsonKeys.__init__(self)

        if QNN.__instance is not None:
            raise Exception("QNN class is a singleton!")
        else:
            QNN.__instance = self

