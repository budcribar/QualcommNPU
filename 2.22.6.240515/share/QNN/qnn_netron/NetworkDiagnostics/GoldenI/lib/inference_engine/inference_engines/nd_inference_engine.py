# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import abc


class InferenceEngine:
    __metaclass__ = abc.ABCMeta

    def __init__(self, context, converter, executor):
        self.context = context
        self.converter = converter
        self.executor = executor

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def get_graph_structure(self):
        pass
