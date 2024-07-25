# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import SnoopingError
from qti.aisw.accuracy_debugger.lib.snooping.nd_layerwise_snooper import LayerwiseSnooping
from qti.aisw.accuracy_debugger.lib.wrapper.nd_tool_setup import ToolConfig


class Snooper:

    def __init__(self, args, logger):

        if logger is None:
            logger = logging.getLogger()  # OR logger = setup_logger(verbose)???
        self.logger = logger
        self.config = args
        self.snooping_type = args.snooping
        self.envToolConfig = ToolConfig()

    def run_layerwise_snooping(self):
        layerwiseSnooper = LayerwiseSnooping(self.config, self.logger, self.envToolConfig)
        layerwiseSnooper.run()

    # def run_fp16_sweep(self):
    #     fp16Sweeper = Fp16Sweep(self.config, self.logger, self.envToolConfig)
    #     fp16Sweeper.run()

    def analyze(self):
        """Runs the analysis on based on given snooping technique."""
        snooperName = self.snooping_type.lower()
        if snooperName == "layerwise":
            self.run_layerwise_snooping()
        # elif snooperName == "fp16sweep":
        #     self.run_fp16_sweep()
        else:
            raise SnoopingError(
                get_message('ERROR_DEEP_ANALYZER_INVALID_ANALYZER_NAME')(self.analyzer_name))
