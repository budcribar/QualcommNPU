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

from lib.utils.nd_logger import setup_logger
from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import DeepAnalyzerError
from lib.deep_analyzer.nd_model_dissection_analyzer import ModelDissectionAnalyzer
from lib.deep_analyzer.nd_reference_code_analyzer import ReferenceAnalyzer
from lib.wrapper.nd_tool_setup import ToolConfig


class DeepAnalyzer:
    def __init__(self, args, logger):

        if logger is None:
            logger = logging.getLogger() # OR logger = setup_logger(verbose)???
        self.logger = logger
        self.config = args
        if args.deep_analyzer is None:
            raise DeepAnalyzerError(get_message('ERROR_DEEP_ANALYZER_INVALID_ANALYZER_NAME')(args.deep_analyzer))
        self.analyzer_name = args.deep_analyzer
        self.verifier_summary = args.result_csv
        self.envToolConfig = ToolConfig()


    def run_model_dissection_analyzer(self, summary_df):
        modelDissectionAnalyzer = ModelDissectionAnalyzer(summary_df, self.config, self.logger, self.envToolConfig)
        modelDissectionAnalyzer.runModelDissection()

    def run_quantization_analyzer(self, summary_df):
        pass

    def run_reference_code_analyzer(self):
        referenceAnalyzer = ReferenceAnalyzer(self.config, self.logger, self.envToolConfig)
        referenceAnalyzer.executeModelwithRefCode()
        referenceAnalyzer.validateAccuracy()

    def analyze(self):
        """Runs the analysis on given analyzer and verifier"""

        summary_df = pd.read_csv(self.verifier_summary)
        analyzerName = self.analyzer_name.lower()
        if analyzerName == "modeldissectionanalyzer":
            self.run_model_dissection_analyzer(summary_df)
        elif analyzerName == "quantizationanalyzer":
            self.run_quantization_analyzer(summary_df)
        elif analyzerName == "referencecodeanalyzer":
            self.run_reference_code_analyzer()
        else:
            raise DeepAnalyzerError(get_message('ERROR_DEEP_ANALYZER_INVALID_ANALYZER_NAME')(self.analyzer_name))
