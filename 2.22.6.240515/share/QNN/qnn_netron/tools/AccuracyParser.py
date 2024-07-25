# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import sys
import pandas as pd
import re
from ParserInterface import ParserInterface


class AccuracyParser(ParserInterface):
    """
    This parser parses the CSV of per-node accuracy metrics from verification.
    The parser expects the CSV file path to be passed in via argv.
    """

    def __init__(self):
        self.verifiers = {'adjustedrtolatol', 'cosinesimilarity', 'l1error', 'mae',
                          'mse', 'meaniou', 'rtolatol', 'topk', 'sqnr', 'scaleddiff','Metric'}

    @staticmethod
    def sanitize_name(name):
        """
        This function modifies a given name to adhere with C++ naming standard as names (node or tensors) are used
        as variable name lookup in generated model.cpp.
        :param name: name to modify
        """
        # All separators should be _ to follow C++ variable naming standard
        name = re.sub(r'\W+', "_", name)
        # prefix needed as C++ variables cant start with numbers
        return name if name[0].isalpha() else "_" + name

    def parse(self, file):
        """
        This method parses the verification accuracy values for each node, for each verifier that is used,
        and then prints out the results.
        :param file: path to the CSV file to parse
        """
        df = pd.read_csv(file)
        verifier_cols = [col for col in df.columns for v in self.verifiers if v in col]
        if len(verifier_cols) > 0:
            for ind in df.index:
                node = self.sanitize_name(os.path.basename(df['Name'][ind]))
                accuracy = [str(df[v][ind]) for v in verifier_cols]
                accuracy = ','.join(accuracy)
                print(node + ',' + accuracy)


if __name__ == "__main__":
    csv_file = sys.argv[1]
    accuracyParser = AccuracyParser()
    accuracyParser.parse(csv_file)
