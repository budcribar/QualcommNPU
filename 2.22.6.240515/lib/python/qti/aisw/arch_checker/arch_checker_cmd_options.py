# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.arch_checker.cmd_options import CmdOptions
import argparse
from argparse import RawTextHelpFormatter

class MainCmdOptions(CmdOptions):
    def __init__(self, command, args):
        super().__init__(command, args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
        self.parser._action_groups.pop()
        required = self.parser.add_argument_group('required arguments')
        optional = self.parser.add_argument_group('optional arguments')

        if self.command.startswith("qnn"):
            required.add_argument('-i', '--input_json', required=True, type=str, help="Path to json file")
            optional.add_argument('-b', '--bin', required=False, type=str, help="Path to a bin file")

        elif self.command.startswith("snpe"):
            required.add_argument('-i', '--input_dlc', required=True, type=str, help="Path to dlc file")

        optional.add_argument('-o', '--output_path', required=False, type=str, help="Path where the output csv should be saved. If not specified, the output csv will be written to the same path as the input file")
        optional.add_argument('-m', '--modify', required=False, type=str, nargs='?',const='', help="The query to select the modifications to apply.\n\
        --modify or --modify show - To see all the possible modifications \n\
        --modify all - To apply all the possible modifications \n\
        --modify apply=rule_name1,rule_name2 - To apply modifications for specified rule names. The list of rules should be comma separated without spaces")
        self.initialized = True

    def parse(self):
        if (not self.initialized):
            self.initialize()
        opts, _ = self.parser.parse_known_args(self.args)
        return opts