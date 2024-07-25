# ==============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils.converter_utils import log_info
from qti.aisw.converters.common.arch_linter.core import linter_tool
import qti.aisw.converters.common.arch_linter.core.config as config

class ArchLinter(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self):
            super(ArchLinter.ArgParser, self).__init__()
            arch_linter_group = self.add_argument_group(title='Architecture Checker Options(Experimental)')

            arch_linter_group.add_argument('--arch_checker', default=False, action='store_true',
                                help='Pass this option to enable architecture checker tool.\
                                This is an experimental option for models that are intended to run on HTP backend.')

    def __init__(self, args):
        self.arch_checker = args.arch_checker
        if not self.arch_checker:
            return

        self.quantized_model = False
        if args.input_list:
            self.quantized_model = True
        else:
            self.quantized_model = False

        if args.output_path is None:
            filename, _ = os.path.splitext(os.path.realpath(args.input_network))
            output_prefix = filename
        else:
            output_prefix = os.path.splitext(os.path.realpath(args.output_path))[0]

        self.model_json_path = output_prefix + "_net.json"
        self.arch_checker_csv_out_path = output_prefix + "_architecture_checker.csv"
        self.arch_checker_html_out_path = output_prefix + "_architecture_checker.html"

        self.node_pool = linter_tool.parse_model(self.model_json_path, self.quantized_model)

    def run_linter(self, optimized_graph, backend):

        if self.arch_checker:
            constraints = os.path.join(os.environ.get('QNN_SDK_ROOT'),
                config.CONSTRAINTS_PATH)
            if not os.path.isfile(constraints):
                log_error("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")

            df = linter_tool.run_checks(self.node_pool, constraints, optimized_graph, backend)
            df.to_csv(self.arch_checker_csv_out_path, index=False)

            log_info("Architecture Checker csv output saved at: %s" % self.arch_checker_csv_out_path)

            linter_tool.get_html(df, self.arch_checker_html_out_path, self.model_json_path)
            log_info("Architecture Checker html output saved at: %s" % self.arch_checker_html_out_path)

