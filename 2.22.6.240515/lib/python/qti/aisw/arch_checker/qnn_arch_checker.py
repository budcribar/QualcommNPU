# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import pandas as pd
from qti.aisw.arch_checker.arch_checker import ArchChecker
from qti.aisw.converters.common import ir_graph
import numpy as np
import qti.aisw.arch_checker.constants as const

class QnnArchChecker(ArchChecker):

    def __init__(self, c_ir_graph, constraints_json, out_path, logger, model_json):
        super(QnnArchChecker, self).__init__(c_ir_graph, constraints_json, out_path, logger)
        self.model_json = model_json

    def format_csv_header(self):
        node_name = const.O_C_GRAPH_NODENAME
        producer_name = const.O_C_PRODUCER_NODE
        consumer_name = const.O_C_CONSUMERS_NODES
        return node_name, producer_name, consumer_name

    def get_output_header(self):
        return dict.fromkeys(const.DF_HEADER_NODES, 'N/A')

    def create_dataframe(self):
        return pd.DataFrame(columns=const.DF_HEADER_NODES, dtype=object)

    def save_to_csv(self, df):
        # Save provided df to csv
        df.to_csv(self.output_file, index=False, columns=const.OUTPUT_CSV_HEADER_NODES)

    def is_8bit(self):
        with open(self.model_json) as f:
            converter_command = json.load(f)["converter_command"]

        if 'act_bw=8' in converter_command and "input_list=None" not in converter_command:
            return True
        else:
            return False

    def is_activation(self, op):
        if op.type == ir_graph.IR_OP_NEURON:
            return True
        return False