# =============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import pandas as pd
from qti.aisw.arch_checker.arch_checker import ArchChecker
from qti.aisw.converters.common import ir_graph
import qti.aisw.arch_checker.constants as const

class SnpeArchChecker(ArchChecker):

    def __init__(self, c_ir_graph, constraints_json, out_path, logger, model_info):
        super(SnpeArchChecker, self).__init__(c_ir_graph, constraints_json, out_path, logger)
        self.model_info = model_info

    def format_csv_header(self):
        node_name = const.O_C_GRAPH_LAYERNAME
        producer_name = const.O_C_PRODUCER_LAYER
        consumer_name = const.O_C_CONSUMERS_LAYERS
        return node_name, producer_name, consumer_name

    def get_output_header(self):
        return dict.fromkeys(const.DF_HEADER_LAYERS, 'N/A')

    def create_dataframe(self):
        return pd.DataFrame(columns=const.DF_HEADER_LAYERS, dtype=object)

    def save_to_csv(self, df):
        df.to_csv(self.output_file, index=False, columns=const.OUTPUT_CSV_HEADER_LAYERS)

    def is_8bit(self):
        if self.model_info.model_reader.quantizer_command == "N/A":
            return False
        if "act_bitwidth=[16]" in self.model_info.model_reader.quantizer_command:
            return False
        return True

    def is_activation(self, op):
        act_list = [ir_graph.QNN_OP_RELU, ir_graph.QNN_OP_SIGMOID,
        ir_graph.QNN_OP_TANH, ir_graph.QNN_OP_HARD_SWISH, ir_graph.QNN_OP_RELU_MIN_MAX]
        if op.type == ir_graph.IR_OP_NEURON:
            try:
                if op.neuron_type in act_list:
                    return True
            except:
                return False
        return False