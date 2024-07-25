# ==============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

INDEX_CHANNEL = 3

MODIFIABLE = "modifiable"
MOD_CONDITION = "mod_condition"
MOD_STRING = "mod_string"
MODIFICATION_STATUS = "Done"

RULE_CTGY_G = "graph"
RULE_CTGY_SN = "single_node"
RULE_CTGY_P = "patterns"
RULE_NAME = "name"

O_C_GRAPH_LAYERNAME = "Graph/Layer_name"
O_C_GRAPH_NODENAME = "Graph/Node_name"
O_C_TYPE = "Type"
O_C_INPUTS = "Input_tensor_name:[dims]"
O_C_OUTPUTS = "Output_tensor_name:[dims]"
O_C_ISSUE = "Issue"
O_C_RECOMM = "Recommendation"
O_C_PARAM = "Parameters"
O_C_PRODUCER_LAYER = "Previous layer"
O_C_CONSUMERS_LAYERS = "Next layers"
O_C_PRODUCER_NODE = "Previous node"
O_C_CONSUMERS_NODES = "Next nodes"
O_C_MODIFICATION = "Modification"
O_C_MODIFICATION_INFO = "Modification_info"
INTERNAL_RULEID = "rule_id"

OUTPUT_CSV_HEADER_LAYERS=[O_C_GRAPH_LAYERNAME, O_C_ISSUE, O_C_RECOMM, O_C_TYPE, O_C_INPUTS, O_C_OUTPUTS, O_C_PARAM, O_C_PRODUCER_LAYER, O_C_CONSUMERS_LAYERS, O_C_MODIFICATION, O_C_MODIFICATION_INFO]
OUTPUT_CSV_HEADER_NODES=[O_C_GRAPH_NODENAME, O_C_ISSUE, O_C_RECOMM, O_C_TYPE, O_C_INPUTS, O_C_OUTPUTS, O_C_PARAM, O_C_PRODUCER_NODE, O_C_CONSUMERS_NODES, O_C_MODIFICATION, O_C_MODIFICATION_INFO]

DF_HEADER_LAYERS = OUTPUT_CSV_HEADER_LAYERS + [INTERNAL_RULEID]
DF_HEADER_NODES = OUTPUT_CSV_HEADER_NODES + [INTERNAL_RULEID]

HTML_STYLE = '''
            <style>
                h1 {
                    font-family: Arial;
                }
                h2 {
                    font-family: Arial;
                    word-break: break-word
                }
                table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: fixed;
                    width: 100%;
                    overflow-y:scroll;
                }
                tr:nth-child(even) {
                    background: #E0E0E0;
                }
                tr:hover{
                    background-color: silver;
                }

                td{
                    word-wrap:break-word;
                    padding: 5px;
                }
                pre{
                    font-family: Arial;
                    overflow-x: auto;
                    overflow-y: scroll;
                    max-height: 250px;
                }
                th:nth-child(7) {
                    width: 15%;
                }
            </style>
        '''
