# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

NODE_IO_TENSORS = "node_io_tensors"
CURRENT_DIMS = "dims"
INPUT_NAMES = "input_names"
OUTPUT_NAMES = "output_names"
SCALAR_PARAMS = "scalar_params"
TENSOR_PARAMS = "tensor_params"
STRIDE = "stride"
FILTER_SIZE = "filter_size"
DATA = "data"
PERM = "perm"
ELTWISE_TYPE = "eltwise_type"

INDEX_BATCH = 0
INDEX_HEIGHT = 1
INDEX_WIDTH = 2
INDEX_CHANNEL = 3
INDEX_FILTER_HEIGHT = 0
INDEX_FILTER_WIDTH = 1
INDEX_FILTER_DEPTH = 2
INDEX_FILTER_DEPTH_OUT = 3
MAX_RANK = 4
INDEX_STRIDE_HEIGHT = 0
INDEX_STRIDE_WIDTH = 1

MULTIPLIER_2 = 2
MULTIPLIER_32 = 32
MULTIPLIER_256 = 256

UNARY_ELWISE_OP = ['ElementWiseAbs', 'ElementWiseCeil', 'ElementWiseCos', 'ElementWiseExp', 'ElementWiseFloor', 'ElementWiseLog', 'ElementWiseNeg', 'ElementWiseNot', 'ElementWiseRound', 'ElementWiseRsqrt', 'ElementWiseSin', 'ElementWiseSquareRoot']
BINARY_ELWISE_OP = ['ElementWiseAdd', 'ElementWiseAnd', 'ElementWiseDivide', 'ElementWiseEqual', 'ElementWiseFloorDiv', 'ElementWiseGreater', 'ElementWiseGreaterEqual', 'ElementWiseLess', 'ElementWiseLessEqual', 'ElementWiseMaximum', 'ElementWiseMinimum', 'ElementWiseMultiply', 'ElementWiseNotEqual', 'ElementWiseOr', 'ElementWisePower', 'ElementWiseSquaredDifference', 'ElementWiseSubtract']
TERNARY_ELWISE_OP = ['ElementWiseSelect']
CONV_OP = ['Conv2d', 'DepthWiseConv2d', 'TransposeConv2d']

QNN_DATATYPE_UINT_32 = '306'

O_C_GRAPH_NODENAME = "Graph/Node_name"
O_C_INPUTS = "Input_tensor_name:[dims]"
O_C_OUTPUTS = "Output_tensor_name:[dims]"
O_C_ISSUE = "Issue"
O_C_RECOMM = "Recommendation"
O_C_PARAMS = "Additional parameters"
OUTPUT_CSV_HEADER=[O_C_GRAPH_NODENAME, O_C_INPUTS, O_C_OUTPUTS, O_C_ISSUE, O_C_RECOMM, O_C_PARAMS]

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
            th:last-child {
                width: 20%;
            }
        </style>
    '''