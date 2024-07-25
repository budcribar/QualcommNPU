# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


from typing import Any, Dict, List


class Node:
    def __init__(
        self, name: str, op_type: str, inputs: List[str], attrs: Dict[str, Any] = {}
    ):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.attrs = attrs


EINSUM_SUPPORTED = {
    "i,d->id": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [0]}),
        Node(
            "transpose_4", "Transpose", inputs=["unsqueeze_3"], attrs={"perm": [1, 0]}
        ),
        Node("unsqueeze_5", "Unsqueeze", inputs=["input_2"], attrs={"axis": [0]}),
        Node("matmul_6", "MatMul", inputs=["transpose_4", "unsqueeze_5"]),
        Node("output", "Output", inputs=["matmul_6"]),
    ],
    "ij,jk->ik": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("matmul_1", "MatMul", inputs=["input_1", "input_2"]),
        Node("output", "Output", inputs=["matmul_1"]),
    ],
    "bm,bhm->bh": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [1]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [0, 2, 1]}),
        Node("matmul_5", "MatMul", inputs=["unsqueeze_3", "transpose_4"]),
        Node("squeeze_6", "Squeeze", inputs=["matmul_5"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["squeeze_6"]),
    ],
    "bl,blh->bh": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [1]}),
        Node("matmul_4", "MatMul", inputs=["unsqueeze_3", "input_2"]),
        Node("squeeze_5", "Squeeze", inputs=["matmul_4"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["squeeze_5"]),
    ],
    "abc,cd->abc": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("mul_5", "Mul", inputs=["transpose_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "abc,dc->abc": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "acb,cd->abc": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "acb,dc->abc": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["unsqueeze_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "bac,cd->bac": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("mul_5", "Mul", inputs=["transpose_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "bac,dc->bac": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "bca,cd->bac": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "bca,dc->bac": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["unsqueeze_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "abc,cd->abd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("matmul_3", "MatMul", inputs=["input_1", "input_2"]),
        Node("output", "Output", inputs=["matmul_3"]),
    ],
    "abc,dc->abd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("matmul_4", "MatMul", inputs=["input_1", "transpose_3"]),
        Node("output", "Output", inputs=["matmul_4"]),
    ],
    "acb,cd->abd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("matmul_4", "MatMul", inputs=["transpose_3", "input_2"]),
        Node("output", "Output", inputs=["matmul_4"]),
    ],
    "acb,dc->abd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("matmul_5", "MatMul", inputs=["transpose_3", "transpose_4"]),
        Node("output", "Output", inputs=["matmul_5"]),
    ],
    "abc,cd->acd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("mul_5", "Mul", inputs=["transpose_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "abc,dc->acd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "acb,cd->acd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "acb,dc->acd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["unsqueeze_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "cab,cd->cad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [1, 2, 0]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node(
            "transpose_6", "Transpose", inputs=["mul_5"], attrs={"perm": [2, 1, 0, 3]}
        ),
        Node("reducesum_7", "ReduceSum", inputs=["transpose_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "cab,dc->cad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [1, 2, 0]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("unsqueeze_5", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "unsqueeze_5"]),
        Node(
            "transpose_7", "Transpose", inputs=["mul_6"], attrs={"perm": [2, 1, 0, 3]}
        ),
        Node("reducesum_8", "ReduceSum", inputs=["transpose_7"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_8"]),
    ],
    "cba,cd->cad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [2, 1, 0]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node(
            "transpose_6", "Transpose", inputs=["mul_5"], attrs={"perm": [2, 1, 0, 3]}
        ),
        Node("reducesum_7", "ReduceSum", inputs=["transpose_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "cba,dc->cad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [2, 1, 0]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("unsqueeze_5", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "unsqueeze_5"]),
        Node(
            "transpose_7", "Transpose", inputs=["mul_6"], attrs={"perm": [2, 1, 0, 3]}
        ),
        Node("reducesum_8", "ReduceSum", inputs=["transpose_7"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_8"]),
    ],
    "cab,cd->cbd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [1, 2, 0]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node(
            "transpose_6", "Transpose", inputs=["mul_5"], attrs={"perm": [0, 2, 1, 3]}
        ),
        Node("reducesum_7", "ReduceSum", inputs=["transpose_6"], attrs={"axis": [0]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "cab,dc->cbd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [2, 1, 0]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("unsqueeze_5", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "unsqueeze_5"]),
        Node(
            "transpose_7", "Transpose", inputs=["mul_6"], attrs={"perm": [2, 1, 0, 3]}
        ),
        Node("reducesum_8", "ReduceSum", inputs=["transpose_7"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_8"]),
    ],
    "cba,cd->cbd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [2, 1, 0]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node(
            "transpose_6", "Transpose", inputs=["mul_5"], attrs={"perm": [0, 2, 1, 3]}
        ),
        Node("reducesum_7", "ReduceSum", inputs=["transpose_6"], attrs={"axis": [0]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "cba,dc->cbd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [1, 2, 0]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("unsqueeze_5", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "unsqueeze_5"]),
        Node(
            "transpose_7", "Transpose", inputs=["mul_6"], attrs={"perm": [2, 1, 0, 3]}
        ),
        Node("reducesum_8", "ReduceSum", inputs=["transpose_7"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_8"]),
    ],
    "bac,cd->bad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("matmul_3", "MatMul", inputs=["input_1", "input_2"]),
        Node("output", "Output", inputs=["matmul_3"]),
    ],
    "bac,dc->bad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("matmul_4", "MatMul", inputs=["input_1", "transpose_3"]),
        Node("output", "Output", inputs=["matmul_4"]),
    ],
    "bca,cd->bad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("matmul_4", "MatMul", inputs=["transpose_3", "input_2"]),
        Node("output", "Output", inputs=["matmul_4"]),
    ],
    "bca,dc->bad": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("matmul_5", "MatMul", inputs=["transpose_3", "transpose_4"]),
        Node("output", "Output", inputs=["matmul_5"]),
    ],
    "bac,cd->bcd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [1, 0, 2]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [2]}),
        Node(
            "transpose_5",
            "Transpose",
            inputs=["unsqueeze_4"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("mul_6", "Mul", inputs=["transpose_5", "input_2"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [0]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "bac,dc->bcd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["transpose_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "bca,cd->bcd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [2, 0, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("mul_5", "Mul", inputs=["unsqueeze_4", "input_2"]),
        Node("reducesum_6", "ReduceSum", inputs=["mul_5"], attrs={"axis": [0]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "bca,dc->bcd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [3]}),
        Node("transpose_5", "Transpose", inputs=["input_2"], attrs={"perm": [1, 0]}),
        Node("mul_6", "Mul", inputs=["unsqueeze_4", "transpose_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "blq,blk->blqk": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 3, 2]},
        ),
        Node("unsqueeze_5", "Unsqueeze", inputs=["input_2"], attrs={"axis": [2]}),
        Node("matmul_6", "MatMul", inputs=["transpose_4", "unsqueeze_5"]),
        Node("output", "Output", inputs=["matmul_6"]),
    ],
    "ibh,hnd->ibnd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["unsqueeze_3"], attrs={"axis": [3]}),
        Node(
            "transpose_5",
            "Transpose",
            inputs=["unsqueeze_4"],
            attrs={"perm": [0, 1, 4, 2, 3]},
        ),
        Node("mul_6", "Mul", inputs=["transpose_5", "input_2"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [2]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "bqc,bchw->bqhw": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [3]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["unsqueeze_3"], attrs={"axis": [4]}),
        Node("unsqueeze_5", "Unsqueeze", inputs=["input_2"], attrs={"axis": [1]}),
        Node("mul_6", "Mul", inputs=["unsqueeze_4", "unsqueeze_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [2]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "blq,bhlk->bhlqk": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [1]}),
        Node("unsqueeze_4", "Unsqueeze", inputs=["unsqueeze_3"], attrs={"axis": [3]}),
        Node(
            "transpose_5",
            "Transpose",
            inputs=["unsqueeze_4"],
            attrs={"perm": [0, 1, 2, 4, 3]},
        ),
        Node("unsqueeze_6", "Unsqueeze", inputs=["input_2"], attrs={"axis": [3]}),
        Node("matmul_7", "MatMul", inputs=["transpose_5", "unsqueeze_6"]),
        Node("output", "Output", inputs=["matmul_7"]),
    ],
    "ibnd,hnd->ibh": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [2]}),
        Node("mul_4", "Mul", inputs=["unsqueeze_3", "input_2"]),
        Node("reducesum_5", "ReduceSum", inputs=["mul_4"], attrs={"axis": [3]}),
        Node("reducesum_6", "ReduceSum", inputs=["reducesum_5"], attrs={"axis": [3]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "abcd,cde->abe": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_1"], attrs={"axis": [4]}),
        Node("mul_4", "Mul", inputs=["unsqueeze_3", "input_2"]),
        Node("reducesum_5", "ReduceSum", inputs=["mul_4"], attrs={"axis": [2]}),
        Node("reducesum_6", "ReduceSum", inputs=["reducesum_5"], attrs={"axis": [2]}),
        Node("output", "Output", inputs=["reducesum_6"]),
    ],
    "ijbn->bnij": [
        Node("input_1", "Input", inputs=[]),
        Node(
            "transpose_2", "Transpose", inputs=["input_1"], attrs={"perm": [2, 3, 0, 1]}
        ),
        Node("output", "Output", inputs=["transpose_2"]),
    ],
    "ibnd,jbnd->bnij": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node(
            "transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [1, 2, 3, 0]}
        ),
        Node(
            "transpose_4", "Transpose", inputs=["input_2"], attrs={"perm": [1, 2, 3, 0]}
        ),
        Node("unsqueeze_5", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [4]}),
        Node("unsqueeze_6", "Unsqueeze", inputs=["transpose_4"], attrs={"axis": [3]}),
        Node("mul_7", "Mul", inputs=["unsqueeze_5", "unsqueeze_6"]),
        Node("reducesum_8", "ReduceSum", inputs=["mul_7"], attrs={"axis": [2]}),
        Node("output", "Output", inputs=["reducesum_8"]),
    ],
    "bnij,jbnd->ibnd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node(
            "transpose_3", "Transpose", inputs=["input_1"], attrs={"perm": [2, 3, 0, 1]}
        ),
        Node("unsqueeze_4", "Unsqueeze", inputs=["transpose_3"], attrs={"axis": [4]}),
        Node("unsqueeze_5", "Unsqueeze", inputs=["input_2"], attrs={"axis": [0]}),
        Node("mul_6", "Mul", inputs=["unsqueeze_4", "unsqueeze_5"]),
        Node("reducesum_7", "ReduceSum", inputs=["mul_6"], attrs={"axis": [1]}),
        Node("output", "Output", inputs=["reducesum_7"]),
    ],
    "bhlqk,bhkd->bhlqd": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_2"], attrs={"axis": [2]}),
        Node("matmul_4", "MatMul", inputs=["input_1", "unsqueeze_3"]),
        Node("output", "Output", inputs=["matmul_4"]),
    ],
    "bhlqd,bhkd->bhlqk": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("unsqueeze_3", "Unsqueeze", inputs=["input_2"], attrs={"axis": [2]}),
        Node(
            "transpose_4",
            "Transpose",
            inputs=["unsqueeze_3"],
            attrs={"perm": [0, 1, 2, 4, 3]},
        ),
        Node("matmul_5", "MatMul", inputs=["input_1", "transpose_4"]),
        Node("output", "Output", inputs=["matmul_5"]),
    ],
    "hbwpc,hbwqc->hbwpq": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node(
            "transpose_3",
            "Transpose",
            inputs=["input_2"],
            attrs={"perm": [0, 1, 2, 4, 3]},
        ),
        Node("matmul_4", "MatMul", inputs=["input_1", "transpose_3"]),
        Node("output", "Output", inputs=["matmul_4"]),
    ],
    "hbwij,hbwjc->hbwic": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("matmul_3", "MatMul", inputs=["input_1", "input_2"]),
        Node("output", "Output", inputs=["matmul_3"]),
    ],
    "bchq,bkhc->bkhq": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node(
            "transpose", "Transpose", inputs=["input_1"], attrs={"perm": [0, 2, 1, 3]}
        ),
        Node(
            "reducesum",
            "ReduceSum",
            inputs=["transpose"],
            attrs={"keepdims": 1, "axis": [1]},
        ),
        Node("matmul", "MatMul", inputs=["input_2", "reducesum"]),
        Node("output", "Output", inputs=["matmul"]),
    ],
    "bid,bjd->bij": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node(
            "transpose", "Transpose", inputs=["input_2"], attrs={"perm": [0, 2, 1]}
        ),
        Node("matmul", "MatMul", inputs=["input_1", "transpose"]),
        Node("output", "Output", inputs=["matmul"]),
    ],
    "bij,bjd->bid": [
        Node("input_1", "Input", inputs=[]),
        Node("input_2", "Input", inputs=[]),
        Node("matmul", "MatMul", inputs=["input_1", "input_2"]),
        Node("output", "Output", inputs=["matmul"]),
    ],
}
