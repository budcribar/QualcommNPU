# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm
from tvm import relay

def get_key_from_expr(expr: relay.expr):
    return hash(expr)

def get_prim_type(v):
    if isinstance(v, tvm.tir.expr.IntImm):
        return v.value
    elif isinstance(v, tvm.tir.expr.FloatImm):
        return v.value
    elif isinstance(v, tvm.ir.container.Array):
        return [get_prim_type(i) for i in list(v)]
    elif isinstance(v, tvm.runtime.container.String):
        return str(v)
    else:
        return v