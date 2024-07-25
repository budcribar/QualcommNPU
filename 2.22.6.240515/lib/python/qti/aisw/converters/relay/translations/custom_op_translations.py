# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


from qti.aisw.converters.common.converter_ir.op_adapter import CustomOp
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.custom_ops.utils.custom_op_helpers import create_custom_op, create_qti_aisw_op
from qti.aisw.converters.common.utils.converter_utils import converter_type
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import OpFactory

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay

import re


# ------------------------------------------------------------------------------
#   Custom Op
# ------------------------------------------------------------------------------
class RelayCustomOpTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCustomOpTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, CustomOp.TRANSLATION_KEY,
                                                CustomOp.LEGACY_TRANSLATION_KEY)
        span = relay_expr.span
        if isinstance(relay_expr.span, tvm.relay.SequentialSpan):
            # for activation we use last one since the order of spans in SequentialSpan is
            # topological order
            span = relay_expr.span.spans[-1]
        # For Pytorch,
        #   The type name of Pytorch source op contains "::".
        #   Discard it along with the namespace because "::" has a different meaning in C++.
        # Other frontends are not affected because "::" is not present in the op type.
        op_type = span.op_type.split('::')[-1]
        custom_frontend_op = OpFactory.op_collection.get_first_of(op_type)

        ir_op = create_custom_op(custom_frontend_op, op_name=op_name, op_type=op_type,
                                 graph=quir_graph)

        return ir_op


RelayTranslations.register_translation(RelayCustomOpTranslation(),
                                       converter_type('custom', 'relay'))


# ------------------------------------------------------------------------------
#   QTI AISW Op
# ------------------------------------------------------------------------------
class RelayAISWOpTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayAISWOpTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, CustomOp.TRANSLATION_KEY)
        attr_dict['node_source_name'] = op_name
        op_type = ''.join(re.findall("\w", relay_expr.span.op_type))
        op_type = re.search(r'qti_aisw([A-Za-z]+)', op_type).group(1)

        ir_op = create_qti_aisw_op(op_type, **attr_dict)

        return ir_op

RelayTranslations.register_translation(RelayAISWOpTranslation(),
                                       converter_type('qti_aisw', 'relay'))
