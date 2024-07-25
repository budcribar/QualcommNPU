# ==============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import re
import numpy as np

from abc import ABCMeta

from qti.aisw.converters.common.converter_ir import translation
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.utils.converter_utils import (
    rename_user_quantization_overrides,
    is_valid_buffer_name,
    log_debug1,
    log_debug3,
    log_warning
)
from qti.aisw.converters.relay.utils import get_key_from_expr
import tvm
from tvm import relay


def validate_const_name(quir_graph, const_name, candidate_const_name):
    if not is_valid_buffer_name(const_name):
        # if tensor name is not valid, replace it by candidate name, and rename encoding in quantization override
        res = candidate_const_name
        log_warning('origin const name {} is not valid, change it to {}'.format(const_name, res))
        rename_user_quantization_overrides(quir_graph, const_name, res)
    elif const_name.startswith('relay_constant'):
        res = candidate_const_name
    else:
        res = const_name
    return res


class RelayTranslationBase(translation.ConversionTranslationBase, metaclass=ABCMeta):

    def __init__(self):
        super(RelayTranslationBase, self).__init__()
        self.extract_parameters = None

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        return {}

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        raise NotImplementedError("translate_op for {} not implemented ".format(str(self.__class__.__name__)))

    # Thin wrapper added so that Op Translation can override if needed
    def extract_input_names(self,
                            relay_expr: relay.expr.Call,
                            converter_context,
                            **kwargs):
        return converter_context.get_input_names(relay_expr)

    # Thin wrapper added so that Op Translation can override if needed
    def extract_output_names(self,
                             relay_expr: relay.expr.Call,
                             converter_context,
                             **kwargs):
        num_outputs = kwargs.get("num_outputs", 1)
        return converter_context.get_output_names(relay_expr, num_outputs)

    def populate_quantization_params(self, relay_expr: relay.expr, converter_context, quir_graph, output_names: list, is_param):
        key = get_key_from_expr(relay_expr)
        encodings = converter_context.expr_to_source_info_dict[key].get_encodings()
        if encodings:
            # TODO: use dict to get encoding instead of using list
            for encoding, output_name in zip(encodings, output_names):
                if isinstance(encoding, list):
                    # per channel quantization
                    quir_graph.set_overridden_encoding(output_name, encoding, is_param=is_param)
                else:
                    quir_graph.set_overridden_encoding(output_name, [encoding], is_param=is_param)

    def add_op(self,
               relay_expr: relay.expr.Call,
               quir_graph: IROpGraph,
               **kwargs):
        converter_context = kwargs.get("converter_context")
        relay_params = kwargs.get("relay_params")

        attr_dict = self.extract_attributes(relay_expr, relay_params)

        input_names = self.extract_input_names(relay_expr,
                                               converter_context=converter_context)

        ir_op = self.translate_op(relay_expr,
                                  relay_params,
                                  converter_context,
                                  quir_graph,
                                  attr_dict,
                                  input_names)

        num_outputs = ir_op.num_outputs
        output_names = self.extract_output_names(relay_expr,
                                                 converter_context=converter_context,
                                                 num_outputs=num_outputs)

        log_debug1("Op {} Type {} inputs {}", ir_op.name, ir_op.type, input_names)
        log_debug1("Op {} Type {} outputs {}", ir_op.name, ir_op.type, output_names[:num_outputs])

        self.populate_quantization_params(relay_expr, converter_context, quir_graph, output_names[:num_outputs], is_param=False)
        ir_node = converter_context.add_op_to_graph(relay_expr, ir_op, input_names, output_names[:num_outputs])
        quir_graph.add_src_op_info(ir_node.op.name, input_names, output_names[:num_outputs])
        return ir_node


class RelayQuantization(object):

    DefaultBw = 8

    QuantTypes = {
        'int8'    : np.int8,
        'uint8'   : np.uint8,
        'int16'   : np.int16,
        'uint16'  : np.uint16}

    @staticmethod
    def get_quantization_params(op_name: str,
                                input_names: list,
                                relay_params: dict,
                                attrs: dict):
        if len(input_names) < 2:
            raise ValueError("Missing quantization params for op {}. Should contain [input, scale, offset(optional)] but got {}"
                             .format(op_name, input_names))

        bw = attrs['bw'] if 'bw' in attrs else 0
        axis = attrs['axis'] if 'axis' in attrs else -1
        offset = int(0)
        if "dtype" not in attrs:
            raise ValueError("Missing quantization dtype attrs for op {}".format(op_name))
        if attrs['dtype'] not in RelayQuantization.QuantTypes:
            raise ValueError("Unsupported quantize dtype {} for op {}, only support int8/uint8/int16/uint16"
                             .format(attrs['dtype'], op_name))
        dtype = RelayQuantization.QuantTypes[attrs['dtype']]

        # Extract the bitwidth from the dtype if necessary
        if bw == 0:
            m = re.search(r'\d+$', attrs['dtype'])
            bw = int(m.group()) if m else RelayQuantization.DefaultBw

        log_debug3('Processing quantization inputs {} for op {}'.format(input_names, op_name))

        # Extract the scale
        scale = relay_params[input_names[1]]
        log_debug3('Found quantization scale {} of type {}'.format(scale, type(scale)))
        if isinstance(scale, tvm.runtime.ndarray.NDArray) or isinstance(scale, tvm.runtime.NDArray):
            scale = scale.asnumpy().astype(np.float32)
        else:
            raise ValueError('Unsupported quantization scale {} for input {}'.format(scale, input_names[1]))


        if len(input_names) > 2:
            offset = relay_params[input_names[2]]
            log_debug3('Found quantization offset {} of type {}'.format(offset, type(offset)))
            if isinstance(offset, tvm.runtime.ndarray.NDArray) or isinstance(offset, tvm.runtime.NDArray):
                offset = offset.asnumpy().astype(np.int32)

        q_params = {'name': op_name,
                    'scale' : scale,
                    'offset': offset,
                    "min": np.float32(np.iinfo(dtype).min - offset) * scale,
                    "max": np.float32(np.iinfo(dtype).max - offset) * scale,
                    'dtype':dtype,
                    'bw': bw,
                    'axis': axis}
        return q_params
