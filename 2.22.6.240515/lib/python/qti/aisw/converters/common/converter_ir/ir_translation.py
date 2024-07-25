# ==============================================================================
#
#  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import numpy as np
from qti.aisw.converters.backend import ir_graph
from qti.aisw.converters.common.utils.converter_utils import log_error

simple_types = {'int': ir_graph.QNN_DATATYPE_INT_64,
                'float': ir_graph.QNN_DATATYPE_FLOAT_32,
                'bool': ir_graph.QNN_DATATYPE_BOOL_8,
                'str': ir_graph.QNN_DATATYPE_UNDEFINED}
complex_types = {'int': ir_graph.QNN_DATATYPE_INT_64,
                 'float': ir_graph.QNN_DATATYPE_FLOAT_32,
                 'bool': ir_graph.QNN_DATATYPE_BOOL_8,
                 'str': ir_graph.QNN_DATATYPE_UNDEFINED,
                 'int8': ir_graph.QNN_DATATYPE_INT_8,
                 'int16': ir_graph.QNN_DATATYPE_INT_16,
                 'int32': ir_graph.QNN_DATATYPE_INT_32,
                 'int64': ir_graph.QNN_DATATYPE_INT_64,
                 'uint8': ir_graph.QNN_DATATYPE_UINT_8,
                 'uint16': ir_graph.QNN_DATATYPE_UINT_16,
                 'uint32': ir_graph.QNN_DATATYPE_UINT_32,
                 'uint64': ir_graph.QNN_DATATYPE_UINT_64,
                 'float16': ir_graph.QNN_DATATYPE_FLOAT_16,
                 'float32': ir_graph.QNN_DATATYPE_FLOAT_32}


class IrTranslation(object):
    def __init__(self):
        self.c_graph = ir_graph.PyIrGraph("model") # Translated graph

    def get_complex_type(self, attr):
        tmp = attr
        if isinstance(attr, list):
            if attr:
                tmp = np.array(attr)
            else:
                return None, None

        if isinstance(tmp, np.ndarray):
            if tmp.size:
                if tmp.dtype == np.float64:
                    tmp = tmp.astype(np.float32)
                return tmp, complex_types[str(tmp.dtype)]
            else:
                return None, None
        else:
            #print('Unsupported type: ', type(attr))
            return None,None

    def get_dict(self, value):
        a = {}
        t = type(value).__name__
        if t in simple_types:
            a['atype'] = ir_graph.QNN_PARAMTYPE_SCALAR
            a['dtype'] = simple_types[type(value).__name__]
            a['data'] = value
        else:
            data, dtype = self.get_complex_type(value)
            if data is None:
                return {}
            a['atype'] = ir_graph.QNN_PARAMTYPE_TENSOR
            a['dtype'] = dtype
            a['data'] = data
        return a

    def translate(self, graph):
        self.c_graph.set_tensor_overrides(graph.quantization_params)
        for node in graph.list_nodes():
            op = node.op
            ignore = ['TRANSLATION_KEY', 'attrs', 'name', 'type', 'data_axis_formats', 'output_dims',
                      'input_tensors', 'output_tensors']
            members = [attr for attr in dir(op) if not callable(getattr(op, attr)) and not attr.startswith("__")]
            member_list = [m for m in members if m not in ignore]
            tmp = {}
            attr_dict = op.list_params()
            for kv in attr_dict.items():
                if kv[0] in ignore:
                    continue
                a = self.get_dict(kv[1])
                if not a:
                    continue
                tmp[kv[0]] = a
            for m in member_list:
                a = self.get_dict(getattr(op, m))
                if not a:
                    continue
                tmp[m] = a
            try:
                # For qnn quantization add these optional outputs
                if op.type == "Lstm" and len(node.output_names) == 3:
                    temp_outputs = ["_input_gate", "_forget_gate", "_cell_gate","_output_gate","_hidden_state"]
                    for o in temp_outputs:
                        node.output_names.append(op.name + o)
                self.c_graph.add_op(op.name, op.type, tmp, node.input_names, node.output_names)
            except Exception as e:
                log_error("Failed translating node {} for quantization.".format(op.name))
                raise e
        return self.c_graph

    def update_params(self, graph, c_graph, quantized_only=True):
        for node in graph.list_nodes():
            op = node.op
            tensors = c_graph.get_static_tensors(op.name, quantized_only)
            for kv in tensors.items():
                t = kv[1]
                if hasattr(op, t.name()):
                    setattr(op, t.name(), t.data())
                elif t.name() in op.attrs:
                    op.attrs[t.name()] = t.data()
                else:
                    raise ValueError("Can't find op: ", op.name, " attribute: ", t.name())


