# ==============================================================================
#
#  Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from .onnx_translations import *
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import OpFactory
from qti.aisw.converters.qnn_backend.custom_ops.core import get_internal_dtype
from qti.aisw.converters.qnn_backend.custom_ops.core import *
import numpy as np


# ------------------------------------------------------------------------------
#   Custom Op
# ------------------------------------------------------------------------------
class OnnxCustomOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.custom_op = None

    def extract_input_names(self, src_op, converter_context):
        return self.custom_op.input_names

    def extract_output_names(self, src_op, converter_context):
        return [str(output.name) for output in self.custom_op.outputs]

    def extract_parameters(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph

        dynamic = kwargs.get("dynamic", False)
        # If dynamic flag is set to True, then create a new custom onnx op for this source op, otherwise directly retrieve
        # the custom onnx op from the custom op collection
        if dynamic:
            model = kwargs['model']
            converter_type = 'onnx'
            # find the corresponding operator which is defined in the XML config for this op type
            operator = None
            for op in OpFactory.custom_opdefs:
                if op.type_name == src_op.op_type:
                    operator = op
            custom_op = OpFactory.create_op_from_operator(operator, src_op, model, converter_type)
        else:
            custom_op = OpFactory.op_collection.get_first_of(src_op.op_type)

        package_name = OpFactory.get_package_name(custom_op.op_type)
        self.custom_op = custom_op

        converter_op_package_lib = None
        if 'converter_op_package_libs' in OpFactory.package_resolver:
            converter_op_package_lib = OpFactory.package_resolver['converter_op_package_libs'][package_name]

        for name, custom_param in custom_op.params.items():
            param = custom_param.param
            if param.data is None:
                if not param.static:
                    raise ValueError(
                        code_to_message.get_error_message("ERROR_CUSTOM_OP_PARAM_NO_DATA")
                        (name, custom_op.op_type))
                elif converter_context.weights.has(name):
                    param.data = np.asarray(converter_context.weights.weight_map[str(name)].weights)
                    param.data_type = get_internal_dtype(param.data, param)
                    param.dimensions = param.data.shape
                    param.rank = len(param.data.shape)
                    converter_context.weights.weight_map[str(name)].consumed = True
                elif param.default_value:
                    param.data = param.default_value
                    param.data_type = get_internal_dtype(param.data, param)
                    param.dimensions = np.asarray(param.data).shape
                    param.rank = len(param.data)
                else:
                    raise LookupError(code_to_message.get_error_message("ERROR_CANNOT"
                                                                        "_INGEST_STATIC_INPUT")
                                      (str(name)))

        inputs, outputs, scalar_params, tensor_params = custom_op.as_dict(graph)

        tensor_param_names = []
        for _, tensor_param in tensor_params.items():
            if tensor_param['static']:
                # if tensor is present in self.params and not in self.param_info
                # it is the input static tensor which was added to params
                if tensor_param['name'] in custom_op.params and (tensor_param['name'] not in [param['name'] for param in custom_op.param_info]):
                    # creates input tensor info
                    input_tensor = TensorInfo()
                    input_tensor.name = tensor_param['name']
                    input_tensor.allowed_data_types = tensor_param['allowed_data_types']
                    # input_tensor.allowed_values = tensor_param['allowed_values']
                    # input_tensor.shape = tensor_param['shape']
                    input_tensor.rank = tensor_param['rank']
                    input_tensor.default_value = tensor_param['default_value']
                    input_tensor.layout = tensor_param['layout']
                    input_tensor.repeated = tensor_param['repeated']
                    input_tensor.dimensions = tensor_param['dimensions']
                    input_tensor.static = tensor_param['static']
                    input_tensor.data = tensor_param['data']
                    input_op_name = custom_op.name + '_' + tensor_param['name']
                    inputs[input_op_name] = input_tensor.as_dict()
                    tensor_param_names.append(tensor_param['name'])
                    # adds the constant op for static inputs and adds the buffer to graph
                    input_op = op_adapter.ConstantOp(input_op_name, tensor=input_tensor.data)
                    axis_format = AxisTracker.AxisFormat.OIHW
                    if input_tensor.rank == 1:
                        axis_format = AxisTracker.AxisFormat.ANY
                    graph.add(input_op, [], [input_op_name], axis_formats=[axis_format])

        # removes the static inputs from tensors params which were added earlier
        for param in tensor_param_names:
            tensor_params.pop(param)

        # adds input_names to custom op to access the updated inputs in extract_input_names
        # after the buffers for static inputs are added to the graph since updated input names
        # cannot be accessed from src_op and custom_op.inputs and custom_op.input_tensor_infos
        self.custom_op.input_names = list(inputs.keys())
        return op_adapter.CustomOp(name=src_op.name,
                                   package_name=package_name,
                                   custom_type=src_op.op_type,
                                   axis_orders=custom_op.axis_orders,
                                   inputs=inputs,
                                   outputs=outputs,
                                   output_dims=custom_op.output_dims,
                                   tensor_params=tensor_params,
                                   scalar_params=scalar_params,
                                   converter_op_package_lib=converter_op_package_lib)

    def add_op(self, src_op, context, **kwargs):
        def add_static_tensor_as_constant_op(node_name):
            if not graph.has_node(node_name):
                constant_op = self.fetch_constant_op(node_name, context)
                graph.add(
                    op=constant_op,
                    input_names=[],
                    output_names=[node_name])

        def is_static_input(input_name):
            return context.weights.has(input_name)

        def add_static_inputs_to_graph(input_names):
            static_input_names = [name for name in input_names if is_static_input(name)]
            for static_input_name in static_input_names:
                add_static_tensor_as_constant_op(static_input_name)

        graph = context.ir_graph
        op = self.extract_parameters(src_op, context, **kwargs)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        add_static_inputs_to_graph(input_names)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

OnnxTranslations.register_translation(OnnxCustomOpTranslation(),
                                      converter_type('custom', 'onnx'),
                                      op_adapter.CustomOp.TRANSLATION_KEY)
