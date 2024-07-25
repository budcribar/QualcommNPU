# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.validation_utils import *
from qti.aisw.converters.common.converter_ir.op_adapter import CustomOp

import numpy as np
import traceback


# ------------------------------------------------------------------------------
#   Custom Op helpers
# ------------------------------------------------------------------------------


def populate_custom_op_collection(model,
                                  converter_type='onnx',
                                  **kwargs):
    if "converter_op_package_lib" in kwargs:
        kwargs["converter_op_package_libs"] = kwargs["converter_op_package_lib"].split(',')[::-1]
        for lib_path in kwargs["converter_op_package_libs"]:
            check_filename_encoding(lib_path)
            io_utils.check_validity(lib_path, is_path=True, must_exist=True)

    # Create a custom op collection based on configs provided by user
    custom_op_config_paths = kwargs.get('custom_op_config_paths', None)
    custom_op_factory = kwargs.get('custom_op_factory', None)
    if custom_op_config_paths is not None and custom_op_factory is not None:
        for config_path in custom_op_config_paths:
            try:
                custom_op_factory.parse_config(config_path,
                                               model=model,
                                               converter_type=converter_type,
                                               **kwargs)
            except Exception as e:
                if not is_log_level_debug():
                    traceback.print_exc()
                log_error("Error populating custom ops from: {}\n {}".format(config_path,
                                                                             str(e)))
                sys.exit(-1)


def create_custom_op(custom_frontend_op, **kwargs):
    """
    This function creates a custom op used to call shape inference function later.
    :param custom_frontend_op: the custom op stored in the op factory
    :return: a custom ir op
    """
    from qti.aisw.converters.qnn_backend.custom_ops.op_factory import OpFactory
    from qti.aisw.converters.qnn_backend.custom_ops.core import get_internal_dtype

    op_name, op_type = kwargs.get('op_name', None), kwargs.get('op_type', None)
    graph = kwargs.get('graph', None)
    if not op_name:
        op_name = custom_frontend_op.op_name
    if not op_type:
        op_type = custom_frontend_op.op_type

    package_name = OpFactory.get_package_name(op_type)
    converter_op_package_lib = None
    if 'converter_op_package_libs' in OpFactory.package_resolver:
        converter_op_package_lib = OpFactory.package_resolver['converter_op_package_libs'][package_name]

    for name, custom_param in custom_frontend_op.params.items():
        param = custom_param.param
        if param.data is None:
            if not param.static:
                raise ValueError(
                    code_to_message.get_error_message("ERROR_CUSTOM_OP_PARAM_NO_DATA")
                    (name, op_type))
            elif param.default_value:
                param.data = param.default_value
                param.data_type = get_internal_dtype(param.data, param)
                param.dimensions = np.asarray(param.data).shape
                param.rank = len(param.data)
            else:
                raise LookupError(code_to_message.get_error_message("ERROR_CANNOT"
                                                                    "_INGEST_STATIC_INPUT")
                                    (str(name)))

    inputs, outputs, scalar_params, tensor_params = custom_frontend_op.as_dict(graph)
    # adds input_names to custom op to access the updated inputs in extract_input_names
    # after the buffers for static inputs are added to the graph since updated input names
    # cannot be accessed from src_op and custom_op.inputs
    custom_frontend_op.input_names = list(inputs.keys())

    ir_op = CustomOp(op_name,
                     package_name,
                     op_type,
                     inputs,
                     outputs,
                     custom_frontend_op.axis_orders,
                     custom_frontend_op.output_dims,
                     scalar_params,
                     tensor_params,
                     converter_op_package_lib=converter_op_package_lib)
    return ir_op


def create_qti_aisw_op(op_type, **kwargs):
    """
    based on the op_type returns the corresponding Ir Op
    :param op_type:
    :param kwargs:
    :return:
    """
    from qti.aisw.converters.common.converter_ir.op_adapter import OpAdapterMap
    supported_types = OpAdapterMap.translations.keys()
    # sub-op types need to be updated
    neuron_types = []
    if 'Neuron' in supported_types:
        neuron_types.append('ReluMinMax')
    op_name = kwargs["node_source_name"]

    if op_type in supported_types:
        ir_op = OpAdapterMap.translations[op_type](op_name, **kwargs)
    elif op_type in neuron_types:
        ir_op = OpAdapterMap.translations['Neuron'](op_name, op_type, **kwargs)
    else:
        ValueError(f"Not supported CCO-QNNIR op:{op_type}", op_type)

    return ir_op
