# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from collections import defaultdict

from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory as BaseFactory

# ------------------------------------------------------------------------------
#   Tflite helpers for relay
# ------------------------------------------------------------------------------


class TfliteCustomOpFactory(BaseFactory):
    def __init__(self):
        super(TfliteCustomOpFactory, self).__init__()

    @classmethod
    def create_ops_from_operator(cls, operator, converter_type, model=None, **kwargs):
        """
        Creates multiples ops from a single Operator object, using a list of src_ops
        in the model that match the operator spec.
        :param operator: The operator to be used
        :param model: The framework model
        :param converter_type: The given converter type
        :return:
        """

        nodes = cls.get_src_ops(str(operator.type_name).lower(), model, converter_type, **kwargs)
        resolved_ops = []
        custom_op_count_dict = defaultdict(lambda: -1)
        for node in nodes:
            # Record the number of occurrences of custom op
            # to later set the output name consistent with the span mechanism, and
            # then create CustomTfliteOp object
            custom_op_count_dict[operator.type_name] += 1
            resolved_ops.append(cls.create_op_from_operator(operator, node, model,
                                                            converter_type,
                                                            custom_op_count_dict=custom_op_count_dict,
                                                            **kwargs))
        return resolved_ops


class TensorInfo(object):
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

    def get_tensor_info(self, tensors_idx_list, subgraph):
        """Get tensor wrapper list from given TFLite tensor index list"""

        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorInfo(tensor_idx, 0, 0))
                continue
            tensor = subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)
            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == int(zero_point[0])):
                                raise Exception(
                                    "TFLite per-axis quantization restricts all zero points to be"
                                    + " the same, but a different value is observed"
                                )
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            "Quantized type {} (scale) and  {} (zero point) not supported".format(
                                type(tflite_scale), type(tflite_zero_point)
                            )
                        )
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(
                        "Quantized type {} not supported".format(type(tflite_scale))
                    )

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = np.array(scale, "float32")
                    qnn_params["zero_point"] = np.array(zero_point, "int32")
            return_list.append(TensorInfo(tensor_idx, tensor, buffer, qnn_params))
        return return_list


def get_custom_op_code(model, op):
    """Get TFLite ops string representation"""
    try:
        from tflite.BuiltinOperator import BuiltinOperator
    except ImportError:
        raise ImportError("The tflite package must be installed")
    builtin_op_code = build_str_map(BuiltinOperator())
    op_code_list_idx = op.OpcodeIndex()
    op_c = model.OperatorCodes(op_code_list_idx)
    try:
        opc = max(op_c.DeprecatedBuiltinCode(), op_c.BuiltinCode())
    except AttributeError:
        opc = op_c.BuiltinCode()
    op_code_id = opc
    try:
        op_code_str = builtin_op_code[op_code_id]
    except KeyError:
        raise NotImplementedError(
            "TFLite operator with code "
            + str(op_code_id)
            + " is not supported by this version of the fbs schema."
        )
    if op_code_id == BuiltinOperator.CUSTOM:
        custom_op_code_str = model.OperatorCodes(op_code_list_idx).CustomCode()
        if custom_op_code_str == b"TFLite_Detection_PostProcess":
            return "DETECTION_POSTPROCESS"
        if custom_op_code_str == b"FlexFakeQuantWithMinMaxVarsPerChannel":
            return "FlexFakeQuantWithMinMaxVarsPerChannel"
        if custom_op_code_str == b"FlexMul":
            return "FlexMul"
        if custom_op_code_str not in builtin_op_code:
            return "CUSTOM"
    return op_code_str

def build_str_map(obj):
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret

