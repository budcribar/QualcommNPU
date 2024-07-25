# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import log_warning
from qti.aisw.converters.qnn_backend.custom_ops.core import BackendCustomOp as CustomOp
from qti.aisw.converters.qnn_backend.custom_ops.core import Param, ScalarParam, TensorParam, \
    StringParam, ParamTypes, convert_to_backend_type_from_numpy
from qti.aisw.converters.pytorch.pytorch_to_ir import get_valid_type_name

from qti.tvm.relay.frontend import pytorch as relay_pytorch


class CustomPytorchOp(CustomOp):
    """
    A subclass of the CustomOp interface which implements framework specific methods defined in
    CustomOp. Calling this class requires that an pytorch module can be imported. Additionally,
    the parameters must be extractable from the op. See CustomOp for all methods that will be
    called when a CustomPytorchOp is instantiated
    """

    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model,
                 node_attr_dict, node_source_name, custom_op_count_dict):
        self.model = model
        self.src_op = src_op
        self.node_attr_dict = node_attr_dict
        self.param_info = param_info
        input_tensors = input_tensor_info
        self.custom_op_idx = custom_op_count_dict[f'{node_source_name}|{src_op.kind()}']
        # The type name of Pytorch source op contains "::".
        # Discard it along with the namespace because "::" has a different meaning in C++.
        self.op_type = get_valid_type_name(src_op.kind())
        output_tensors = self.set_output_dims(src_op, output_tensor_info, model)
        # The output name should be the same as the output name in span.
        for i in range(len(output_tensors)):
            # node source name, op type, number of this node, the index of output
            output_tensors[i].name = f"{node_source_name}." + "_".join([str(self.op_type),
                                                                        str(self.custom_op_idx),
                                                                        str(i)])
        self.op_name = f"{node_source_name}." + "_".join([str(self.op_type),
                                                          str(self.custom_op_idx)])
        super(CustomPytorchOp, self).__init__(op_type=str(self.op_type), src_op=src_op,
                                              name=self.op_name, input_tensors=input_tensors,
                                              output_tensors=output_tensors, param_info=param_info)

    def extract_attrs(self, src_op, param_infos):
        """
        This method extracts attributes from the provided pytorch src_op, using a list of param_infos
        that have been obtained from the operator spec.

        :param src_op: The pytorch src_op
        :param param_infos: The list of parameter information which should be a list of
        CustomTensorInfo objects.
        :return: a dictionary of attributes, where the key is the attribute name and the value
        is a CustomParam object
        """
        attrs = dict()
        param = None

        def is_iterable(attr_value):
            try:
                iter(attr_value)
            except TypeError:
                return False
            return True

        inputs = [self.node_attr_dict[name] for name in
                  [inp.debugName() for inp in src_op.inputs()]]
        # If "prim::GetAttr" appears in the value, it means that this element is an input and
        # not an attribute, and the input will always appear at the beginning of the list.
        # So, remove it from the list.
        # E.g., inputs = [['prim::GetAttr', '1'], 3, [1, 1], False, None]
        #   -> The list is full of actual values instead of the debug name.
        #   -> The first one containing 'prim::GetAttr' indicates that it is an input to this op
        #      rather than an attr.
        src_attrs = [x for x in inputs if str(x).find("prim::GetAttr") == -1]

        for param_info in param_infos:
            # The origin name of the attribute is not stored after running "torch._C._jit_pass_inline"
            # on the Pytorch script model.
            # The order of attributes we used here is directly the order of param_info, so user has to
            # make sure that the order of parameters in param_info is the same as the order of parameters
            # in the official pytorch op definition.
            name = param_info.name
            if src_attrs:
                attr_value = src_attrs.pop(0)
                while attr_value is None:
                    attr_value = src_attrs.pop(0)
            elif param_info.static:
                attr_value = []
            elif param_info.default_value is not None:
                attr_value = param_info.default_value
            else:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(
                    param_info.name, str(src_op.kind())))

            if not is_iterable(attr_value):
                if isinstance(attr_value, (int, float, bool)):
                    param = Param(name, ParamTypes.SCALAR,
                                  ScalarParam(attr_value))
            else:
                if isinstance(attr_value, (str, bytes)):
                    if isinstance(attr_value, bytes):
                        # assuming unicode or bytes and utf-8 encoding
                        attr_value = attr_value.decode('utf-8') + '\0'
                    param = Param(name, ParamTypes.STRING, StringParam(attr_value))
                else:
                    param = Param(name, ParamTypes.TENSOR,
                                  TensorParam(attr_value, param_info))
            attrs[name] = param

        return attrs

    def infer_output_shapes(self, node, model=None, **kwargs):
        """
         This method infers the shape of a PyTorch Node's output tensors using the node itself,
         a user provided model containing the node.

        :param node: The PyTorch Node object
        :param model: A required field which should be an PyTorch TorchScripted object
        :return: a list of lists which contains output dimensions for each output tensor
        in the PyTorch Node.
        """
        output_dims = []
        log_warning("PyTorch shape inference failed. Output shapes will be inferred using"
                    " shape inference function from custom op package library if provided.")
        return output_dims

    def set_tensor_data_types(self, node):
        for output, node_output in zip(self.outputs, node.outputs()):
            typ = node_output.type()
            dtype = relay_pytorch._get_pytorch_value_type(typ)
            output.data_type = convert_to_backend_type_from_numpy(dtype)

        for input_, node_input in zip(self.inputs, node.inputs()):
            typ = node_input.type()
            dtype = relay_pytorch._get_pytorch_value_type(typ)
            input_.data_type = convert_to_backend_type_from_numpy(dtype)

    def validate(self, *args, **kwargs):
        self.validate_params(self.src_op, self.param_info)

    def validate_params(self, src_op, param_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec.
        Note that unlike tensors, params must be named in the config spec.
        If the param is not present in the op, a KeyError is raised.
        Likewise, if a param not provided in the config spec is included, a warning is raised.
        :param src_op: The PyTorch op containing the params
        :param param_info: The list of param information as defined in the config spec.
        :raises: a KeyError if the param is missing or an param is present in the op.
        """
        inputs = [self.node_attr_dict[name] for name in
                  [inp.debugName() for inp in src_op.inputs()]]
        # If "prim::GetAttr" appears in the value, it means that this element is an input and
        # not an attribute, and the input will always appear at the beginning of the list.
        # So, remove it from the list.
        src_attrs = [x for x in inputs if str(x).find("prim::GetAttr") == -1]

        # attrs may contain "None"
        # Record the index whose value is None
        useless_idx = [i for i in range(len(src_attrs)) if src_attrs[i] is None]

        # The origin name of the attribute is not stored after running "torch._C._jit_pass_inline"
        # on the PyTorch script model.
        # So, we compare the length of the param_info in the config and the length of attributes in the
        # source op here instead of checking the existence of the parameter directly using its name.
        # Case 1: If the param provided in the config spec is not present in the op, a KeyError is raised.
        #       - Check the "static" and "default value" fields to get the missing attribute
        if len(param_info) > len(src_attrs) - len(useless_idx):
            for param in param_info:
                if not param.static and param.default_value is None:
                    raise KeyError(
                        code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param.name,
                                                                                     str(src_op.kind())))
        # Case 2: If a param not provided in the config spec is included, a warning is raised.
        elif len(param_info) < len(src_attrs) - len(useless_idx):
            num_extra_attrs = len(src_attrs) - len(useless_idx) - len(param_info)
            log_warning("{} attribute(s) was found in the op: {} but has not been defined in "
                        "the op config. The attribute(s) will be ignored!",
                        num_extra_attrs, str(src_op.kind()))

