# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.op_package_generator.generator import *
import re

SCALAR_PARAM_DEFAULT_VARS = dict()


# ------------------------------------------------------------------------------
#   Qnn Package Generator Template Helpers
# ------------------------------------------------------------------------------
def is_valid_cpp_identifier(candidate_str):
    # C++ identifiers must begin with a letter or underscore and
    # contain only letters or underscores
    valid_cpp_regex = re.compile('^[a-zA-z_][a-zA-Z0-9_]*$')

    if valid_cpp_regex.search(candidate_str) is None:
        raise NameError("Name: {} is not valid. Name must be usable as"
                        " a valid C++ function identifier".format(candidate_str))


def get_unique_datatypes(operator):
    all_datatypes = list()
    unique_datatypes = list()
    for output in operator.output:
        all_datatypes.extend(output.allowed_data_types)
    for input_ in operator.input:
        all_datatypes.extend(input_.allowed_data_types)

    for data_type in all_datatypes:
        if all_datatypes.count(data_type) == 1:
            unique_datatypes.append(data_type)
    return unique_datatypes


def _template_builder(operator):
    tensor_types = ["typename TensorType"]
    num_of_unique_datatypes = len(get_unique_datatypes(operator))

    # handles the number of tensor types based on number of inputs and outputs
    if num_of_unique_datatypes > len(operator.output) + len(operator.input):
        num_of_unique_datatypes = len(operator.input) + len(operator.output)
    if num_of_unique_datatypes >= 1:
        tensor_types.extend(["typename TensorType{}".format(str(idx)) for idx in
                             range(1, num_of_unique_datatypes + 1)])
    return "template<{}>".format(','.join(tensor_types))


def get_tensor_type_mapping(operator: Operator, data_types: List[QnnDatatype], cur_idx: int,
                            prefix) -> str:
    """
     This maps an operator and its data-types into a variable signature. The idea is that for each unique datatype
     in the operator input and output datatypes, there will be a corresponding TensorType template element.

     See usage in format output

    """
    if any(data_type in get_unique_datatypes(operator) for data_type in data_types):
        cur_idx = cur_idx + 1
        tensor_str = "TensorType{} &{},".format(str(cur_idx), prefix)
    else:
        if cur_idx == 0:
            tensor_str = "TensorType& {},".format(prefix)
        else:
            tensor_str = "TensorType{}& {},".format(str(cur_idx), prefix)
    return tensor_str


def get_hexnn_tensor_sig(operator: Operator,
                         func_tab_size: int,
                         cur_idx: int,
                         *,
                         output_only=False,
                         input_only=False) -> str:
    """
    Formats the input and output names that appear in the wrapper.cpp file. It creates variable signatures for
    all inputs and outputs belonging to an operator.

    :param input_only: Boolean control to return just the output strings if True
    :param output_only: Boolean control to return just the input strings if True
    :param operator: The name of the operator
    :param func_tab_size: A tab size for the function signature to ensure the variable signature is aligned
    :param cur_idx: The current index for the tensor type
    :return: output tensor signatures, input tensor signatures or both
    """
    out_strings = ""
    in_strings = ""

    """
     The idea here is to produce a signature for each output/input in the operator. The resulting function will be 
     templated meaning that signature must take a template type. An example is shown below:

     The first template type is always TensorType. Then subsequently, there will be additional template types for
     each unique (once and only once) datatype in [all_input_datatypes, all_output_datatypes]

     e.x. two input and one output op
     unique_datatypes = [QNN_DATATYPE_UINT_32]

     first output:
     output_0_datatypes = [QNN_DATATYPE_INT_32, QNN_DATATYPE_FLOAT_32]
     will produce TensorType &out since the datatypes for this output are not unique

     first input
     input_0_datatypes = [QNN_DATATYPE_UINT_32, QNN_DATATYPE_FLOAT_32] -> const TensorType1 &in

     second input
     input_1_datatypes = [QNN_DATATYPE_INT_32, QNN_DATATYPE_FLOAT_32] -> const TensorType &in_1
    """

    def get_tensor_string(data_types: List[QnnDatatype],
                          prefix: str,
                          qualifier: str = "",
                          container_type: str = "",
                          container_value_type_qualifier=""):
        tensor_mapping = str(get_tensor_type_mapping(operator,
                                                     data_types, cur_idx,
                                                     prefix)).rstrip()
        if container_type:
            split_tensor_mapping = tensor_mapping.split("&")
            tensor_mapping = "{}<{} {}*>".format(container_type, container_value_type_qualifier,
                                                 split_tensor_mapping[0]) + "&" + \
                             split_tensor_mapping[1]

        return " " * func_tab_size + qualifier + " " + tensor_mapping + '\n'

    for idx, output in enumerate(operator.output):
        out_name = "out" if "out" in output.name else output.name
        if output.repeated:
            raise TypeError("Cannot auto-generate HTP code for variadic operation: {}".format(
                operator.type_name))
            # TODO: This code works but is limited by the variadic op headers in HTP. We can re-enable this code
            # once support is added
            # out_strings = out_strings + get_tensor_string(output.data_types,
            #                                               prefix="{}_first".format(out_name))
            # out_strings = out_strings + get_tensor_string(output.data_types,
            #                                               prefix="{}_rest".format(out_name),
            #                                               container_type="VECTOR",
            #                                               container_value_type_qualifier="const")
        else:
            prefix = "{}_{}".format(out_name, idx) if out_name == "out" else out_name
            out_strings = out_strings + get_tensor_string(output.allowed_data_types, prefix)
    if output_only:
        return out_strings
    for idx, input_ in enumerate(operator.input):
        in_name = "in" if "in" in input_.name else input_.name
        if input_.repeated:
            # TODO: This code works but is limited by the variadic op headers in HTP. We can re-enable this code
            # once support is added
            raise TypeError("Cannot auto-generate HTP code for variadic operation: {}".format(
                operator.type_name))
            # in_strings = in_strings + get_tensor_string(input_.data_types,
            #                                             prefix="{}_first".format(in_name),
            #                                             qualifier="const")
            # in_strings = in_strings + get_tensor_string(input_.data_types,
            #                                             prefix="{}_rest".format(in_name),
            #                                             qualifier="const",
            #                                             container_type="VECTOR",
            #                                             container_value_type_qualifier="const")
        else:
            prefix = "{}_{}".format(in_name, idx) if in_name == "in" else in_name
            in_strings = in_strings + get_tensor_string(input_.allowed_data_types, prefix,
                                                        qualifier="const")
    if input_only:
        return in_strings

    # remove trailing commas
    if not operator.param:
        in_strings = in_strings.rstrip('\n,') + ')'
    return str(out_strings + in_strings)


def get_hexnn_param_sig(operator, func_tab_size):
    """
    Produces a parameter argument signature based on its datatype
    :param operator: The operator instance
    :param func_tab_size: The function signature tab size for far (note the signatures for inputs
    and outputs are computed previously)
    :return: Th string signature for all the params for the operator instance
    """
    param_string = ''

    def get_param_dtype(param):
        dtypes = set()
        dtype = "Tensor"
        for data_type in param.allowed_data_types:
            if data_type in QnnDatatype.get_types("unsigned_quantized"):
                dtypes.add("QuantUint16Tensor")
            elif data_type in QnnDatatype.get_types("signed_quantized"):
                dtypes.add("QuantInt32Tensor")
            elif data_type in QnnDatatype.get_types("integer"):
                dtypes.add("Int32Tensor")
            elif data_type in QnnDatatype.get_types("float"):
                dtypes.add("PlainFloatTensor")
            elif data_type in QnnDatatype.get_types("float_fp16"):
                dtypes.add("F16CroutonTensor")
            else:
                dtypes.add("Tensor")
            if len(dtypes) > 1:
                return dtype

        return dtypes.pop()

    for param in operator.param:
        dtype = get_param_dtype(param)
        param_string += " " * func_tab_size + "const {}& {},".format(dtype, param.name) + '\n'
    return param_string.rstrip('\n,') + ')'


def get_param_order_sig(params: List[TensorInfo], func_tab_size):
    operator_param_order_str = ""
    for param in params:
        param_order_str = "\n" + " " * func_tab_size + "\"{}\",".format(param.name)
        # if the parameter has a default value and is a scalar type
        if param.default_value is not None and param.name in SCALAR_PARAM_DEFAULT_VARS:
            param_order_str = param_order_str + "\n" + " " * func_tab_size + "false,"
            param_order_str = param_order_str + "\n" + " " * func_tab_size + "&{}".format(
                SCALAR_PARAM_DEFAULT_VARS[param.name]) + ","
        else:
            # tensor params, string params and mandatory params all go here
            param_order_str = param_order_str + "\n" + " " * func_tab_size + "true,"
            param_order_str = param_order_str + "\n" + " " * func_tab_size + "nullptr,"
        operator_param_order_str = operator_param_order_str + param_order_str

    return operator_param_order_str.rstrip(',') + ')'


def is_default_scalar_param(param):
    return param.default_value is not None and \
           isinstance(param.default_value, (int, float, bool))


def build_scalar_param(param: TensorInfo):
    data_type = param.allowed_data_types[0]
    scalar_var = "sg_opDefault{}Scalar".format(param.name.title())
    orig_param_str = "static Qnn_Scalar_t {} =".format(scalar_var)
    param_str = orig_param_str + " {{.dataType = Qnn_DataType_t::{}," \
        .format(str(param.allowed_data_types[0]).split(".")[-1])

    if not is_default_scalar_param(param):
        raise TypeError("Parameter {} has no default value".format(param.name))
    if data_type == QnnDatatype.QNN_DATATYPE_FLOAT_32:
        param_str = param_str + "\n" + " " * len(orig_param_str) + ".floatValue = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_UINT_32:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .uint32Value = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_INT_32:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .int32Value = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_UINT_16:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .uint16Value = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_INT_16:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .int16Value = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_INT_8:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .int8Value = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_UINT_8:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .uint8Value = {}}};".format(
            param.default_value)
    elif data_type == QnnDatatype.QNN_DATATYPE_BOOL_8:
        param_str = param_str + "\n" + " " * len(orig_param_str) + " .bool8Value = {}}};".format(
            param.default_value)
    else:
        raise TypeError("Parameter datatype {} is not supported".format(data_type))

    param_var = "sg_opDefault{}".format(param.name.title())
    param_var_decl = "static Qnn_Param_t {} = ".format(param_var)
    len_param_var_decl = len(param_var_decl)
    param_var_decl = param_var_decl + "{.paramType = QNN_PARAMTYPE_SCALAR,\n"
    param_var_decl = param_var_decl + " " * len_param_var_decl + ".scalarParam = {}}};".format(
        scalar_var)

    global SCALAR_PARAM_DEFAULT_VARS
    SCALAR_PARAM_DEFAULT_VARS[param.name] = param_var

    return param_str + "\n" + param_var_decl
