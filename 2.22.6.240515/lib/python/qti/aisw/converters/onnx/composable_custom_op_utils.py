# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import log_warning
import copy
from onnx.helper import make_model, make_graph, make_tensor_value_info
from onnx import shape_inference


class ComposableCustomOpCollection(object):
    """
    This class stores the definition of all the functions present in the model.
    """

    def __init__(self):
        self.functions = dict()

    def parse_functions_from_model(self, model):
        """
        extract all the functions from the model and store them in a dict with domain and name as a key
        :param model: a ONNX ModelProto
        :return:
        """
        functions_key = [(function.name, function.domain) for function in model.functions]
        self.functions = dict(zip(functions_key, model.functions))

    def get_function(self, node):
        """
        find the function definition of a node base on domain and name
        :param node: NodeProto
        :return: return FunctionProto
        """
        return self.functions[(node.op_type, node.domain)]

    def is_composable_op(self, node):
        """
        check whether a node is composable or not
        :param node: NodeProto
        :return: return True if node is composable else false
        """
        if (node.op_type, node.domain) in self.functions.keys():
            return True
        return False

class ComposableCustomOpSrcInfo(object):
    def __init__(self):
        self.input_mapping = dict()
        self.output_mapping = dict()
        self.attributes = dict()
        # this mapping stores the new name of the intermediate outputs.
        self.intermediate_output_mapping = dict()

def update_input_names(node, node_info):
    for index, inp_name in enumerate(node.input):
        if inp_name in node_info.input_mapping.keys():
            node.input[index] = node_info.input_mapping[inp_name]
        elif inp_name in node_info.output_mapping.keys():
            node.input[index] =node_info.output_mapping[inp_name]
        else:
            node.input[index] = node_info.intermediate_output_mapping[inp_name]

def update_output_names(function_node_name, node, node_info):
    for index, out_name in enumerate(node.output):
        if out_name in node_info.output_mapping.keys():
            node.output[index] = node_info.output_mapping[out_name]
        else:
            new_output_name = function_node_name + "_" + out_name
            node_info.intermediate_output_mapping[out_name] = new_output_name
            node.output[index] = new_output_name

def update_attribute_info(node, node_info):
    for index, attr in enumerate(node.attribute):
        # replace the attribute reference with the corresponding attribute value in the function node
        if attr.ref_attr_name:
            new_attr = copy.deepcopy(node_info.attributes[attr.ref_attr_name])
            new_attr.name = attr.name
            node.attribute.remove(attr)
            node.attribute.insert(index, new_attr)

def transform(function_node_name, node, node_info):
    """
    Update the node according to the function node info
    :param function_node_name: name of the actual function node
    :param node: a ONNX NodeProto
    :param mappings: a ComposableCustomOpSrcInfo object
    :return: a NodeProto with updated infos
    """
    # update input infos in the elementary node
    update_input_names(node, node_info)

    # update output infos in the elementary node
    update_output_names(function_node_name, node, node_info)

    # update attribute infos
    update_attribute_info(node, node_info)

    # update name for elementary node
    if node.name:
        node.name = function_node_name + "_" + node.name
    else:
        node.name = node.output[0]

    return node


def expand(node, composable_custom_op_collection):
    """
    Expand the node and update name, inputs names, output names and Attribute of all elementary nodes
    :param node: a NodeProto
    :param composable_custom_op_collection: a ComposableCustomOpCollection object
    :return: list of expanded NodeProtos with updated information
    """

    # Create a copy of  NodeProtos otherwise it will change the actual FunctionProto.
    function = composable_custom_op_collection.get_function(node)
    elementary_nodes = copy.deepcopy(function.node)

    node_info = ComposableCustomOpSrcInfo()
    node_info.input_mapping = dict(zip(function.input, node.input))
    node_info.output_mapping = dict(zip(function.output, node.output))
    node_info.attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))

    final_elementary_nodes = []

    for elem_node in elementary_nodes:

        elem_node = transform(node.name, elem_node, node_info)

        # Function can have reference to some other onnx functions (Nested Functions).
        # first check whether it's a composable custom op or not.
        if composable_custom_op_collection.is_composable_op(elem_node):
            final_elementary_nodes += expand(elem_node, composable_custom_op_collection)
        else:
            final_elementary_nodes.append(elem_node)

    return final_elementary_nodes


def create_model_from_function(function_node, expanded_nodes, composable_custom_op_collection, main_model):
    """
    create a model from the function node using the expansion of the node
    :param function_node: a NodeProto of a function node
    :param expanded_nodes: expansion of the function node according to the defintion
    :param composable_custom_op_collection: a ComposableCustomOpCollection object
    :param main_model: an Onnx ModelProto object
    :return: an ONNX ModelProto object created from the subgraph of the function node
    """
    function = composable_custom_op_collection.get_function(function_node)
    input_shape = dict()
    input_dtype = dict()

    for input_name in function_node.input:
        found = False
        for value_info in main_model.graph.input:
            if value_info.name == input_name:
                input_shape[input_name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
                input_dtype[input_name] = value_info.type.tensor_type.elem_type
                found = True
                break

        if found:
            continue

        for value_info in main_model.graph.value_info:
            if value_info.name == input_name:
                input_shape[input_name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
                input_dtype[input_name] = value_info.type.tensor_type.elem_type
                break

    output_shape = dict()
    output_dtype = dict()
    for output_name in function_node.output:
        found = False
        for value_info in main_model.graph.output:
            if value_info.name == output_name:
                output_shape[output_name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
                output_dtype[output_name] = value_info.type.tensor_type.elem_type
                found = True
                break

        if found:
            continue

        for value_info in main_model.graph.value_info:
            if value_info.name == output_name:
                output_shape[output_name] = [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]
                output_dtype[output_name] = value_info.type.tensor_type.elem_type
                break

    # Subgraph can only be generated if all the output shapes are present in the inferred model.
    # If any output shapes is empty or any output shapes contains a zero value, then return None.
    if not output_shape or [] in output_shape.values() or any([0 in value for value in output_shape.values()]):
        log_warning("Output shapes is empty for composable custom node {}. Shape inference not possible in the "
                    "subgraph.".format(function_node.name))
        return None

    input_tensor_info = []
    for input in function_node.input:
        input_tensor_info.append(make_tensor_value_info(input, input_dtype[input], input_shape[input]))

    output_tensor_info = []
    for output in function_node.output:
        output_tensor_info.append(make_tensor_value_info(output, output_dtype[output], output_shape[output]))

    graph = make_graph(expanded_nodes, 'sub_graph', input_tensor_info, output_tensor_info)

    model = make_model(
        graph, opset_imports=function.opset_import)
    return model
