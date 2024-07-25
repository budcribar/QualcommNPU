# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from collections import defaultdict
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
from qti.aisw.converters.pytorch.pytorch_to_ir import get_valid_type_name
from qti.tvm.relay.frontend import pytorch as relay_pytorch


# ------------------------------------------------------------------------------
#   PyTorch helpers for relay
# ------------------------------------------------------------------------------


class PytorchCustomOpFactory(QnnCustomOpFactory):
    def __init__(self):
        super(PytorchCustomOpFactory, self).__init__()

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

        input_model_path = kwargs.get('input_model_path', None)
        use_parser_friendly_name = kwargs.get('use_parser_friendly_name', False)
        custom_op_count_dict = defaultdict(lambda: -1)

        model, op_converter = cls.preprocess_pytorch_model_with_relay(model,
                                                                      input_model_path,
                                                                      use_parser_friendly_name)

        nodes, node_attr_dict = cls.get_src_ops(str(operator.type_name).lower(), model,
                                                converter_type, **kwargs)
        resolved_ops = []
        for node in nodes:
            # Record the number of occurrences of custom op and the node source name
            # to later set the output name consistent with the span mechanism, and
            # then create CustomPytorchOp object
            custom_op_count_dict[node.kind()] += 1
            node_source_name, op_converter = cls.get_pytorch_node_source_name(node, op_converter)
            custom_op_count_dict[f'{node_source_name}|{node.kind()}'] += 1
            resolved_ops.append(cls.create_op_from_operator(operator, node, model,
                                                            converter_type,
                                                            node_source_name=node_source_name,
                                                            node_attr_dict=node_attr_dict,
                                                            custom_op_count_dict=custom_op_count_dict,
                                                            **kwargs))
        return resolved_ops

    @staticmethod
    def get_pytorch_node_source_name(src_op, op_converter):
        """
        :param src_op: PyTorch source op
        :param op_converter: PyTorchOpConverter, a helper class for holding PyTorch op
                             converters in tvm
        :return source node name for Pytorch op
        """
        node_source_name = op_converter._get_op_name_for_span(src_op)
        # Convert the type name to a valid name as a cpp identifier
        node_source_name = get_valid_type_name(node_source_name)
        return node_source_name, op_converter

    @staticmethod
    def preprocess_pytorch_model_with_relay(model, input_model_path, use_parser_friendly_name):
        """
        This function pre-process the PyTorch source model using the method in tvm relay.
        :param model: TorchScripted PyTorch model
        :param input_model_path: path to torchscript
        :param use_parser_friendly_name:
            When True, replace '.' with `_' in a original parameter name.
            The Relay text parser treats a variable name followed by a period as a tuple
            element access, so a variable name like "dense.weight" cannot be parsed correctly.
            Use this option when you want to run the AnnotateSpans pass on the imported module.
        :return: PyTorch model, PyTorchOpConverter
        """
        # Call _run_jit_passes as used in TVM
        enable_lower_all_tuples = True
        # Check if lower_all_tuples pass can be enabled
        graph_inputs = list(model.graph.inputs())
        for inp in graph_inputs:
            if inp.type().kind() == "TupleType" or inp.type().kind() == "ListType":
                enable_lower_all_tuples = False
                break
        # The inline pass is necessary to unwrap prim::CallMethod
        relay_pytorch._run_jit_passes(model.graph, enable_lower_all_tuples)

        # Rename the debug name and set the source map following the steps we do in TVM
        # to record the node name further
        #  - Example of node name -> aten::convolution_<count>
        op_converter = relay_pytorch.PyTorchOpConverter(None, 'float32', use_parser_friendly_name)
        # rename _C.Graph here for constructing meaningful source name of graph nodes
        # source_map is the map between node and source node name
        #  - get source name of operator and rename all of its outputs
        #  - e.g. node.kind(): aten::adaptive_max_pool2d
        #  - node_src_name -> aten::adaptive_max_pool2d_x
        #  - output_1 -> aten::adaptive_max_pool2d_x_0
        #  - output_2 -> aten::adaptive_max_pool2d_x_1
        op_converter.source_map = relay_pytorch._debug_rename(model.graph, use_parser_friendly_name)
        # Get mapping form debugName to layer_name, e.g., model.conv1, which is used for quantization override.
        #  - e.g., given following IR
        #    graph(%model : __torch__.torch.fx.graph_module.___torch_mangle_0.GraphModule,
        #          %input : Tensor):
        #       %1 : __torch__.torch.nn.modules.linear.___torch_mangle_1.Linear = prim::GetAttr[name="fc"](%model)
        #       %2 : __torch__.torch.nn.modules.activation.___torch_mangle_2.ReLU = prim::GetAttr[name="relu1"](%model)
        #       %3 : Tensor = prim::GetAttr[name="weight"](%1)
        #       %4 : Tensor = prim::GetAttr[name="bias"](%1)
        #       %5 : Tensor = aten::linear(%input, %3, %4)
        #       %6 : Tensor = aten::relu(%5) # sourceRange contain __torch__.torch.nn.modules.activation.___torch_mangle_2.ReLU
        #       return (%6)
        #   populate mapping {'5': 'fc', '6': 'relu1'} to output_name_to_node_name
        relay_pytorch._parse_graph(model.graph, model, op_converter.output_name_to_node_name,
                                   input_model_path, op_converter.torch_source_map)
        return model, op_converter

    @staticmethod
    def extract_attributes_from_src_node(node, node_attr_dict):
        """
        Extract the actual values of all inputs and attributes of a PyTorch source node,
        and update them in node_attr_dict

        :param node: The node in the graph
        :param node_attr_dict: {node_name : attr_values}
                               A dictionary storing a mapping from node names to the actual
                               values of all inputs and attributes, for each node of the
                               parsed model
        """
        # E.g., for upsample,
        #   After multiple iterations and the current node is "upsample":
        #       node_attr_dict = {
        #           'input.1': ['prim::GetAttr', '1'],   # '1' is the input name
        #           'prim::Constant_0_0': None,
        #           'prim::Constant_1_0': 2.0
        #           'prim::ListConstruct_0_0': [2.0, 2.0]
        #       },
        #       which records the mapping from node names to actual values of all inputs and attributes of nodes.
        #
        #       input_names (for upsample) = [
        #           'input.1', 'prim::Constant_0_0', 'prim::ListConstruct_0_0'
        #       ]
        #
        #       -> The corresponding values for input names are: [
        #               ['prim::GetAttr', '1'], # -> the value in node_attr_dict['input.1']                  -> the value of input
        #               None,                   # -> the value in node_attr_dict['prim::Constant_0_0']       -> the value of output_size
        #               [2.0, 2.0]              # -> the value in node_attr_dict['prim::ListConstruct_0_0']  -> the value of scale_factor
        #           ]
        #
        #       Then, update the node_attr_dict to store the actual values of all inputs and attributes
        #       of this upsample node
        #           node_attr_dict = {
        #               ...,
        #               'aten::upsample_nearest2d_0_0': [['prim::GetAttr', '1'], None, [2.0, 2.0]]
        #           }
        from qti.tvm.relay import expr as _expr

        input_names = [i.debugName() for i in list(node.inputs())]
        if node.outputsSize() > 1:
            node_name = "_".join(relay_pytorch._get_output_names(node))
        else:
            node_name = relay_pytorch._get_output_name(node)
        if node.kind() != "prim::GetAttr":
            if node.kind() == "prim::Constant":
                constant_val = relay_pytorch._get_constant(node)
                if isinstance(constant_val, _expr.Constant):
                    constant_val = constant_val.data.numpy()
                node_attr_dict.update({node_name: constant_val})
            elif node.kind() in ["prim::TupleConstruct", "prim::ListUnpack", \
                                    "prim::TupleUnpack", "prim::If", "prim::Loop"]:
                pass
            elif node.kind() == "prim::prim::RaiseException":
                node_attr_dict.update({node_name: None})
            else:
                if len(input_names) > 1:
                    # Update node_attr_dict to record the actual values of all inputs and attributes
                    # of the current node.
                    # e.g., for upsample,
                    #   [node_attr_dict[inp] for inp in input_names]
                    #                       ||
                    #    [['prim::GetAttr', '1'], None, [2.0, 2.0]]
                    node_attr_dict.update({node_name: [node_attr_dict[inp] for inp in input_names]})
                else:
                    # If the length of the input name is 1, the length of the stored values
                    # corresponding to the node is also 1.
                    value = node_attr_dict[input_names[0]]
                    # In order to extract the real value,
                    # the case where the original value is a list and there are multiple types
                    # in it is handled by extracting the first value.
                    # E.g., [array([[0, 0], [0, 0]], dtype=int32), 'cpu', False, None],
                    #       where other values are supplementary and useless
                    if isinstance(value, list) and len(set(type(val) for val in value)) > 1:
                        value = value[0]
                    node_attr_dict.update({node_name: value})
        else:
            # Record the op type, i.e., "prim::GetAttr", to identify inputs easily later
            if node_name in node_attr_dict:
                node_attr_dict[node_name].insert(0, "prim::GetAttr")
            else:
                node_attr_dict[node_name] = "prim::GetAttr"

