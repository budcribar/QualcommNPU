# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Tuple

import numpy as np
import onnx
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_error
from qti.aisw.converters.common.utils.einsum_eqn import EINSUM_SUPPORTED
from qti.aisw.converters.onnx.util import (
    create_node_name,
    create_tensor_name,
    get_opset_version,
    make_node,
)


class ONNXEinsumPatternOptimizer:
    """
    ONNXEinsumPatternOptimizer class is responsible for:
    Replacing einsum operators by its mathematical equivalent sequence of operators.
    """

    def __init__(self, loader):
        """
        Constructor of ONNXEinsumPatternOptimizer
        """
        self.loader = loader
        self.model_opset = get_opset_version(self.loader.model)
        self.__validate_mapping(EINSUM_SUPPORTED)
        self.einsum_equation_bank = EINSUM_SUPPORTED

    def __validate_mapping(self, data: Dict) -> None:
        """
        Function to check the validity of the einsum equations mapping file.
        :param data (Dict): Dict containing einsum equation's nodes and edges info.
        """
        for equation in data:
            equ_node_list = data[equation]
            equ_node_names = [n.name for n in equ_node_list]
            if "input_1" not in equ_node_names:
                error = f"Input-1 not found for equation: {equation}."
                raise ValueError(error)
            for equ_node in equ_node_list:
                if "input" in equ_node.name or "output" in equ_node.name:
                    continue
                eqn_node_op_type = equ_node.op_type

    def __get_matching_equation(self, node_eqn: str) -> str:
        """
        Function to get the equivalent matching einsum equation from the
        bank.
        :param node_eqn (str): Einsum equation to query.
        :returns str: Fetched equivalent einsum equation.
        """
        matched_eqn = ""
        for database_eqn in self.einsum_equation_bank:
            if ONNXEinsumPatternOptimizer.match_eqn(node_eqn, database_eqn):
                matched_eqn = database_eqn
                break
        return matched_eqn

    @staticmethod
    def match_eqn(str1: str, str2: str) -> bool:
        """
        Function to match two equations.
        e.g: bm,bhm->bh is equal to ij,ikj->ik
        Operation is same, letters are different.
        :param str1 (str): String representation of equation 1.
        :param str2 (str): String representation of equation 2.
        :returns bool: Boolean status indicating whether equations are matching or not.
        """
        char_to_index1 = {}
        char_to_index2 = {}
        if len(str1) != len(str2):
            return False
        values1 = list(str1)
        values2 = list(str2)
        for i, val in enumerate(values1):
            if val not in char_to_index1.keys():
                char_to_index1[val] = i
        for i, val in enumerate(values2):
            if val not in char_to_index2.keys():
                char_to_index2[val] = i
        hash_map1 = {}
        for i, val in enumerate(values1):
            if char_to_index1[val] not in hash_map1.keys():
                hash_map1[char_to_index1[val]] = []
            l = hash_map1[char_to_index1[val]]
            l.append(i)
            hash_map1[char_to_index1[val]] = l
        hash_map2 = {}
        for i, val in enumerate(values2):
            if char_to_index2[val] not in hash_map2.keys():
                hash_map2[char_to_index2[val]] = []
            l = hash_map2[char_to_index2[val]]
            l.append(i)
            hash_map2[char_to_index2[val]] = l
        status = hash_map1 == hash_map2
        return status

    def __add_axes(
        self, node: str, new_init_list: List[onnx.TensorProto], node_args: Dict
    ) -> Tuple[List, Dict]:
        """
        Function to identify axis value and add the same into node args' input
        or axis attributes.
        :param node (str): String representation of node
        :param new_init_list (List[onnx.TensorProto]): List of new initializer to be added in the model.
        :param node_args (Dict): Node argument dict containing info about node.
        :returns Tuple[List, Dict]: Tuple of updated list of initializer and update node_args dict.
        """
        axes = node.attrs.get("axis", None)
        if axes is None:
            raise RuntimeError(
                f"For node: {node.name} in einsum mapping, axis attribute is not found."
            )
        node_name = node_args["name"]
        if self.model_opset >= 13:
            axes_initializer = onnx.numpy_helper.from_array(
                np.array(axes, dtype="int64"), name=f"{node_name}.axes"
            )
            new_init_list.append(axes_initializer)
            node_args["inputs"].append(axes_initializer.name)
        else:
            node_args["axes"] = axes
        return new_init_list, node_args

    def get_einsum_equivalent_nodes(
        self,
        einsum_node_input_names: List[str],
        einsum_node_output_name: str,
        node_eqn: str,
        node_name_mapping_dict: Dict,
        output_dtype: int,
        einsum_node_name: str,
    ) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
        """
        Function to get the equivalent nodes for given einsum equation.
        :param einsum_node_input_names (List[str]): Input names of einsum node.
        :param einsum_node_output_name (str): Output name of einsum node.
        :param node_eqn (str): Einsum equation string representation.
        :param node_name_mapping_dict (Dict): Node mapping dict to check naming
                conflicts.
        :param output_dtype (int): Output datatype of the einsum node.
        :param einsum_node_name (str): Einsum onnx node name.
        :returns Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]: Tuple of
                List of nodes to be replaced with einsum node and List of
                new initializer to be added in the graph.
        """
        new_nodes = {}
        new_initializers = []
        unique_name_set = set()

        for node in self.loader.get_nodes():
            for act_name in node.output:
                unique_name_set.add(act_name)
            for in_act in node.input:
                unique_name_set.add(in_act)

            unique_name_set.add(node.name)

        node_eqn = node_eqn.replace(" ", "")
        matched_eqn = self.__get_matching_equation(node_eqn)
        if not matched_eqn:
            raise ValueError(
                f"Einsum equation : {node_eqn} for node : {einsum_node_name} is not implemented currently"
            )

        equ_node_list = self.einsum_equation_bank[matched_eqn]

        equ_node_name_to_onnx_node_name = {}
        for equ_node in equ_node_list:
            if equ_node.name in ["input_1", "input_2", "output"]:
                continue
            op_type = equ_node.op_type
            # Get a unique node name based on existing node names and op type.
            name_prefix = op_type + "_" + einsum_node_name
            onnx_node_name, node_name_mapping_dict = create_node_name(
                self.loader.model.graph,
                op_type,
                node_name_mapping_dict,
                name_prefix=name_prefix,
            )
            equ_node_name_to_onnx_node_name[equ_node.name] = onnx_node_name
            output, unique_name_set = create_tensor_name(
                onnx_node_name, unique_name_set
            )
            node_args = {
                "op_type": op_type,
                "inputs": [],
                "outputs": [output],
                "name": onnx_node_name,
            }
            if op_type == "Unsqueeze":
                new_initializers, node_args = self.__add_axes(
                    equ_node, new_initializers, node_args
                )
            elif op_type == "Transpose":
                perm_axes = equ_node.attrs.get("perm", None)
                if perm_axes is None:
                    raise RuntimeError(
                        f"For node {equ_node.name} in einsum mapping, perm attribute is not found."
                    )
                node_args["perm"] = perm_axes
            elif op_type == "Cast":
                dtype = equ_node.attrs.get("dtype", None)
                if dtype is None:
                    raise RuntimeError(
                        f"For node {equ_node.name} in einsum mapping, dtype attribute is not found."
                    )
                # Checking for cast node datatype. Cast node before Matmul
                # node is used for converting the data into float32 dtype
                # and cast node after Matmul is used for converting the data
                # back to original datatype of the einsum node.
                if dtype == "original":
                    onnx_dtype = output_dtype
                else:
                    onnx_dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
                node_args["to"] = onnx_dtype
            elif op_type == "ReduceSum":
                node_args["keepdims"] = equ_node.attrs.get("keepdims", 0)
                new_initializers, node_args = self.__add_axes(
                    equ_node, new_initializers, node_args
                )
            elif op_type == "Squeeze":
                new_initializers, node_args = self.__add_axes(
                    equ_node, new_initializers, node_args
                )
            elif op_type not in ["MatMul", "Mul"]:
                error = f"Unknown node type: '{op_type}' found for equation: {node_eqn}"
                raise ValueError(error)
            new_node = make_node(**node_args)

            new_node_inputs = []
            for equ_node_input in equ_node.inputs:
                if equ_node_input == "input_1":
                    new_node_inputs.append(einsum_node_input_names[0])
                elif equ_node_input == "input_2":
                    new_node_inputs.append(einsum_node_input_names[1])
                else:
                    if equ_node_input not in equ_node_name_to_onnx_node_name:
                        raise ValueError(
                            f"For node {node.name} in einsum "
                            f"mapping, its input {equ_node_input} not found."
                        )

                    node_outputs = new_nodes[
                        equ_node_name_to_onnx_node_name[equ_node_input]
                    ].output
                    if len(node_outputs) > 1:
                        error = "Current implementation only supports many to one node relationship."
                        raise ValueError(error)

                    new_node_inputs.append(node_outputs[0])

            new_node_inputs.extend(new_node.input)
            del new_node.input[:]
            new_node.input.extend(new_node_inputs)

            new_nodes[onnx_node_name] = new_node

        equ_output_node = [n for n in equ_node_list if n.op_type == "Output"]
        if len(equ_output_node) != 1:
            raise RuntimeError(
                "Einsum mapping should possess only 1 Output op_type node."
            )
        onnx_output_node_name = equ_node_name_to_onnx_node_name[
            equ_output_node[0].inputs[0]
        ]
        onnx_output_node = new_nodes[onnx_output_node_name]
        onnx_output_node.output[0] = einsum_node_output_name
        return new_nodes.values(), new_initializers

    def __str__(self) -> str:
        """
        Function to get the string representation of ONNXEinsumPatternOptimizer
        class.
        :returns str: String representation of class.
        """
        return "ONNX - Einsum PatternOptimizer"

    def optimize(self):
        """
        Function to apply Einsum optimization logic and replace einsum node with
        equivalent nodes.
        :returns ONNXLoader instance.
        :raises:
            e: Exception raised in case of checker failure.
        """
        _node_name_suffix = {}
        einsum_node_output_dtype = 0
        for i, node in enumerate(self.loader.get_nodes()):
            if node.op_type == "Einsum":
                einsum_node_output_name = node.output[0]
                for value in self.loader.model.graph.value_info:
                    if value.name == einsum_node_output_name:
                        einsum_node_output_dtype = value.type.tensor_type.elem_type
                if einsum_node_output_dtype == 0:
                    log_debug1(
                        f"Einsum node {node.name} output datatype not found. \
                        Proceeding with Float32 dtype."
                    )
                    einsum_node_output_dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
                        np.dtype("float32")
                    ]
                einsum_node_input_names = []
                if len(node.input) == 1:
                    # einsum_ip1
                    einsum_node_input_names.append(node.input[0])
                else:
                    # einsum_ip1, einsum_ip2
                    einsum_node_input_names.append(node.input[0])
                    einsum_node_input_names.append(node.input[1])
                equation = node.attribute[0].s.decode("UTF-8")
                log_debug1(f"Einsum node found with {equation} at node {str(i)}")
                replacement_nodes, new_initializers = self.get_einsum_equivalent_nodes(
                    einsum_node_input_names=einsum_node_input_names,
                    einsum_node_output_name=einsum_node_output_name,
                    node_eqn=equation,
                    node_name_mapping_dict=_node_name_suffix,
                    output_dtype=einsum_node_output_dtype,
                    einsum_node_name=node.name,
                )
                self.loader.utils.add_nodes(replacement_nodes)
                for init in new_initializers:
                    self.loader.utils.add_initializer(init)
                self.loader.utils.remove_node(node)
                # Cleanup model and apply topological sort.
                self.loader.utils.clean_model().topological_sort()
        try:
            self.loader.native_checker()
        except Exception as e:
            log_error(f"The Onnx native checker failed with Exception : {e} ")
            raise e

        self.loader.utils.native_shape_inference()
        return self.loader
