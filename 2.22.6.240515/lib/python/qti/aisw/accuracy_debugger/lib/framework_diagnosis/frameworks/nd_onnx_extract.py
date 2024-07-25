# This text retained for license compliance purposes only
# SPDX-License-Identifier: Apache-2.0

##############################################################################
#
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from typing import List, Tuple, Text

import onnx.checker
import onnx.helper
import onnx.shape_inference

from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto
from onnx.utils import Extractor


class FastExtractor(onnx.utils.Extractor):
    """Optimizes the recursive DFS used in the Parent class, by using
    memoization technique.

    Updated method uses dictonary of output to node mapping to find
    respective node of a given output in O(1) time. Also a dictionary is
    used to check if a given node is already reached in O(1) time.
    """

    def __init__(self, model):  # type: (ModelProto) -> None
        super().__init__(model)

    def _dfs_search_reachable_nodes(
            self,
            node_output_name,  # type: Text
            graph_input_names,  # type: List[Text]
            reachable_nodes,  # type: List[NodeProto]
            output_node_map,  # type: Dict[Text, NodeProto]
            reachable_nodes_dict,  # type: Dict[NodeProto, ""]
    ):  # type: (...) -> None

        if node_output_name in graph_input_names:
            return

        curr_node = output_node_map.get(node_output_name, None)

        if curr_node is not None and curr_node.output[0] not in reachable_nodes_dict:
            reachable_nodes.append(curr_node)
            reachable_nodes_dict[curr_node.output[0]] = None

            for name in curr_node.input:
                self._dfs_search_reachable_nodes(name, graph_input_names, reachable_nodes,
                                                 output_node_map, reachable_nodes_dict)

    def _collect_reachable_nodes(
            self,
            input_names,  # type: List[Text]
            output_names,  # type: List[Text]
    ):  # type: (...) -> List[NodeProto]

        reachable_nodes = list()  # type: ignore
        reachable_nodes_dict = dict()

        output_node_map = dict()
        for node in self.graph.node:
            for out in node.output:
                output_node_map[out] = node

        for name in output_names:
            self._dfs_search_reachable_nodes(name, input_names, reachable_nodes, output_node_map,
                                             reachable_nodes_dict)

        # needs to be topology sorted.
        nodes = [n for n in self.graph.node if n in reachable_nodes]
        return nodes


def extract_model(
        model,  # type: Text
        output_path,  # type: Text
        input_names,  # type: List[Text]
        output_names,  # type: List[Text]
):  # type: (...) -> None
    """Extracts sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.

    Arguments:
        input_path (string): The path to original ONNX model.
        output_path (string): The path to save the extracted ONNX model.
        input_names (list of string): The names of the input tensors that to be extracted.
        output_names (list of string): The names of the output tensors that to be extracted.
    """
    # if not os.path.exists(input_path):
    #     raise ValueError("Invalid input model path: %s" % input_path)
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    # onnx.checker.check_model(input_path)
    # model = onnx.load(input_path)

    e = FastExtractor(model)
    extracted = e.extract_model(input_names, output_names)

    onnx.save(extracted, output_path)
    onnx.checker.check_model(output_path)
