# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from collections import defaultdict
from typing import Union

from lib.utils.nd_graph_structure import GraphStructure


class TensorDAG(defaultdict):
    def __init__(self, graph_struct, available_tensors):
        # type: (GraphStructure, list[str]) -> dict[str, dict[str, set]]
        """returns TensorDAG given graph_struct

        DAG is represented as a dictionary where key is tensor name and
        value is nested dictionary. With regard to inner dictionary, keys
        are 'parents' and 'children', and value is set of tensor names.

        E.g. For a node with input A and outputs B,C:
        dag = {
            'tensorA': {
                'parents': {},
                'children': {'tensorB','tensorC'}
            },
            'tensorB': {
                'parents': {'tensorA'},
                'children': {}
            },
            'tensorC': {
                'parents': {'tensorA'},
                'children':{}
            }
        }

        :param graph_struct: GraphStructure object showing structure of model
        :param available_tensors: list of available inference tensor names
        :return: defaultdict graph representation
        """

        def default_val():  # type: () -> dict[str, set]
            return {"parents": set(), "children": set()}

        super().__init__(default_val)

        for component in graph_struct._components:
            if set(component.input_tensors.keys()) == set(component.output_tensors.keys()):
                # component is most likely input node, do not add parents and children
                continue
            for output_tensor in component.output_tensors.keys():
                if output_tensor in available_tensors:
                    self[output_tensor]["parents"].update([t for t in component.input_tensors.keys() if t in available_tensors])
            for input_tensor in component.input_tensors.keys():
                if input_tensor in available_tensors:
                    self[input_tensor]["children"].update([t for t in component.output_tensors.keys() if t in available_tensors])

    @staticmethod
    def getAncestors(dag, tensor):
        # type: (dict[str, dict[str, set]], str) -> set
        """Returns set of ancestors (including self) for given tensor

        :param dag: TensorDAG instance
        :param tensor: tensor name
        :return: set of ancestor names
        """

        # Breadth First Search
        ancestors = set([tensor]) # tensor can be own ancestor
        if dag[tensor]:
            to_visit = list(dag[tensor]["parents"])
            while to_visit:
                ancestor = to_visit.pop(0)
                if ancestor not in ancestors:
                    ancestors.add(ancestor)
                    to_visit.extend(list(dag[ancestor]["parents"]))

        return ancestors

    @staticmethod
    def getLCA(dag, tensor_list):
        # type: (dict[str, dict[str, set]], list[str]) -> Union[str, None]
        """returns lowest common ancestor given TensorDAG and iterable of tensor names

        :param dag: TensorDAG instance
        :param tensor_list: list of tensor names
        :return: tensor name
        """

        tensor_ancestors = [TensorDAG.getAncestors(dag, tensor) for tensor in tensor_list]
        common_ancestors = tensor_ancestors[0].intersection(*tensor_ancestors)

        if common_ancestors:
            # pick arbitrary tensor and get common descendants until LCA
            ancestor = list(common_ancestors)[0]
            lca = None
            while not lca:
                found_lower = False
                for child in dag[ancestor]['children']:
                    if child in common_ancestors:
                        ancestor = child
                        found_lower = True
                        break
                if not found_lower:
                    lca = ancestor
            return lca
        else:
            # no common ancestors
            return None

    @staticmethod
    def isDescendant(dag, t1, t2):
        # type: (dict[str, dict[str, set]], str, str) -> bool
        """Returns True if tensor1 is descendant or equivalent to tensor2.

        :param dag: TensorDAG instance
        :param t1: tensor name
        :param t2: tensor name
        :return: boolean
        """
        return TensorDAG.getLCA(dag, [t1,t2]) == t2
