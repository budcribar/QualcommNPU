# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
import os
import json
from collections import OrderedDict


class GraphStructure:
    class Component:
        """
        Component represents either an op for a framework (such as Tensorflow)
        or a layer for an engine (such as SNPE) alogn with their associated input and output tensors
        """
        def __init__(self, key, info):
            self.name = key
            self.type = info[0]
            # dicts of tensor name to dimension
            if len(info[1:]) == 3:
                self.input_tensors, self.output_tensors, self.output_encodings = info[1:]
            else:
                self.input_tensors, self.output_tensors = info[1:]
                self.output_encodings = None
            self.all_tensors = self.input_tensors.copy()
            self.all_tensors.update(self.output_tensors)

    def __init__(self, graph_structure):
        """
        Instantiation of GraphStructure should be done through load_graph_structure.
        :param graph_structure: OrderedDict
        """

        self._components = []
        self._tensor_list = []
        self._types = OrderedDict()
        self._tensor_dimension_dict = OrderedDict()
        self._all_output_tensor_encoding_dict = OrderedDict()

        self._initialize(graph_structure)

    @classmethod
    def load_graph_structure(cls, json_graph_file):
        """
        Retrieves the structure of a graph from file
        :param json_graph_file: File to retrieve data from.
        :return: GraphStructure
        """

        graph_structure = json.load(open(json_graph_file), object_pairs_hook=OrderedDict)
        return cls(graph_structure)

    def _initialize(self, graph_structure):
        # Initialize components of graph
        self._components = [GraphStructure.Component(key, info) for key, info in graph_structure.items()]

        for component in self._components:
            # Initialize list of all tensors in graph structure without duplicates
            self._tensor_list = self._tensor_list + list(component.output_tensors.keys())
            for output_name in component.output_tensors:
                self._types[output_name] = component.type
            # Initialize dictionary of all tensors to their dimensions
            self._tensor_dimension_dict.update(component.all_tensors)
            if component.output_encodings:
                self._all_output_tensor_encoding_dict.update(component.output_encodings)

    @staticmethod
    def save_graph_structure(graph_file, graph_struct):
        """
        Save the structure of a graph to file.
        :param graph_file: JSON file to save graph structure to
        :param graph_struct:    OrderedDict of op/layer name to list of two lists (input tensors, output tensors).
                                The value of an inference engine or framework's get_graph_structure should be passed in.
        """

        os.makedirs(os.path.dirname(graph_file), exist_ok=True)

        with open(graph_file, 'w+') as f:
            json.dump(graph_struct, f, indent=4)

    def get_all_types(self):
        return self._types

    def get_all_tensors(self):
        return copy.deepcopy(self._tensor_list)

    def get_tensor_dimension_dict(self):
        return copy.deepcopy(self._tensor_dimension_dict)

    def get_all_output_tensor_encodings_dict(self):
        return self._all_output_tensor_encoding_dict
