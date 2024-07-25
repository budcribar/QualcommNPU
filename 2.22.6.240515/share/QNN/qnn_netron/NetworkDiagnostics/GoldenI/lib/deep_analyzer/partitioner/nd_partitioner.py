# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
from typing import Union, Tuple

import pandas as pd

from lib.utils.nd_constants import Framework
from lib.utils.nd_graph_structure import GraphStructure
from lib.utils.nd_tensor_dag import TensorDAG
from lib.utils.nd_exceptions import DeepAnalyzerError
# from lib.inference_engine.util.nd_constants import Engine, Framework, Runtime
from lib.deep_analyzer.partitioner.nd_verifier_analyzer import VerifierAnalyzer


class PartitionedModel:
    """A PartitionedModel object represents any partitioned model.

    A PartitionedModel object contains the necessary information to define a
    subgraph within a model. A PartitionedModel object allows for deterministic
    dissection of a model. A PartitionedModel's tensors are always defined using
    Golden tensor names.
    """

    def __init__(self, input_tensors, output_tensors, verifier_type, original_accuracy=None, actual_accuracy=None):
        # type: (list[str], list[str], str, float, float) -> None
        """
        :param input_tensors: A list of input tensors of the partitioned model
        :param output_tensors: A list of output tensors of the partitioned model
        :param verifier_type: The verifier type (str) used to generate the accuracies
        :param original_accuracy: Float indicating the original inference engine accuracy
        :param actual_accuracy: Float indicating the actual accuracy from running the partitioned model
        """

        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.verifier_type = verifier_type
        self.original_accuracy = original_accuracy
        self._actual_accuracy = actual_accuracy

    @property
    def actual_accuracy(self):
        return self._actual_accuracy

    @actual_accuracy.setter
    def actual_accuracy(self, actual_accuracy_score):
        self._actual_accuracy = actual_accuracy_score


class Partitioner:
    """A class to analyze and create new partitioned models."""

    def __init__(self, verifier, summary_df, graph_struct, golden_tensor_paths, golden_to_inference_mapping, inference_to_golden_mapping, framework, logger):
        # type: (str, pd.DataFrame, GraphStructure, logging.Logger, dict[str, str], dict[str, str], dict[str, str], str) -> None
        """
        :param verifier: The verifier type (str) used to generate the accuracies
        :param summary_df: pandas DataFrame containing the verifier summary
        :param graph_struct: GraphStructure object showing structure of model
        :param logger: logging.Logger object
        :param golden_tensor_paths: dict mapping golden tensor names to paths
        :param golden_to_inference_mapping: dict mapping golden tensor names to inference tensor names
        :param inference_to_golden_mapping: dict mapping inference tensor names to golden tensor names
        """

        self.verifier = verifier
        self.summary_df = summary_df
        self.graph_struct = graph_struct
        self.logger = logger
        self.golden_tensor_paths = golden_tensor_paths
        self.golden_to_inference_mapping = golden_to_inference_mapping
        self.inference_to_golden_mapping = inference_to_golden_mapping
        self.dim_dict = self.graph_struct.get_tensor_dimension_dict()
        self.inference_tensors = self.graph_struct.get_all_tensors()
        self.dag = TensorDAG(graph_struct, self.inference_tensors)
        self.framework = framework

    def get_extended_inputs(self, input_tensor_names, output_tensor_names):
        # type: (list[str], list[str]) -> Tuple[list[str], list[str]]
        """Get subgraph input and output tensor names.

        Extends inputs to a previous layer while avoiding input conflicts,
        such that additional nodes are added to the PartitionedModel for systematic
        analysis. Returns tuple of arrays of input and output tensors names,
        respectively, for original tensor names and TensorDAG.

        :param input_tensor_names: list of input tensor names
        :param output_tensor_names: list of output tensor names
        :return: Tuple of new input and output tensor names
        """

        # Extending Inputs Behaviour:
        # Input (plural if multiple branches) will be exteded upwards to the nearest
        # possible valid input(s) whose golden tensor(s) exists. If inputs on different
        # branches are about to covnerge, they will be prevented from extending to the
        # common ancestor until all inputs can advance simultaneously
        #
        # Example:
        #   Graph Subsection           Tensor DAG
        #       Node A
        #       /   \                   Tensor A
        #   Node B  Node C               /   \
        #      |     |             Tensor B  Tensor C
        #      |    Node D              |     |
        #       \   /                   |    Tensor D
        #      Node E                    \   /
        #                               Tensor E
        #
        # Order of PartitionedModel inputs over time:
        # 1. Tensors E
        # 2. Tensors B, D
        # 3. Tensors B, C
        # 4. Tensors A
        #
        # Extending Inputs Algorithm:
        # For each existing input tensor t, extend inputs to t's parent tensors if and only if
        # all the parent tensors' children (i) that are also ancestors of the output tensor (ii)
        # are a subset of the existing input tensors. (iii) Else, keep the input the same (iv)
        # to wait for other inputs to 'catch up'.
        # Furthermore, t's parent tensors are only included if valid corresponding golden tensors
        # can be found (v). Otherwise, it may be an inference-only intermediate tensor. In these cases,
        # further search will be performed (vi) such that the parent's lowest valid ancestors are used.

        input_tensor_names = set(input_tensor_names)
        outputs = set(output_tensor_names)
        new_inputs = set()

        while input_tensor_names:
            t = list(input_tensor_names)[0] # retrieve any tensor
            parents = self.dag[t]["parents"]

            all_relevant_children = set()
            for parent in parents:
                for child in self.dag[parent]["children"]: # (i)
                    if any([TensorDAG.isDescendant(self.dag, output, child) for output in list(outputs)]): # (ii)
                        # typically len(list(outputs)) == 1
                        all_relevant_children.add(child)

            if all_relevant_children.issubset(input_tensor_names): # (iii)
                # (v) only add parents whose golden tensors exist
                valid_parents = [p for p in parents if p in self.inference_to_golden_mapping and self.inference_to_golden_mapping[p] in self.golden_tensor_paths]
                new_inputs.update(valid_parents)

                input_tensor_names.update(set(parents) - set(valid_parents)) # (vi) need to iterate further on invalid parents
                input_tensor_names.difference_update(all_relevant_children) # remove children whose parents are already accounted for
            else:
                new_inputs.add(t) # (iv)

            input_tensor_names.discard(t) # do not reprocess input t

        return list(new_inputs), list(outputs)

    def partition_analyze(self, problem_tensor_name=None, extend_inputs=False, partitioned_model=None, auto_stop=False, verifier_threshold=None):
        # type: (str, bool, PartitionedModel, bool, float) -> Union[PartitionedModel, None]
        """Determine initial or subsequent PartitionedModel for use in Model Dissection Analysis.

        Uses VerifierAnalyzers and summary dataframe of original model to
        determine the ideal location for dissection. If a previous Partitioned Model
        is given, determine the next Partitioned Model to be used. Returns a new
        Partitioned Model object.

        :param problem_tensor_name: manually specified problematic inference tensor name
        :param extend_inputs: boolean flag to show inputs of old PartitionedModel should be extended
        :param partitioned_model: latest PartitionedModel analyzed
        :param verifier_threshold: threshold as float value to use in VerifierAnalyzer
        :param auto_stop: boolean flag to automatically prevent new PartitionedModel from
                          being created if error in existing PartitionedModel is considered significant
        :return: new PartitionedModel
        """


        if extend_inputs and (partitioned_model is None):
            raise DeepAnalyzerError("Need old partitioned model to extend inputs.")

        if not extend_inputs:
            # initial iteration
            if problem_tensor_name:
                # given initial tensor
                if problem_tensor_name not in self.summary_df['Name'].values:
                    raise DeepAnalyzerError("SPECIFIED PROBLEM TENSOR DOES NOT EXIST")
                metric_val = VerifierAnalyzer.get_metric(self.verifier, self.summary_df, problem_tensor_name)
            else:
                # find problematic tensor
                problem_tensor_name, metric_val = VerifierAnalyzer.analyze(self.verifier, self.summary_df, threshold=verifier_threshold)
                if not problem_tensor_name:
                    self.logger.error("Initial Problematic Tensor not found")
                    return None
            self.logger.info("Initial Problematic Tensor: {}".format(problem_tensor_name))
            latest_inputs = [problem_tensor_name]
            latest_outputs = [problem_tensor_name]
        else:
            latest_inputs = [self.golden_to_inference_mapping[tensor_name] for tensor_name,_ in partitioned_model.input_tensors]
            latest_outputs = [self.golden_to_inference_mapping[tensor_name] for tensor_name,_ in partitioned_model.output_tensors]
            metric_val = partitioned_model.original_accuracy

            actual_accuracy = partitioned_model.actual_accuracy
            if auto_stop and VerifierAnalyzer.is_change_significant(self.verifier, self.summary_df, metric_val, actual_accuracy):
                return None
        inputs, outputs = self.get_extended_inputs(latest_inputs, latest_outputs)

        try:
            inputs = [[self.inference_to_golden_mapping[name], self.dim_dict[name]] for name in inputs]
            outputs = [[self.inference_to_golden_mapping[name], self.dim_dict[name]] for name in outputs]
        except KeyError:
            self.logger.error("KeyError, tensor not found in inference to golden mapping.")
            return None
        return PartitionedModel(inputs, outputs, self.verifier, original_accuracy=metric_val)

    def create_partitioned_model(self, partition):
        tensorflow_suffix = ""
        if self.framework == Framework.tensorflow.value:
            tensorflow_suffix = ":0"
        inputs = [[input+tensorflow_suffix, dim] for input, dim in zip(partition['inputs'],partition['inputs_dims'])]
        outputs = [partition['outputs']]
        return PartitionedModel(inputs, outputs, self.verifier)

    @staticmethod
    def get_relevant_nodes_from_inputs(graph_struct, golden_to_inference_mapping, input_tensors):
        # type: (GraphStructure, dict[str, str], list[str]) -> list[str]
        """Get node names associated with given list of input tensor names

        :param graph_struct: GraphStructure object showing structure of model
        :param golden_to_inference_mapping: dict mapping golden tensor names to inference tensor names
        :param input_tensors: list of input tensor names
        :return: list of node names
        """

        inference_inputs = set([golden_to_inference_mapping[tensor_name] for tensor_name in input_tensors])
        node_names = []
        for component in graph_struct._components:
            overlap = inference_inputs & set(component.input_tensors.keys())
            if bool(overlap):
                node_names.append(component.name)
                inference_inputs = inference_inputs - overlap
        return node_names

    # @staticmethod
    # def get_node_io(graph_struct, tensor_names, output_tensor_names=None):
    #     """Get subgraph input and output tensor names.

    #     Extends each input to a previous layer.
    #     Returns tuple of arrays of input and output tensors names,
    #     respectively, for a given output name and graph_struct.
    #     """
    #     if output_tensor_names is None:
    #         output_tensor_names = tensor_names

    #     available_tensors = graph_struct.get_all_tensors()
    #     tensor_names = set(tensor_names)
    #     output_tensor_names = set(output_tensor_names)
    #     outputs = set()
    #     inputs = set()

    #     for component in graph_struct._components:
    #         overlap = output_tensor_names & set(component.output_tensors.keys())
    #         if bool(overlap):
    #             outputs.update([name for name in component.output_tensors.keys()])
    #             output_tensor_names = output_tensor_names - overlap

    #         overlap = tensor_names & set(component.output_tensors.keys())
    #         if bool(overlap):
    #             inputs.update([name for name in component.input_tensors.keys() if name in available_tensors])
    #             tensor_names = tensor_names - overlap

    #     return list(inputs), list(outputs)
