# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABCMeta
from abc import abstractmethod
from typing import List, Tuple, Dict, Union
import numpy


class BaseFramework(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    def load_model(self, model_path):  # type: (str) -> None
        """ Loads a machine learning inference model into the class

        Takes in model paths (both relative or absolute paths works) to
        the model files, and loads the model into the class.

        :param model_path: A string which documents the relative or
        absolute path to the files.
        :return: None
        """

        raise NotImplementedError('Method load_model must be implemented to use this base class')

    @abstractmethod
    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        # type: (numpy.ndarray, List[str], List[str]) -> Dict[str, numpy.ndarray]
        """ Runs a singular operator in the network and retrieves its output data

        Finds the operator which takes in input_tensor_names, and runs it once
        by feeding it input_data. Returns a dictionary with tensor name and data.
        This dictionary will only contain tensors which are included in
        output_tensor_names.

        :param input_data: a numpy ndarray of properly-formatted tensors
        :param input_tensor_names: a list of input tensor names, which respectively
        correspond with input_data
        :param output_tensor_names: a list of output tensor names
        :return: a dictionary of tensor data, indexed by tensor name
        """

        raise NotImplementedError('Method run_inference must be implemented to use this base class')

    @abstractmethod
    def get_intermediate_tensors(self, input_tensors, output_tensors):
        # type: (List[str], List[str]) -> List[Tuple[List[str]]]
        """ Traces the graph from the output_ops to input_ops, tracking intermediate tensors

        Traverses the network to gather a list of tensors which are passed to get from the
        input tensors to the output tensors. May include tensors that aren't necessarily
        required to get from input_tensors to output_tensors.

        :param input_tensors: list of input tensor names
        :param output_tensors: list of output tensor names
        :return: list of tuples, each tuple represents an operator and contains two lists:
        the op's list of inputs, and the op's list of outputs
        """

        raise NotImplementedError('Method get_intermediate_tensors must be implemented to use this '
                                  'base class')

    @abstractmethod
    def get_dimensions(self, tensor_name):  # type: (str) -> List[int]
        """ Returns shape of the given tensor

        :param tensor_name: the name of the desired tensor
        :return: the tensor's shape as a list
        """

        raise NotImplementedError('Method get_dimensions must be implemented to use this base '
                                  'class')

    @abstractmethod
    def get_graph_structure(self):
        # type: () -> Dict[Union[str, int], Tuple[Union[int, str], str, List[str], List[str]]]
        """ creates a detailed list of the network's operators

        Iterates through the operators in the net, and retrieves every
        operator index/name, as well as its type, inputs, and outputs

        :return: dictionary keyed by operator index or name, with tuple values which
        include the op's index/name, type, list of inputs, and list of outputs
        """

        raise NotImplementedError('Method get_graph_structure must be implemented to use this base '
                                  'class')

    @abstractmethod
    def get_version(self):  # type: () -> str
        """ returns framework version

        :return: version of framework as string
        """
        raise NotImplementedError('Method get_version must be implemented to use this base '
                                  'class')
    @abstractmethod
    def get_mapping_for_qnn_node(self, qnn_output):  # type: (str) -> str
        """ returns framework node name

        :return: the node name of qnn_output in the framework
        """
        raise NotImplementedError('get_mapping_for_qnn_node must be implemented to use this base '
                                  'class')


    def get_mapping_for_snpe_node(self, snpe_output):  # type: (str) -> str
        """ returns framework node name

        :return: the node name of snpe_output in the framework
        """
        return snpe_output
