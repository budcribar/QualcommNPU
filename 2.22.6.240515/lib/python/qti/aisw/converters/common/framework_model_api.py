# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import List
from abc import ABC, abstractmethod


class FrameworkModelAPI(ABC):
    def __init__(self):
        """
        Initialize the FrameworkModelAPI Object.
        :param FrameworkLoader loader: onnx Loader object.
        """
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """
        Perform default optimizations.
        """
        pass

    @abstractmethod
    def add_inputs(self, input_names: List):
        """
        add the inputs to the model.

        :param: input_names: List containing tensor names
        """
        pass

    @abstractmethod
    def add_outputs(self, out_names: List):
        """
        Adds the output to the model.

        :param: out_names: List containing tensor names
        """
        pass

    @abstractmethod
    def remove_inputs(self, input_names_to_remove: List[str]):
        """
        Remove the inputs.

        :param List[str] input_names_to_remove: Input names to remove.
        """
        pass

    @abstractmethod
    def remove_outputs(self, output_names_to_remove: List[str]):
        """
        Remove the output from the model.

        :param List[str] output_names_to_remove: output names to remove.
        """
        pass

    @abstractmethod
    def native_shape_inference(self, *args, **kwargs):
        """
        Runs the native shape inference.

        :param bool delete_existing_shapes: Delete exiting shapes or not, defaults to True
        """
        pass

    @abstractmethod
    def clean_model(self):
        """
        Cleans up the model by removing unused nodes and dangling inputs/outputs.
        """
        pass

