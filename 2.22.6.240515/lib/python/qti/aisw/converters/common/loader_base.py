# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Text

from numpy import ndarray
from qti.aisw.converters.common.common_base import ConverterBase
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
from qti.aisw.converters.common.custom_ops.utils.custom_op_helpers import (
    populate_custom_op_collection,
)
from qti.aisw.converters.common.utils.framework_utils import (
    FrameworkSummary,
    TensorInfo,
)


class FrameworkModelLoader(ConverterBase):
    def __init__(
        self, converter_type: Text, custom_op_factory: CustomOpFactory, args: Any
    ) -> None:
        """
        Base class of model loader

        :param Text converter_type: Type of converter
        :param CustomOpFactory custom_op_factory: CustomOPFactory instance.
        :param args: argument pass to the converters.
        """
        super(FrameworkModelLoader, self).__init__(args, custom_op_factory=custom_op_factory)
        populate_custom_op_collection(
            self.model,
            converter_type,
            custom_op_config_paths=self.custom_op_config_paths,
            custom_op_factory=self.custom_op_factory,
            converter_op_package_lib=self.converter_op_package_lib,
        )

    @property
    def has_custom_op(self) -> bool:
        """
        Check if underlying model has custom op or not.

        :return bool: False if model don't have custom op else True.
        """

        return (
            self.custom_op_factory != None
            and len(self.custom_op_factory.op_collection) != 0
        )

    @abstractmethod
    def clone_loader(self):
        """
        Clones the Model Loader object safely along with the internal model instance.
        """
        pass

    @abstractmethod
    def get_nodes(self):
        """
        get nodes for underlying model.
        """
        pass

    @abstractmethod
    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets input tensor information.

        :return Dict[Text, TensorInfo]: Input names to tensor information.
        """
        pass

    @abstractmethod
    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets output tensor information

        :return Dict[Text, TensorInfo]: Output names to tensor information.
        """
        pass

    @abstractmethod
    def native_checker(self) -> bool:
        """
        Check the model if it valid or not.

        :return bool: return True only if model is valid.
        """
        pass

    @abstractmethod
    def save_model(self, path: Text, **kwargs: Optional[dict]) -> None:
        """
        saves the underlying model

        :param Text path: Path or BytesIO object.
        """
        pass

    def summarize_model(self, summary) -> FrameworkSummary:
        """
        Returns the summary object.
        """
        return summary
