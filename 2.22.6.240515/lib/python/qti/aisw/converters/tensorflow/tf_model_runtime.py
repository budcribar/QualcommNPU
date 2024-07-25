# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Dict, List, Optional, Text, Union

import numpy as np
from qti.aisw.converters.common.qnn_runtime_base import IModelRuntime
from qti.aisw.converters.common.utils.framework_utils import TensorInfo
from qti.aisw.converters.tensorflow.loader import Model
from qti.aisw.converters.tensorflow.util import (get_input_info,
                                                 get_output_info,
                                                 run_tf_inference)


class TensorflowModelRuntime(IModelRuntime):
    def __init__(
        self,
        model_or_path: Union[Text, Model],
        input_nodes_names: List[str] = None,
        input_nodes_shapes: List[int] = None,
        out_node_names: List[str] = None,
        saved_model_tag: str = "",
        saved_model_signature_key: str = "",
    ) -> None:
        """
        Initializing Tensorflow runtime object.

        :param Union[Text, Model] model_or_path: Path of the tensorflow model
            or Model class representation of tensorflow model.
        :param List[str] input_nodes_names: List of input node names, required
            in case of loading the model from path, defaults to None
        :param List[int] input_nodes_shapes: List of input node shapes,
            required in case of loading the model from path, defaults to None
        :param List[str] out_node_names: List of output node names, required in
            case of loading the model from path, defaults to None
        :param str saved_model_tag: Saved model tag for loading saved model
            format, required in case of loading the model from path, defaults to ""
        :param str saved_model_signature_key: Saved model signature key for
            loading saved model format, required in case of loading the model
            from path, defaults to ""
        :raises AttributeError: If the provided model is not a path of model or
            a Model class instance.
        """
        if isinstance(model_or_path, str):
            self.model = Model(
                model_or_path,
                input_nodes_names,
                input_nodes_shapes,
                out_node_names,
                saved_model_tag,
                saved_model_signature_key,
            )
        elif isinstance(model_or_path, Model):
            self.model = model_or_path
        else:
            raise AttributeError(
                f"Expected Path or Model object but received: {type(model_or_path)}"
            )

    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets input tensor information.

        :return Dict[Text, TensorInfo]: Input names to tensor information.
        """
        return get_input_info(self.model.graph_def)

    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets output tensor information

        :return Dict[Text, TensorInfo]: Output names to tensor information.
        """
        return get_output_info(self.model.graph_def)

    def execute_inference(
        self, inputs: Dict[Text, np.ndarray], output_names: Optional[List] = []
    ) -> Dict[Text, np.ndarray]:
        """
        Run the inference of given model.

        :param Dict[Text, np.ndarray] inputs: Dict containing input tensor name
            to corresponding tensor data as numpy array.
        :param Optional[List] output_names: List of output names for
            inference, defaults to []
        :return Dict[Text, np.ndarray]: Dict containing output tensor name as
            key and its computed numpy array output as value.
        """
        return run_tf_inference(
            self.model.graph_def, input_data=inputs, output_node_names=output_names
        )
