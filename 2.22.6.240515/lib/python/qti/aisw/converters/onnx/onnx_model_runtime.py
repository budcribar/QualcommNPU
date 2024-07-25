# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import onnx
import numpy as np
import copy
from typing import Dict, Text, Union, List, Optional
from qti.aisw.converters.common.utils.framework_utils import TensorInfo
from qti.aisw.converters.common.qnn_runtime_base import IModelRuntime
from onnx import ModelProto
from qti.aisw.converters.onnx.util import (
    get_input_info,
    get_output_info,
    run_ort_inference,
)

class ONNXModelRuntime(IModelRuntime):
    def __init__(self, model_or_proto: Union[Text, ModelProto]) -> None:
        """
        Initializing ONNX runtime object.

        :param Union[Text,ModelProto] model_path: Onnx model path or model proto.
        """
        if isinstance(model_or_proto, str):
            self.model = onnx.load(model_or_proto)
        elif isinstance(model_or_proto, ModelProto):
            self.model = model_or_proto
        else:
            raise AttributeError(
                f"Expected Path or Model Proto but received: {type(model_or_proto)}"
            )

    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets input tensor information.

        :return Dict[Text, TensorInfo]: Input names to tensor information.
        """
        return get_input_info(self.model)

    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets output tensor information

        :return Dict[Text, TensorInfo]: Output names to tensor information.
        """
        return get_output_info(self.model)

    def execute_inference(
        self, inputs: Dict[Text, np.ndarray], output_names: Optional[List]=[]
    ) -> Dict[Text, np.ndarray]:
        """
        Run the inference of given model.

        :param Dict[Text, np.ndarray] input_data: Dict containing input tensor name
            to corresponding tensor data.
        :param Optional[List] output_names:Optional list of output names for
            inference.
        :return Dict[Text, np.ndarray]: Dict containing output tensor name as key and
            its computed numpy array output as value.
        """
        return run_ort_inference(
            self.model, input_data=inputs, output_node_names=output_names
        )
