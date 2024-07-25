# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import os
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Text, Union

import numpy as np
import onnx
import qti.aisw.converters.onnx.util as Utils
from onnx import ModelProto, NodeProto
from packaging.version import parse as parse_version
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning
from qti.aisw.converters.common.utils.framework_utils import (
    FrameworkSummary,
    TensorInfo,
)
from qti.aisw.converters.onnx import model_evaluator
from qti.aisw.converters.onnx.onnx_model_api import ONNXModelUtils


class ONNXLoader(FrameworkModelLoader):
    def __init__(
        self,
        args: Any,
        custom_op_factory: Optional[CustomOpFactory],
    ) -> None:
        """
        Creates the onnx loader instance.
        :param custom_op_factory: CustomOpFactory Instance, defaults to None
        """
        self.init(args.input_network)
        super(ONNXLoader, self).__init__(
            converter_type="onnx", custom_op_factory=custom_op_factory, args=args
        )

    def init(
        self,
        path_or_proto: Union[Text, ModelProto],
    ):
        """
        Initialize the ONNX Loader class.

        :param Union[Text,ModelProto] path_or_proto: Path of onnx model or model proto.
        """

        self.model_name = "qnn_prepared.onnx"
        if isinstance(path_or_proto, str):
            self._model = onnx.load(path_or_proto)
            if parse_version(onnx.version.version) <= parse_version("1.6.0"):
                # For onnx-1.6, the load API loads the external data and keeps
                # the external data as a part of ModelProto which is wrong.
                self._model = Utils.remove_external_data_from_model(self._model)
            self.model_name = os.path.basename(path_or_proto)
        else:
            self._model = path_or_proto
        self.size = Utils.get_model_size(self._model)
        self.utils = ONNXModelUtils(self)

    def clone_loader(self):
        """
        Clone the onnx loader safely

        :return ONNXLoader: Returns the deep copied ONNXLoader instance.
        """
        model = copy.deepcopy(self._model)
        loader = copy.deepcopy(self)
        loader.init(model)
        return loader

    def update_model(self, model: ModelProto) -> None:
        """
        Update the current object's ModelProto with given ModelProto.

        :param ModelProto model: ModelProto instance to be copied.
        """
        self.size = Utils.get_model_size(model)
        self._model = copy.deepcopy(model)

    def update_model_shallow(self, model: ModelProto) -> None:
        """
        Update the given instance of onnx ModelProto to current model.
        It will do direct assignment which is shallow copy.

        :param ModelProto model: ModelProto instance to be shallow copied.
        """
        self.size = Utils.get_model_size(model)
        self._model = model

    @property
    def model(self) -> ModelProto:
        """
        Get the ModelProto instance from loader.

        :return ModelProto: Onnx ModelProto instance.
        """
        return self._model

    def get_inputs(self) -> Dict[str, onnx.ValueInfoProto]:
        """
        Get the ONNX Model input tensors dict.

        :return Dict[str, onnx.ValueInfoProto]: Dict with input tensor name as
            key and input tensor as value.
        """
        return {inp.name: inp for inp in Utils.get_inputs(self.model)}

    def get_outputs(self) -> Dict[str, onnx.ValueInfoProto]:
        """
        Get the ONNX Model output tensors dict.

        :return Dict[str, onnx.ValueInfoProto]: Dict with output tensor name as
            key and output tensor as value.
        """
        return {inp.name: inp for inp in Utils.get_outputs(self.model)}

    def get_input_names(self) -> List[str]:
        """
        Get the ONNX Model input names.

        :returns:List[str]: list of input names.
        """
        return [inp.name for inp in Utils.get_inputs(self.model)]

    def get_output_names(self) -> List[str]:
        """
        Get the Onnx Model output names.

        :returns:List[str]: list of output names.
        """
        return [out.name for out in Utils.get_outputs(self.model)]

    def get_nodes(self) -> List[NodeProto]:
        """
        Get all the nodes from underlying onnx model.

        :returns:Underlying onnx nodes.
        """
        return Utils.get_nodes(self.model)

    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Get input name to TensorInfo Mappings. e.g. shape, dtype, layout etc.

        :return Dict[Text, TensorInfo]: TensorInfo mapping for inputs.
        """
        return Utils.get_input_info(self.model)

    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Get output name to TensorInfo Mappings. e.g. shape, dtype, layout etc.

        :return Dict[Text, TensorInfo]: TensorInfo mapping for outputs.
        """
        return Utils.get_output_info(self.model)

    def native_checker(self, dry_run=None) -> bool:
        """
        This method will return the result of onnx model checker as well as evaluate the model.
        :return: Boolean indicating the success/failure of the Native Onnx checker
        """
        success = True
        # Calling graph checker for sanity checking about the graph's node names,
        # initializer names etc.
        graph_check_status = Utils.graph_checker(self.model)
        if not graph_check_status:
            log_warning("Duplicate naming found in the graph.")

        try:
            if self.size < 2:
                onnx.checker.check_model(self.model)
            else:
                # large models try to convert through a temporary file
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_model_path = os.path.join(tmpdirname, "model.onnx")
                    self.save_model(temp_model_path)
                    onnx.checker.check_model(temp_model_path)
        except Exception as e:
            log_warning("The model is invalid: %s" % e)
            return False

        if dry_run:
            log_info(
                "Proceeding with model evaluation...................................\n"
            )
            model_evaluator.setup_dry_run(self.model, dry_run)
        return success

    def save_model(self, path: Text, **kwargs: Optional[dict]) -> None:
        """
        Save The ONNX model on to the disc.

        :param Text path: to save onnx model.
        """
        if not path.endswith(".onnx"):
            prepared_name = self.model_name + "_preparator_qnn.onnx"
            path = os.path.join(path, prepared_name)
        Utils.save_model(self.model, path, **kwargs)

    def summarize_model(self, stage_info: Dict = {}) -> FrameworkSummary:
        """
        Populates summary of the onnx model.

        :param Dict stage_info: Dict: Stages information of the onnx model.
        :return FrameworkSummary: Returns the framework summary object.
        """
        summary = FrameworkSummary()
        summary.total_parameters = Utils.get_model_params(self.model)
        model_proto = self.model

        summary.ops_counter = Utils.get_unique_ops(model_proto)
        summary.ir_version = model_proto.ir_version

        # opset_details = []
        # for opi in model_proto.opset_import:
        #     opset_details.append(str(opi.version) + " " + opi.domain)
        # summary.opset_version = ", ".join(opset_details)

        summary.producer_name = ""
        # TODO ADD QNN Version here.

        summary.model_name = self.model_name

        inp_specs = self.get_input_info()
        out_specs = self.get_output_info()

        summary.inp_specs = {
            k: (v.shape, "input", v.dtype) for k, v in inp_specs.items()
        }
        summary.out_specs = {
            k: (v.shape, "output", v.dtype) for k, v in out_specs.items()
        }
        summary.stage_info = stage_info
        return super().summarize_model(summary)
