# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.framework_diagnosis.frameworks.nd_base_framework import BaseFramework
from lib.utils.nd_exceptions import FrameworkError
from lib.utils.nd_errors import get_message, get_warning_message
import caffe.proto.caffe_pb2 as caffe_pb2
from typing import List, Tuple, Dict, Union
from collections import OrderedDict
from google.protobuf import text_format
import caffe
import os
import logging

class CaffeFramework_1_0_0(BaseFramework):
    __VERSION__ = '1.0.0'

    def __init__(self, logger):
        self.logger = logger
        self._model = None
        self._deploy = None
        self.net = None

    def load_model(self, model_path):  # type: (str) -> None
        """ Loads a caffe inference model into the class

        Takes in model paths (both relative or absolute paths works) to deploy.prototxt and model.caffemodel, and loads
        the model into the class.

        :param model_path: A string delimited by comma which documents the relative or
        absolute path to the files. The first section of the string is the path
        for deploy.prototxt, and the second section is for model.caffemodel.
        :return: None
        """
        deploy_prototxt = None
        caffemodel_filename = None
        try:
            paths = model_path.split(',', 2)
            if len(paths)!=2:
                raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE_MODEL_INPUTS"))
            if 'prototxt' in paths[0] and 'caffemodel' in paths[1]:
                deploy_prototxt = paths[0]
                caffemodel_filename = paths[1]
            elif  'prototxt' in paths[1] and 'caffemodel' in paths[0]:
                deploy_prototxt = paths[1]
                caffemodel_filename = paths[0]
        except RuntimeError:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE_MODEL_INPUTS"))

        if not os.path.isfile(deploy_prototxt):
            raise FrameworkError("Prototxt '{}' not found".format(deploy_prototxt))
        if not os.path.isfile(caffemodel_filename):
            raise FrameworkError("Caffemodel '{}' not found".format(caffemodel_filename))

        self.net = caffe.Net(deploy_prototxt, caffemodel_filename, caffe.TEST)
        self._deploy = deploy_prototxt

        self._model = caffe_pb2.NetParameter()

        with open(self._deploy, 'rb') as f:
            text_format.Merge(f.read(), self._model)

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        if len(input_data) != len(input_tensor_names) or len(input_tensor_names) >1:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE_MISMATCH_INPUTS"))

        for input_tensor in input_tensor_names:
            if input_tensor not in self.net.layer_dict.keys() and input_tensor !="data":
                raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE_MISMATCH_INPUTS"))

        for input_name, data in zip(input_tensor_names,input_data):
            if input_name=="input" or input_name=="data":
                input_name = "data"
            if self.net.blobs[input_name].data.shape != data.shape:
                raise FrameworkError("ERROR_FRAMEWORK_CAFFE_MISMATCH_INPUTS, need {}, but input shape is{}"
                .format(self.net.blobs[input_name].data.shape, data.shape))
            self.net.blobs[input_name].data[...] = data

        results = {}
        for out_blob in output_tensor_names:
            try:
                blob_data = None
                if out_blob not in self.net.layer_dict.keys():
                    raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE_MISMATCH_OUTPUT"))
                if input_tensor_names[0] == "input" or input_tensor_names[0] == "data":
                    blob_data = self.net.forward(end=out_blob)
                else:
                    blob_data = self.net.forward(start=input_tensor_names[0],end=out_blob)
                if not hasattr(blob_data, 'decode'):
                    for output_tensor, data in blob_data.items():
                        results[out_blob] = data
                else:
                    self.logger.warn(get_warning_message("WARNING_FRAMEWORK_CAFFE_ENCOUNTERED_BYTES_OBJECT")(out_blob))
            except RuntimeError:
                self.logger.warn(get_warning_message("WARNING_FRAMEWORK_CAFFE_FAILED_TO_FETCH_BLOB")(out_blob))
        return results

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        outputs = output_tensors.copy()
        inputs = input_tensors.copy()
        visited =[]
        for layer in self._model.layer:
            bottoms = []
            if layer.name in inputs:
                inputs.remove(layer.name)
            if len(inputs) > 0:
                continue
            for bottom in layer.bottom:
                bottoms.append(bottom)

            if layer.name in outputs:
                outputs.remove(layer.name)
            if len(bottoms) > 0:
                visited.append(([] ,[layer.name]))

            if len(outputs) < 1:
                break
        return visited

    def get_dimensions(self, tensor_name):  # type: (str) -> List[int]
        """ Returns shape of the given tensor

        :param tensor_name: the name of the desired tensor
        :return: the tensor's shape as a list
        """
        if tensor_name in self.net.blobs.keys():
            return list(self.net.blobs[tensor_name].data.shape)
        else:
            return None

    def get_graph_structure(self):
        # type: () -> Dict[Union[str, int], Tuple[Union[int, str], str, List[str], List[str]]]
        """ creates a detailed list of the network's operators

        Iterates through the operators in the net, and retrieves every
        operator index/name, as well as its type, inputs, and outputs

        :return: dictionary keyed by operator index or name, with tuple values which
        include the op's index/name, type, list of inputs, and list of outputs
        """
        op_dict = OrderedDict()
        for layer in self._model.layer:
            tops = []
            bottoms = []
            for top in layer.top:
                tops.append(top)
            for bottom in layer.bottom:
                bottoms.append(bottom)
            op_dict[layer.name] = (layer.type, tops, bottoms)
        return op_dict

    def get_mapping_for_qnn_node(self, qnn_output):
        for layer in self._model.layer:
            for bottom in layer.bottom:
                if layer.name == qnn_output or (layer.name in qnn_output and bottom in qnn_output):
                    return layer.name
        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_CAFFE_MISMATCH_TENSOR")(qnn_output))
        return " "

    def get_mapping_for_snpe_node(self, snpe_output):  # type: (str) -> str
        for layer in self._model.layer:
            for bottom in layer.bottom:
                if layer.name == snpe_output or (layer.name in snpe_output and bottom in snpe_output):
                    return layer.name
        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_CAFFE_MISMATCH_TENSOR")(snpe_output))
        return " "

    def get_version(self):  # type: () -> str
        return caffe.__version__
