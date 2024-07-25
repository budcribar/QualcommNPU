# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from collections import OrderedDict
import onnx
import numpy as np
import onnxruntime

from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_base_framework import BaseFramework
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import ModelHelper
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name


class OnnxFramework_1_3_0(BaseFramework):
    __VERSION__ = '1.3.0'
    FRAMEWORK_SUFFIX = '.onnx'

    def __init__(self, logger, custom_op_lib=None):
        super(OnnxFramework_1_3_0, self).__init__(logger)
        self._model = None
        self._graph = None
        self.ort_outs = None
        self.onnx_custom_op_lib = custom_op_lib
        self.is_custom_op_registered = False
        self.ort_custom_op_sess_options = None

    @property
    def graph(self):
        return self._graph

    def load_model(self, model_path):
        # Import graph and save to instance variable
        self._model = onnx.load_model(model_path)
        try:
            onnx.checker.check_model(self._model)
        except ValueError as e:
            self.logger.warning(str(e))
        self._graph = self._model.graph

    def optimize(self, model_path, optimized_model_path):
        """
        This method applies basic graph optimizations like const-folding, etc.. and
        returns the transformed model. Uses OnnxRT to optimize the model. Refer to the
        below link for all the optimizations that OnnxRT does.
        Reference Doc: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
        Args :
            model_path              : path to the model
            optimized_model_path    : path to optimized model
        Returns:
            ret_status        : status as boolean type
            transformed_model : path to transformed model
        """
        sess_options = onnxruntime.SessionOptions()
        if self.onnx_custom_op_lib is not None:
            self.ort_custom_op_sess_options = self.register_custom_op(sess_options)
            sess_options = self.ort_custom_op_sess_options
        # Set graph optimization level to basic
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = optimized_model_path
        providers = ['CPUExecutionProvider']
        try:
            session = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        except Exception as e:
            self.logger.error('rt_session init failed')
        transformed_model = self._fix_old_ir_versions(optimized_model_path)
        return True, transformed_model

    def _fix_old_ir_versions(self, model_path):
        """
        Onnx Runtime doesn't handle the older ir_versions(<4) properly.
        This method adds the initializers to the inputs which is required by
        ir_version < 4.
        Args:
            model_path: path to the Onnx model
        Returns:
            model_path: path to the updated Onnx model.
        """
        model = onnx.load(model_path)
        if model.ir_version < 4:
            # Add initializers to the inputs.
            initializers = [i.name for i in model.graph.initializer]
            graphInputs = [i.name for i in model.graph.input]
            diff = np.setdiff1d(initializers, graphInputs, assume_unique=True).tolist()
            new_inputs = [onnx.helper.make_tensor_value_info(init.name, init.data_type, init.dims)
                            for
                            init in model.graph.initializer if init.name in diff]
            model.graph.input.extend(new_inputs)
            onnx.save(model, model_path)
        return model_path

    def add_outputs(self):
        # adds all intermediate nodes of model as output nodes
        if len(self.get_output_layers(names_only=True)) >= len(self.get_layer_identifiers()):
            # Do not modify the model if #outputnodes >= #modelnodes
            return
        for node in self._model.graph.node:
            for output in node.output:
                self._model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    def register_custom_op(self, sess_opts=None):
        if sess_opts is None:
            sess_opts = onnxruntime.SessionOptions()

        if self.is_custom_op_registered:
            return sess_opts
        try:
            sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_opts.register_custom_ops_library(self.onnx_custom_op_lib)
            self.is_custom_op_registered=True
            self.logger.info(f'Registered given custom op lib : {self.onnx_custom_op_lib}')
        except Exception as e:
            raise FrameworkError(f'Registration of custom op:{self.onnx_custom_op_lib} failed due to {e}.')
        return sess_opts

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_ONNX_MISMATCH_INPUTS"))
        ort_inputs = {}

        if self.is_custom_op_registered and self.ort_custom_op_sess_options:
            ort_sess_opts = self.ort_custom_op_sess_options
        else:
            if self.onnx_custom_op_lib is not None:
                self.ort_custom_op_sess_options = self.register_custom_op()
                ort_sess_opts = self.ort_custom_op_sess_options
            else:
                ort_sess_opts = onnxruntime.SessionOptions()
        ort_session = onnxruntime.InferenceSession(self._model.SerializeToString(), ort_sess_opts)
        for data, input_ele in zip(input_data, ort_session.get_inputs()):
            ort_inputs[input_ele.name] = data

        outputs = [x.name for x in ort_session.get_outputs()]
        if self.ort_outs is None:
            self.ort_outs = ort_session.run(outputs, ort_inputs)

        result = {}
        for output_tensor, data in zip(outputs, self.ort_outs):
            if str(output_tensor_names[0]) == str(output_tensor):
                result.update({output_tensor: data})

        return result

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        tensor_pairs = []
        input_initializer = [node.name for node in self._model.graph.initializer]
        for node in self._model.graph.node:
            inputs = []
            outputs = []
            for input in node.input:
                input_name = onnx.ValueInfoProto(name=input).name
                if input_name not in input_initializer:
                    inputs.append(input_name)
            for output in node.output:
                outputs.append(onnx.ValueInfoProto(name=output).name)
            tensor_pairs.append((inputs, outputs))

        return tensor_pairs

    def get_dimensions(self, tensor_name):
        pass

    def get_graph_structure(self):
        """Creates a detailed list of the network's operators.

        Iterates through the operators in the net, and retrieves every
        operator's index , as well as its type, inputs, and outputs

        :return: dictionary indexed by op index with values containing
            the index, tuple of list of inputs and list of outputs
        """
        op_dict = OrderedDict()
        i = 0
        input_initializer = [node.name for node in self._model.graph.initializer]
        for node in self._model.graph.node:
            inputs = []
            outputs = []
            for input in node.input:
                input_name = onnx.ValueInfoProto(name=input).name
                if input_name not in input_initializer:
                    inputs.append(input_name)
            for output in node.output:
                outputs.append(onnx.ValueInfoProto(name=output).name)
            op_dict[i] = (node.op_type, inputs, outputs)
            i += 1
        return op_dict

    def get_mapping_for_qnn_node(self, qnn_output):  # type: (str) -> str
        """Returns framework node name :return: the node name of qnn_output in
        the framework."""
        if qnn_output[1:].isdigit():
            qnn_output = qnn_output[1:]
        check_conv_batch_norm = False
        for node in self._model.graph.node:
            if not check_conv_batch_norm:
                for output in node.output:
                    tensor_name = onnx.ValueInfoProto(name=output).name
                    tensor_replace = tensor_name.replace(".", "_")
                    tensor_replace = tensor_replace.replace("/", "_")
                    if qnn_output == tensor_replace:
                        if node.op_type == 'Conv':
                            check_conv_batch_norm = True
                            break
                        else:
                            return qnn_output
            else:
                check_conv_batch_norm = False
                if node.op_type == 'BatchNormalization':
                    return onnx.ValueInfoProto(name=node.output[0]).name  # node.output[0]
                else:
                    return qnn_output

        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR")(qnn_output))
        return " "

    def get_mapping_for_snpe_node(self, snpe_output_tensor):  # type: (str) -> str
        """Returns framework node name :return: the node name of
        snpe_output_tensor in the framework."""
        check_conv_batch_norm = False
        for node in self._model.graph.node:
            if not check_conv_batch_norm:
                for output in node.output:
                    tensor_name = santize_node_name(onnx.ValueInfoProto(name=output).name)
                    if tensor_name == snpe_output_tensor or \
                            tensor_name == "_"+snpe_output_tensor:
                        if node.op_type == 'Conv':
                            check_conv_batch_norm = True
                            break
                        else:
                            return tensor_name
            else:
                check_conv_batch_norm = False
                if node.op_type == 'BatchNormalization':
                    # node.output[0]
                    return santize_node_name(onnx.ValueInfoProto(name=node.output[0]).name)
                else:
                    return tensor_name

        # if no matching, some warning will occur.
        logging.warning(
            get_warning_message("WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR")(snpe_output_tensor))
        return " "

    def get_version(self):
        return onnx.__version__

    def extract(self, start_layer_output_name, end_layer_output_name=None, out_model_path=None):
        raise NotImplementedError('Method extract is not implemented for onnx version < 1.8.0')

    ################################Layerwise_snooping utility methods ####################################
    def get_layer_identifiers(self, op_types_only=False):
        """This method returns list of layer name, output name and type in the
        onnx model.

        Returns:
            layers : list of tuples containing layer_name, output_name, layer_op_type.
        """
        layer_info = []
        model = self._model
        for node in model.graph.node:
            if op_types_only:
                if node.op_type not in layer_info:
                    layer_info.append(node.op_type)
            else:
                if node.op_type in ['Constant', 'Identity']:
                    continue
                layer_info.append((node.name, node.output[0], node.op_type))
        return layer_info

    def get_output_layers(self, names_only=False):
        """This method returns list of output layers and their datatype of
        provided onnx model.

        Args:
            names_only : boolean flag to return just list of output layer names
        Returns:
            output_layers_info : list of tuple containing output layer names and corresponding
            numpy datatype.
        """
        output_layers_info = []
        model = self._model

        layer_out_type_map = {}
        if not names_only:
            for node in model.graph.node:
                for idx in range(len(node.output)):
                    layer_out_type_map[node.output[idx]] = node.op_type

        # form list of tuple containing output layer names and corresponding numpy datatype
        for vi in model.graph.output:
            out_name = vi.name
            if names_only:
                output_layers_info.append(out_name)
            else:
                dim = []
                for i in range(len(vi.type.tensor_type.shape.dim)):
                    dim.append(vi.type.tensor_type.shape.dim[i].dim_value)
                try:
                    (out_dtype,
                     _) = ModelHelper.onnx_type_to_numpy(str(vi.type.tensor_type.elem_type))
                except Exception as e:
                    logging.error(e)
                output_layers_info.append((out_name, out_dtype, layer_out_type_map[out_name], dim))
        return output_layers_info

    def get_input_layers(self, names_only=False):
        """This method returns list of inputs in the onnx model.

        Args:
            names_only: only return list of names
        Returns:
            input_layers_info : list of tuple containing input layer names and corresponding
            numpy datatype.
        """
        input_layers_info = []
        model = self._model
        # form list of tuple containing input layer names and corresponding numpy datatype
        for vi in model.graph.input:
            inp_name = vi.name
            if names_only:
                input_layers_info.append(inp_name)
            else:
                (inp_dtype, _) = ModelHelper.onnx_type_to_numpy(str(vi.type.tensor_type.elem_type))
                dim = []
                for i in range(len(vi.type.tensor_type.shape.dim)):
                    if vi.type.tensor_type.shape.dim[i].dim_value:
                        dim.append(vi.type.tensor_type.shape.dim[i].dim_value)
                    elif vi.type.tensor_type.shape.dim[i].dim_param:
                        dim.append(vi.type.tensor_type.shape.dim[i].dim_param)
                input_layers_info.append((inp_name, inp_dtype, dim))
        return input_layers_info
