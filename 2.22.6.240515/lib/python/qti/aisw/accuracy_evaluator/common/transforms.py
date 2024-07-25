##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import errno
import logging
import os
import sys

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc


class supress_stdout:

    def __init__(self):
        self.prev_fd = None
        self.prev_stdout = None

    def __enter__(self):
        #Creating null file descriptor
        temp = open(os.devnull, "w")
        self.prev_fd = os.dup(sys.stdout.fileno())
        #Making original stdout fd point to same file as temp fd
        os.dup2(temp.fileno(), sys.stdout.fileno())
        self.prev_stdout = sys.stdout
        sys.stdout = os.fdopen(self.prev_fd, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.prev_fd, self.prev_stdout.fileno())
        sys.stdout = self.prev_stdout


class ModelHelper:

    @classmethod
    def _remove_initializer_from_input(cls, model_path, out_model_path):
        """Removes the initializers from the onnx models with ir_version>3."""
        onnx = Helper.safe_import_package("onnx")
        M = onnx.load(model_path)
        if M.ir_version < 4:
            qacc_file_logger.debug(
                'Model with ir_version below 4 requires to include initializer in graph input')
            return

        inputs = M.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in M.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

        qacc_file_logger.debug("Removed initializers from the graph input.")
        onnx.save(M, out_model_path)

    @classmethod
    def simplify_onnx(cls, onnx_model_mem):
        '''
        Method to simplify a onnx model already loaded in memory
        Returns : Simplified onnx model (memory object)
        '''
        onnxsim = Helper.safe_import_package("onnxsim", "0.4.36")
        onnxruntime = Helper.safe_import_package("onnxruntime", "1.17.1")
        simplified_model, simplified_check_status = onnxsim.simplify(onnx_model_mem)
        if not simplified_check_status:
            raise ce.ModelTransformationException(
                'Check for simplified model failed for given model')
        return simplified_model

    @classmethod
    def clean_onnx(cls, model_path, out_path, symbols, replace_special_chars=True, check_model=True,
                   simplify_model=True):

        onnx = Helper.safe_import_package("onnx")
        try:
            onnx_model = onnx.load(model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
        if simplify_model:
            onnx_model = ModelHelper.simplify_onnx(onnx_model)
        try:
            if check_model:
                onnx.checker.check_model(onnx_model)
        except Exception as e:
            qacc_file_logger.error('check_model failed for given model : {}'.format(model_path))
            qacc_file_logger.exception(e)
            raise ce.ModelTransformationException(
                'check_model failed for given model : {}'.format(model_path))

        # remove special characters in node inputs and outputs.
        if replace_special_chars:
            for i in range(len(onnx_model.graph.node)):
                for j in range(len(onnx_model.graph.node[i].input)):
                    onnx_model.graph.node[i].input[j] = Helper.sanitize_node_names(
                        onnx_model.graph.node[i].input[j])
                for j in range(len(onnx_model.graph.node[i].output)):
                    onnx_model.graph.node[i].output[j] = Helper.sanitize_node_names(
                        onnx_model.graph.node[i].output[j])

            # remove special characters in graph inputs.
            for ip in onnx_model.graph.input:
                ip.name = Helper.sanitize_node_names(ip.name)
            # remove special characters in graph outputs.
            for ip in onnx_model.graph.output:
                ip.name = Helper.sanitize_node_names(ip.name)
            # remove special characters in value-info and initializers.
            for i in range(len(onnx_model.graph.value_info)):
                onnx_model.graph.value_info[i].name = Helper.sanitize_node_names(
                    onnx_model.graph.value_info[i].name)
            for i in range(len(onnx_model.graph.initializer)):
                onnx_model.graph.initializer[i].name = Helper.sanitize_node_names(
                    onnx_model.graph.initializer[i].name)

        # remove symbols with provided values (default 1)
        for ip in onnx_model.graph.input:
            dim_len = len(ip.type.tensor_type.shape.dim)
            for i in range(dim_len):
                if len(ip.type.tensor_type.shape.dim[i].dim_param) > 0:
                    _symbol = ip.type.tensor_type.shape.dim[i].dim_param
                    if _symbol in symbols:
                        ip.type.tensor_type.shape.dim[i].dim_value = \
                            symbols[_symbol]
                    else:
                        ip.type.tensor_type.shape.dim[i].dim_value = 1
                    qacc_file_logger.debug('Replaced symbol {} with value {} in cleaned'
                                           ' model'.format(
                                               _symbol, ip.type.tensor_type.shape.dim[i].dim_value))

        for op in onnx_model.graph.output:
            dim_len = len(op.type.tensor_type.shape.dim)
            for i in range(dim_len):
                if len(op.type.tensor_type.shape.dim[i].dim_param) > 0:
                    _symbol = op.type.tensor_type.shape.dim[i].dim_param
                    if _symbol in symbols:
                        op.type.tensor_type.shape.dim[i].dim_value = \
                            symbols[_symbol]
                    else:
                        op.type.tensor_type.shape.dim[i].dim_value = 1
                    qacc_file_logger.debug('Replaced symbol {} with value {} in cleaned'
                                           ' model'.format(
                                               _symbol, op.type.tensor_type.shape.dim[i].dim_value))

        onnx.save(onnx_model, out_path)
        # to remove initializers from onnx model inputs
        ModelHelper._remove_initializer_from_input(out_path, out_path)
        return out_path

    @classmethod
    def clean_tf(cls, model_path, out_path, symbols):
        tf = Helper.safe_import_package("tensorflow")
        qacc_file_logger.info('Preparing model for qnn')
        # load tf model
        with tf.io.gfile.GFile(model_path, "rb") as f:
            M = tf.compat.v1.GraphDef()
            M.ParseFromString(f.read())

        for i in range(len(M.node)):
            # Remove colocate information from GraphDef
            if "_class" in M.node[i].attr:
                del M.node[i].attr["_class"]
            # remove special characters from node names
            M.node[i].name = Helper.sanitize_node_names(M.node[i].name)
            for j in range(len(M.node[i].input)):
                # remove special characters except ':' from node inputs
                # Otherwise graph connection breaks in case of node with multiple outputs
                M.node[i].input[j] = Helper.sanitize_node_names(M.node[i].input[j])

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # save the cleaned model
        tf.io.write_graph(graph_or_graph_def=M, logdir=out_path, name='cleanmodel.pb',
                          as_text=False)
        qacc_file_logger.info('Prepared model for qnn')
        cleanmodel_path = os.path.join(out_path, 'cleanmodel.pb')
        return cleanmodel_path

    @classmethod
    def clean_torchscript(cls, model_path, out_path, symbols):
        """Return the path to the cleaned model."""
        qacc_file_logger.info('Preparing model for qnn')
        torch = Helper.safe_import_package("torch")
        # load torchscript model
        loaded = torch.jit.load(model_path)
        # loaded = loaded.eval()
        # Freeze and inline graph
        # loaded._c = torch._C._freeze_module(loaded._c)
        # torch._C._jit_pass_inline(loaded.graph)
        torch.jit.save(loaded, out_path)
        return out_path

    @classmethod
    def clean_tflite(cls, model_path, out_path, symbols):
        """Return the path to the cleaned model."""
        #TODO: Do we need to clean the tflite model?
        qacc_file_logger.info('Preparing model')

        return out_path

    @classmethod
    def clean_model_for_qnn(cls, model_path, out_dir, symbols={}, replace_special_chars=True,
                            check_model=True, simplify_model=True):
        """Clean up the model for qnn."""
        mtype = Helper.get_model_type(model_path)

        if mtype == ModelType.ONNX:
            if simplify_model:
                out_path = out_dir + '/' + qcc.SIMPLIFIED_CLEANED_MODEL_ONNX
            else:
                out_path = out_dir + '/' + qcc.CLEANED_MODEL_ONNX
            return ModelHelper.clean_onnx(model_path, out_path, symbols, replace_special_chars,
                                          check_model=check_model, simplify_model=simplify_model)
        elif mtype == ModelType.TENSORFLOW:
            return ModelHelper.clean_tf(model_path, out_dir, symbols)
        elif mtype == ModelType.TORCHSCRIPT:
            out_path = os.path.join(out_dir, "cleanmodel.pt")
            return ModelHelper.clean_torchscript(model_path, out_path, symbols)
        elif mtype == ModelType.TFLITE:
            #out_path = os.path.join(out_dir, "cleanmodel.tflite")
            return ModelHelper.clean_tflite(model_path, model_path, symbols)
        else:
            raise ce.UnsupportedException('Unsupported model type {}'.format(mtype))

    @classmethod
    def get_onnx_batch_size(cls, model_path, input_name):
        onnx = Helper.safe_import_package("onnx")
        try:
            m = onnx.load(model_path)
            for inp in m.graph.input:
                modified_input_name = Helper.sanitize_node_names(input_name)
                if inp.name == modified_input_name:
                    try:
                        return inp.type.tensor_type.shape.dim[0].dim_value
                    except Exception as e:
                        qacc_file_logger.warning(
                            'batchsize not found for model: {}'.format(model_path))
                        qacc_file_logger.warning('Setting model batchsize to 1')
                        return 1
                else:
                    raise ce.ConfigurationException('Incorrect model input name provided in config.'
                                                    ' Given: {}, Using: {}, Actual: {}'.format(
                                                        input_name, modified_input_name, inp.name))
        except FileNotFoundError as e:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    @classmethod
    def get_tf_batch_size(cls, model_path, input_name):
        try:
            tf = Helper.safe_import_package("tensorflow")
            with tf.io.gfile.GFile(model_path, "rb") as f:
                model = tf.compat.v1.GraphDef()
                model.ParseFromString(f.read())
            for node in model.node:
                modified_input_name = Helper.sanitize_node_names(input_name)
                if node.name == modified_input_name:
                    try:
                        return node.attr['_output_shapes'].list.shape[0].dim[0].size
                    except Exception as e:
                        qacc_file_logger.warning(
                            'batchsize not found for model : {}'.format(model_path))
                        qacc_file_logger.warning('Setting model batchsize to 1')
                        return 1
                else:
                    raise ce.ConfigurationException('Incorrect model input name provided in config.'
                                                    ' Given: {}, Using: {} , Actual: {}'.format(
                                                        input_name, modified_input_name, node.name))
        except FileNotFoundError as e:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    @classmethod
    def get_torchscript_batch_size(cls, model_path, input_name):
        return 1

    @classmethod
    def get_model_batch_size(cls, model_path, input_name):
        mtype = Helper.get_model_type(model_path)

        if mtype == ModelType.ONNX:
            return ModelHelper.get_onnx_batch_size(model_path, input_name)
        elif mtype == ModelType.TENSORFLOW:
            return ModelHelper.get_tf_batch_size(model_path, input_name)
        elif mtype == ModelType.TORCHSCRIPT:
            return ModelHelper.get_torchscript_batch_size(model_path, input_name)
        else:
            raise ce.UnsupportedException('Unsupported model type {}'.format(mtype))
