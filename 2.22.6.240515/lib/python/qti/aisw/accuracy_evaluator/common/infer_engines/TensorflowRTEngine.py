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
import logging
import numpy as np
import os
import time
# to avoid printing logs on console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import sys

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.infer_engines.infer_engine import InferenceEngine
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class TensorflowInferenceEngine(InferenceEngine):
    """TensorflowInferenceEngine class takes required inputs supplied by user
    from commandline options and calls validate and execute methods.

    TODO: Add an extra dictionary parameter to class which enables to uses extra tensorflow
    options
    To use:
    >>> engine = TensorflowInferenceEngine(model, inputlistfile, output_path, multithread,
    input_info,
                output_info, gen_out_file, extra_params)
    >>> engine.validate()
    >>> engine.execute()
    """

    def __init__(self, model, inputlistfile, output_path, multithread, input_info, output_info=None,
                 gen_out_file=None, extra_params=None):
        super().__init__(model, inputlistfile, output_path, multithread, input_info, output_info,
                         gen_out_file, extra_params)
        tf = Helper.safe_import_package("tensorflow")
        self.validate()

    def wrap_frozen_graph(self, graph_def, inputs, outputs):
        """This method converts frozen graph to ConcreteFunction."""
        tf = Helper.safe_import_package("tensorflow")

        def _imports_graph_def():
            tf = Helper.safe_import_package("tensorflow")
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

    def execute(self):
        """
        This method runs the given model on tensorflow
        Returns:
            status: execution status
            res   : dictionary containing ort_session
        """
        tf = Helper.safe_import_package("tensorflow")
        qacc_file_logger.debug("TensorflowInferenceEngine start execution")
        # capture inference time
        inf_time = 0
        res = {}
        inp_shapes_map = {}
        do_profile = False
        save_intermediate_outputs = True
        profile_data = None
        if self.extra_params:
            profile_data = {}
            if '-profile' in self.extra_params:
                do_profile = self.extra_params['-profile']
            if '-save-intermediate-outputs' in self.extra_params:
                save_intermediate_outputs = self.extra_params['-save-intermediate-outputs']

        # Load the tensorflow frozen graph
        with tf.io.gfile.GFile(self.model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Create a list of input nodes so that it can be used to get node dtype and shapes while
        # running the inference
        inp_nodes = [node for node in graph_def.node if node.op == "Placeholder"]

        # getting graph output nodes
        graph_nodes = {}
        for node in graph_def.node:
            graph_nodes[node.name] = node

        for node in graph_def.node:
            for elem in node.input:
                if elem in graph_nodes.keys():
                    del graph_nodes[elem]
        out_nodes = graph_nodes

        # create a dictionary mapping input node names to corresponding shapes
        for node in inp_nodes:
            shape = []
            for j in range(len(node.attr['_output_shapes'].list.shape[0].dim)):
                shape.append(node.attr['_output_shapes'].list.shape[0].dim[j].size)
            inp_shapes_map[node.name] = tuple(shape)

        out_node_names = []
        if self.extra_params and '-save-single-layer-output-name' in self.extra_params:
            for i, node in enumerate(graph_def.node):
                if node.op == 'Const' or node.op == 'Identity':
                    continue
                if node.op != "Placeholder" and (self.extra_params['-save-single-layer-output-name']
                                                 in node.name):
                    out_node_names.append(node.name)
                    break
        elif self.extra_params and '-save-input' in self.extra_params:
            for i, node in enumerate(graph_def.node):
                for input_layer in self.extra_params['-save-input']:
                    if node.op == 'Const':
                        continue
                    if node.op != "Placeholder" and (input_layer in node.name):
                        out_node_names.append(node.name)
        elif save_intermediate_outputs:
            for i, node in enumerate(graph_def.node):
                if node.op == 'Const' or (node.op == 'Identity' and '_class' in node.attr.keys()):
                    continue
                if node.op != "Placeholder":
                    out_node_names.append(node.name)
        else:
            # original graph outputs alone are considered. No intermediate outputs are dumped
            out_node_names = out_nodes.keys()

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = self.wrap_frozen_graph(
            graph_def=graph_def,
            inputs=[node.name + ':0' for node in inp_nodes],
            outputs=[out_name + ':0' for out_name in out_node_names],
        )

        # Create the output file if requested.
        out_list_file = None
        if self.gen_out_file:
            out_list_file = open(self.gen_out_file, 'w')

        start_time = time.time()

        # Create input dictionary and run the inference for each input.
        with open(self.input_path) as f:
            for iter, line in enumerate(f):
                input_list = []
                inps = line.strip().split()
                inps = [inp.split(':=')[-1].strip() for inp in inps if inp.strip()]

                for idx, inp in enumerate(inps):
                    if self.input_info is None:
                        # When input shapes and dtypes are not passed by user,inputs dictionary
                        # is formed using inp_nodes list
                        try:
                            input_np = np.fromfile(
                                inp, dtype=Helper.tf_type_to_numpy(
                                    inp_nodes[idx].attr['dtype'].type)).reshape(
                                        inp_shapes_map[inp_nodes[idx].name])
                            input_list.append(
                                tf.convert_to_tensor(
                                    input_np,
                                    Helper.tf_type_to_numpy(inp_nodes[idx].attr['dtype'].type)))

                        except Exception as e:
                            qacc_logger.error('Unable to extract input info from model.Please try '
                                              'passing input-info ')
                            qacc_file_logger.exception(e)
                            raise ce.InferenceEngineException(
                                "Unable to extract input info from model.Please try "
                                "passing input-info", e)
                    else:
                        if inp_nodes[idx].name not in self.input_info:
                            qacc_file_logger.error(
                                'Input info name not valid for this model. expected: {} '.format(
                                    inp_nodes[idx].name))
                            raise ce.ConfigurationException("Invalid Configuration")
                        input_np = np.fromfile(
                            inp, dtype=(Helper.get_np_dtype(
                                self.input_info[inp_nodes[idx].name][0],
                                map_tf=True))).reshape(self.input_info[inp_nodes[idx].name][1])

                        input_list.append(
                            tf.convert_to_tensor(
                                input_np,
                                Helper.get_np_dtype(self.input_info[inp_nodes[idx].name][0],
                                                    map_tf=True)))

                # Run inference
                try:
                    frozen_graph_predictions = frozen_func(*input_list)
                    outputs = []
                    for elem in frozen_graph_predictions:
                        outputs.append(elem.numpy())

                except Exception as e:
                    qacc_file_logger.error('inference.run failed')
                    qacc_file_logger.exception(e)
                    raise ce.InferenceEngineException("inference failed", e)

                tf_outputs = frozen_func.outputs

                # Store the names of the outputs. Use the same for raw file name generation.
                if self.output_info:
                    output_names = list(self.output_info.keys())
                    # reorder tf outputs if needed as per output info names
                    _temp = []
                    _temp_outs = []
                    for name in output_names:
                        _name_found = False
                        for out, node in zip(outputs, tf_outputs):
                            # ignore :0 at the end of output node names added by tf session
                            # internally
                            if str(node.name[:-2]) == str(name):
                                _temp.append(node)
                                _temp_outs.append(out)
                                _name_found = True
                                break
                        if not _name_found:
                            qacc_logger.error('Output name {} in config is incorrect.'.format(name))
                            raise ce.InferenceEngineException(
                                "Invalid model config. Please fix output_info {}".format(name))
                    tf_outputs = _temp
                    outputs = _temp_outs
                else:
                    # set same as tf session
                    output_names = []
                    for node in tf_outputs:
                        output_names.append(node.name[:-2])

                if self.output_path:
                    path = self.output_path + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # Check if the output info is configured correctly.
                    if len(output_names) != len(tf_outputs):
                        raise ce.ConfigurationException('The number of outputs in config file'
                                                        '({}) does not match with tf model'
                                                        ' outputs ({})'.format(
                                                            len(output_names), len(tf_outputs)))

                    # Write output files and get profile data.
                    profile_data, _paths = self.save_outputs_and_profile(
                        output_names, outputs, iter, True, do_profile)

                    # generate output text file for each of the inputs.
                    if self.gen_out_file:
                        out_list_file.write(','.join(_paths) + '\n')

        if self.gen_out_file:
            out_list_file.close()

        #For generating histogram profile
        output_dtypes = [op.dtype for op in outputs]
        output_array_map = list(zip(output_names, output_dtypes, outputs))

        res['tf_session'] = frozen_func
        res['profile'] = profile_data

        if do_profile:
            qacc_logger.info('Captured tf profile')

        inf_time = time.time() - start_time
        qacc_file_logger.debug("TensorflowInferenceEngine execution success")
        return True, res, inf_time, output_array_map

    def get_profile(self):
        return self.profile_data

    def validate(self):
        """
        This method checks whether the given model_path ,model,input_path and output_path are
        valid or not
        Returns:
            status: validation status
        """
        qacc_file_logger.debug("TensorflowInferenceEngine validation")
        # check the existence of model path and its authenticity
        if not os.path.exists(self.model_path):
            qacc_file_logger.error('Model path : {} does not exist '.format(self.model_path))
            raise ce.InferenceEngineException('Model path : {} does not exist '.format(
                self.model_path))

        # check whether the output path exists and create the path otherwise
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # check the existence of input path
        if not os.path.exists(self.input_path):
            qacc_logger.error('Input path : {} does not exist '.format(self.input_path))
            raise ce.InferenceEngineException('Input path : {} does not exist '.format(
                self.input_path))
        qacc_file_logger.debug("TensorflowInferenceEngine validation success")
        return True
