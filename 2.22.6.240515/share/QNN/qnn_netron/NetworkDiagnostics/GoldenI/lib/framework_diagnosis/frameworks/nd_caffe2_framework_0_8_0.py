# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from lib.framework_diagnosis.frameworks.nd_base_framework import BaseFramework
from caffe2.python import workspace
from caffe2.proto.caffe2_pb2 import NetDef

from lib.utils.nd_errors import get_message, get_warning_message
from lib.utils.nd_exceptions import FrameworkError

import os


class Caffe2Framework_0_8_0(BaseFramework):
    __VERSION__ = '0.8.0'

    def __init__(self, logger):
        super(Caffe2Framework_0_8_0, self).__init__(logger)
        self.init_net = None
        self.predict_net = None
        self.input_tensors = []
        self.initialized_predict_net = False

    def load_model(self, model_path):
        """ Loads a Caffe2 inference model into the class

        Takes in model paths (both relative or absolute paths works) to init_net.pb and predict_net.pb, and loads
        the model into the class.

        :param model_path: A string delimited by comma which documents the relative or
        absolute path to the files. The first section of the string is the path
        for init_net.pb, and the second section is for predict_net.pb.
        :return: None
        """

        paths = model_path.split(',', 2)

        with open(os.path.abspath(paths[0]), 'rb') as f:
            self.init_net = NetDef()
            self.init_net.ParseFromString(f.read())
            self.init_net.name = "init_net"

        with open(os.path.abspath(paths[1]), 'rb') as g:
            self.predict_net = NetDef()
            self.predict_net.ParseFromString(g.read())
            self.predict_net.name = "predict_net"

        workspace.RunNetOnce(self.init_net)

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        """ Obtains the specified tensor values from the graph.

        Reads in input_data for the model's input tensors (e.g. data), to initialize and run predict_net.
        Once predict_net has been run, blob values can be easily returned.
        Returns a dictionary of blob data keyed by output_tensor_names, of the blob data.

        :param input_data: a numpy ndarray of properly-formatted blobs for the model's input tensors
        :param input_tensor_names: a list of input blobs' names, which respectively correspond with input_data
        :param output_tensor_names: a list of output blobs' names
        :return: a dictionary of blobs indexed by blob name
        """

        # Feeds model's input blobs and runs predict_net if all input blobs have been fed
        if not self.initialized_predict_net:
            for tensor_name, tensor_data in zip(input_tensor_names, input_data):
                if tensor_name in self.input_tensors:
                    workspace.FeedBlob(tensor_name, tensor_data)
                    self.input_tensors.remove(tensor_name)
            if not self.input_tensors:
                try:
                    workspace.RunNetOnce(self.predict_net)
                    self.initialized_predict_net = True
                except RuntimeError:
                    raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE2_PREDICT_NET_INITIALIZATION_FAILED"))

        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE2_MISMATCH_INPUTS"))
        elif len(input_tensor_names) > len(set(input_tensor_names)) or \
                len(output_tensor_names) > len(set(output_tensor_names)):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE2_DUPLICATE_INPUTS_OR_OUTPUTS"))

        results = {}
        for out_blob in output_tensor_names:
            try:
                blob_data = workspace.FetchBlob(out_blob)
                if not hasattr(blob_data, 'decode'):
                    results[out_blob] = blob_data
                else:
                    self.logger.warn(get_warning_message("WARNING_FRAMEWORK_CAFFE2_ENCOUNTERED_BYTES_OBJECT")(out_blob))
            except RuntimeError:
                self.logger.warn(get_warning_message("WARNING_FRAMEWORK_CAFFE2_FAILED_TO_FETCH_BLOB")(out_blob))
        return results

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        """ Gets all the blobs in the model

        Traces the graph from the output tensor(s) to the input tensor(s).
        Uses DFS to traverse the network to return a list of ALL the blobs in the model.

        :param input_tensors: list of input blob names
        :param output_tensors: list of output blob names
        :return: list of tuples, each tuple represents an operator and contains two lists:
        the op's list of inputs, and the op's list of output blobs
        """

        # add input_tensor keys to self.input_tensors
        self.input_tensors = input_tensors.copy()

        visited, stack, relus = {}, [], {}
        blobs_to_find = list(input_tensors)
        only_relu = set(input_tensors) == set(output_tensors)  # case when only looking for one relu op

        # dictionary which stores all relus in model
        for index, op in enumerate(self.predict_net.op):
            if any(blob in op.input for blob in op.output):
                relus[index] = (op.input, op.output)

        for out_blob in output_tensors:
            stack.append((None, out_blob))  # (index_of_op_which_outputs_blob, blob_name)

        # depth-first search for graph traversal (output_tensors to input_tensors)
        while stack:
            curr_blob = stack.pop()
            blob_parent_op, blob_name = curr_blob

            # check if blob_name is found in a relu's output
            for index, blob_inputs_outputs in relus.items():
                if blob_name in blob_inputs_outputs[1]:
                    visited[index] = blob_inputs_outputs

            for index, op in enumerate(self.predict_net.op):
                if blob_name in op.output:

                    if only_relu:  # only looking for one relu op
                        blobs_to_find.clear()

                    if blob_parent_op is not None:
                        visited[blob_parent_op][0].append(blob_name)

                    if index in visited or blob_name in input_tensors:
                        break

                    visited[index] = ([], op.output)

                    for new_input in op.input:
                        stack.append((index, new_input))
                        if new_input in blobs_to_find:
                            blobs_to_find.remove(new_input)
                        if index == 0 and new_input in input_tensors:
                            visited[index][0].append(new_input)

                    break

        if blobs_to_find:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE2_NON-EXISTENT_BLOBS"))
        return list(visited.values())

    def get_dimensions(self, tensor_name):
        """ Returns the shape of the specified blob

        For get_dimensions() to successfully work, get_intermediate_tensors() must
        be called on the entire graph, followed by run_inference() which enters the
        model's input tensors.

        :param tensor_name: the name of the desired blob
        :return: Tuple of blob shape
        """

        if not self.initialized_predict_net:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE2_UNINITIALIZED_PREDICT_NET"))

        try:
            return workspace.FetchBlob(tensor_name).shape
        except RuntimeError:
            raise FrameworkError(get_message("ERROR_FRAMEWORK_CAFFE2_FAILED_TO_GET_DIMENSIONS")(tensor_name))

    def get_graph_structure(self):
        """ creates a detailed list of the network's operators

        Iterates through the operators in the net, and retrieves every
        operator's index , as well as its type, inputs, and outputs

        :return: dictionary indexed by op index with values containing the
        index, tuple of list of inputs and list of outputs
        """

        graph_structure = {}
        for i, op in enumerate(self.predict_net.op):
            graph_structure[i] = (i, op.type, op.input, op.output)

        return graph_structure

    def get_mapping_for_qnn_node(self, qnn_output):
        raise FrameworkError(get_message("ERROR_FRAMEWORK_TENSORFLOW_MISMATCH_TENSOR")(qnn_output))
        return None

    def get_version(self):
        """ returns Caffe2 version

        :return: version of framework as string
        """
        return Caffe2Framework_0_8_0.__VERSION__
