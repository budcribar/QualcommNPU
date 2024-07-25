# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_onnx_framework_1_3_0 import OnnxFramework_1_3_0
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message
from onnx import helper, shape_inference, version_converter
from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_onnx_extract import extract_model
import logging
import numpy as np
import os
import onnx


class OnnxFramework_1_8_0(OnnxFramework_1_3_0):
    __VERSION__ = '1.8.0'

    def __init__(self, logger, custom_op_lib=None):
        super(OnnxFramework_1_8_0, self).__init__(logger, custom_op_lib)

    def reorder_inputs(self, model):
        """This method returns the list of input node names of provided model
        in traversal order .

        Args :
            model : path to the model
        Returns:
            sorted_inputs : list of model input names
        """
        sorted_inputs = []
        M = onnx.load(model)
        initializers = [i.name for i in M.graph.initializer]
        graphInputs = [i.name for i in M.graph.input]

        if M.ir_version < 4:
            # In older onnx IR ,inputs also has initializers
            graphInputs = np.setdiff1d(graphInputs, initializers, assume_unique=True).tolist()
            sorted_inputs = graphInputs.copy()
        else:
            sorted_inputs = graphInputs.copy()
        return sorted_inputs

    def extract(self, start_layer_output_name, end_layer_output_name=None, out_model_path=None):
        """
        This method extracts the layers of the model from given start_layer to given end_layer
        and returns transformed model
        Args :
            start_layer_output_name : output name of partition start point
            end_layer_output_name   : output name of partition end point
            out_model_path : output extracted model path (default :
            partitions/extracted_<name>.onnx)

        Returns:
            status : status of sub model extraction
            transformed_model : path to transformed model
            new_g_inputpaths : list of inputs to extracted model
        """
        logging.debug('started model extraction')
        m = self._model
        assert start_layer_output_name or end_layer_output_name, 'Extract: start and end' \
                                                                 ' layer both cannot be None'
        # onnx shape inference
        try:
            m = onnx.shape_inference.infer_shapes(m)
        except Exception as e:
            logging.warning(e)

        # initializers are at graph level tensors - used for constants and graph inputs.
        # initializers wont contain the intermediate activations.
        initializers = [i.name for i in m.graph.initializer]
        graphInputs = [i.name for i in m.graph.input]

        if m.ir_version < 4:
            # In older onnx IR ,inputs also has initializers
            graphInputs = np.setdiff1d(graphInputs, initializers, assume_unique=True).tolist()
            original_inputs = graphInputs.copy()
        else:
            original_inputs = graphInputs.copy()

        # populate the traversed graph output names till start_layer_out_name into processed_outputs
        processed_outputs = []
        if start_layer_output_name:
            for node in m.graph.node:
                if start_layer_output_name in [str(n) for n in node.output]:
                    if start_layer_output_name not in processed_outputs:
                        processed_outputs.extend([str(n) for n in node.output])
                    break
                else:
                    processed_outputs.extend([str(n) for n in node.output])

        # Create mapping between each activation input node to connected node(s) output.
        mapping = {}
        for node in m.graph.node:
            # Find only the activation inputs for this node - not graph inputs or constants.
            ipnames = [ip for ip in node.input if ip not in initializers]
            for ip in ipnames:
                if ip in mapping:
                    mapping[ip].extend([str(n) for n in node.output])
                else:
                    mapping[ip] = [str(n) for n in node.output]

        # visit all nodes topologically if start node provided.
        new_g_inputs = []
        g_outputnames = [i.name for i in m.graph.output]
        opname_processed = False

        logging.debug('original model inputs before extract: ' + str(original_inputs))
        logging.debug('original model outputs before extract: ' + str(g_outputnames))

        if start_layer_output_name is None:
            # Find the new set of inputs to be preserved.
            for node in m.graph.node:
                ips = [ip for ip in node.input if ip in graphInputs]
                for i in ips:
                    if (i in original_inputs):
                        new_g_inputs.append(i)
                if end_layer_output_name in node.output:
                    break
            start_layer_output_name = 'top'
            logging.debug('preserved inputs:{}'.format(new_g_inputs))
        else:
            # Find the new set of inputs needed.
            for node in m.graph.node:

                node_outs = [str(n) for n in node.output]

                if start_layer_output_name in node.output:
                    # If start_layer_out_name encounters. New graph starts from this output node.
                    if start_layer_output_name not in new_g_inputs:
                        new_g_inputs.append(start_layer_output_name)
                    # remove all graph inputs for this node if any.
                    ips = [ip for ip in node.input if ip in graphInputs]
                    for i in ips:
                        if i in mapping:
                            for cn in mapping[i]:
                                if (cn in processed_outputs) and (i in original_inputs):
                                    logging.debug('Removing original input (post op): ' + i)
                                    original_inputs.remove(i)

                    # for any of the inputs (not graph inputs) to this node which are
                    # also used later -need to preserve them.
                    ips = [str(ip) for ip in node.input if ip not in initializers]
                    for ip in ips:
                        if ip in mapping:
                            for cn in mapping[ip]:
                                if (cn not in processed_outputs) and (ip not in new_g_inputs):
                                    new_g_inputs.append(ip)
                                    break
                    opname_processed = True

                elif opname_processed:
                    # for all nodes after start layer
                    # check if any required inputs and preserve if not in original inputs.
                    ips = [ip for ip in node.input if ip in graphInputs]
                    for i in ips:
                        if i in mapping:
                            for cn in mapping[i]:
                                if (i not in original_inputs) and (i not in new_g_inputs):
                                    new_g_inputs.append(i)

                    # if any node input is already processed, use it directly.
                    ips = [ip for ip in node.input if ip not in initializers]
                    for i in ips:
                        if i in mapping:
                            for cn in mapping[i]:
                                if (i in processed_outputs) and (i not in new_g_inputs):
                                    new_g_inputs.append(i)

                else:
                    # for all nodes before start_layer_output_name
                    # confirm all outputs processed for this node.
                    pnames = list(set(node_outs).intersection(processed_outputs))

                    if len(pnames) == 0:  # not processed
                        logging.error(
                            'Topological Node : {} not processed before reaching current op '
                            ''.format(node.name))
                        raise ce.ModelTransformationException(
                            'Topological Node : {} not processed before reaching current op '
                            ''.format(node.name))
                    else:
                        # remove graph inputs for this node unless it is used by some later
                        # unprocessed node
                        ips = [ip for ip in node.input if ip in graphInputs]
                        for i in ips:
                            for cn in mapping[i]:
                                if (cn in processed_outputs) and (i in original_inputs):
                                    logging.debug('Removing original input (pre op): ' + i)
                                    original_inputs.remove(i)

                        # if node input activations used later by unprocessed nodes, they need to be
                        # preserved.
                        ips = [str(ip) for ip in node.input if ip not in initializers]
                        for ip in ips:
                            for cn in mapping[ip]:
                                if (cn not in processed_outputs) and (ip not in new_g_inputs):
                                    new_g_inputs.append(ip)
                                    break
                if end_layer_output_name in node.output:
                    break

            new_g_inputs = list(set(new_g_inputs))
            original_inputs = list(set(original_inputs))
            new_g_inputs.extend(original_inputs)

        # Remove any duplicates.
        new_g_inputs = list(set(new_g_inputs))

        # Path of the partitioned model.
        # if not out_model_path:
        part_model_path = os.path.join(out_model_path, "extracted_model.onnx")

        # new outputs for the sub-model are formed
        all_outputs = g_outputnames.copy()

        if end_layer_output_name is None:
            new_g_outputs = set(all_outputs) - set(processed_outputs)
        else:
            # populate the traversed graph outputnames after end_layer_out_name into
            # outputs_after_end_layer
            all_outputs.append(end_layer_output_name)
            reached_end_layer = False
            outputs_after_end_layer = []
            for node in m.graph.node:
                if reached_end_layer:
                    outputs_after_end_layer.extend([str(n) for n in node.output])
                    continue
                if end_layer_output_name in [str(n) for n in node.output]:
                    reached_end_layer = True
            new_g_outputs = list(set(all_outputs) - set(outputs_after_end_layer))

        # the new inputs/output shape must be fixed if needed.
        inout = []
        inout.extend(new_g_inputs)
        inout.extend(new_g_outputs)
        # if self._fix_shape_of_req_nodes(model_path, inout):
        #     return (False, None, None)

        # extraction of submodel from given onnx model
        try:
            if m.ir_version < 4:
                # initializers are also inputs for previous onnx IR
                new_g_inputs_temp = new_g_inputs.copy()
                new_g_inputs.extend(initializers)
            logging.debug('{} : new graph inputs: {}, new graph outputs {}'.format(
                part_model_path, str(new_g_inputs), str(new_g_outputs)))
            extract_model(m, part_model_path, new_g_inputs, new_g_outputs)
            if m.ir_version < 4:
                # remove unused initializer inputs.
                m = onnx.load(part_model_path)
                m_initializers = [i.name for i in m.graph.initializer]
                m_graphInputs = [i.name for i in m.graph.input]
                diff = np.setdiff1d(m_graphInputs, m_initializers, assume_unique=True).tolist()
                vi_list = []
                for ip in np.setdiff1d(diff, new_g_inputs_temp).tolist():
                    vi_list.extend([vi for vi in m.graph.input if vi.name == ip])
                for vi in vi_list:
                    m.graph.input.remove(vi)
                onnx.save(m, part_model_path)
                # always return the actual inputs and not initializers.
                new_g_inputs = new_g_inputs_temp

        except Exception as e:
            logging.error('Extraction of model failed. start {}, end {}'.format(
                start_layer_output_name, end_layer_output_name))
            logging.error('new inputs: {}  outputs {}'.format(new_g_inputs, new_g_outputs))
            logging.exception(e)
            return (False, None, None)
        all_inputs = []  # Removing dangling inputs
        part_model = onnx.load(part_model_path)
        [all_inputs.extend(node.input) for node in part_model.graph.node]
        dangling_inputs = [inp for inp in part_model.graph.input if inp.name not in all_inputs]
        [part_model.graph.input.remove(dinp) for dinp in dangling_inputs]
        new_g_inputs = [inp for inp in new_g_inputs if inp not in [i.name for i in dangling_inputs]]
        onnx.save(part_model, part_model_path)

        # self._populate_dynamic_input_shape(part_model_path, new_g_inputs)

        # if m.ir_version > 4:
        #     new_g_inputs = self._fix_reshape_nodes(part_model_path)

        # get graph inputs of extracted model in model traversal sequence
        new_g_inputs = self.reorder_inputs(part_model_path)
        logging.debug('OnnxModelTransformation ::extract() success')
        return (True, part_model_path, new_g_inputs)
