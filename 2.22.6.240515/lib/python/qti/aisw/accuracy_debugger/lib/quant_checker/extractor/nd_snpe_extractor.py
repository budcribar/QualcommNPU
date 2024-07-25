# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import re
import json
import numpy as np
import tarfile
from collections import OrderedDict

from qti.aisw.accuracy_debugger.lib.quant_checker.nd_op import Op
from .nd_base_extractor import BaseExtractor
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
try:
    from qti.aisw.dlc_utils.snpe_dlc_utils import OpRow, ModelInfo
except:
    pass
from qti.aisw.converters.common import ir_graph


def getDataTypeBasedOnBitWidth(bitWidth):
    dataType = np.uint8
    if bitWidth == 16:
        dataType = np.uint16
    elif bitWidth == 32:
        dataType = np.uint32
    return dataType


class SnpeExtractor(BaseExtractor):

    def __init__(self, args, quant_schemes_dir_map, logger=None):
        super().__init__(args, quant_schemes_dir_map, logger)

    def extract(self):
        self._logger.info(get_message('Extracting weights and biases from dlc.'))
        self._parseOpsFromDlc()
        self._extract_input_data()
        self._extractSnpeActivations()

    def _parseOpsFromDlc(self):
        for quant_scheme, path in self._quant_schemes_dir_map.items():
            opsInfo = self._getOpsFromDlcForQuantizationVariation(quant_scheme, path)
            if opsInfo is not None:
                self._opMap[quant_scheme] = opsInfo

    def _getOpsFromDlcForQuantizationVariation(self, quant_scheme, path):
        dlc_file_path = os.path.join(path, quant_scheme + '.dlc')
        if not os.path.exists(dlc_file_path):
            return
        return self._parseOpDataFromDlc(self._loadSnpeModel(dlc_file_path), quant_scheme)

    def _loadSnpeModel(self, dlc_file_path):
        self._logger.info(get_message('Loading the following model: ' + dlc_file_path))
        model = ModelInfo()
        model.load(dlc_file_path)
        return model

    def _parseOpDataFromDlc(self, model, quant_scheme):
        opMap = {}
        graph = model.model_reader.get_ir_graph()
        nodes = graph.get_ops()
        for node in nodes:
            layer = OpRow(node, [])
            if layer.type == 'data':
                continue
            op = Op(layer.name)
            op.setActivationNodeName(node.outputs()[0].name())
            if quant_scheme != 'unquantized':
                activation_encoding = node.outputs()[0].get_encoding().encInfo
                op.setActivationScale(activation_encoding.scale)
                op.setActivationOffset(activation_encoding.offset)
                Op.setActivationWidth(activation_encoding.bw)
            if layer.get_input_list():
                input_names = layer.get_input_list()
                if layer.type.upper() in (type.upper()
                                          for type in Op.getOpTypesWithWeightsBiases()):
                    op.setInputNodeName(input_names[0])
                    op.setWeightName(layer.name + '_weight')
                    op.setBiasName(layer.name + '_bias')
                    op.setNode(node)
                    self._extractSnpeWeights(op, quant_scheme)
                    self._extractSnpeBiases(op, quant_scheme)
            opMap[layer.name] = op
        return opMap

    def _extractSnpeWeights(self, op, quant_scheme):
        if op.getWeightName() not in (None, ''):
            weights = op.getNode().inputs()[1]
            dataType = np.uint8
            if quant_scheme != 'unquantized':
                quant_encoding = weights.get_encoding().encInfo
                # quantEncoding format:
                # bw, min, max, scale, offset : uint32_t, float, float, float, int32_t
                op.setIsQuantizedPerChannel(False)
                weights_scale_offset = {}
                weights_scale_offset['scale'] = quant_encoding.scale
                weights_scale_offset['offset'] = quant_encoding.offset
                op.setWeightsScaleOffset(weights_scale_offset)
                bit_width = quant_encoding.bw
                dataType = getDataTypeBasedOnBitWidth(bit_width)
                Op.setWeightWidth(str(bit_width))
            else:
                Op.setWeightWidth('32')
                dataType = np.float32
            try:
                weights_data = ir_graph.PyIrStaticTensor(weights)
                op.setWeights(np.frombuffer(weights_data.data().flatten(), dtype=dataType))
                op.setWeightsDims(weights_data.data().shape)
            except Exception as e:
                self._logger.info("Weight Tensor {} is of type PyIrTensor".format(
                    op.getWeightName()))

    def _extractSnpeBiases(self, op, quant_scheme):
        if op.getBiasName() not in (None, ''):
            dataType = np.uint8
            biases = op.getNode().inputs()[2]
            if quant_scheme != 'unquantized':
                quant_encoding = biases.get_encoding().encInfo
                # quantEncoding format:
                # bw, min, max, scale, offset : uint32_t, float, float, float, int32_t
                op.setBiasScale(quant_encoding.scale)
                op.setBiasOffset(quant_encoding.offset)
                bit_width = quant_encoding.bw
                dataType = getDataTypeBasedOnBitWidth(bit_width)
                Op.setBiasWidth(str(bit_width))
            else:
                Op.setBiasWidth('32')
                dataType = np.float32
            biases_data = ir_graph.PyIrStaticTensor(biases)
            op.setBiases(np.frombuffer(biases_data.data().flatten(), dtype=dataType))

    def _extractSnpeActivations(self):
        for quant_scheme in self._quant_schemes_dir_map:
            # skip quantized models which are failed to get converted correctly
            if quant_scheme not in self._opMap:
                continue

            for op_name, op in self._opMap[quant_scheme].items():
                activation_node_name = op.getActivationNodeName()
                if activation_node_name is None:
                    continue
                raw_file_name = activation_node_name
                if quant_scheme == 'unquantized':
                    activation_path = self._args.golden_output_reference_directory
                    with os.scandir(activation_path) as all_results:
                        activation_list = []
                        for result_dir in all_results:
                            if result_dir.is_dir() and result_dir.name != 'latest':
                                activation_file = os.path.join(activation_path, result_dir.name,
                                                               raw_file_name + '.raw')
                                if os.path.exists(activation_file) and os.path.isfile(
                                        activation_file):
                                    activation_list.append(
                                        (result_dir.name,
                                         np.fromfile(activation_file, dtype='float32')))
                        op.setActivations(activation_list)
                self._opMap[quant_scheme][op_name] = op
