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
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message


def getDataTypeBasedOnBitWidth(bitWidth):
    dataType = np.uint8
    if bitWidth == 16:
        dataType = np.uint16
    elif bitWidth == 32:
        dataType = np.uint32
    return dataType


class QnnExtractor(BaseExtractor):

    def __init__(self, args, quant_schemes_dir_map, logger=None):
        super().__init__(args, quant_schemes_dir_map, logger)

    def extract(self):
        # For each quant_scheme, extract the model weights, biases, and corresponding encodings into main memory
        self._extractQnn()
        self._logger.info(get_progress_message('Extracting input file names and input file data.'))
        self._extract_input_data()
        # For each input file, extract the intermediate op activations and load them in main memory
        self._extractQnnActivations()

    def _extractQnnActivations(self):
        '''
        For each quant_scheme in all possible quant schemes:
            for each op in the model:
                if quant_scheme is unquantized:
                    load the fp32 op activation
                    op.setActivation(fp32_activatio)
                save the op's activation's scale and offset
        '''
        for quant_scheme, path in self._quant_schemes_dir_map.items():
            if quant_scheme not in self._opMap:
                self._logger.info(
                    get_message('Quantization Scheme {} not in consideration due to failure'.format(
                        quant_scheme)))
                continue
            with open(os.path.join(path, quant_scheme + '_net.json')) as f:
                modelMeta = json.load(f, object_pairs_hook=OrderedDict)
            for op_name, op in self._opMap[quant_scheme].items():
                activationNodeName = op.getActivationNodeName()
                raw_file_name = activationNodeName
                if activationNodeName is None:
                    continue
                if quant_scheme == 'unquantized':
                    activationPath = self._args.golden_output_reference_directory
                    with os.scandir(activationPath) as allResults:
                        activationList = []
                        for resultDir in allResults:
                            if resultDir.is_dir() and resultDir.name != 'latest':
                                activationFile = os.path.join(activationPath, resultDir.name,
                                                              raw_file_name + '.raw')
                                if os.path.exists(activationFile) and os.path.isfile(
                                        activationFile):
                                    activationList.append(
                                        (resultDir.name,
                                         np.fromfile(activationFile, dtype='float32')))
                        op.setActivations(activationList)
                op.setActivationScale(modelMeta['graph']['tensors'][activationNodeName]
                                      ['quant_params']['scale_offset']['scale'])
                op.setActivationOffset(modelMeta['graph']['tensors'][activationNodeName]
                                       ['quant_params']['scale_offset']['offset'])
                if op.getInputNodeName() is not None:
                    op.setInputNodeScale(modelMeta['graph']['tensors'][op.getInputNodeName()]
                                         ['quant_params']['scale_offset']['scale'])
                self._opMap[quant_scheme][op_name] = op

    def _extractQnn(self):
        self._logger.info(get_progress_message('Unpacking weights and biases from bin files.'))
        self._unpackWeightsAndBiasesFiles()
        self._parseAllOpsFromJson()
        self._logger.info(
            get_progress_message('Extracting weights and biases data from raw files.'))
        self._extractQnnWeights()
        self._extractQnnBiases()

    def _extractQnnWeights(self):
        '''
        for each quant_scheme in all possible quant schemes:
            for each op in the model:
                if op is weight node:
                    get it's quantization encodings from net.json
                    save it's weights, dims, scale, offset
        '''
        for quant_scheme, path in self._quant_schemes_dir_map.items():
            if quant_scheme not in self._opMap:
                self._logger.info(
                    get_message('Quantization Scheme {} not in consideration due to failure'.format(
                        quant_scheme)))
                continue
            with open(os.path.join(path, quant_scheme + '_net.json')) as f:
                modelMeta = json.load(f, object_pairs_hook=OrderedDict)
            for op_name, op in self._opMap[quant_scheme].items():
                if op.getWeightName() not in (None, ''):
                    weightName = op.getWeightName()
                    dtype = None
                    quantEncoding = modelMeta['graph']['tensors'][weightName]['quant_params'][
                        'encoding']
                    op.setIsQuantizedPerChannel(quantEncoding)
                    if 'dims' in modelMeta['graph']['tensors'][weightName]:
                        op.setWeightsDims(modelMeta['graph']['tensors'][weightName]['dims'])
                    elif 'current_dims' in modelMeta['graph']['tensors'][weightName]:
                        op.setWeightsDims(modelMeta['graph']['tensors'][weightName]['current_dims'])
                    else:
                        self.logger.info(
                            get_message(
                                'Extracting weight values failed due to keyError while retrieving weight dimension.'
                            ))
                        exit(-1)
                    # quantization encoding=0 for non-pcq weights
                    if quantEncoding == 0:
                        op.setWeightsScaleOffset(modelMeta['graph']['tensors'][weightName]
                                                 ['quant_params']['scale_offset'])
                        dtype = 'uint8'
                    # quantization encoding=1 for pcq weights
                    elif quantEncoding == 1:
                        op.setWeightsScaleOffset(modelMeta['graph']['tensors'][weightName]
                                                 ['quant_params']['axis_scale_offset'])
                        dtype = 'int8'
                    try:
                        if quant_scheme == 'unquantized':
                            op.setWeights(
                                np.fromfile(os.path.join(path, weightName + '.raw'), dtype='float32'))
                        else:
                            op.setWeights(
                                np.fromfile(os.path.join(path, weightName + '.raw'), dtype=dtype))
                    except Exception:
                        self._logger.info(
                            get_message('No tensor dump for {} was not found in bin file extract for quant_scheme {}'.format(
                                weightName, quant_scheme)))
                self._opMap[quant_scheme][op_name] = op

    def _extractQnnBiases(self):
        '''
        for each quant_scheme in all possible quant schemes:
            for each op in the model:
                if op has bias node:
                    get it's quantization encodings from net.json
                    save it's weights, dims, scale, offset
        '''
        for quant_scheme, path in self._quant_schemes_dir_map.items():
            if quant_scheme not in self._opMap:
                self._logger.info(
                    get_message('Quantization Scheme {} not in consideration due to failure'.format(
                        quant_scheme)))
                continue
            with open(os.path.join(path, quant_scheme + '_net.json')) as f:
                modelMeta = json.load(f, object_pairs_hook=OrderedDict)
            for op_name, op in self._opMap[quant_scheme].items():
                if op.getBiasName() not in (None, ''):
                    biasName = op.getBiasName()
                    op.setBiasScale(modelMeta['graph']['tensors'][biasName]['quant_params']
                                    ['scale_offset']['scale'])
                    op.setBiasOffset(modelMeta['graph']['tensors'][biasName]['quant_params']
                                     ['scale_offset']['offset'])
                    dataType = modelMeta['graph']['tensors'][biasName]['data_type']
                    # TODO: check if bias raw file exists and report if not...
                    biasRawFilePath = os.path.join(path, biasName + '.raw')
                    if not os.path.exists(biasRawFilePath):
                        continue
                    if quant_scheme == 'unquantized':
                        op.setBiases(np.fromfile(biasRawFilePath, dtype='float32'))
                    elif dataType == Op.getUint8QnnCode():
                        op.setBiases(np.fromfile(biasRawFilePath, dtype='uint8'))
                    else:
                        op.setBiases(np.fromfile(biasRawFilePath, dtype='int32'))
                self._opMap[quant_scheme][op_name] = op

    def _unpackWeightsAndBiasesFiles(self):
        for quant_scheme, path in self._quant_schemes_dir_map.items():
            fileToExtract = os.path.join(path, quant_scheme + '.bin')
            # untar the bin file
            binFile = tarfile.open(fileToExtract, 'r')
            extractDir = os.path.dirname(fileToExtract)
            binFile.extractall(extractDir)

    def _parseAllOpsFromJson(self):
        for quant_scheme, path in self._quant_schemes_dir_map.items():
            opsInfo = self._getOpsFromJsonForQuantizationVariation(quant_scheme, path)
            if opsInfo is not None:
                self._opMap[quant_scheme] = opsInfo

    def _getOpsFromJsonForQuantizationVariation(self, quant_scheme, path):
        jsonFilePath = os.path.join(path, quant_scheme + '_net.json')
        if not os.path.exists(jsonFilePath):
            return
        with open(jsonFilePath) as f:
            modelMeta = json.load(f, object_pairs_hook=OrderedDict)
        return self._parseOpDataFromJsonMeta(modelMeta)

    def _parseOpDataFromJsonMeta(self, modelMeta):
        nodes = modelMeta['graph']['nodes']
        opMap = {}
        for node in nodes.keys():
            op = Op(node)
            activationNodeName = nodes[node]['output_names'][0]
            op.setActivationNodeName(activationNodeName)
            if nodes[node]['input_names']:
                inputNames = nodes[node]['input_names']
                if nodes[node]['type'] == 'LSTM':
                    itr = 0
                    for inputName in inputNames:
                        if Op.isLSTMBias(itr):
                            op.setBiasName(inputName)
                        else:
                            op.setWeightName(inputName)
                        itr += 1
                elif nodes[node]['type'] in Op.getOpTypesWithWeightsBiases():
                    op.setInputNodeName(inputNames[0])
                    op.setWeightName(inputNames[1])
                    op.setBiasName(inputNames[2])
            opMap[node] = op
        return opMap
