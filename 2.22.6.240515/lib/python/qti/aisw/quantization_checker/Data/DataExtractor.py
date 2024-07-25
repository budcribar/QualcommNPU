#=============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import re
import json
import numpy as np
import tarfile
from collections import OrderedDict
from qti.aisw.converters.common import ir_graph
from qti.aisw.quantization_checker.utils.Op import Op
try:
    from qti.aisw.dlc_utils.snpe_dlc_utils import OpRow, ModelInfo
except ImportError:
    # do nothing
    pass
from qti.aisw.quantization_checker.utils.Logger import PrintOptions
import qti.aisw.quantization_checker.utils.Constants as Constants

def getDataTypeBasedOnBitWidth(bitWidth):
    dataType = np.uint8
    if bitWidth == 16:
        dataType = np.uint16
    elif bitWidth == 32:
        dataType = np.uint32
    return dataType

class DataExtractor:
    def __init__(self, sdkType, quantizationVariations, inputNetwork, inputList, outputDir, logger):
        self.sdkType = sdkType
        self.quantizationVariations = quantizationVariations
        self.inputNetwork = inputNetwork
        self.inputList = inputList
        self.inputFileNames = []
        self.outputDir = outputDir
        self.opMap = {}
        self.inputData = {}
        self.logger = logger

    def getAllOps(self):
        return self.opMap

    def __extractSnpeWeights(self, op, quantizationVariation):
        if op.getWeightName() not in (None, ''):
            weights = op.getNode().inputs()[1]
            dataType = np.uint8
            if quantizationVariation != Constants.UNQUANTIZED:
                quantEncoding = weights.get_encoding().encInfo
                # quantEncoding format:
                # bw, min, max, scale, offset : uint32_t, float, float, float, int32_t
                op.setIsQuantizedPerChannel(False)
                weightsScaleOffset = {}
                weightsScaleOffset['scale'] = quantEncoding.scale
                weightsScaleOffset['offset'] = quantEncoding.offset
                op.setWeightsScaleOffset(weightsScaleOffset)
                bitWidth = quantEncoding.bw
                dataType = getDataTypeBasedOnBitWidth(bitWidth)
                Op.setWeightWidth(str(bitWidth))
            else:
                Op.setWeightWidth('32')
                dataType = np.float32
            weightsData = ir_graph.PyIrStaticTensor(weights)
            op.setWeights(np.frombuffer(weightsData.data().flatten(), dtype=dataType))
            op.setWeightsDims(weightsData.data().shape)

    def __extractSnpeBiases(self, op, quantizationVariation):
        if op.getBiasName() not in (None, ''):
            dataType = np.uint8
            biases = op.getNode().inputs()[2]
            if quantizationVariation != Constants.UNQUANTIZED:
                quantEncoding = biases.get_encoding().encInfo
                # quantEncoding format:
                # bw, min, max, scale, offset : uint32_t, float, float, float, int32_t
                op.setBiasScale(quantEncoding.scale)
                op.setBiasOffset(quantEncoding.offset)
                bitWidth = quantEncoding.bw
                dataType = getDataTypeBasedOnBitWidth(bitWidth)
                Op.setBiasWidth(str(bitWidth))
            else:
                Op.setBiasWidth('32')
                dataType = np.float32
            biasesData = ir_graph.PyIrStaticTensor(biases)
            op.setBiases(np.frombuffer(biasesData.data().flatten(), dtype=dataType))

    def extractActivations(self):
        if self.sdkType.upper() == Constants.SNPE:
            self.__extractSnpeActivations()
        else:
            self.__extractQnnActivations()

    def __extractQnnActivations(self):
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.opMap:
                continue
            with open(os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '_net.json')) as f:
                modelMeta = json.load(f, object_pairs_hook=OrderedDict)
            for item in self.opMap[quantizationVariation].items():
                op = self.opMap[quantizationVariation][item[0]]
                activationNodeName = op.getActivationNodeName()
                if activationNodeName is None:
                    continue
                if quantizationVariation == Constants.UNQUANTIZED:
                    activationPath = os.path.join(self.outputDir, Constants.NET_RUN_OUTPUT_DIR, Constants.UNQUANTIZED)
                    resultCount = 0
                    with os.scandir(activationPath) as allResults:
                        activationList = []
                        for resultDir in allResults:
                            if resultDir.is_dir():
                                activationFile = os.path.join(activationPath, resultDir.name, activationNodeName + '.raw')
                                if os.path.exists(activationFile) and os.path.isfile(activationFile):
                                    activationList.append((self.inputFileNames[resultCount], np.fromfile(activationFile, dtype='float32')))
                                    resultCount += 1
                        op.setActivations(activationList)
                op.setActivationScale(modelMeta['graph']['tensors'][activationNodeName]['quant_params']['scale_offset']['scale'])
                op.setActivationOffset(modelMeta['graph']['tensors'][activationNodeName]['quant_params']['scale_offset']['offset'])
                if op.getInputNodeName() is not None:
                    op.setInputNodeScale(modelMeta['graph']['tensors'][op.getInputNodeName()]['quant_params']['scale_offset']['scale'])
                self.opMap[quantizationVariation][item[0]] = op

    def __extractSnpeActivations(self):
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.opMap:
                continue

            for key in self.opMap[quantizationVariation].keys():
                op = self.opMap[quantizationVariation][key]
                activationNodeName = op.getActivationNodeName()
                if activationNodeName is None:
                    continue
                if quantizationVariation == Constants.UNQUANTIZED:
                    activationPath = os.path.join(self.outputDir, Constants.NET_RUN_OUTPUT_DIR, Constants.UNQUANTIZED)
                    resultCount = 0
                    with os.scandir(activationPath) as items:
                        activationList = []
                        for entry in items:
                            if entry.is_dir():
                                for root, _, files in os.walk(entry):
                                    for file in files:
                                        if str(os.path.join(root, file)) == str(os.path.join(activationPath, entry.name, activationNodeName + '.raw')):
                                            activationList.append((self.inputFileNames[resultCount], np.fromfile(os.path.join(root, file), dtype='float32')))
                                resultCount += 1
                        op.setActivations(activationList)
                self.opMap[quantizationVariation][key] = op

    def extract(self):
        if self.sdkType.upper() == Constants.SNPE:
            self.__extractSnpe()
        else:
            self.__extractQnn()
        self.logger.print('Extracting input file names and input file data.')
        self.__extractInputData()

    def __extractSnpe(self):
        self.logger.print('Extracting weights and biases from dlc.')
        self.__parseOpsFromDlc()

    def __extractQnn(self):
        self.logger.print('Unpacking weights and biases from bin files.')
        self.__unpackWeightsAndBiasesFiles()
        self.__parseAllOpsFromJson()
        self.logger.print('Extracting weights and biases data from raw files.')
        self.__extractQnnWeights()
        self.__extractQnnBiases()

    def __extractQnnWeights(self):
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.opMap:
                continue
            with open(os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '_net.json')) as f:
                modelMeta = json.load(f, object_pairs_hook=OrderedDict)
            for item in self.opMap[quantizationVariation].items():
                op = self.opMap[quantizationVariation][item[0]]
                if op.getWeightName() not in (None, ''):
                    weightName = op.getWeightName()
                    dtype = None
                    quantEncoding = modelMeta['graph']['tensors'][weightName]['quant_params']['encoding']
                    op.setIsQuantizedPerChannel(quantEncoding)
                    if 'dims' in modelMeta['graph']['tensors'][weightName]:
                        op.setWeightsDims(modelMeta['graph']['tensors'][weightName]['dims'])
                    elif 'current_dims' in modelMeta['graph']['tensors'][weightName]:
                        op.setWeightsDims(modelMeta['graph']['tensors'][weightName]['current_dims'])
                    else:
                        self.logger.print('Extracting weight values failed due to keyError while retrieving weight dimension.')
                        exit(-1)
                    # quantization encoding=0 for non-pcq weights
                    if quantEncoding == 0:
                        op.setWeightsScaleOffset(modelMeta['graph']['tensors'][weightName]['quant_params']['scale_offset'])
                        dtype = 'uint8'
                    # quantization encoding=1 for pcq weights
                    elif quantEncoding == 1:
                        op.setWeightsScaleOffset(modelMeta['graph']['tensors'][weightName]['quant_params']['axis_scale_offset'])
                        dtype = 'int8'
                    if quantizationVariation == 'unquantized':
                        op.setWeights(np.fromfile(os.path.join(self.outputDir, quantizationVariation, weightName + '.raw'), dtype='float32'))
                    else:
                        op.setWeights(np.fromfile(os.path.join(self.outputDir, quantizationVariation, weightName + '.raw'), dtype=dtype))
                self.opMap[quantizationVariation][item[0]] = op

    def __extractQnnBiases(self):
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.opMap:
                continue
            with open(os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '_net.json')) as f:
                modelMeta = json.load(f, object_pairs_hook=OrderedDict)
            for item in self.opMap[quantizationVariation].items():
                op = self.opMap[quantizationVariation][item[0]]
                if op.getBiasName() not in (None, ''):
                    biasName = op.getBiasName()
                    op.setBiasScale(modelMeta['graph']['tensors'][biasName]['quant_params']['scale_offset']['scale'])
                    op.setBiasOffset(modelMeta['graph']['tensors'][biasName]['quant_params']['scale_offset']['offset'])
                    dataType = modelMeta['graph']['tensors'][biasName]['data_type']
                    # TODO: check if bias raw file exists and report if not...
                    biasRawFilePath = os.path.join(self.outputDir, quantizationVariation, biasName + '.raw')
                    if not os.path.exists(biasRawFilePath):
                        continue
                    if quantizationVariation == 'unquantized':
                        op.setBiases(np.fromfile(biasRawFilePath, dtype='float32'))
                    elif dataType == Op.getUint8QnnCode():
                        op.setBiases(np.fromfile(biasRawFilePath, dtype='uint8'))
                    else:
                        op.setBiases(np.fromfile(biasRawFilePath, dtype='int32'))
                self.opMap[quantizationVariation][item[0]] = op

    def __unpackWeightsAndBiasesFiles(self):
        for quantizationVariation in self.quantizationVariations:
            fileToExtract = os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '.bin')
            if not os.path.exists(fileToExtract):
                continue
            # untar the bin file
            binFile = tarfile.open(fileToExtract, 'r')
            extractDir = os.path.dirname(fileToExtract)
            binFile.extractall(extractDir)

    def __parseAllOpsFromJson(self):
        for quantizationVariation in self.quantizationVariations:
            opsInfo = self.__getOpsFromJsonForQuantizationVariation(quantizationVariation)
            if opsInfo is not None:
                self.opMap[quantizationVariation] = opsInfo

    def __getOpsFromJsonForQuantizationVariation(self, quantizationVariation):
        jsonFilePath = os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '_net.json')
        if not os.path.exists(jsonFilePath):
            return
        with open(jsonFilePath) as f:
            modelMeta = json.load(f, object_pairs_hook=OrderedDict)
        return self.__parseOpDataFromJsonMeta(modelMeta)

    def __parseOpDataFromJsonMeta(self, modelMeta):
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

    def __parseOpsFromDlc(self):
        for quantizationVariation in self.quantizationVariations:
            opsInfo = self.__getOpsFromDlcForQuantizationVariation(quantizationVariation)
            if opsInfo is not None:
                self.opMap[quantizationVariation] = opsInfo

    def __getOpsFromDlcForQuantizationVariation(self, quantizationVariation):
        dlcFilePath = os.path.join(self.outputDir, quantizationVariation, quantizationVariation + '.dlc')
        if not os.path.exists(dlcFilePath):
            return
        return self.__parseOpDataFromDlc(self.__loadSnpeModel(dlcFilePath), quantizationVariation)

    def __loadSnpeModel(self, dlcFilePath):
        self.logger.print('Loading the following model: ' + dlcFilePath)
        model = ModelInfo()
        model.load(dlcFilePath)
        return model

    def __parseOpDataFromDlc(self, model, quantizationVariation):
        opMap = {}
        graph = model.model_reader.get_ir_graph()
        nodes = graph.get_ops()
        for node in nodes:
            layer = OpRow(node, [])
            if layer.type == 'data':
                continue
            op = Op(layer.name)
            op.setActivationNodeName(node.outputs()[0].name())
            if quantizationVariation != Constants.UNQUANTIZED:
                activationEncoding = node.outputs()[0].get_encoding().encInfo
                op.setActivationScale(activationEncoding.scale)
                op.setActivationOffset(activationEncoding.offset)
                Op.setActivationWidth(activationEncoding.bw)
            if layer.get_input_list():
                inputNames = layer.get_input_list()
                if layer.type.upper() in (type.upper() for type in Op.getOpTypesWithWeightsBiases()):
                    op.setInputNodeName(inputNames[0])
                    op.setWeightName(layer.name + '_weight')
                    op.setBiasName(layer.name + '_bias')
                    op.setNode(node)
                    self.__extractSnpeWeights(op, quantizationVariation)
                    self.__extractSnpeBiases(op, quantizationVariation)
            opMap[layer.name] = op
        return opMap

    def __extractInputData(self):
        try:
            with open(self.inputList) as file:
                inputDirPath = os.path.dirname(self.inputNetwork)
                inputFileNames = file.readlines()
                for line in inputFileNames:
                    filenames = line.rstrip()
                    for file in filenames.split():
                        if file:
                            file = re.split('=|:', file)[-1]
                            if not os.path.exists(os.path.join(inputDirPath, file)):
                                self.logger.print('The following file from the input list (' + file + ') could not be found. Exiting...')
                                exit(-1)
                            self.inputFileNames.append(file)
                            self.inputData[file] = np.fromfile(os.path.join(inputDirPath, file), dtype=np.float32)
        except Exception as e:
            self.logger.print("Unable to open input list file, please check the file path! Exiting...")
            self.logger.print(e, PrintOptions.LOGFILE)
            exit(-1)

    def getInputFiles(self):
        return self.inputFileNames

    def getInputData(self):
        return self.inputData
