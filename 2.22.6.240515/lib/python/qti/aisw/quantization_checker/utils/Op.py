#=============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import numpy as np

class Op:
    biasWidth = str(np.iinfo(np.uint8).bits)
    weightWidth = str(np.iinfo(np.uint8).bits)
    activationWidth = str(np.iinfo(np.uint8).bits)
    uint8QnnCode = 1032
    def __init__(self, opName, weightName=None, weights=[], weightsDims=[], isQuantizedPerChannel=0, weightsScaleOffset=None, biasName=None, biases=[], biasScale=None, biasOffset=None, activations=[], activationScale=None, activationOffset=None, activationNodeName=None, inputNodeName=None, inputNodeScale=None):
        self.opName = opName
        self.weightName = weightName
        self.weights = weights
        self.weightsDims = weightsDims
        self.isQuantizedPerChannel = bool(isQuantizedPerChannel)
        self.weightsScaleOffset = weightsScaleOffset
        self.dequantizedWeights = []
        self.biasName = biasName
        self.biases = biases
        self.dequantizedBiases = []
        self.biasScale = biasScale
        self.biasOffset = biasOffset
        self.activations = activations
        self.dequantizedActivations = {}
        self.activationScale = activationScale
        self.activationOffset = activationOffset
        self.activationNodeName = activationNodeName
        self.inputNodeName = inputNodeName
        self.inputNodeScale = inputNodeScale
        self.node = None

    @staticmethod
    def getOpTypesWithWeightsBiases():
        return [
            "Conv2d",
            "Conv3d",
            "Batchnorm",
            "FullyConnected",
            "DepthWiseConv2d",
            "TransposeConv2d",
            "TransposeConv3d",
            "InstanceNorm",
            "LayerNorm",
            "LSTM",
            "Convolutional"
        ]

    @staticmethod
    def isLSTMBias(index : int):
        # LSTM has 24 input params, if the input index is between 7 and 9, 21 or 23 we consider it as bias otherwise weight
        if (index >= 7 and index <= 9) or index == 21 or index == 23:
            return True
        return False

    def setNode(self, node):
        self.node = node

    def getNode(self):
        return self.node

    @staticmethod
    def setActivationWidth(width):
        Op.activationWidth = width

    @staticmethod
    def getActivationWidth():
        return Op.activationWidth

    @staticmethod
    def setWeightWidth(width):
        Op.weightWidth = width

    @staticmethod
    def getWeightWidth():
        return Op.weightWidth

    @staticmethod
    def setBiasWidth(width):
        Op.biasWidth = width

    @staticmethod
    def getBiasWidth():
        return Op.biasWidth

    @staticmethod
    def getUint8QnnCode():
        return Op.uint8QnnCode

    def setInputNodeName(self, inputNodeName):
        self.inputNodeName = inputNodeName

    def getInputNodeName(self):
        return self.inputNodeName

    def setWeightName(self, weightName):
        self.weightName = weightName

    def getWeightName(self):
        return self.weightName

    def setWeights(self, weights):
        self.weights = weights

    def getWeights(self):
        return self.weights

    def setWeightsDims(self, weightsDims):
        self.weightsDims = weightsDims

    def getWeightsDims(self):
        return self.weightsDims

    def setIsQuantizedPerChannel(self, isQuantizedPerChannel):
        self.isQuantizedPerChannel = bool(isQuantizedPerChannel)

    def getIsQuantizedPerChannel(self):
        return self.isQuantizedPerChannel

    def setWeightsScaleOffset(self, weightsScaleOffset):
        self.weightsScaleOffset = weightsScaleOffset

    def getWeightsScaleOffset(self):
        return self.weightsScaleOffset

    def setDequantizedWeights(self, dequantizedWeights):
        self.dequantizedWeights = dequantizedWeights

    def getDequantizedWeights(self):
        return self.dequantizedWeights

    def setBiasName(self, biasName):
        self.biasName = biasName

    def setBiasScale(self, biasScale):
        self.biasScale = biasScale

    def setBiasOffset(self, biasOffset):
        self.biasOffset = biasOffset

    def setBiases(self, biases):
        self.biases = biases

    def getBiases(self):
        return self.biases

    def setDequantizedBiases(self, dequantizedBiases):
        self.dequantizedBiases = dequantizedBiases

    def getBiasName(self):
        return self.biasName

    def getBiasScale(self):
        return self.biasScale

    def getBiasOffset(self):
        return self.biasOffset

    def getDequantizedBiases(self):
        return self.dequantizedBiases

    def setInputNodeScale(self, inputNodeScale):
        self.inputNodeScale = inputNodeScale

    def getInputNodeScale(self):
        return self.inputNodeScale

    def setActivationScale(self, activationScale):
        self.activationScale = activationScale

    def setActivationOffset(self, activationOffset):
        self.activationOffset = activationOffset

    def setActivations(self, activations):
        self.activations = activations

    def getActivationScale(self):
        return self.activationScale

    def getActivationOffset(self):
        return self.activationOffset

    def getActivations(self):
        return self.activations

    # TODO: Add per input dequantizedActivations
    def setDequantizedActivations(self, dequantizedActivations):
        self.dequantizedActivations = dequantizedActivations

    def getDequantizedActivations(self):
        return self.dequantizedActivations

    def setOpName(self, opName):
        self.opName = opName

    def getOpName(self):
        return self.opName

    def setActivationNodeName(self, activationNodeName):
        self.activationNodeName = activationNodeName

    def getActivationNodeName(self):
        return self.activationNodeName
