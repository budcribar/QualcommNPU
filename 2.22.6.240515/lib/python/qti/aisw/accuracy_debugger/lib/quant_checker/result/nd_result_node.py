# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Dict


# Class to hold the comparison data and related data useful for displaying to the user
class ResultNode:

    def __init__(self, nodeName: str, tensorName: str, tensorType: str, results: Dict,
                 scale: float = None, offset: int = None):
        self.__nodeName = nodeName
        self.__tensorName = tensorName
        self.__tensorType = tensorType
        self.__results = results
        self.__scale = scale
        self.__offset = offset

    def getNodeName(self) -> str:
        return self.__nodeName

    def getTensorName(self) -> str:
        return self.__tensorName

    def getTensorType(self) -> str:
        return self.__tensorType

    def getResults(self) -> Dict:
        return self.__results

    def getScale(self) -> float:
        return self.__scale

    def setScale(self, scale: float) -> None:
        self.__scale = scale

    def getOffset(self) -> int:
        return self.__offset

    def setOffset(self, offset: int) -> None:
        self.__offset = offset

    def hasVerifier(self, verifierName):
        if (verifierName in self.__results.keys()) and self.__results[verifierName]:
            return True
        else:
            return False

    def getResultsByVerifier(self, verifierName):
        return self.__results[verifierName]

    def isFailed(self, verifierName) -> bool or None:
        comparisonResults = self.__results[verifierName]
        if comparisonResults:
            return bool(comparisonResults['pass']) == False
        else:
            return None


class WeightResultNode(ResultNode):

    def __init__(self, nodeName: str, tensorName: str, results: Dict, scale: float = None,
                 offset: int = None) -> None:
        super().__init__(nodeName, tensorName, "Weight", results, scale, offset)


class BiasResultNode(ResultNode):

    def __init__(self, nodeName: str, tensorName: str, results: Dict, scale: float = None,
                 offset: int = None) -> None:
        super().__init__(nodeName, tensorName, "Bias", results, scale, offset)


class ActivationResultNode(ResultNode):

    def __init__(self, nodeName: str, tensorName: str, input: str, results: Dict,
                 scale: float = None, offset: int = None) -> None:
        super().__init__(nodeName, tensorName, "Activation", results, scale, offset)
        self.input = input

    def getInputFilename(self):
        return self.input


class InputResultNode(ResultNode):

    def __init__(self, nodeName: str, results: Dict) -> None:
        super().__init__(nodeName, 'Input', 'Input', results)
