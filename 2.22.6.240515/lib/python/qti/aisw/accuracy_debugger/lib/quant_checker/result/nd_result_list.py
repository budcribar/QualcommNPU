# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from .nd_result_node import ActivationResultNode, WeightResultNode, BiasResultNode


# Wrapper around the list type specific to our comparison results
# The main idea here is to add helpful methods to make life easier for manipulating the result data
class ResultList:

    def __init__(self) -> None:
        self.__results = []
        self.__currentIndex = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.__currentIndex += 1
        try:
            return self.__results[self.__currentIndex]
        except IndexError:
            self.__currentIndex = 0
            raise StopIteration

    def append(self, result: ActivationResultNode or WeightResultNode or BiasResultNode) -> None:
        self.__results.append(result)

    def get(self):
        return self.__results

    def length(self):
        return len(self.__results)

    def isEmpty(self):
        return bool(len(self.__results) == 0)

    def getFailureCountForVerifier(self, verifierName):
        failureCount = 0
        for resultNode in self.__results:
            if resultNode.hasVerifier(verifierName):
                if resultNode.isFailed(verifierName):
                    failureCount = failureCount + 1
        return failureCount

    def getTotalCountForVerifier(self, verifierName):
        totalCount = 0
        for resultNode in self.__results:
            if resultNode.hasVerifier(verifierName):
                totalCount = totalCount + len(resultNode.getResultsByVerifier(verifierName))
        return totalCount
