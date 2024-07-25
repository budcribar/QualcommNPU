# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Any
from .nd_result_list import ResultList


# Wrapper around the dictionary type specific to our comparison results
# The main idea here is to add helpful methods to make life easier for manipulating the result data
class ResultSet:

    def __init__(self) -> None:
        self.__set = {}

    def __getitem__(self, key):
        try:
            return self.__set[key]
        except Exception as e:
            return None

    def add(self, key: str, results: ResultList) -> None:
        self.__set[key] = results

    def get(self, key: str) -> Any:
        try:
            return self.__set[key]
        except Exception as e:
            return None

    def keys(self):
        return self.__set.keys()

    def values(self):
        return self.__set.values()

    def items(self):
        return self.__set.items()
