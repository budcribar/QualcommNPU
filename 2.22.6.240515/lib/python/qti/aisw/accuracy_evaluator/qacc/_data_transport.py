##############################################################################
#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
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
import os
import numpy as np
from numpy import ndarray
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Union, Dict


# Custom implementation of a dict like object.
class PipelineItem:

    def __init__(self, data: Union[ndarray, str, List[ndarray], List[str]],
                 meta: Optional[Union[Dict,
                                      DefaultDict]] = defaultdict(), input_idx: int = -1, **kwargs):
        self.data = data  # Infrastructure always takes a list of inputs. ie [inp_node1,inp_node2 ...]
        self.meta = meta
        self.input_idx = input_idx
        self.__dict__.update(kwargs)
        # TODO: Add a field to indicate whether the data is batched or not
        # TODO: Support file extensions, data type and annotation information fields

    # Dict like behavior
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    # format the object for pretty printing
    def __repr__(self):
        return repr(self.__dict__)
