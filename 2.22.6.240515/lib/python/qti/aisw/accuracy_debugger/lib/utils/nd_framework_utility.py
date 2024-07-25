# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from typing import Tuple
import json
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import UnsupportedError


def load_inputs(data_path, data_type, data_dimension=None):
    # type:  (str, str, Tuple) -> np.ndarray
    data = np.fromfile(data_path, data_type)
    if data_dimension is not None:
        data = data.reshape(data_dimension)
    return data


def save_outputs(data, data_path, data_type):
    # type:  (np.ndarray, str, str) -> None
    data.astype(data_type).tofile(data_path)


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def dump_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def transpose_to_nhwc(data, data_dimension):
    # type:  (np.ndarray, list) ->np.ndarray
    if len(data_dimension) == 4:
        data = np.reshape(
            data, (data_dimension[0], data_dimension[1], data_dimension[2], data_dimension[3]))
        data = np.transpose(data, (0, 2, 3, 1))
        data = data.flatten()
    return data


class ModelHelper:

    @classmethod
    def onnx_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given onnx tensor element type
        Args:
            type : onnx tensor element type
        Returns:
            corresponding onnx datatype
        """
        onnx_to_numpy = {
            '1': (np.float32, 4),
            '2': (np.uint8, 1),
            '3': (np.int8, 1),
            '4': (np.uint16, 2),
            '5': (np.int16, 2),
            '6': (np.int32, 4),
            '7': (np.int64, 8),
            '9': (np.bool_, 1)
        }
        if type in onnx_to_numpy:
            return onnx_to_numpy[type]
        else:
            raise UnsupportedError('Unsupported type : {}'.format(str(type)))
