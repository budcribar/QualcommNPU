# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from typing import Tuple


def load_inputs(data_path, data_type, data_dimension=None):
    # type:  (str, str, Tuple) -> np.ndarray
    data = np.fromfile(data_path, data_type)
    if data_dimension is not None:
        data = data.reshape(data_dimension)
    return data


def save_outputs(data, data_path, data_type):
    # type:  (np.ndarray, str, str) -> None
    data.astype(data_type).tofile(data_path)
