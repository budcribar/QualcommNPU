# =============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import AxisFormat


def permute_tensor_data_axis_order(src_axis_format, axis_format, tensor_dims, golden_tensor_data):
    """Permutes intermediate tensors goldens to spatial-first axis order for
    verification :param src_axis_format: axis format of source framework tensor
    :param axis_format: axis format of QNN tensor :param tensor_dims: current
    dimensions of QNN tensor :param golden_tensor_data: golden tensor data to
    be permuted :return: np.array of permuted golden tensor data."""

    # base case for same axis format / nontrivial
    if src_axis_format == axis_format or src_axis_format == 'NONTRIVIAL' or axis_format == 'NONTRIVIAL':
        return golden_tensor_data, False
    # reshape golden data to spatial-last axis format
    golden_tensor_data = np.reshape(
        golden_tensor_data,
        tuple([
            tensor_dims[i]
            for i in AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][0]
        ]))
    # transpose golden data to spatial-first axis format
    golden_tensor_data = np.transpose(
        golden_tensor_data,
        AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][1])
    # return flatten golden data
    return golden_tensor_data.flatten(), True


def to_csv(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, encoding='utf-8', index=False)


def to_html(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_html(file_path, classes='table', index=False)


def to_json(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, orient='records', indent=4)


def save_to_file(data, filename) -> None:
    """Save data to file in CSV, HTML and JSON formats :param data: Data to be
    saved to file :param filename: Name of the file."""
    filename = Path(filename)
    to_csv(data, filename.with_suffix(".csv"))
    to_html(data, filename.with_suffix(".html"))
    to_json(data, filename.with_suffix(".json"))
