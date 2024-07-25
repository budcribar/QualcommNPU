# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from lib.utils.nd_exceptions import ParameterError
from pathlib import Path

def get_absolute_path(dir, checkExist=True, pathPrepend=None):
    """
    Returns a absolute path
    :param dir: the relate path or absolute path
           checkExist: whether to check whether the path exists
    :return: absolute path
    """
    if not dir:
        return dir

    absdir = os.path.expandvars(dir)
    if not os.path.isabs(absdir):
        if not pathPrepend:
            absdir = os.path.abspath(absdir)
        else:
            absdir = os.path.join(pathPrepend,dir)

    if not checkExist:
        return absdir

    if os.path.exists(absdir):
        return absdir
    else:
        raise ParameterError(dir + "(relpath) and " + absdir + "(abspath) are not existing")

def get_tensor_paths(tensors_path):
    """
    Returns a dictionary indexed by k, of tensor paths
    :param tensors_path: path to output directory with all the tensor raw data
    :return: Dictionary
    """
    tensors = {}
    for dir_path, sub_dirs, files in os.walk(tensors_path):
        for file in files:
            if file.endswith(".raw"):
                tensor_path = os.path.join(dir_path, file)
                # tensor name is part of it's path
                path = os.path.relpath(tensor_path, tensors_path)

                # remove .raw extension
                tensor_name = str(Path(path).with_suffix(''))
                tensors[tensor_name] = tensor_path
    return tensors
