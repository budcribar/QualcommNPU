# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import re
import shutil
from pathlib import Path

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, InferenceEngineError


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
            absdir = os.path.join(pathPrepend, dir)

    if not checkExist:
        return absdir

    if os.path.exists(absdir):
        return absdir
    else:
        raise ParameterError(dir + "(relpath) and " + absdir + "(abspath) are not existing")


def get_tensor_paths(tensors_path):
    """Returns a dictionary indexed by k, of tensor paths :param tensors_path:
    path to output directory with all the tensor raw data :return:
    Dictionary."""
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


def format_args(additional_args, ignore_args=[]):
    """Returns a formatted string to append to qnn converter/netrun command
    :param additional_args: extra options to be addded to converter/netrun
    commands :param ignore_args: list of args to be ignored :return: String."""
    extra_options = additional_args.split(';')
    extra_cmd = ''
    for item in extra_options:
        arg = item.strip(' ').split('=')
        if arg[0].rstrip(' ') in ignore_args:
            continue
        if len(arg) == 1:
            extra_cmd += '--' + arg[0].rstrip(' ') + ' '
        else:
            extra_cmd += '--' + arg[0].rstrip(' ') + ' ' + arg[1].lstrip(' ') + ' '
    return extra_cmd

def retrieveQnnSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <QNN root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        qnn_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (qnn_sdk_dir.match('qnn-*') or qnn_sdk_dir.match('qaisw-*')):
            return str(qnn_sdk_dir)
        else:
            qnn_path = filePath
            for _ in range(len(filePath.parts)):
                qnn_path = qnn_path.parent
                if (qnn_path.match('qnn-*') or qnn_path.match('qaisw-*')):
                    return str(qnn_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QNN'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QNN'))


def retrieveSnpeSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <SNPE root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        snpe_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (snpe_sdk_dir.match('snpe-*')) or snpe_sdk_dir.match('qaisw-*'):
            return str(snpe_sdk_dir)
        else:
            snpe_path = filePath
            for _ in range(len(filePath.parts)):
                snpe_path = snpe_path.parent
                if (snpe_path.match('snpe-*') or snpe_path.match('qaisw-*')):
                    return str(snpe_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('SNPE'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('SNPE'))


def santize_node_name(node_name):
    """Santize the Node Names to follow Converter Node Naming Conventions
    All Special Characters will be replaced by '_' and numeric node names
    will be modified to have preceding '_'."""
    if not isinstance(node_name, str):
        node_name = str(node_name)
    sanitized_name = re.sub(pattern='\\W+', repl='_', string=node_name)
    if not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
        sanitized_name = "_" + sanitized_name
    return sanitized_name


def sanitize_output_tensor_files(output_directory):
    """
    Moves output raw files from sub folders to 'output_directory' and sanitizes raw file names
    Example: output_directory/conv/tensor_0.raw will become output_directory/_conv_tensor_0.raw
    param output_directory: path to output raw file directory.
    return: status code 0 for success and -1 for failure.
    """

    if not output_directory or not os.path.isdir(output_directory):
        return -1
    for root, folders, files in os.walk(output_directory):

        if not files:
            continue
        rel_path = os.path.relpath(root, output_directory)
        if rel_path == ".":
            rel_path = ""

        for file in files:
            if not file.endswith(".raw"):
                continue
            new_file = os.path.join(rel_path, file)
            new_file, file_extension = os.path.splitext(new_file)
            new_file = santize_node_name(new_file) + file_extension
            shutil.move(os.path.join(root, file), os.path.join(output_directory, new_file))

    for item in os.listdir(output_directory):
        full_path = os.path.join(output_directory, item)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
    return 0
