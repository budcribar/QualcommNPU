##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
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
import enum
import logging
import numpy as np
import os
import re
import sys
import shutil
import importlib
import pkg_resources
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
import qti.aisw.accuracy_evaluator.common.defaults as df

# to avoid printing logs on console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

defaults = df.Defaults.getInstance()

mismatched_packages = []


class ModelType(enum.Enum):
    ONNX = 0
    TORCHSCRIPT = 1
    TENSORFLOW = 2
    TFLITE = 3
    FOLDER = 4


class Helper:
    """
    Utility class contains common utility methods
    To use:
    >>>Helper.get_np_dtype(type)
    >>>Helper.get_model_type(path)
    """

    @classmethod
    def safe_import_package(cls, package_name, recommended_package_version=None):
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            qacc_logger.error(
                f"Failed to import {package_name}. Kindly refer to SDK documentation and install supported version of {package_name}"
            )
            sys.exit(1)
        else:
            if recommended_package_version:
                try:
                    detected_package_version = pkg_resources.get_distribution(package_name).version
                except:
                    detected_package_version = package.__version__
                if detected_package_version != recommended_package_version and package_name not in mismatched_packages:
                    qacc_logger.warning(
                        f"{package_name} installed version: {detected_package_version}, and Recommended version: {recommended_package_version}"
                    )
                    mismatched_packages.append(package_name)
            return package

    @classmethod
    def get_np_dtype(cls, dtype, map_tf=False):
        # TODO: TF import?
        """
        This method gives the appropriate numpy datatype for given data type
        Args:
            dtype  : onnx data type

        Returns:
            corresponding numpy datatype
        """
        # returns dtype if it is already a numpy dtype
        # else get the corresponding numpy datatype
        try:
            if dtype.__module__ == np.__name__:
                return dtype
        except AttributeError as e:
            if dtype.__class__ == np.dtype:
                dtype = dtype.name

        if (dtype == 'tensor(float)' or dtype == 'float' or dtype == 'float32'):
            return np.float32
        elif (dtype == 'tensor(int)' or dtype == 'int'):
            return np.int
        elif (dtype == 'tensor(float64)' or dtype == 'float64'):
            return np.float64
        elif (dtype == 'tensor(int64)' or dtype == 'int64'):
            return np.int64
        elif (dtype == 'tensor(int32)' or dtype == 'int32'):
            return np.int32
        elif dtype == 'tensor(bool)' or dtype == 'bool':
            return bool
        else:
            assert False, "Unsupported OP type " + str(dtype)
        if map_tf:
            tf = Helper.safe_import_package("tensorflow")
            if dtype == tf.float32: return np.float32
            elif dtype == tf.float64: return np.float64
            elif dtype == tf.int64: return np.int64
            elif dtype == tf.int32: return np.int32
            elif dtype == tf.bool: return bool

    @classmethod
    def get_model_type(cls, path):
        if os.path.isdir(path):
            return ModelType.FOLDER
        else:
            extn = os.path.splitext(path)[1]
        if extn == '.onnx':
            return ModelType.ONNX
        elif extn == '.pt':
            return ModelType.TORCHSCRIPT
        elif extn == '.pb':
            return ModelType.TENSORFLOW
        elif extn == ".tflite":
            return ModelType.TFLITE
        else:
            # TODO : support other model types.
            raise ce.UnsupportedException('model type not supported :' + path)

    @classmethod
    def onnx_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given onnx tensor element type
        Args:
            type : onnx tensor element type
        Returns:
            corresponding numpy datatype and size
        """
        if type == '1':
            return (np.float32, 4)
        elif type == '2':
            return (np.uint8, 1)
        elif type == '3':
            return (np.int8, 1)
        elif type == '4':
            return (np.uint16, 2)
        elif type == '5':
            return (np.int16, 2)
        elif type == '6':
            return (np.int32, 4)
        elif type == '7':
            return (np.int64, 8)
        elif type == '9':
            return (bool, 1)
        else:
            raise ce.UnsupportedException('Unsupported type : {}'.format(str(type)))

    @classmethod
    def tf_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given tensorflow tensor element type
        Args:
            type : tensorflow tensor element type
        Returns:
            corresponding tensorflow datatype
        """
        # TODO: Add QINT dtypes
        tf_to_numpy = {
            1: np.float32,
            2: np.float64,
            3: np.int32,
            4: np.uint8,
            5: np.int16,
            6: np.int8,
            9: np.int64,
            10: bool
        }
        if type in tf_to_numpy:
            return tf_to_numpy[type]
        else:
            raise ce.UnsupportedException('Unsupported type : {}'.format(str(type)))

    @classmethod
    def ort_to_tensorProto(cls, type):
        """
        This method gives the appropriate numpy datatype for given onnx data type
        Args:
            type  : onnx data type

        Returns:
            corresponding numpy datatype
        """
        onnx = Helper.safe_import_package("onnx")
        if (type == 'tensor(float)' or type == 'float'):
            return onnx.TensorProto.FLOAT
        elif (type == 'tensor(int)' or type == 'int'):
            return onnx.TensorProto.INT8
        elif (type == 'tensor(float64)' or type == 'float64'):
            return onnx.TensorProto.DOUBLE
        elif (type == 'tensor(int64)' or type == 'int64'):
            return onnx.TensorProto.INT64
        elif (type == 'tensor(int32)' or type == 'int32'):
            return onnx.TensorProto.INT32
        else:
            assert ("TODO: fix unsupported OP type " + str(type))

    @classmethod
    def get_average_match_percentage(cls, outputs_match_percentage, output_comp_map):
        """Return the average match for all the outputs for a given
        comparator."""
        all_op_match = []
        for op, match in outputs_match_percentage.items():
            comparator = output_comp_map[op]
            comp_name = comparator.display_name()
            all_op_match.append(match[comp_name])

        return sum(all_op_match) / len(all_op_match)

    @classmethod
    def show_progress(cls, total_count, cur_count, info='', key='='):
        """Displays the progress bar."""
        completed = int(round(80 * cur_count / float(total_count)))
        percent = round(100.0 * cur_count / float(total_count), 1)
        bar = key * completed + '-' * (80 - completed)

        sys.stdout.write('[%s] %s%s (%s)\r' % (bar, percent, '%', info))
        sys.stdout.flush()

    @classmethod
    def validate_aic_device_id(self, device_ids):
        '''
        device_ids: list containing the device ids
        '''
        try:
            valid_devices = [
                d.strip()
                for d in os.popen('/opt/qti-aic/tools/qaic-util -q |grep "QID"').readlines()
            ]
            device_count = len(valid_devices)
        except:
            raise ce.ConfigurationException(
                'Failed to get Device Count. Check Devices are connected and Platform SDK '
                'Installation')
        for dev_id in device_ids:
            if f'QID {dev_id}' not in valid_devices:
                raise ce.ConfigurationException(
                    f'Invalid Device Id(s) Passed. Device used must be one of '
                    f'{", ".join(valid_devices)}')
        return True

    @classmethod
    def prepare_work_dir(self, work_dir):
        temp_dir = os.path.join(work_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # create empty temp dir
        os.makedirs(temp_dir)
        defaults = df.Defaults.getInstance()
        defaults.set_log(work_dir + '/qacc.log')

    @classmethod
    def dump_stage_error_log(self, logfile):
        with open(logfile) as f:
            log = f.read()
        qacc_file_logger.error(log)

    @classmethod
    def sanitize_node_names(cls, node_name):
        """Sanitize the node names to follow converter's node naming
        conventions.

        All consecutive special characters will be replaced by an
        underscore '_' and node names not beginning with an alphabet
        will be prepended with an underscore '_'.
        """
        if not isinstance(node_name, str):
            node_name = str(node_name)
        sanitized_name = re.sub(pattern='\\W+', repl='_', string=node_name)
        if not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
            sanitized_name = "_" + sanitized_name
        return sanitized_name

    @classmethod
    def sanitize_native_tensor_names(cls, tensor_names):
        """Sanitize the tensor names to follow converter's node naming
        conventions.
        tensor_names would be in the format graphName0:tensorName0,tensorName1;graphName1:tensorName0,tensorName1
        """
        tensor_names_list = tensor_names.split(';')
        sanitized_tensor_names = ''
        for tlist_str in tensor_names_list:
            # find the first occurrence of ':' as individual tensor names could have ':' in them
            tlist = tlist_str.split(':', 1)
            graph_name = tlist[0]
            tensors = tlist[1].split(',')
            for i, tensor in enumerate(tensors):
                tensors[i] = cls.sanitize_node_names(tensor)
            sanitized_tensors = ','.join(tensors)
            sanitized_tensor_names += graph_name + ':' + sanitized_tensors + ';'
        # remove the last ';' from the sanitized tensor names
        return sanitized_tensor_names[:-1]

    @classmethod
    def cli_params_to_list(cls, params: dict) -> list:
        """Convert given dictionary of QNN converter params to list of its CLI args.
        """
        args = []
        for param, value in params.items():
            if isinstance(value, bool):
                if value:
                    args.append(f'--{param}')
            # InferenceSchemaManager converts all values, including Boolean to str
            elif value == "True":
                args.append(f'--{param}')
            elif value == "False":
                continue
            else:
                # algorithms: default is used by Evaluator and not a valid converter option
                if param == "algorithms" and value == "default":
                    continue
                if param == "native_input_tensor_names" or param == "set_output_tensors":
                    value = cls.sanitize_native_tensor_names(value)
                    args.append(f'--{param} {value}')
                elif param == "extra_args":
                    args.append(value)
                else:
                    args.append(f'--{param} {value}')
        return args
