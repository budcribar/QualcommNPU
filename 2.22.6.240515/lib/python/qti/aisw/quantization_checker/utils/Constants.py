#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

import os

# Platform
WINDOWS = 'Windows'
LINUX = 'Linux'

# Executable and library paths
SNPE = 'SNPE'
QNN = 'QNN'
UNQUANTIZED = 'unquantized'
NET_RUN_OUTPUT_DIR = 'output'
X86_LINUX_CLANG = 'x86_64-linux-clang'
WINDOWS_X86 = 'windows-x86_64'
BIN_PATH_IN_SDK_LINUX = os.path.join('bin', X86_LINUX_CLANG)
BIN_PATH_IN_SDK_WINDOWS = os.path.join('bin','x86_64-windows-msvc')
LIB_PATH_IN_SDK_LINUX = os.path.join('lib', X86_LINUX_CLANG)
LIB_PATH_IN_SDK_WINDOWS = os.path.join('lib','x86_64-windows-msvc')
PYTHONPATH = os.path.join('lib', 'python')
CONFIG_PATH = '/qti/aisw/quantization_checker/configs'

QNN_TF_CONVERTER_BIN_NAME = 'qnn-tensorflow-converter'
QNN_TFLITE_CONVERTER_BIN_NAME = 'qnn-tflite-converter'
QNN_ONNX_CONVERTER_BIN_NAME = 'qnn-onnx-converter'
QNN_MODEL_LIB_GENERATOR_BIN_NAME = 'qnn-model-lib-generator'
QNN_NET_RUN_BIN_NAME_LINUX = 'qnn-net-run'
QNN_NET_RUN_BIN_NAME_WINDOWS = 'qnn-net-run.exe'

SNPE_TF_CONVERTER_BIN_NAME = 'snpe-tensorflow-to-dlc'
SNPE_TFLITE_CONVERTER_BIN_NAME = 'snpe-tflite-to-dlc'
SNPE_ONNX_CONVERTER_BIN_NAME = 'snpe-onnx-to-dlc'
SNPE_QUANTIZER_BIN_NAME_LINUX = 'snpe-dlc-quantize'
SNPE_QUANTIZER_BIN_NAME_WINDOWS = 'snpe-dlc-quant.exe'
SNPE_NET_RUN_BIN_NAME_LINUX = 'snpe-net-run'
SNPE_NET_RUN_BIN_NAME_WINDOWS = 'snpe-net-run.exe'
SNPE_UDO_ROOT = os.path.join('share', 'SNPE', 'SnpeUdo')

BACKEND_LIB_NAME_LINUX = 'libQnnCpu.so'
BACKEND_LIB_NAME_WINDOWS = 'QnnCpu.dll'

MODEL_SO_OUTPUT_PATH = X86_LINUX_CLANG
MODEL_LIB_PATH = 'model_lib'
MODEL_DLL_OUTPUT_PATH = os.path.join('model_lib','x64')

TENSORFLOW = 'TENSORFLOW'
ONNX = 'ONNX'
TFLITE = 'TFLITE'
# Environment paths
# TENSORFLOW
TF_DISTRIBUTE = 'distribute'
TF_PYTHON_PATH = os.path.join('dependencies', 'python')
# TFLITE
TFLITE_DISTRIBUTE = 'distribute'
TFLITE_PYTHON_PATH = os.path.join('dependencies', 'python')
# ONNX
ONNX_DISTRIBUTE = 'distribute'
ONNX_PYTHON_PATH = os.path.join('dependencies', 'python')
