#=============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import sys
import platform
import qti.aisw.quantization_checker.utils.Constants as Constants

def getEnvironment(configParams : dict, sdkDir : str, targetSdk : str, mlFramework=None, pythonPath=None):
    environ = dict()
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if targetSdk.upper() == Constants.SNPE:
        environ[targetSdk.upper() + '_ROOT'] = sdkDir
    else:
        environ[targetSdk.upper() + '_SDK_ROOT'] = sdkDir

    environ['PYTHONPATH'] = os.path.join(sdkDir, Constants.PYTHONPATH)
    if pythonPath:
        environ['PYTHONPATH'] = os.path.join(sdkDir, pythonPath)

    if "CLANG_PATH" not in configParams:
        print("ERROR: Please provide CLANG PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["CLANG_PATH"]
    if "BASH_PATH" not in configParams:
        print("ERROR: Please provide BASH PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["BASH_PATH"] + ':' + environ['PATH']
    if "PY3_PATH" not in configParams:
        print("ERROR: Please provide Python3 environment bin PATH to config_file.", flush=True)
        exit(-1)
    if "BIN_PATH" not in configParams:
        print("ERROR: Please provide BIN PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams['PY3_PATH'] + ':' + environ['PATH'] + ':' + configParams["BIN_PATH"]

    if targetSdk.upper() == Constants.SNPE:
        environ['SNPE_UDO_ROOT'] = os.path.join(sdkDir, Constants.SNPE_UDO_ROOT)

    environ['PATH'] = os.path.join(sdkDir, Constants.BIN_PATH_IN_SDK_LINUX) + ':' + environ['PATH']
    environ['LD_LIBRARY_PATH'] = os.path.join(sdkDir, Constants.LIB_PATH_IN_SDK_LINUX)

    if mlFramework == Constants.TENSORFLOW:
        if 'TENSORFLOW_HOME' not in configParams:
            print('ERROR: Please provide TENSORFLOW PATH to config_file.', flush=True)
            exit(-1)
        environ['TENSORFLOW_HOME'] = configParams['TENSORFLOW_HOME']
        environ['PYTHONPATH'] = environ['TENSORFLOW_HOME'] + ':' + environ['PYTHONPATH']
        environ['PYTHONPATH'] = os.path.join(environ['TENSORFLOW_HOME'], Constants.TF_DISTRIBUTE) + ':' + environ['PYTHONPATH']
        environ['PYTHONPATH'] = os.path.join(environ['TENSORFLOW_HOME'], Constants.TF_PYTHON_PATH) + ':' + environ['PYTHONPATH']
    elif mlFramework == Constants.TFLITE:
        if 'TFLITE_HOME' not in configParams:
            print('ERROR: Please provide TFLITE PATH to config_file.', flush=True)
            exit(-1)
        environ['TFLITE_HOME'] = configParams['TFLITE_HOME']
        environ['PYTHONPATH'] = environ['TFLITE_HOME'] + ':' + environ['PYTHONPATH']
        environ['PYTHONPATH'] = os.path.join(environ['TFLITE_HOME'], Constants.TFLITE_DISTRIBUTE) + ':' + environ['PYTHONPATH']
        environ['PYTHONPATH'] = os.path.join(environ['TFLITE_HOME'], Constants.TFLITE_PYTHON_PATH) + ':' + environ['PYTHONPATH']
    elif mlFramework == Constants.ONNX:
        if 'ONNX_HOME' not in configParams:
            print('ERROR: Please provide ONNX path.', flush=True)
            exit(-1)
        environ['ONNX_HOME'] = configParams['ONNX_HOME']
        environ['PYTHONPATH'] = environ['ONNX_HOME'] + ':' + environ['PYTHONPATH']
        environ['PYTHONPATH'] = os.path.join(environ['ONNX_HOME'], Constants.ONNX_DISTRIBUTE) + ':' + environ['PYTHONPATH']
        environ['PYTHONPATH'] = os.path.join(environ['ONNX_HOME'], Constants.ONNX_PYTHON_PATH) + ':' + environ['PYTHONPATH']

    if platform.system() == Constants.WINDOWS:
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.environ.get('PYTHONPATH')
        environment['PATH'] = os.path.join(sdkDir, Constants.BIN_PATH_IN_SDK_WINDOWS) + ',' + environment['PATH']
        environment['LD_LIBRARY_PATH'] = os.path.join(sdkDir, Constants.LIB_PATH_IN_SDK_WINDOWS)
        return environment

    return environ

def setEnvironment(configParams, sdkDir, mlFramework=None):
    curr_sys_path = []
    for path in sys.path:
        curr_sys_path.append(path)
    sys.path.clear()
    sys.path.append(os.path.join(sdkDir, Constants.PYTHONPATH))
    if mlFramework == Constants.TENSORFLOW:
        if 'TENSORFLOW_HOME' not in configParams:
            print('ERROR: Please provide TENSORFLOW PATH to config_file.', flush=True)
            exit(-1)
        sys.path.append(configParams['TENSORFLOW_HOME'])
        sys.path.append(os.path.join(configParams['TENSORFLOW_HOME'], Constants.TF_DISTRIBUTE))
        sys.path.append(os.path.join(configParams['TENSORFLOW_HOME'], Constants.TF_PYTHON_PATH))
    elif mlFramework == Constants.TFLITE:
        if 'TFLITE_HOME' not in configParams:
            print('ERROR: Please provide TFLITE PATH to config_file.', flush=True)
            exit(-1)
        sys.path.append(configParams['TFLITE_HOME'])
        sys.path.append(os.path.join(configParams['TFLITE_HOME'], Constants.TFLITE_DISTRIBUTE))
        sys.path.append(os.path.join(configParams['TFLITE_HOME'], Constants.TFLITE_PYTHON_PATH))
    elif mlFramework == Constants.ONNX:
        if 'ONNX_HOME' not in configParams:
            print('ERROR: Please provide ONNX path.', flush=True)
            exit(-1)
        sys.path.append(configParams['ONNX_HOME'])
        sys.path.append(os.path.join(configParams['ONNX_HOME'], Constants.ONNX_DISTRIBUTE))
        sys.path.append(os.path.join(configParams['ONNX_HOME'], Constants.ONNX_PYTHON_PATH))
    for path in curr_sys_path:
        if path not in sys.path:
            sys.path.append(path)
