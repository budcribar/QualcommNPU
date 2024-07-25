# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum

class Engine(Enum):
    SNPE = 'SNPE'
    ANN = 'ANN'
    QNN = 'QNN'


class Framework(Enum):
    tensorflow = 'tensorflow'
    caffe2 = 'caffe2'
    tflite = 'tflite'
    onnx = 'onnx'
    caffe = "caffe"


class Runtime(Enum):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    aip = 'aip'
    dspv65 = 'dspv65'
    dspv66 = 'dspv66'
    dspv68 = 'dspv68'
    dspv69 = 'dspv69'
    dspv73 = 'dspv73'


class ComponentType(Enum):
    converter = "converter"
    context_binary_generator = "context_binary_generator"
    executor = "executor"
    inference_engine = "inference_engine"
    devices = "devices"
    snpe_quantizer = "snpe_quantizer"


class Status(Enum):
    off = "off"
    on = "on"
    always = "always"

class Devices_list(Enum):
    devices = ["linux-embedded", "android", "x86"]
