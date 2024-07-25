#===========================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#===========================================================================

import json
import numpy as np
import argparse
import sys

qnn_to_numpy = {
    776: np.int8,
    790: np.int16,
    818: np.int32,
    1032: np.uint8,
    1046: np.uint16,
    1074: np.uint32
}
quant_types = [776, 790, 818, 1032, 1046, 1074]
QNN_TENSOR_TYPE_APP_WRITE = 0
dynamic_weight_substring = "_as_input"

def run():
    parser = argparse.ArgumentParser(description='Quantize weights using model .json quant params')
    parser.add_argument('--model_json',type = str, help='json for quantized model')

    try:
        global args
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    model_json = args.model_json
    with open(model_json, 'r') as json_file:
        quant_json = json.load(json_file)
        for tensor_name in quant_json["graph"]["tensors"]:
            tensor = quant_json["graph"]["tensors"][tensor_name]
            tensor_data_type = tensor["data_type"]
            tensor_type = tensor["type"]
            if (tensor_type == QNN_TENSOR_TYPE_APP_WRITE and
                tensor_data_type in quant_types and
                dynamic_weight_substring in tensor_name):
                float_file_name = "{}.raw".format(tensor_name)
                quant_file_name = "{}_quant.raw".format(tensor_name)
                numpy_type = qnn_to_numpy[tensor_data_type]
                float_data = np.fromfile(float_file_name, dtype=np.float32)
                scale = tensor["quant_params"]["scale_offset"]["scale"]
                offset = tensor["quant_params"]["scale_offset"]["offset"]
                type_min = np.iinfo(numpy_type).min
                type_max = np.iinfo(numpy_type).max
                quant_data = (np.clip((float_data / scale).round() - offset, type_min, type_max)).astype(numpy_type)
                quant_data.tofile(quant_file_name)

if __name__=="__main__":
    run()