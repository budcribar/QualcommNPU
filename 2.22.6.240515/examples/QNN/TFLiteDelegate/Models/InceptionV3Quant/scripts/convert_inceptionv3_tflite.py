#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import os
import tensorflow.compat.v1 as tf
import numpy as np


if 'QNN_SDK_ROOT' not in os.environ:
        raise RuntimeError('QNN_SDK_ROOT not setup.  Please run the SDK env setup script.')

global QNN_ROOT
QNN_ROOT = os.path.abspath(os.environ['QNN_SDK_ROOT'])

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = QNN_ROOT + '/examples/Models/InceptionV3/tensorflow/inception_v3_2016_08_28_frozen_opt.pb', 
    input_arrays = ['input'],
    output_arrays = ['InceptionV3/Predictions/Reshape_1'] 
)
encode_signature_dataset = []
   
def representative_dataset():
    with open(QNN_ROOT + '/examples/Models/InceptionV3/data/target_raw_list.txt', 'r') as file:
        for line in file:
            line = line.strip()
            line = QNN_ROOT + '/examples/Models/InceptionV3/data/' + line
            raw_image = np.fromfile(line, dtype=np.float32).reshape(1, 299, 299, 3)
            encode_signature_dataset.append(raw_image)

    for data in encode_signature_dataset:
        yield [data]


converter.experimental_new_converter = True #True for MLIR / False for TOCO
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_type = tf.int8 #tf.uint8 or tf.int8
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()    
#Save the model.
tflite_mode_path = QNN_ROOT + '/examples/QNN/TFLiteDelegate/Models/InceptionV3Quant/inception_v3_quant.tflite'
with open(tflite_mode_path, 'wb') as f:
    f.write(tflite_model)
