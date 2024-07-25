#===========================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#===========================================================================

import tensorflow as tf
from models import sesr, model_utils
import utils
import tf2onnx

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS.__delattr__('int_features')
tf.compat.v1.flags.DEFINE_string('model_name', 'SESR', 'Name of the model')
tf.compat.v1.flags.DEFINE_bool('quant_W', False, 'Quantize weights')
tf.compat.v1.flags.DEFINE_bool('quant_A', False, 'Quantize activations')
tf.compat.v1.flags.DEFINE_bool('gen_tflite', True, 'Generate TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_height', 256, 'Height of LR image in TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_width', 256, 'Width of LR image in TFLITE')
tf.compat.v1.flags.DEFINE_integer('int_features', 32, 'Number of intermediate features within SESR (parameter f in paper)')

model = sesr.SESR(
   m=11,
   feature_size=64,
   LinearBlock_fn=model_utils.LinearBlock_c,
   quant_W=FLAGS.quant_W,
   quant_A=FLAGS.quant_A,
   gen_tflite=FLAGS.gen_tflite,
   mode='infer')
model.compile()

input_image = tf.random.uniform([1, FLAGS.tflite_height, FLAGS.tflite_width, 1])
temp = model(input_image)
tf2onnx.convert.from_keras(model, output_path="original_model.onnx")