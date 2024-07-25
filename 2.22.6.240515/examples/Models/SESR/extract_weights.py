#===========================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#===========================================================================

import onnx
from onnx import helper, numpy_helper

onnx_model = onnx.load('original_model.onnx')

nodes = onnx_model.graph.node
inputs = onnx_model.graph.input
weights = onnx_model.graph.initializer

for i,node in enumerate(nodes):
   if node.op_type == "Conv":
      weight_name = node.input[1]
      bias_name = node.input[2]
      new_weight_name = (weight_name + "_as_input").replace("/", "_").replace(":", "_")
      new_bias_name = (bias_name + "_as_input").replace("/", "_").replace(":", "_")

      [weights_tensor] = [w for w in weights if w.name == weight_name]
      [biases_tensor] = [b for b in weights if b.name == bias_name]
      weights_tensor = numpy_helper.to_array(weights_tensor)
      biases_tensor = numpy_helper.to_array(biases_tensor)

      weights_tensor.tofile(new_weight_name + ".raw")
      biases_tensor.tofile(new_bias_name + ".raw")