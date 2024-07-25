#===========================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#===========================================================================

import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

onnx_model = onnx.load('original_model.onnx')

nodes = onnx_model.graph.node
inputs = onnx_model.graph.input
weights = onnx_model.graph.initializer

for i,node in enumerate(nodes):
   if node.op_type == "Conv":
      weight_name = node.input[1]
      bias_name = node.input[2]
      new_weight_name = weight_name + "_as_input"
      new_bias_name = bias_name + "_as_input"

      [weights_tensor] = [w for w in weights if w.name == weight_name]
      [biases_tensor] = [b for b in weights if b.name == bias_name]
      weights_tensor = numpy_helper.to_array(weights_tensor)
      biases_tensor = numpy_helper.to_array(biases_tensor)

      # create new tensors to be used as graph inputs
      weights_as_input = helper.make_tensor_value_info(new_weight_name, TensorProto.FLOAT, weights_tensor.shape)
      biases_as_input = helper.make_tensor_value_info(new_bias_name, TensorProto.FLOAT, biases_tensor.shape)
      inputs.append(weights_as_input)
      inputs.append(biases_as_input)

      # create new copy of conv node that uses graph inputs for weights/biases instead
      new_conv_node = onnx.helper.make_node(
            'Conv',
            name=node.name + "_dynamic",
            inputs=[node.input[0], new_weight_name, new_bias_name],
            outputs=node.output
      )

      # copy over attribute data to new conv node
      for attribute in node.attribute:
         if (attribute.type == AttributeProto.INTS):
            new_conv_node.attribute.append(helper.make_attribute(attribute.name, attribute.ints))
         if (attribute.type == AttributeProto.INT):
            new_conv_node.attribute.append(helper.make_attribute(attribute.name, attribute.i))

      # replace old conv node in the node list
      nodes.remove(node)
      nodes.insert(i, new_conv_node)

try:
   onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
   print('The modified model is not valid: %s' % e)
else:
   print('The modified model is valid')

onnx.save(onnx_model, 'modified_model.onnx')