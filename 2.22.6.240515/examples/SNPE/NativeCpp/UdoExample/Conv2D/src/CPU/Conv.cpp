//==============================================================================
// Auto Generated Code for Conv2DPackage
//==============================================================================
#include <iostream>
#include <string>
#include <string.h>
#include <cmath>
#include <algorithm>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace conv {

void floatToInt(float realMultiplier, int32_t* outputMultiplier, int32_t* outputShift)
{
    if (realMultiplier == 0.) {
        *outputMultiplier = 0;
        *outputShift      = 0;
    } else {
        const double q = std::frexp(realMultiplier, outputShift);
        auto qFixed    = static_cast<int64_t>(::round(q * (1LL << 31)));
        if (qFixed == (1LL << 31)) {
            qFixed /= 2;
            *outputShift += 1;
        }
        if (*outputShift < -31) {
            *outputShift = 0;
            qFixed      = 0;
        }
        if (*outputShift > 30) {
            *outputShift = 30;
            qFixed      = (1LL << 31) - 1;
        }
        *outputMultiplier = static_cast<int32_t>(qFixed);
    }
}

int32_t inline evalQuantizedMultiplier(uint32_t input, int32_t output_offset,
    int32_t quantized_multiplier, int shift)
{
    int32_t unclamped_result = input;
    const int32_t total_shift = 31 - shift;
    const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
    int64_t result = unclamped_result * static_cast<int64_t>(quantized_multiplier) + round;
    result = result >> total_shift;
    unclamped_result = static_cast<int32_t>(result) - output_offset;
    const int32_t clamped_output = std::min(255, std::max(0, unclamped_result));
    return clamped_output;
}

Qnn_ErrorHandle_t evaluateQuantized(CustomOp* operation) {
    int32_t groups = 1;
    int32_t* pad = nullptr;
    int32_t padH = 0;
    int32_t padW = 0;
    int32_t* stride = nullptr;
    int32_t strideH = 1;
    int32_t strideW = 1;
    int32_t* dilation = nullptr;
    int32_t* kernel_shape = nullptr;
    auto m_Inputs = operation->getInput(0);
    auto m_Outputs = operation->getOutput(0);

    uint8_t* in  = (uint8_t*)m_Inputs->data;
    uint8_t* out  = (uint8_t*)m_Outputs->data;
    uint8_t* filter  = (uint8_t*)(operation->getInput(1))->data;
    uint8_t* bias  = (uint8_t*)(operation->getInput(2))->data;

    groups = (int32_t)(operation->getParam("group")->scalarParam);
    pad = ((int32_t*)(operation->getParam("pads")->tensorParam->data));
    stride = ((int32_t*)(operation->getParam("strides")->tensorParam->data));
    dilation = ((int32_t*)(operation->getParam("dilations")->tensorParam->data));
    kernel_shape = ((int32_t*)(operation->getParam("kernel_shape")->tensorParam->data));

    if (pad != nullptr)
    {
        padH = pad[0];
        padW = pad[1];
    }

    if (stride != nullptr)
    {
        strideH = stride[0];
        strideW = stride[1];
    }

    //Input height, width and depth.
    int32_t inputHeight = m_Inputs->currentDimensions[1];
    int32_t inputWidth = m_Inputs->currentDimensions[2];
    int32_t inputDepth = m_Inputs->currentDimensions[3];
    int32_t input_offset = m_Inputs->quantizeParams.scaleOffsetEncoding.offset;
    float input_scale = m_Inputs->quantizeParams.scaleOffsetEncoding.scale;

    //Output height, width and depth
    int32_t outputHeight = m_Outputs->currentDimensions[1];
    int32_t outputWidth = m_Outputs->currentDimensions[2];
    int32_t outputDepth = m_Outputs->currentDimensions[3];
    int32_t output_offset = m_Outputs->quantizeParams.scaleOffsetEncoding.offset;
    float output_scale = m_Outputs->quantizeParams.scaleOffsetEncoding.scale;

    //Filter height, width and depth
    int32_t filterHeight  = (operation->getInput(1))->currentDimensions[0];
    int32_t filterWidth = (operation->getInput(1))->currentDimensions[1];
    int32_t filterDepth = (operation->getInput(1))->currentDimensions[2];
    int32_t filter_offset = (operation->getInput(1))->quantizeParams.scaleOffsetEncoding.offset;
    float filter_scale = (operation->getInput(1))->quantizeParams.scaleOffsetEncoding.scale;

    // set the depth for each group of filters
    int32_t outputGroupDepth = outputDepth / groups;
    float realMultiplier = 0.0;
    if (output_scale)
    {
        realMultiplier = (input_scale * filter_scale) / output_scale;
    }
    int32_t output_multiplier=0;
    int32_t shift=0;
    floatToInt(realMultiplier, &output_multiplier, &shift);
    int32_t outputActivationMin = std::numeric_limits<uint8_t>::lowest();
    int32_t outputActivationMax = std::numeric_limits<uint8_t>::max();
    for(int32_t oh = 0; oh < outputHeight; oh++) {
       for(int32_t ow = 0; ow < outputWidth; ow++) {
          for (int32_t g = 0; g < groups; g++) {
              for (int32_t d = 0; d < outputGroupDepth; d++) {
                  int offset = g * outputGroupDepth + d;
                  int32_t sum = 0;
                  for(int32_t fh = 0; fh < filterHeight; fh++) {
                     int32_t inputH = oh * strideH - padH + fh;
                     if(inputH < 0) {
                       continue;
                     }
                     if(inputH >= inputHeight) {
                        break;
                     }

                     for(int32_t fw = 0; fw < filterWidth; fw++) {
                        int32_t inputW = ow * strideW - padW + fw;
                        if(inputW < 0) {
                          continue;
                        }
                        if(inputW >= inputWidth) {
                           break;
                        }

                        for(int32_t fd = 0; fd < filterDepth; fd++) {
                            int32_t inOffset = (inputH * inputWidth + inputW) * inputDepth + fd + g * filterDepth;
                            int32_t fOffset = (fh * filterWidth + fw) * filterDepth * outputDepth + fd * outputDepth;
                            sum += (in[inOffset] + input_offset) * (filter[fOffset + offset] + filter_offset);
                        }//fd
                     }//fw
                  }// end of loop fh
                  if (bias) {
                      sum += bias[offset];
                  }
                  sum = evalQuantizedMultiplier(sum, output_offset, output_multiplier, shift);
                  out[d] = static_cast<uint8_t>(sum);
              }// d
              out += outputGroupDepth;
          }//g
       }// end of loop ox
    }// end of loop oy
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t evaluateFloat(CustomOp* operation) {
    int32_t groups = 1;
    int32_t* pad = nullptr;
    int32_t padH = 0;
    int32_t padW = 0;
    int32_t* stride = nullptr;
    int32_t strideH = 1;
    int32_t strideW = 1;
    int32_t* dilation = nullptr;
    int32_t* kernel_shape = nullptr;

    auto m_Inputs = operation->getInput(0);
    auto m_Outputs = operation->getOutput(0);

    const float* in  = (float*)m_Inputs->data;
    float* out  = (float*)m_Outputs->data;

    float* filter  = (float*)(operation->getInput(1))->data;
    float* bias  = (float*)(operation->getInput(2))->data;

    groups = (int32_t)(operation->getParam("group")->scalarParam);
    pad = ((int32_t*)(operation->getParam("pads")->tensorParam->data));
    stride = ((int32_t*)(operation->getParam("strides")->tensorParam->data));
    dilation = ((int32_t*)(operation->getParam("dilations")->tensorParam->data));
    kernel_shape = ((int32_t*)(operation->getParam("kernel_shape")->tensorParam->data));

    if (pad != nullptr)
    {
        padH = pad[0];
        padW = pad[1];
    }

    if (stride != nullptr)
    {
        strideH = stride[0];
        strideW = stride[1];
    }

    //Input height, width and depth.
    int32_t inputHeight = m_Inputs->currentDimensions[1];
    int32_t inputWidth = m_Inputs->currentDimensions[2];
    int32_t inputDepth = m_Inputs->currentDimensions[3];

    //Output height, width and depth
    int32_t outputHeight = m_Outputs->currentDimensions[1];
    int32_t outputWidth = m_Outputs->currentDimensions[2];
    int32_t outputDepth = m_Outputs->currentDimensions[3];

    //Filter height, width and depth
    int32_t filterHeight  = (operation->getInput(1))->currentDimensions[0];
    int32_t filterWidth = (operation->getInput(1))->currentDimensions[1];
    int32_t filterDepth = (operation->getInput(1))->currentDimensions[2];

    // set the depth for each group of filters
    int32_t outputGroupDepth = outputDepth / groups;

    float outputActivationMin = std::numeric_limits<float>::lowest();
    float outputActivationMax = std::numeric_limits<float>::max();
    for(int32_t oh = 0; oh < outputHeight; oh++) {
       for(int32_t ow = 0; ow < outputWidth; ow++) {
          for (int32_t g = 0; g < groups; g++) {
              for (int32_t d = 0; d < outputGroupDepth; d++) {
                  int offset = g * outputGroupDepth + d;
                  float sum = 0.0f;
                  for(int32_t fh = 0; fh < filterHeight; fh++) {
                     int32_t inputH = oh * strideH - padH + fh;
                     if(inputH < 0) {
                       continue;
                     }
                     if(inputH >= inputHeight) {
                        break;
                     }

                     for(int32_t fw = 0; fw < filterWidth; fw++) {
                        int32_t inputW = ow * strideW - padW + fw;
                        if(inputW < 0) {
                          continue;
                        }
                        if(inputW >= inputWidth) {
                           break;
                        }

                        for(int32_t fd = 0; fd < filterDepth; fd++) {
                            int32_t inOffset = (inputH * inputWidth + inputW) * inputDepth + fd + g * filterDepth;
                            int32_t fOffset = (fh * filterWidth + fw) * filterDepth * outputDepth + fd * outputDepth;
                            sum += in[inOffset] * filter[fOffset + offset];
                        }//fd
                     }//fw
                  }// end of loop fh
                  sum += bias[offset];
                  sum = std::max(std::min(sum, outputActivationMax), outputActivationMin);
                  out[d] = sum;
              }// d
              out += outputGroupDepth;
          }//g
       }// end of loop ox
    }// end of loop oy
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

    /**
    * Add code here
    **/

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto input  = operation->getInput(0);

  if (input->dataType == QNN_CPU_DATATYPE_FLOAT_32) {
    evaluateFloat(operation);
  } else if (input->dataType == QNN_CPU_DATATYPE_UINT_8) {
    evaluateQuantized(operation);
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

  // Add params
   // The getParam function returns a pair -> hasParam, paramValue
   // Check that parameter has be retrieved. Pair.first is false if it was not found and the paramValue is nullptr

   auto groupPair = getParam(node, "group");

   QNN_CUSTOM_BE_ENSURE(groupPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("group", groupPair.second);


   auto padsPair = getParam(node, "pads");

   QNN_CUSTOM_BE_ENSURE(padsPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("pads", padsPair.second);


   auto stridesPair = getParam(node, "strides");

   QNN_CUSTOM_BE_ENSURE(stridesPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("strides", stridesPair.second);


   auto dilationsPair = getParam(node, "dilations");

   QNN_CUSTOM_BE_ENSURE(dilationsPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("dilations", dilationsPair.second);


   auto kernel_shapePair = getParam(node, "kernel_shape");

   QNN_CUSTOM_BE_ENSURE(kernel_shapePair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("kernel_shape", kernel_shapePair.second);


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Conv"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace conv

CustomOpRegistration_t* register_ConvCustomOp() {
  using namespace conv;
  static CustomOpRegistration_t ConvRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &ConvRegister;
}

REGISTER_OP(Conv, register_ConvCustomOp);
