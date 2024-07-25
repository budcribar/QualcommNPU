//=============================================================================
//
//  Copyright (c) 2020-2021,2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <limits>

#include "ops/QnnCpuOpPkgRelu.hpp"

Qnn_ErrorHandle_t QnnCpuOpPkgRelu::finalize() {
  QNN_CPU_BE_ENSURE_EQ(numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CPU_BE_ENSURE_EQ(numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  auto input  = getInput(0);
  auto output = getOutput(0);
  QNN_CPU_BE_ENSURE_EQ(input->dataType, output->dataType, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  // Supporting upto 4D input tensor
  const int numInDims  = numTensorDim(input);
  const int numOutDims = numTensorDim(output);
  QNN_CPU_BE_ENSURE(numInDims == numOutDims, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CPU_BE_ENSURE(numInDims >= 1 && numInDims <= 4, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  setIsFinalize(true);

  return QNN_SUCCESS;
}

template <typename T_Ttype>
void QnnCpuOpPkgRelu::evaluateActivation(const T_Ttype* in, T_Ttype* out, T_Ttype limiter) {
  *out = std::max(std::numeric_limits<T_Ttype>::min(), *in);
  *out = std::min(std::numeric_limits<T_Ttype>::max(), *in);

  // Clamp the output to be always above Limiter for Relu
  if (*out < limiter) *out = limiter;
}

void QnnCpuOpPkgRelu::executeQuantized(QnnCpuOpPackage_Tensor_t* in,
                                       QnnCpuOpPackage_Tensor_t* out) {
  const uint8_t* inData = (const uint8_t*)in->data;
  uint8_t* outData      = (uint8_t*)out->data;
  int32_t inputOffset = 0u, outputOffset = 0u;
  float inputScale = 0.0f, outputScale = 0.0f;
  int32_t outputMultiplier = 0, outputShift = 0;

  if (in->quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED) {
    if (in->quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      inputOffset  = in->quantizeParams.scaleOffsetEncoding.offset;
      inputScale   = in->quantizeParams.scaleOffsetEncoding.scale;
      outputOffset = out->quantizeParams.scaleOffsetEncoding.offset;
      outputScale  = out->quantizeParams.scaleOffsetEncoding.scale;
    }

    // Calculate multiplier from input and output scale
    double realMultiplier = inputScale / outputScale;
    if (realMultiplier == 0.) {
      outputMultiplier = 0;
      outputShift      = 0;
    } else {
      const double q = std::frexp(realMultiplier, &outputShift);
      auto qFixed    = static_cast<int64_t>(std::round(q * (1LL << 31)));
      if (qFixed == (1LL << 31)) {
        qFixed /= 2;
        ++outputShift;
      }
      if (outputShift < -31) {
        outputShift = 0;
        qFixed      = 0;
      }
      if (outputShift > 30) {
        outputShift = 30;
        qFixed      = (1LL << 31) - 1;
      }
      outputMultiplier = static_cast<int32_t>(qFixed);
    }

    // Calculate number of element of output
    size_t numOutputs = 1;
    for (int i = 0; i < (int)out->rank; ++i) {
      numOutputs *= out->currentDimensions[i];
    }

    const int64_t totalShift = 31 - outputShift;
    const int64_t round      = static_cast<int64_t>(1) << (totalShift - 1);

    // Calculate individual output from multiplier
    for (size_t i = 0; i < numOutputs; i++) {
      int64_t result  = (inData[i] + inputOffset) * static_cast<int64_t>(outputMultiplier) + round;
      result          = result >> totalShift;
      int32_t clamped = (static_cast<int32_t>(result) - outputOffset);
      int32_t limiter = 0;
      int32_t res     = 0;
      evaluateActivation<int32_t>(&clamped, &res, limiter);
      outData[i] = static_cast<uint8_t>(res);
    }
  }
}

void QnnCpuOpPkgRelu::executeFloat(QnnCpuOpPackage_Tensor_t* input,
                                   QnnCpuOpPackage_Tensor_t* output) {
  const float* in     = (const float*)input->data;
  const int inputSize = nunTensorSize(input);
  float* out          = (float*)output->data;
  float limiter       = static_cast<float>(0);

  for (int32_t s = 0; s < inputSize; ++s) {
    evaluateActivation<float>(in, out, limiter);
    in++;
    out++;
  }
}

Qnn_ErrorHandle_t QnnCpuOpPkgRelu::execute() {
  QNN_CPU_BE_ENSURE(getIsFinalize(), QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED);
  auto input  = getInput(0);
  auto output = getOutput(0);

  // Call execute as per datatype
  if (input->dataType == QNN_CPU_DATATYPE_FLOAT_32) {
    executeFloat(input, output);
  } else if (input->dataType == QNN_CPU_DATATYPE_UINT_8) {
    executeQuantized(input, output);
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnCpuOpPkgRelu::setOpNode(QnnCpuOpPackage_Node_t* node) {
  // Add input
  for (uint32_t i = 0; i < node->numOfInputs; i++) {
    addInput(node->inputs[i]);
  }

  // Add output
  for (uint32_t i = 0; i < node->numOfOutputs; i++) {
    addOutput(node->outputs[i]);
  }

  return QNN_SUCCESS;
}
