//=============================================================================
//
//  Copyright (c) 2022-2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <iostream>
#include <limits>
#include <string>
#include <cmath>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace relu {

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  const auto input  = operation->getInput(0);
  const auto output = operation->getOutput(0);

  QNN_CUSTOM_BE_ENSURE_EQ(
      input->dataType, output->dataType, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  // Supporting only 1D and 2D input tensor
  QNN_CUSTOM_BE_ENSURE(numDimensions(input) >= 1 && numDimensions(input) <= 2,
                       QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  return QNN_SUCCESS;
}

template <typename T_Ttype>
void evaluateActivation(const T_Ttype* in, T_Ttype* out, T_Ttype limiter) {
  *out = std::max(std::numeric_limits<T_Ttype>::min(), *in);
  *out = std::min(std::numeric_limits<T_Ttype>::max(), *in);

  // Clamp the output to be always above Limiter for Relu
  if (*out < limiter) {
    *out = limiter;
  }
}

void evaluateQuantized(CustomOp* operation) {
  auto in  = operation->getInput(0);
  auto out = operation->getOutput(0);
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

void evaluateFloat(CustomOp* operation) {
  auto input  = operation->getInput(0);
  auto output = operation->getOutput(0);
  const float* in     = (const float*)input->data;
  const int inputSize = numTensorSize(input);
  float* out          = (float*)output->data;
  float limiter       = static_cast<float>(0);

  for (uint32_t s = 0; s < inputSize; ++s) {
    evaluateActivation<float>(in, out, limiter);
    in++;
    out++;
  }
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

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Relu"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  return QNN_SUCCESS;
}
}  // namespace relu

CustomOpRegistration_t* register_ReluCustomOp() {
  static CustomOpRegistration_t reluRegister = {
      relu::execute, relu::finalize, nullptr, relu::validateOpConfig, relu::populateFromNode};
  return &reluRegister;
}

REGISTER_OP(Relu, register_ReluCustomOp);
