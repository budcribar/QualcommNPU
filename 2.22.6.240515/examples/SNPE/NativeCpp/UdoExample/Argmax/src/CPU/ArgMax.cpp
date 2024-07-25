//=============================================================================
//
//  Copyright (c) 2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace argmax {
template <typename T_Ttype>
void evaluateActivation(const T_Ttype* in, uint32_t* out, size_t depth) {
  float maxElt = in[0];
  out[0]       = 0;
  for (size_t i = 1; i < depth; ++i) {
    if (in[i] > maxElt) {
      maxElt = in[i];
      out[0] = i;
    }
  }
}

void evaluateQuantized(CustomOp* operation) {
  auto in               = operation->getInput(0);
  auto out              = operation->getOutput(0);
  const uint8_t* inData = (const uint8_t*)in->data;
  uint32_t* outData     = (uint32_t*)out->data;

  // Calculate number of element of output
  size_t numOutputs = 1;
  for (int i = 0; i < (int)out->rank; ++i) {
    numOutputs *= out->currentDimensions[i];
  }

  size_t numInputs = 1;
  for (int i = 0; i < (int)in->rank; ++i) {
    numInputs *= in->currentDimensions[i];
  }

  size_t depth = numInputs / numOutputs;

  // Calculate individual output from multiplier
  for (size_t i = 0; i < numOutputs; i++) {
    evaluateActivation<uint8_t>(&inData[i * depth], &outData[i], depth);
  }
}

Qnn_ErrorHandle_t evaluateFloat(CustomOp* operation) {
  auto m_Input   = operation->getInput(0);
  auto m_Outputs = operation->getOutput(0);

  const uint32_t rank   = m_Input->rank;
  const size_t depth    = m_Input->currentDimensions[rank - 1];
  uint32_t tensorLength = 1;
  for (uint32_t j = 0; j < rank; ++j) {
    tensorLength *= (uint32_t)(m_Input->currentDimensions[j]);
  }
  const size_t numPixels = tensorLength / depth;
  for (size_t pix = 0; pix < numPixels; ++pix) {
    const float* in = (float*)m_Input->data + pix * depth;
    uint32_t* out   = (uint32_t*)m_Outputs->data + pix;
    evaluateActivation<float>(in, out, depth);
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto input = operation->getInput(0);

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
      strcmp(opConfig.v1.typeName, "ArgMax"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace argmax

CustomOpRegistration_t* register_ArgmaxCustomOp() {
  using namespace argmax;
  static CustomOpRegistration_t ArgmaxRegister = {
      execute, finalize, free, validateOpConfig, populateFromNode};
  return &ArgmaxRegister;
}

REGISTER_OP(ArgMax, register_ArgmaxCustomOp);
