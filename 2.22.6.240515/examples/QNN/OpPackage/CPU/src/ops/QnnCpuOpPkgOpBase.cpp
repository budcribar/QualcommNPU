//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "ops/QnnCpuOpPkgOpBase.hpp"

Qnn_ErrorHandle_t QnnCpuOpPkgOpBase::addInput(QnnCpuOpPackage_Tensor_t* inTensor) {
  m_inputTensor.emplace_back(inTensor);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnCpuOpPkgOpBase::addOutput(QnnCpuOpPackage_Tensor_t* outTensor) {
  m_outputTensor.emplace_back(outTensor);
  return QNN_SUCCESS;
}

uint32_t QnnCpuOpPkgOpBase::nunTensorSize(QnnCpuOpPackage_Tensor_t* tensor) {
  uint32_t size = 1;

  for (uint32_t i = 0; i < numTensorDim(tensor); i++) {
    size *= tensor->currentDimensions[i];
  }

  return size;
}