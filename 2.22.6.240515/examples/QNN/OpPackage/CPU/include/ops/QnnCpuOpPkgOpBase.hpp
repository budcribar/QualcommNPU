//=============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <string>
#include <vector>

#include "CPU/QnnCpuOpPackage.h"
#include "QnnCpuMacro.hpp"

class QnnCpuOpPkgOpBase {
 public:
  QnnCpuOpPkgOpBase() { m_isFinalize = false; }
  QnnCpuOpPkgOpBase(const char* name, const char* typeName) : m_name(name), m_typeName(typeName) {
    m_isFinalize = false;
  }

  virtual Qnn_ErrorHandle_t finalize() { return QNN_OP_PACKAGE_ERROR_GENERAL; };

  virtual Qnn_ErrorHandle_t execute() { return QNN_OP_PACKAGE_ERROR_GENERAL; };

  virtual Qnn_ErrorHandle_t setOpNode(QnnCpuOpPackage_Node_t* node) {
    return QNN_OP_PACKAGE_ERROR_GENERAL;
  };

  Qnn_ErrorHandle_t addInput(QnnCpuOpPackage_Tensor_t* in_tensor);

  Qnn_ErrorHandle_t addOutput(QnnCpuOpPackage_Tensor_t* out_tensor);

  std::string getName() { return m_name; }

  std::string getTypeName() { return m_name; }

  QnnCpuOpPackage_Tensor_t* getInput(uint32_t index) { return m_inputTensor[index]; }

  QnnCpuOpPackage_Tensor_t* getOutput(uint32_t index) { return m_outputTensor[index]; }

  uint32_t numInput() { return m_inputTensor.size(); }

  uint32_t numOutput() { return m_outputTensor.size(); }

  uint32_t numTensorDim(QnnCpuOpPackage_Tensor_t* tensor) { return tensor->rank; }

  uint32_t nunTensorSize(QnnCpuOpPackage_Tensor_t* tensor);

  void setIsFinalize(bool isFinalize) { m_isFinalize = isFinalize; }

  bool getIsFinalize() { return m_isFinalize; }

 private:
  std::string m_name;
  std::string m_typeName;
  bool m_isFinalize;
  std::vector<QnnCpuOpPackage_Tensor_t*> m_inputTensor;
  std::vector<QnnCpuOpPackage_Tensor_t*> m_outputTensor;
};