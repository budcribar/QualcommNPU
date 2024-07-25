//=============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Operation.hpp"

class ReluOperation : public Operation {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t* node,
                                           Qnn_ErrorHandle_t* status);

  static const std::string s_operationType;

 private:
  ReluOperation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status);

  QnnGpu_Kernel_t setKernelInfo(size_t numElements, Qnn_ErrorHandle_t* status);

  std::string m_kernelName;
  std::string m_kernelSource;
  std::vector<QnnGpu_KernelArg_t> m_kernelArgs;
  std::vector<QnnGpu_KernelArg_t*> m_kernelArgPtrs;
};
