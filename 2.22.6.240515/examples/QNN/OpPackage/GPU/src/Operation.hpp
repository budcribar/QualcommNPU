//=============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <vector>

#include "QnnGpuOpPackage.h"

class Operation {
 public:
  virtual ~Operation() {}

  QnnGpu_Operation_t* getOperationInfo() {
    m_operation = QNN_GPU_OPERATION_INIT;

    if (m_kernels.size() > 0u) {
      m_kernelPtrs.clear();
      for (uint32_t i = 0u; i < m_kernels.size(); i++) {
        m_kernelPtrs.push_back(&m_kernels[i]);
      }
      m_kernelPtrs.push_back(nullptr);
      m_operation.kernels = m_kernelPtrs.data();
    } else {
      m_operation.kernels = nullptr;
    }

    return &m_operation;
  }

 protected:
  Operation() {}

  QnnGpu_Operation_t m_operation;
  std::vector<QnnGpu_Kernel_t> m_kernels;
  std::vector<QnnGpu_Kernel_t*> m_kernelPtrs;
};
