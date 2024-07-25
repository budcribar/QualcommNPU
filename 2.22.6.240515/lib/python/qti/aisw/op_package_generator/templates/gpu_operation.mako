<%doc>//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================</%doc>

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "QnnGpuOpPackage.h"

class Operation {
 public:
  virtual ~Operation() {}

  QnnGpu_Operation_t* getOperationInfo() {
    m_operation = QNN_GPU_OPERATION_INIT;

    if (m_kernels.size() > 0u) {
      m_kernelPtrs.clear();
      for (uint32_t i = 0u; i < m_kernels.size(); ++i) {
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

  std::string m_kernelName;
  std::string m_kernelSource;

  std::vector<QnnGpu_KernelArg_t> m_kernelArgs;
  std::vector<QnnGpu_KernelArg_t*> m_kernelArgPtrs;
};

%for operator in package_info.operators:
class ${operator.type_name}Operation : public Operation {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t* node,
                                           Qnn_ErrorHandle_t* status);
  static const std::string s_operationType;

 private:
  ${operator.type_name}Operation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status);
  QnnGpu_Kernel_t setKernelInfo(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status);
};

%endfor

#define RETURN(errCode)  ${"\\"}
  do {                   ${"\\"}
    if (status) {        ${"\\"}
      *status = errCode; ${"\\"}
    }                    ${"\\"}
    return;              ${"\\"}
  } while (0);

#define KERNEL_RETURN(errCode) ${"\\"}
  do {                         ${"\\"}
    if (status) {              ${"\\"}
      *status = errCode;       ${"\\"}
    }                          ${"\\"}
    return gpuKernel;          ${"\\"}
  } while (0)

#define ADD_KERNEL(kernel)                     ${"\\"}
  do {                                         ${"\\"}
    if (kernel.name && kernel.kernelSource){   ${"\\"}
      m_kernels.push_back(kernel);             ${"\\"}
    }                                          ${"\\"}
  } while(0)                                   ${"\\"}
