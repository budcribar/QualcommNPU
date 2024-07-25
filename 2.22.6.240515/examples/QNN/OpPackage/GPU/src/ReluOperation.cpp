//=============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Common.hpp"
#include "QnnGpuOpPackage.h"
#include "QnnGpuTypeMacros.hpp"
#include "QnnOpPackage.h"
#include "QnnTypes.h"
#include "ReluOperation.hpp"

#define RETURN(errCode)  \
  do {                   \
    if (status) {        \
      *status = errCode; \
    }                    \
    return;              \
  } while (0);

#define KERNEL_RETURN(errCode) \
  do {                         \
    if (status) {              \
      *status = errCode;       \
    }                          \
    return gpuKernel;          \
  } while (0);

const std::string ReluOperation::s_operationType = "Relu";

std::shared_ptr<Operation> ReluOperation::create(const QnnGpuOpPackage_Node_t* node,
                                                 Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<ReluOperation>(new (std::nothrow) ReluOperation(node, status));
};

ReluOperation::ReluOperation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status)
    : Operation() {
  ////////////////////////////////////////////////////////////////////////
  /// Extract the input tensor
  ////////////////////////////////////////////////////////////////////////

  if (QNN_GPU_OP_CFG_GET_NUM_INPUTS(*(node->configs[0])) != 1u) {
    RETURN(QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);
  }

  Qnn_Tensor_t input        = QNN_GPU_OP_CFG_GET_INPUT_TENSOR(*(node->configs[0]), 0u);
  uint32_t inputRank        = QNN_GPU_TENSOR_GET_RANK(input);
  uint32_t* inputDimensions = QNN_GPU_TENSOR_GET_DIMENSIONS(input);
  size_t numInputElements   = 1u;

  for (size_t idx = 0u; idx < inputRank; ++idx) {
    numInputElements *= inputDimensions[idx];
  }

  ////////////////////////////////////////////////////////////////////////
  /// Extract the output tensor
  ////////////////////////////////////////////////////////////////////////

  if (QNN_GPU_OP_CFG_GET_NUM_OUTPUTS(*(node->configs[0])) != 1u) {
    RETURN(QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);
  }

  Qnn_Tensor_t output        = QNN_GPU_OP_CFG_GET_OUTPUT_TENSOR(*(node->configs[0]), 0u);
  uint32_t outputRank        = QNN_GPU_TENSOR_GET_RANK(output);
  uint32_t* outputDimensions = QNN_GPU_TENSOR_GET_DIMENSIONS(output);
  size_t numOutputElements   = 1;

  for (size_t idx = 0u; idx < outputRank; ++idx) {
    numOutputElements *= outputDimensions[idx];
  }

  ////////////////////////////////////////////////////////////////////////
  /// Verify the flattened tensors sizes are the same
  ////////////////////////////////////////////////////////////////////////

  if (numInputElements != numOutputElements) {
    RETURN(QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);
  }

  ////////////////////////////////////////////////////////////////////////
  /// Create the kernel
  ////////////////////////////////////////////////////////////////////////

  auto kernel = setKernelInfo(numInputElements, status);
  if (QNN_SUCCESS != *status) {
    return;
  }
  m_kernels.push_back(kernel);

  RETURN(QNN_SUCCESS);
}

QnnGpu_Kernel_t ReluOperation::setKernelInfo(size_t numElements, Qnn_ErrorHandle_t* status) {
  QnnGpu_Kernel_t gpuKernel = QNN_GPU_KERNEL_INIT;

  ////////////////////////////////////////////////////////////////////////
  /// Set the kernel work sizes
  ////////////////////////////////////////////////////////////////////////

  gpuKernel.globalWorkDim      = 3u;
  gpuKernel.globalWorkSizes[0] = numElements;
  gpuKernel.globalWorkSizes[1] = 1u;
  gpuKernel.globalWorkSizes[2] = 1u;

  gpuKernel.localWorkDim      = 3u;
  gpuKernel.localWorkSizes[0] = std::min((size_t)64u, numElements);
  gpuKernel.localWorkSizes[1] = 1u;
  gpuKernel.localWorkSizes[2] = 1u;

  ////////////////////////////////////////////////////////////////////////
  /// Set up the input tensor kernel argument
  ////////////////////////////////////////////////////////////////////////

  QnnGpu_KernelArg_t inputTensorArg   = QNN_GPU_KERNEL_ARG_INIT;
  inputTensorArg.type                 = QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ;
  inputTensorArg.tensor.opConfigIndex = 0u;
  inputTensorArg.tensor.tensorIndex   = 0u;
  inputTensorArg.tensor.element       = 0u;
  m_kernelArgs.push_back(inputTensorArg);

  ////////////////////////////////////////////////////////////////////////
  /// Set up the output tensor kernel argument
  ////////////////////////////////////////////////////////////////////////

  QnnGpu_KernelArg_t outputTensorArg   = QNN_GPU_KERNEL_ARG_INIT;
  outputTensorArg.type                 = QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE;
  outputTensorArg.tensor.opConfigIndex = 0u;
  outputTensorArg.tensor.tensorIndex   = 0u;
  outputTensorArg.tensor.element       = 0u;
  m_kernelArgs.push_back(outputTensorArg);

  ////////////////////////////////////////////////////////////////////////
  /// Add kernel argument pointers
  ////////////////////////////////////////////////////////////////////////

  m_kernelArgPtrs.clear();
  for (uint32_t i = 0u; i < m_kernelArgs.size(); i++) {
    m_kernelArgPtrs.push_back(&m_kernelArgs[i]);
  }
  m_kernelArgPtrs.push_back(nullptr);
  gpuKernel.args = m_kernelArgPtrs.data();

  ////////////////////////////////////////////////////////////////////////
  /// Set the kernel source information
  ////////////////////////////////////////////////////////////////////////

  m_kernelSource = R"(
  __kernel void relu_activation_kernel(__global float* input,
                                       __global float* output) {
    int x = get_global_id(0);
    output[x] = max(input[x], 0.0);
  })";

  m_kernelName           = "relu_activation_kernel";
  gpuKernel.name         = m_kernelName.c_str();
  gpuKernel.sourceType   = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  gpuKernel.kernelSource = m_kernelSource.c_str();
  gpuKernel.sourceLength = m_kernelSource.size();
  gpuKernel.buildOptions = "";

  KERNEL_RETURN(QNN_SUCCESS);
}
