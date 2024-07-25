//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Operation.hpp"
#include "QnnGpuOpPackage.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

const std::string ReluOperation::s_operationType = "Relu";

std::shared_ptr<Operation> ReluOperation::create(const QnnGpuOpPackage_Node_t* node,
                                                 Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<ReluOperation>(new (std::nothrow) ReluOperation(node, status));
}

ReluOperation::ReluOperation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status)
    : Operation() {
  ////////////////////////////////////////////////////////////////////////
  /// Extract the input tensor
  ////////////////////////////////////////////////////////////////////////

  if (node->configs[0]->v1.numOfInputs != 1u) {
    RETURN(QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);
  }
  Qnn_Tensor_t input      = node->configs[0]->v1.inputTensors[0];
  size_t numInputElements = 1u;
  for (size_t idx = 0u; idx < input.v1.rank; ++idx) {
    numInputElements *= input.v1.dimensions[idx];
  }

  ////////////////////////////////////////////////////////////////////////
  /// Extract the output tensor
  ////////////////////////////////////////////////////////////////////////

  if (node->configs[0]->v1.numOfOutputs != 1u) {
    RETURN(QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT);
  }
  Qnn_Tensor_t output      = node->configs[0]->v1.outputTensors[0];
  size_t numOutputElements = 1;
  for (size_t idx = 0u; idx < output.v1.rank; ++idx) {
    numOutputElements *= output.v1.dimensions[idx];
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

  auto kernel = setKernelInfo(node, status);
  if (QNN_SUCCESS != *status) {
    return;
  }
  ADD_KERNEL(kernel);

  RETURN(QNN_SUCCESS);
}

QnnGpu_Kernel_t ReluOperation::setKernelInfo(const QnnGpuOpPackage_Node_t* node,
                                             Qnn_ErrorHandle_t* status) {
  QnnGpu_Kernel_t gpuKernel = QNN_GPU_KERNEL_INIT;

  Qnn_Tensor_t input = node->configs[0]->v1.inputTensors[0];
  size_t numElements = 1;
  for (size_t idx = 0; idx < input.v1.rank; ++idx) {
    numElements *= input.v1.dimensions[idx];
  }

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
