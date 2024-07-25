//=============================================================================
//
//  Copyright (c) 2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Operation.hpp"
#include "QnnGpuOpPackage.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

const std::string ArgMaxOperation::s_operationType = "ArgMax";

std::shared_ptr<Operation> ArgMaxOperation::create(const QnnGpuOpPackage_Node_t* node,
                                                   Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<ArgMaxOperation>(new (std::nothrow) ArgMaxOperation(node, status));
}

ArgMaxOperation::ArgMaxOperation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status)
    : Operation() {
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

QnnGpu_Kernel_t ArgMaxOperation::setKernelInfo(const QnnGpuOpPackage_Node_t* node,
                                               Qnn_ErrorHandle_t* status) {
  QnnGpu_Kernel_t gpuKernel = QNN_GPU_KERNEL_INIT;

  Qnn_Tensor_t input  = node->configs[0]->v1.inputTensors[0];
  size_t tensorLength = 1;
  for (size_t idx = 0; idx < input.v1.rank - 1; ++idx) {
    tensorLength *= input.v1.dimensions[idx];
  }

  int numChannels = input.v1.dimensions[input.v1.rank - 1];

  ////////////////////////////////////////////////////////////////////////
  /// Set the kernel work sizes
  ////////////////////////////////////////////////////////////////////////

  gpuKernel.globalWorkDim      = 3u;
  gpuKernel.globalWorkSizes[0] = tensorLength;
  gpuKernel.globalWorkSizes[1] = 1;
  gpuKernel.globalWorkSizes[2] = 1;

  gpuKernel.localWorkDim      = 3u;
  gpuKernel.localWorkSizes[0] = 1024;
  gpuKernel.localWorkSizes[1] = 1;
  gpuKernel.localWorkSizes[2] = 1;

  ////////////////////////////////////////////////////////////////////////
  /// Set up the input tensor kernel argument
  ////////////////////////////////////////////////////////////////////////

  QnnGpu_KernelArg_t inputTensorArg   = QNN_GPU_KERNEL_ARG_INIT;
  inputTensorArg.type                 = QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ;
  inputTensorArg.tensor.opConfigIndex = 0u;
  inputTensorArg.tensor.tensorIndex   = 0u;
  inputTensorArg.tensor.element       = 0u;
  m_kernelArgs.push_back(inputTensorArg);

  QnnGpu_KernelArg_t tensorLengthArg = QNN_GPU_KERNEL_ARG_INIT;
  tensorLengthArg.type               = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  tensorLengthArg.data.type          = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  tensorLengthArg.data.qnnInt        = static_cast<int32_t>(numChannels);
  m_kernelArgs.push_back(tensorLengthArg);

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

  m_kernelSource         = R"(
  __kernel void argmax_udo_kernel(
     __global float* input,
     int numChannels,
     __global int* output)
  {
     int x = get_global_id(0);
     float maxPixelValue = -MAXFLOAT;
     for (int ch = 0; ch< numChannels; ch++)
     {
         if (input[ch + x * numChannels] > maxPixelValue) {
             maxPixelValue = input[ch + x * numChannels];
             output[x] = ch;
         }
     }
  })";
  m_kernelName           = "argmax_udo_kernel";
  gpuKernel.name         = m_kernelName.c_str();
  gpuKernel.sourceType   = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  gpuKernel.kernelSource = m_kernelSource.c_str();
  gpuKernel.sourceLength = m_kernelSource.size();
  gpuKernel.buildOptions = "-cl-std=CL2.0";

  KERNEL_RETURN(QNN_SUCCESS);
}
