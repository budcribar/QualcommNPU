//<!--
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//-->
#include "Operation.hpp"
#include "QnnGpuOpPackage.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

const std::string SoftmaxOperation::s_operationType = "Softmax";

std::shared_ptr<Operation> SoftmaxOperation::create(const QnnGpuOpPackage_Node_t* node,
                                                    Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<SoftmaxOperation>(new (std::nothrow) SoftmaxOperation(node, status));
}

SoftmaxOperation::SoftmaxOperation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status)
    : Operation() {
  /*
   * validation code if needed.
   */

  ////////////////////////////////////////////////////////////////////////
  /// Create the kernel
  ////////////////////////////////////////////////////////////////////////

  auto kernel = setKernelInfo(node, status);
  if (QNN_SUCCESS != *status) {
    return;
  }
  m_kernels.push_back(kernel);

  RETURN(QNN_SUCCESS);
}

QnnGpu_Kernel_t SoftmaxOperation::setKernelInfo(const QnnGpuOpPackage_Node_t* node,
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
  gpuKernel.localWorkSizes[0] = tensorLength;
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

  m_kernelSource = R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void softmax_udo_kernel(
     __global float* input,
     int numChannels,
     __global float* output)
  {
     int x = get_global_id(0);
     float maxPixelValue = -MAXFLOAT;
     for(int ch = 0 ; ch< numChannels; ch++)
     {
        maxPixelValue = max(maxPixelValue, input[ch+x*numChannels]);
     }
     float expSum = 0.0;
     for(int ch =0 ; ch< numChannels; ch++)
     {
        float  tempIn = input[ch+x*numChannels] - maxPixelValue;
        float expIn = native_exp(tempIn);
        output[ch+x*numChannels] = expIn;
        expSum+=expIn;
     }
     for(int ch =0 ; ch< numChannels; ch++)
     {
        output[ch+x*numChannels] = output[ch+x*numChannels]/expSum;
     }
  })";

  m_kernelName           = "softmax_udo_kernel";
  gpuKernel.name         = m_kernelName.c_str();
  gpuKernel.sourceType   = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  gpuKernel.kernelSource = m_kernelSource.c_str();
  gpuKernel.sourceLength = m_kernelSource.size();
  gpuKernel.buildOptions = "";

  KERNEL_RETURN(QNN_SUCCESS);
}
