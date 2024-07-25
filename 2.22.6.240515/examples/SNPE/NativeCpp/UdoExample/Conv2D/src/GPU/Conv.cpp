//==============================================================================
// Auto Generated Code for Conv2DPackage
//==============================================================================
#include "Operation.hpp"
#include "QnnGpuOpPackage.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"
#include <numeric>
#include <algorithm>

const std::string ConvOperation::s_operationType = "Conv";

std::shared_ptr<Operation> ConvOperation::create(const QnnGpuOpPackage_Node_t* node,
                                                 Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<ConvOperation>(new (std::nothrow) ConvOperation(node, status));
}

ConvOperation::ConvOperation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status)
    : Operation() {

  /**
   * Add code here
   **/

  auto kernel = setKernelInfo(node, status);
  if (QNN_SUCCESS != *status) {
    return;
  }
  ADD_KERNEL(kernel);

  RETURN(QNN_SUCCESS);
}

QnnGpu_Kernel_t ConvOperation::setKernelInfo(const QnnGpuOpPackage_Node_t* node,
                                            Qnn_ErrorHandle_t* status) {
  QnnGpu_Kernel_t gpuKernel = QNN_GPU_KERNEL_INIT;

  /**
   * Add code here
   **/

  Qnn_Tensor_t input = node->configs[0]->v1.inputTensors[0];
  Qnn_Tensor_t weights = node->configs[0]->v1.inputTensors[1];
  Qnn_Tensor_t bias = node->configs[0]->v1.inputTensors[2];
  Qnn_Tensor_t output = node -> configs[0]->v1.outputTensors[0];

  Qnn_Param_t dilations = node->configs[0]->v1.params[0];
  Qnn_Param_t group = node->configs[0]->v1.params[1];
  Qnn_Param_t pads = node->configs[0]->v1.params[3];
  Qnn_Param_t stride = node->configs[0]->v1.params[4];

  size_t tensorLength =1;
  for (size_t idx = 0; idx < input.v1.rank - 1; ++idx) {
    tensorLength *= input.v1.dimensions[idx];
  }

   ////////////////////////////////////////////////////////////////////////
  /// Set the kernel work sizes
  ////////////////////////////////////////////////////////////////////////

  gpuKernel.globalWorkDim      = 3u;
  gpuKernel.globalWorkSizes[0] = output.v1.dimensions[0];
  gpuKernel.globalWorkSizes[1] = output.v1.dimensions[1];
  gpuKernel.globalWorkSizes[2] = output.v1.dimensions[2];

  uint32_t z = std::min((uint32_t)1024 , output.v1.dimensions[2]);
  uint32_t y = std::min((uint32_t)floor(1024/z) , output.v1.dimensions[1]);
  uint32_t x = std::min((uint32_t)floor(1024/(z*y)) , output.v1.dimensions[0]);

  gpuKernel.localWorkDim      = 3u;
  gpuKernel.localWorkSizes[0] = x;
  gpuKernel.localWorkSizes[1] = y;
  gpuKernel.localWorkSizes[2] = z;

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
  /// Set up the param tensor kernel argument
  ////////////////////////////////////////////////////////////////////////

  QnnGpu_KernelArg_t weightsTensor = QNN_GPU_KERNEL_ARG_INIT;
  weightsTensor.type = QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ;
  inputTensorArg.tensor.opConfigIndex = 0u;
  weightsTensor.tensor.tensorIndex = 1u;
  weightsTensor.tensor.element = 0u;
  m_kernelArgs.push_back(weightsTensor);

  QnnGpu_KernelArg_t biasTensor = QNN_GPU_KERNEL_ARG_INIT;
  if (bias.v1.type == QNN_TENSOR_TYPE_NULL) {
    biasTensor.type =  QNN_GPU_KERNEL_ARG_TYPE_NULL_PTR;
  }
  else {
    biasTensor.type = QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ;
  }
  inputTensorArg.tensor.opConfigIndex = 0u;
  biasTensor.tensor.tensorIndex = 2u;
  biasTensor.tensor.element = 0u;
  m_kernelArgs.push_back(biasTensor);

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
  /// Set up the scalar params argument
  ////////////////////////////////////////////////////////////////////////

  QnnGpu_KernelArg_t inputWidthTensor = QNN_GPU_KERNEL_ARG_INIT;
  inputWidthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  inputWidthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  inputWidthTensor.data.qnnInt = static_cast<int32_t>(input.v1.dimensions[2]);
  m_kernelArgs.push_back(inputWidthTensor);

  QnnGpu_KernelArg_t inputHeightTensor = QNN_GPU_KERNEL_ARG_INIT;
  inputHeightTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  inputHeightTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  inputHeightTensor.data.qnnInt = static_cast<int32_t>(input.v1.dimensions[1]);
  m_kernelArgs.push_back(inputHeightTensor);

  QnnGpu_KernelArg_t inputdepthTensor = QNN_GPU_KERNEL_ARG_INIT;
  inputdepthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  inputdepthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  inputdepthTensor.data.qnnInt = static_cast<int32_t>(input.v1.dimensions[3]);
  m_kernelArgs.push_back(inputdepthTensor);

  QnnGpu_KernelArg_t filterDepthTensor = QNN_GPU_KERNEL_ARG_INIT;
  filterDepthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  filterDepthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  filterDepthTensor.data.qnnInt = static_cast<int32_t>(weights.v1.dimensions[3]);
  m_kernelArgs.push_back(filterDepthTensor);

  QnnGpu_KernelArg_t filterWidthTensor = QNN_GPU_KERNEL_ARG_INIT;
  filterWidthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  filterWidthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  filterWidthTensor.data.qnnInt = static_cast<int32_t>(weights.v1.dimensions[1]);
  m_kernelArgs.push_back(filterWidthTensor);

  QnnGpu_KernelArg_t filterHeightTensor = QNN_GPU_KERNEL_ARG_INIT;
  filterHeightTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  filterHeightTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  filterHeightTensor.data.qnnInt = static_cast<int32_t>(weights.v1.dimensions[0]);
  m_kernelArgs.push_back(filterHeightTensor);

  QnnGpu_KernelArg_t paddingWidthTensor = QNN_GPU_KERNEL_ARG_INIT;
  paddingWidthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  paddingWidthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  paddingWidthTensor.data.qnnInt = static_cast<int32_t>(static_cast<int32_t*>(
                                      pads.tensorParam.v1.clientBuf.data)[1]);
  m_kernelArgs.push_back(paddingWidthTensor);

  QnnGpu_KernelArg_t paddingHeightTensor = QNN_GPU_KERNEL_ARG_INIT;
  paddingHeightTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  paddingHeightTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  paddingHeightTensor.data.qnnInt = static_cast<int32_t>(static_cast<int32_t*>(
                                        pads.tensorParam.v1.clientBuf.data)[0]);
  m_kernelArgs.push_back(paddingHeightTensor);

  QnnGpu_KernelArg_t strideWidthTensor = QNN_GPU_KERNEL_ARG_INIT;
  strideWidthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  strideWidthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  strideWidthTensor.data.qnnInt = static_cast<int32_t>(static_cast<int32_t*>(
                                    stride.tensorParam.v1.clientBuf.data)[1]);
  m_kernelArgs.push_back(strideWidthTensor);

  QnnGpu_KernelArg_t strideHeightTensor = QNN_GPU_KERNEL_ARG_INIT;
  strideHeightTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  strideHeightTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  strideHeightTensor.data.qnnInt =static_cast<int32_t>(static_cast<int32_t*>(
                                   stride.tensorParam.v1.clientBuf.data)[0]);
  m_kernelArgs.push_back(strideHeightTensor);

  QnnGpu_KernelArg_t groupTensor = QNN_GPU_KERNEL_ARG_INIT;
  groupTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  groupTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  groupTensor.data.qnnInt = static_cast<int32_t>(group.scalarParam.int32Value);
  m_kernelArgs.push_back(groupTensor);

  QnnGpu_KernelArg_t dilationXTensor = QNN_GPU_KERNEL_ARG_INIT;
  dilationXTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  dilationXTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  dilationXTensor.data.qnnInt = static_cast<int32_t>(static_cast<int32_t*>(
                                dilations.tensorParam.v1.clientBuf.data)[1]);
  m_kernelArgs.push_back(dilationXTensor);

  QnnGpu_KernelArg_t dilationYTensor = QNN_GPU_KERNEL_ARG_INIT;
  dilationYTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  dilationYTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  dilationYTensor.data.qnnInt = static_cast<int32_t>(static_cast<int32_t*>(
                              dilations.tensorParam.v1.clientBuf.data)[0]);
  m_kernelArgs.push_back(dilationYTensor);

  uint32_t inSizePerBatch =  input.v1.dimensions[1] * input.v1.dimensions[2] * input.v1.dimensions[3];
  uint32_t outSizePerBatch = output.v1.dimensions[1] * output.v1.dimensions[2] * output.v1.dimensions[3];

  QnnGpu_KernelArg_t inSizePerBatchTensor = QNN_GPU_KERNEL_ARG_INIT;
  inSizePerBatchTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  inSizePerBatchTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  inSizePerBatchTensor.data.qnnInt = static_cast<int32_t>(inSizePerBatch);
  m_kernelArgs.push_back(inSizePerBatchTensor);

  QnnGpu_KernelArg_t outSizePerBatchTensor = QNN_GPU_KERNEL_ARG_INIT;
  outSizePerBatchTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  outSizePerBatchTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  outSizePerBatchTensor.data.qnnInt = static_cast<int32_t>(outSizePerBatch);
  m_kernelArgs.push_back(outSizePerBatchTensor);

  QnnGpu_KernelArg_t outputWidthTensor = QNN_GPU_KERNEL_ARG_INIT;
  outputWidthTensor.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  outputWidthTensor.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  outputWidthTensor.data.qnnInt = static_cast<int32_t>(output.v1.dimensions[2]);
  m_kernelArgs.push_back(outputWidthTensor);

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
    #pragma OPENCL EXTENSION
    __kernel void conv2d_udo_kernel(
    __global float* input,
    __global float* filters,
    __constant float* bias,
    __global float* output,
    int input_width,
    int input_height,
    int input_depth,
    int numFilters,
    int filterX,
    int filterY,
    int paddingX,
    int paddingY,
    int strideX,
    int strideY,
    int groups,
    int dilationX,
    int dilationY,
    int inSizePerBatch,
    int outSizePerBatch,
    int outputWidth
    )
{
  int batchIdx        = get_global_id(0);
  int outputHeightIdx = get_global_id(1);
  int outputWidthIdx  = get_global_id(2);

  int offsetIn = mad24(batchIdx ,inSizePerBatch, 0);
  int offsetOut = mad24(batchIdx, outSizePerBatch, 0);
  __global float *inputPtr = input + offsetIn;

  int iGroupDepth = native_divide(input_depth, groups);
  int oGroupDepth = native_divide(numFilters ,groups);
  int filterExt   = mad24(numFilters, iGroupDepth, 0);

  int realY = mad24(outputHeightIdx, strideY, -paddingY);
  int aymin = ceil(native_divide((float)-realY , dilationY));
  int aymax = ceil(native_divide((float)(input_height - realY), dilationY));
  aymin = aymin > 0 ? aymin : 0;
  aymax = filterY < aymax ? filterY : aymax;

  int realX = mad24(outputWidthIdx, strideX, -paddingX);
  int axmin = ceil(native_divide((float)-realX, (float)dilationX));
  int axmax = ceil(native_divide((float)(input_width - realX), (float)dilationX));
  axmin = axmin > 0 ? axmin : 0;
  axmax = filterX < axmax ? filterX : axmax;

  __global float *out = output + mad24(numFilters ,mad24(outputWidth, outputHeightIdx,
                                                           outputWidthIdx), offsetOut);

  if (bias != NULL) {
    for(int biasIdx = 0; biasIdx < numFilters/4; biasIdx++) {
      vstore4(vload4(0, bias + biasIdx * 4), 0, out + biasIdx * 4);
    }
    for (int idx = 0; idx < numFilters%4; idx++) {
      out[idx] = bias[mad24((numFilters/4), 4, idx)];
    }
  }

  for (int ky = aymin; ky < aymax; ++ky) {
    int y = realY + ky * dilationY;
    for (int kx = axmin; kx < axmax; ++kx) {
      int x = realX + kx * dilationX;
      __global float* filter = filters + filterExt * mad24(filterX, ky, kx);
      for (int g = 0; g < groups; ++g) {
        for (int c = 0; c < iGroupDepth; ++c) {
          float f = inputPtr[mad24(g , iGroupDepth , mad24(input_depth, mad24(input_width,y,x), c))];
          float outputValue = 0.0f;
          for (int z = (g * oGroupDepth); z < ((g + 1) * oGroupDepth); z++) {
            out[z] += filter[mad24(numFilters, c, z)] * f;
          }
        }
      }
    }
  }
}
  )";

  m_kernelName           = "conv2d_udo_kernel";
  gpuKernel.name         = m_kernelName.c_str();
  gpuKernel.sourceType   = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  gpuKernel.kernelSource = m_kernelSource.c_str();
  gpuKernel.sourceLength = m_kernelSource.size();
  gpuKernel.buildOptions = "-cl-std=CL2.0";
  KERNEL_RETURN(QNN_SUCCESS);
}