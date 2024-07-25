//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <stdio.h>

#include "QnnOpPackage.h"
#include "QnnTypes.h"

#ifndef EXPORT_API
#if defined _MSC_VER
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif
#endif

#define NUM_OF_INPUTS  (1)
#define NUM_OF_OUTPUTS (1)
#define NUM_OF_PARAMS  (0)

// Sample implementation to populate output shapes of Custom Relu op
extern "C" {
EXPORT_API Qnn_ErrorHandle_t ReluShapeInference(Qnn_OpConfig_t *op) {
  printf("[RELU_IMPL] : CustomOp Infer Output shapes started..\n");

  // optional validation to validate number of inputs
  if (op->v1.numOfInputs != NUM_OF_INPUTS) {
    printf("[RELU_IMPL] : No. of inputs mismatch, expected : %d, got %d : \n",
           NUM_OF_INPUTS,
           op->v1.numOfInputs);
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }

  // optional validation to validate number of outputs
  if (op->v1.numOfOutputs != NUM_OF_OUTPUTS) {
    printf("[RELU_IMPL] : No. of outputs mismatch, expected : %d, got %d : \n",
           NUM_OF_OUTPUTS,
           op->v1.numOfOutputs);
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }

  // optional validation to validate number of params
  if (op->v1.numOfParams != NUM_OF_PARAMS) {
    printf("[RELU_IMPL] : No. of params mismatch, expected : %d, got %d : \n",
           NUM_OF_PARAMS,
           op->v1.numOfParams);
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }

  for (uint32_t i = 0; i < op->v1.numOfOutputs; i++) {
    Qnn_Tensor_t *out     = &op->v1.outputTensors[i];
    Qnn_Tensor_t *in      = &op->v1.inputTensors[i];
    uint32_t *input_dims  = in->v1.dimensions;
    uint32_t *output_dims = out->v1.dimensions;
    for (uint32_t j = 0; j < in->v1.rank; j++) {
      output_dims[j] = input_dims[j];
    }
    out->v1.dimensions = output_dims;
  }

  printf("[RELU_IMPL] : CustomOp Infer Output shapes done!\n");
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*RelushapeInferencePtr)(Qnn_OpConfig_t *) = &ReluShapeInference;

}  // extern "C"