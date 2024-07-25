//=============================================================================
//
//  Copyright (c) 2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"

// operations info
char g_reluOpType[]                                    = "Relu";
uint32_t g_reluStaticParamsNum                         = 0;
uint32_t g_reluInputsNum                               = 1;
uint32_t g_reluOutputsNum                              = 1;
Udo_QuantizationType_t g_reluInputQuantizationTypes[]  = {UDO_QUANTIZATION_TF};
Udo_QuantizationType_t g_reluOutputQuantizationTypes[] = {UDO_QUANTIZATION_TF};
Udo_HexNNTensorLayout_t *g_reluLayout                  = NULL;

Udo_ErrorType_t relu_createOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                     Udo_CoreType_t udoCoreType,
                                     void *perFactoryInfrastructure,
                                     Udo_String_t operationType,
                                     uint32_t numOfStaticParams,
                                     Udo_Param_t *staticParams,
                                     Udo_OpFactory_t *opFactory) {
  if (operationType == NULL || opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (strcmp(operationType, g_reluOpType) == 0) {
    ReluOpFactory_t *thisFactory = (ReluOpFactory_t *)(*(
        globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(ReluOpFactory_t));
    int size                     = strlen(operationType) + 1;  // +1 to hold the '\0' character
    thisFactory->opType =
        (Udo_String_t)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
    strlcpy((thisFactory->opType), operationType, size);
    thisFactory->numOfStaticParams = numOfStaticParams;
    /*
     * if this op has static params, add code here
     */
    *opFactory = (Udo_OpFactory_t)thisFactory;
  } else {
    return UDO_INVALID_ARGUMENT;
  }
  return UDO_NO_ERROR;
}

Udo_ErrorType_t relu_releaseOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                      Udo_OpFactory_t opFactory) {
  if (opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  ReluOpFactory_t *thisFactory = (ReluOpFactory_t *)(opFactory);
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->opType));
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(thisFactory);
  /*
   * if this op has static params, add code here
   */
  return UDO_NO_ERROR;
}

Udo_ErrorType_t relu_validateOperation(Udo_String_t operationType,
                                       uint32_t numOfStaticParams,
                                       const Udo_Param_t *staticParams) {
  if (strcmp(operationType, g_reluOpType) == 0) {
    if (numOfStaticParams != g_reluStaticParamsNum) {
      return UDO_INVALID_ARGUMENT;
    }
    /*
     * If this op should validate others, add code here
     */
  } else {
    return UDO_INVALID_ARGUMENT;
  }
  return UDO_NO_ERROR;
}

static inline int32_t quantize_int(float val, float minval, float maxval) {
  /* We want 0.0 -- 255.0 to resize to 0..255 */
  float range     = fmaxf(1e-18f, maxval - minval);
  float resizeAmt = 255.0f / (range);
  float valueF    = (val - minval) * resizeAmt;
  int32_t value   = roundf(valueF);
  return (-1) * value;
}

void workerThreadReluQuant(void *perOpInfrastructure, void *userData) {
  ReluOpInfo_t *data = (ReluOpInfo_t *)userData;
  uint8_t *input     = data->input;
  uint32_t inputsLen = data->inputsLen;
  float inputMin     = data->inputMin;
  float inputMax     = data->inputMax;
  float deltaIn      = (inputMax - inputMin) / 255;
  float offsetIn     = (float)quantize_int(0.0f, inputMin, inputMax);

  uint8_t *output = data->output;
  float outputMin = data->outputMin;
  float outputMax = data->outputMax;
  float deltaOut  = (outputMax - outputMin) / 255;
  float offsetOut = (float)quantize_int(0.0f, outputMin, outputMax);

  for (size_t i = 0; i < inputsLen; i++) {
    float in  = (input[i] + offsetIn) * deltaIn;
    float out = in > 0.0 ? in : 0.0;
    output[i] = (out / deltaOut) - offsetOut;
  }
}

Udo_ErrorType_t relu_executeOp(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                               Udo_Operation_t operation,
                               bool blocking,
                               const uint32_t ID,
                               Udo_ExternalNotify_t notifyFunc) {
  if (operation == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  OpParams_t *m_Operation = (OpParams_t *)operation;
  const char *opType      = ((ReluOpFactory_t *)(m_Operation->opFactory))->opType;
  if (opType == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (strcmp(opType, g_reluOpType) == 0) {
    /*
     * add code here
     */
    Udo_TensorParam_t *in_tensor  = &(m_Operation->InputParams[0]);
    Udo_TensorParam_t *out_tensor = &(m_Operation->outputParams[0]);

    if (in_tensor->layout == UDO_LAYOUT_NULL || out_tensor->layout == UDO_LAYOUT_NULL) {
      return UDO_UNSUPPORTED_FEATURE;
    }

    uint32_t inputLen = sizeof(uint8_t);
    for (int k = 0; k < in_tensor->tensorRank; k++) {
      inputLen *= in_tensor->currDimensions[k];
      out_tensor->currDimensions[k] = in_tensor->currDimensions[k];
    }

    float inputMin = in_tensor->quantizeParams.TFParams.minValue;
    float inputMax = in_tensor->quantizeParams.TFParams.maxValue;

    float outputMin = out_tensor->quantizeParams.TFParams.minValue;
    float outputMax = out_tensor->quantizeParams.TFParams.maxValue;

    uint8_t *inputTensorData  = (uint8_t *)(in_tensor->tensorData);
    uint8_t *outputTensorData = (uint8_t *)(out_tensor->tensorData);

    out_tensor->dataType = UDO_DATATYPE_FIXED_8;
    if ((*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoSetOutputTensorSize))(
            m_Operation->opInfra, 0, inputLen) != 0) {
      return UDO_UNSUPPORTED_FEATURE;
    }
    ReluOpInfo_t workerThreadIn = {
        inputTensorData, inputLen, inputMin, inputMax, outputMin, outputMax, outputTensorData};
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoRunWorkerThreads))(
        m_Operation->opInfra, 1, workerThreadReluQuant, &workerThreadIn);
    return UDO_NO_ERROR;
  } else {
    return UDO_INVALID_ARGUMENT;
  }
}

Udo_ErrorType_t relu_queryOperation(Udo_String_t operationType,
                                    uint32_t numOfStaticParams,
                                    const Udo_Param_t *staticParams,
                                    uint32_t *numOfInputs,
                                    Udo_QuantizationType_t **inputsQuantTypes,
                                    Udo_HexNNTensorLayout_t **inputsLayouts,
                                    uint32_t *numOfOutputs,
                                    Udo_QuantizationType_t **outputsQuantTypes,
                                    Udo_HexNNTensorLayout_t **outputsLayouts) {
  if (strcmp(operationType, g_reluOpType) == 0) {
    *numOfInputs       = g_reluInputsNum;
    *inputsQuantTypes  = g_reluInputQuantizationTypes;
    *inputsLayouts     = g_reluLayout;
    *numOfOutputs      = g_reluOutputsNum;
    *outputsQuantTypes = g_reluOutputQuantizationTypes;
    *outputsLayouts    = g_reluLayout;
  } else {
    return UDO_WRONG_OPERATION;
  }
  return UDO_NO_ERROR;
}

UdoDspShared *new_relu(QnnOpPackage_GlobalInfrastructure_t globalInfra) {
  UdoDspShared *pOpObj = (UdoDspShared *)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(
      sizeof(UdoDspShared));
  if (pOpObj == NULL) {
    return NULL;
  }
  pOpObj->opType            = g_reluOpType;
  pOpObj->numOfStaticParams = g_reluStaticParamsNum;
  pOpObj->numOfInputs       = g_reluInputsNum;
  pOpObj->numOfOutputs      = g_reluOutputsNum;

  pOpObj->createOpFactory  = relu_createOpFactory;
  pOpObj->releaseOpFactory = relu_releaseOpFactory;
  pOpObj->validateOp       = relu_validateOperation;
  pOpObj->executeOp        = relu_executeOp;
  pOpObj->queryOp          = relu_queryOperation;
  return pOpObj;
}

Udo_ErrorType_t free_relu(QnnOpPackage_GlobalInfrastructure_t globalInfra, UdoDspShared *opInfo) {
  if (opInfo == NULL) {
    return UDO_NO_ERROR;
  }
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(opInfo);
  return UDO_NO_ERROR;
}
