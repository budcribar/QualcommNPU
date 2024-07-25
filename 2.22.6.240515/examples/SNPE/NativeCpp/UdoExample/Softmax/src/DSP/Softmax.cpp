//<!--
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//-->

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"

// operations info
char g_softmaxOpType[]                                    = "Softmax";
uint32_t g_softmaxStaticParamsNum                         = 1;
uint32_t g_softmaxInputsNum                               = 1;
uint32_t g_softmaxOutputsNum                              = 1;
Udo_QuantizationType_t g_softmaxInputQuantizationTypes[]  = {UDO_QUANTIZATION_TF};
Udo_QuantizationType_t g_softmaxOutputQuantizationTypes[] = {UDO_QUANTIZATION_TF};
Udo_HexNNTensorLayout_t *g_softmaxLayout                  = NULL;

Udo_ErrorType_t softmax_createOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                        Udo_CoreType_t udoCoreType,
                                        void *perFactoryInfrastructure,
                                        Udo_String_t operationType,
                                        uint32_t numOfStaticParams,
                                        Udo_Param_t *staticParams,
                                        Udo_OpFactory_t *opFactory) {
  if (operationType == NULL || opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (strcmp(operationType, g_softmaxOpType) == 0) {
    softmaxOpFactory_t *thisFactory = (softmaxOpFactory_t *)(*(
        globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(softmaxOpFactory_t));
    int size                        = strlen(operationType) + 1;  // +1 to hold the '\0' character
    thisFactory->opType =
        (Udo_String_t)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
    strlcpy((thisFactory->opType), operationType, size);
    thisFactory->numOfStaticParams = numOfStaticParams;
    *opFactory                     = (Udo_OpFactory_t)thisFactory;
  } else {
    return UDO_INVALID_ARGUMENT;
  }
  return UDO_NO_ERROR;
}

Udo_ErrorType_t softmax_releaseOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                         Udo_OpFactory_t opFactory) {
  if (opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  softmaxOpFactory_t *thisFactory = (softmaxOpFactory_t *)(opFactory);
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->opType));
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(thisFactory);
  return UDO_NO_ERROR;
}

Udo_ErrorType_t softmax_validateOperation(Udo_String_t operationType,
                                          uint32_t numOfStaticParams,
                                          const Udo_Param_t *staticParams) {
  if (strcmp(operationType, g_softmaxOpType) == 0) {
    if (numOfStaticParams != g_softmaxStaticParamsNum) {
      return UDO_INVALID_ARGUMENT;
    }
  } else {
    return UDO_INVALID_ARGUMENT;
  }
  return UDO_NO_ERROR;
}

typedef struct SoftmaxOpInfo_t {
  uint8_t *input;
  uint32_t inputsLen;
  float inputMin;
  float inputMax;
  float outputMin;
  float outputMax;
  uint32_t depth;
  uint8_t *output;
} SoftmaxOpInfo_t;

static float expfApprox(float x) {
  float val  = x * (float)(16384.0 / 0.69314718056);
  int xfrac  = (int)(val + copysignf(0.5f, val));
  float xf   = (xfrac & 0x3FFF) * (float)(1. / 16384.);
  float exp0 = 1.0f + xf * (0.69051585f + xf * (0.23793659f + xf * 0.07154756f));
  float exp  = powf(2.0f, xfrac >> 14);

  return (exp * exp0);
}

void workerThreadSoftmaxQuant(void *perOpInfrastructure, void *userData) {
  SoftmaxOpInfo_t *data = (SoftmaxOpInfo_t *)userData;
  uint8_t *input        = data->input;
  uint32_t inputsLen    = data->inputsLen;
  uint8_t *output       = data->output;
  float inputMin        = data->inputMin;
  float inputMax        = data->inputMax;
  uint32_t depth        = data->depth;

  float stepsize = (inputMax - inputMin) * (1.0f / 255.0f);

  float sum = 0.0f;
  uint8_t maxval;
  float temp_slice[depth];
  uint8_t *in;
  uint8_t *out;
  float recip;

  if (stepsize < 0.63529f) {
    for (size_t i = 0; i < inputsLen / depth; i++) {
      in  = input + i * depth;
      out = output + i * depth;
      sum = 0.0f;

      for (size_t j = 0; j < depth; j++) {
        float exp     = expfApprox(stepsize * in[j] - 83.0f);
        temp_slice[j] = exp;
        sum += exp;
      }
      recip = 255.0f / sum;
      for (size_t j = 0; j < depth; j++) {
        int val = roundf(recip * temp_slice[j]);
        out[j]  = (val < 0) ? 0 : ((val > 255) ? 255 : val);
      }
    }
  } else {
    for (size_t i = 0; i < inputsLen / depth; i++) {
      in     = input + i * depth;
      out    = output + i * depth;
      sum    = 0.0f;
      maxval = (uint8_t)in[0];

      for (size_t j = 0; j < depth; j++) {
        maxval = (maxval < in[j]) ? in[j] : maxval;
      }

      for (size_t j = 0; j < depth; j++) {
        float exp     = expfApprox(stepsize * (in[j] - maxval));
        temp_slice[j] = exp;
        sum += exp;
      }
      recip = 255.0f / sum;
      for (size_t j = 0; j < depth; j++) {
        int val = roundf(recip * temp_slice[j]);
        out[j]  = (val < 0) ? 0 : ((val > 255) ? 255 : val);
      }
    }
  }
}

Udo_ErrorType_t softmax_executeOp(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                  Udo_Operation_t operation,
                                  bool blocking,
                                  const uint32_t ID,
                                  Udo_ExternalNotify_t notifyFunc) {
  if (operation == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  OpParams_t *mOperation = (OpParams_t *)operation;
  const char *opType     = ((softmaxOpFactory_t *)(mOperation->opFactory))->opType;
  if (opType == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (strcmp(opType, g_softmaxOpType) == 0) {
    Udo_TensorParam_t *input = &(mOperation->InputParams[0]);
    Udo_TensorParam_t *out   = mOperation->outputParams;

    if (input->layout == UDO_LAYOUT_NULL || out->layout == UDO_LAYOUT_NULL) {
      return UDO_UNSUPPORTED_FEATURE;
    }

    uint32_t inputLen = sizeof(uint8_t);
    for (int k = 0; k < input->tensorRank; k++) {
      inputLen *= input->currDimensions[k];
      out->currDimensions[k] = input->currDimensions[k];
    }

    float inputMin = input->quantizeParams.TFParams.minValue;
    float inputMax = input->quantizeParams.TFParams.maxValue;

    float outputMin = out->quantizeParams.TFParams.minValue;
    float outputMax = out->quantizeParams.TFParams.maxValue;

    uint8_t *inputTensorData  = (uint8_t *)(input->tensorData);
    uint8_t *outputTensorData = (uint8_t *)(out->tensorData);

    out->dataType = UDO_DATATYPE_FIXED_8;
    // required to set output tensor sizes
    if ((*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoSetOutputTensorSize))(
            mOperation->opInfra, 0, inputLen) != 0) {
      return UDO_UNSUPPORTED_FEATURE;
    }

    SoftmaxOpInfo_t workerThreadIn = {inputTensorData,
                                      inputLen,
                                      inputMin,
                                      inputMax,
                                      outputMin,
                                      outputMax,
                                      input->currDimensions[3],
                                      outputTensorData};

    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoRunWorkerThreads))(
        mOperation->opInfra, 1, workerThreadSoftmaxQuant, &workerThreadIn);

    return UDO_NO_ERROR;
  } else {
    return UDO_INVALID_ARGUMENT;
  }
}

Udo_ErrorType_t softmax_queryOperation(Udo_String_t operationType,
                                       uint32_t numOfStaticParams,
                                       const Udo_Param_t *staticParams,
                                       uint32_t *numOfInputs,
                                       Udo_QuantizationType_t **inputsQuantTypes,
                                       Udo_HexNNTensorLayout_t **inputsLayouts,
                                       uint32_t *numOfOutputs,
                                       Udo_QuantizationType_t **outputsQuantTypes,
                                       Udo_HexNNTensorLayout_t **outputsLayouts) {
  if (strcmp(operationType, g_softmaxOpType) == 0) {
    *numOfInputs       = g_softmaxInputsNum;
    *inputsQuantTypes  = g_softmaxInputQuantizationTypes;
    *inputsLayouts     = g_softmaxLayout;
    *numOfOutputs      = g_softmaxOutputsNum;
    *outputsQuantTypes = g_softmaxOutputQuantizationTypes;
    *outputsLayouts    = g_softmaxLayout;
  } else {
    return UDO_WRONG_OPERATION;
  }
  return UDO_NO_ERROR;
}

UdoDspShared *new_softmax(QnnOpPackage_GlobalInfrastructure_t globalInfra) {
  UdoDspShared *pOpObj = (UdoDspShared *)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(
      sizeof(UdoDspShared));
  if (pOpObj == NULL) {
    return NULL;
  }
  pOpObj->opType            = g_softmaxOpType;
  pOpObj->numOfStaticParams = g_softmaxStaticParamsNum;
  pOpObj->numOfInputs       = g_softmaxInputsNum;
  pOpObj->numOfOutputs      = g_softmaxOutputsNum;

  pOpObj->createOpFactory  = softmax_createOpFactory;
  pOpObj->releaseOpFactory = softmax_releaseOpFactory;
  pOpObj->validateOp       = softmax_validateOperation;
  pOpObj->executeOp        = softmax_executeOp;
  pOpObj->queryOp          = softmax_queryOperation;
  return pOpObj;
}

Udo_ErrorType_t free_softmax(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                             UdoDspShared *opInfo) {
  if (opInfo == NULL) {
    return UDO_NO_ERROR;
  }
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(opInfo);
  return UDO_NO_ERROR;
}
