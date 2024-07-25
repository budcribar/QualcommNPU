//=============================================================================
//
//  Copyright (c) 2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"

// operations info
char g_argmaxOpType[]                                    = "ArgMax";
uint32_t g_argmaxStaticParamsNum                         = 0;
uint32_t g_argmaxInputsNum                               = 1;
uint32_t g_argmaxOutputsNum                              = 1;
Udo_QuantizationType_t g_argmaxInputQuantizationTypes[]  = {UDO_QUANTIZATION_TF};
Udo_QuantizationType_t g_argmaxOutputQuantizationTypes[] = {UDO_QUANTIZATION_NONE};
Udo_HexNNTensorLayout_t* g_argmaxLayout                  = NULL;

Udo_ErrorType_t argmax_createOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                       Udo_CoreType_t udoCoreType,
                                       void* perFactoryInfrastructure,
                                       Udo_String_t operationType,
                                       uint32_t numOfStaticParams,
                                       Udo_Param_t* staticParams,
                                       Udo_OpFactory_t* opFactory) {
  if (operationType == NULL || opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (strcmp(operationType, g_argmaxOpType) == 0) {
    argmaxOpFactory_t* thisFactory = (argmaxOpFactory_t*)(*(
        globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(argmaxOpFactory_t));
    int size                       = strlen(operationType) + 1;  // +1 to hold the '\0' character
    thisFactory->opType =
        (Udo_String_t)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
    strlcpy((thisFactory->opType), operationType, size);
    thisFactory->numOfStaticParams = numOfStaticParams;

    *opFactory = (Udo_OpFactory_t)thisFactory;
  } else {
    return UDO_INVALID_ARGUMENT;
  }
  return UDO_NO_ERROR;
}

Udo_ErrorType_t argmax_releaseOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                        Udo_OpFactory_t opFactory) {
  if (opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  argmaxOpFactory_t* thisFactory = (argmaxOpFactory_t*)(opFactory);
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->opType));
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(thisFactory);

  return UDO_NO_ERROR;
}

Udo_ErrorType_t argmax_validateOperation(Udo_String_t operationType,
                                         uint32_t numOfStaticParams,
                                         const Udo_Param_t* staticParams) {
  if (strcmp(operationType, g_argmaxOpType) == 0) {
    if (numOfStaticParams != g_argmaxStaticParamsNum) {
      return UDO_INVALID_ARGUMENT;
    }

  } else {
    return UDO_INVALID_ARGUMENT;
  }
  return UDO_NO_ERROR;
}

typedef struct ArgMaxOpInfo_t {
  uint8_t* input;
  uint32_t inputsLen;
  uint32_t inputDepth;  // This would be different for input and output. OutputDepth is 1
  uint32_t* output;
} ArgMaxOpInfo;

void worker_thread_ArgMaxQuant(void* perOpInfrastructure, void* userData) {
  ArgMaxOpInfo* data = (ArgMaxOpInfo*)userData;
  uint8_t* input     = data->input;
  int32_t inputsLen  = data->inputsLen;
  int32_t inputDepth = data->inputDepth;
  uint32_t* output   = data->output;
  uint32_t outputLen = inputsLen / inputDepth;

  const uint8_t* in;
  uint32_t* out;
  uint8_t maxval;
  for (uint32_t i = 0; i < outputLen; i++) {
    in     = input + i * inputDepth;
    out    = output + i;
    maxval = in[0];
    out[0] = 0;
    for (uint32_t j = 1; j < inputDepth; j++) {
      if (in[j] > maxval) {
        maxval = in[j];
        out[0] = j;
      }
    }
  }
}

Udo_ErrorType_t argmax_executeOp(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                 Udo_Operation_t operation,
                                 bool blocking,
                                 const uint32_t ID,
                                 Udo_ExternalNotify_t notifyFunc) {
  if (operation == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  OpParams_t* m_Operation = (OpParams_t*)operation;
  const char* opType      = ((argmaxOpFactory_t*)(m_Operation->opFactory))->opType;
  if (opType == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (strcmp(opType, g_argmaxOpType) == 0) {
    Udo_TensorParam_t* input = &(m_Operation->InputParams[0]);
    Udo_TensorParam_t* out   = m_Operation->outputParams;

    if (input->layout == UDO_LAYOUT_NULL || out->layout == UDO_LAYOUT_NULL) {
      return UDO_UNSUPPORTED_FEATURE;
    }

    uint32_t inputLen = sizeof(uint8_t);
    for (int k = 0; k < input->tensorRank; k++) {
      inputLen *= input->currDimensions[k];
      out->currDimensions[k] = input->currDimensions[k];
    }
    out->currDimensions[input->tensorRank - 1] = 1;  // Last dimension of out is 1

    uint8_t* inputTensorData   = (uint8_t*)(input->tensorData);
    uint32_t* outputTensorData = (uint32_t*)(out->tensorData);

    // required to set output tensor sizes
    if ((*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoSetOutputTensorSize))(
            m_Operation->opInfra,
            0,
            (inputLen / input->currDimensions[input->tensorRank - 1]) * sizeof(uint32_t)) != 0) {
      return UDO_UNSUPPORTED_FEATURE;
    }

    ArgMaxOpInfo workerThreadIn = {
        inputTensorData, inputLen, (uint32_t)input->currDimensions[3], outputTensorData};
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoRunWorkerThreads))(
        m_Operation->opInfra, 1, worker_thread_ArgMaxQuant, &workerThreadIn);

    return UDO_NO_ERROR;
  } else {
    return UDO_INVALID_ARGUMENT;
  }
}

Udo_ErrorType_t argmax_queryOperation(Udo_String_t operationType,
                                      uint32_t numOfStaticParams,
                                      const Udo_Param_t* staticParams,
                                      uint32_t* numOfInputs,
                                      Udo_QuantizationType_t** inputsQuantTypes,
                                      Udo_HexNNTensorLayout_t** inputsLayouts,
                                      uint32_t* numOfOutputs,
                                      Udo_QuantizationType_t** outputsQuantTypes,
                                      Udo_HexNNTensorLayout_t** outputsLayouts) {
  if (strcmp(operationType, g_argmaxOpType) == 0) {
    *numOfInputs       = g_argmaxInputsNum;
    *inputsQuantTypes  = g_argmaxInputQuantizationTypes;
    *inputsLayouts     = g_argmaxLayout;
    *numOfOutputs      = g_argmaxOutputsNum;
    *outputsQuantTypes = g_argmaxOutputQuantizationTypes;
    *outputsLayouts    = g_argmaxLayout;
  } else {
    return UDO_WRONG_OPERATION;
  }
  return UDO_NO_ERROR;
}

UdoDspShared* new_argmax(QnnOpPackage_GlobalInfrastructure_t globalInfra) {
  UdoDspShared* pOpObj =
      (UdoDspShared*)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(UdoDspShared));
  if (pOpObj == NULL) {
    return NULL;
  }
  pOpObj->opType            = g_argmaxOpType;
  pOpObj->numOfStaticParams = g_argmaxStaticParamsNum;
  pOpObj->numOfInputs       = g_argmaxInputsNum;
  pOpObj->numOfOutputs      = g_argmaxOutputsNum;

  pOpObj->createOpFactory  = argmax_createOpFactory;
  pOpObj->releaseOpFactory = argmax_releaseOpFactory;
  pOpObj->validateOp       = argmax_validateOperation;
  pOpObj->executeOp        = argmax_executeOp;
  pOpObj->queryOp          = argmax_queryOperation;
  return pOpObj;
}

Udo_ErrorType_t free_argmax(QnnOpPackage_GlobalInfrastructure_t globalInfra, UdoDspShared* opInfo) {
  if (opInfo == NULL) {
    return UDO_NO_ERROR;
  }
  (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(opInfo);
  return UDO_NO_ERROR;
}
