//=============================================================================
//
//  Copyright (c) 2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#pragma once

#include "DSP/QnnDspOpPackage.h"

typedef Udo_ErrorType_t (*fptrQueryOperation)(Udo_String_t,
                                              uint32_t,
                                              const Udo_Param_t *,
                                              uint32_t *,
                                              Udo_QuantizationType_t **,
                                              Udo_HexNNTensorLayout_t **,
                                              uint32_t *,
                                              Udo_QuantizationType_t **,
                                              Udo_HexNNTensorLayout_t **);

typedef Udo_ErrorType_t (*fptrValidateOperation)(Udo_String_t, uint32_t, const Udo_Param_t *);
typedef Udo_ErrorType_t (*fptrCreateOpFactory)(QnnOpPackage_GlobalInfrastructure_t,
                                               Udo_CoreType_t,
                                               void *,
                                               Udo_String_t,
                                               uint32_t,
                                               Udo_Param_t *,
                                               Udo_OpFactory_t *);

typedef Udo_ErrorType_t (*fptrReleaseOpFactory)(QnnOpPackage_GlobalInfrastructure_t,
                                                Udo_OpFactory_t);

typedef Udo_ErrorType_t (*fptrExecuteOp)(QnnOpPackage_GlobalInfrastructure_t,
                                         Udo_Operation_t,
                                         bool,
                                         const uint32_t,
                                         Udo_ExternalNotify_t);

typedef struct UdoDspShared {
  char *opType;
  uint32_t numOfStaticParams;
  uint32_t numOfInputs;
  uint32_t numOfOutputs;
  fptrQueryOperation queryOp;
  fptrValidateOperation validateOp;
  fptrCreateOpFactory createOpFactory;
  fptrReleaseOpFactory releaseOpFactory;
  fptrExecuteOp executeOp;
} UdoDspShared_t;

typedef struct OpFactory {
  Udo_String_t opType;
} OpFactory_t;

typedef struct OpParams {
  Udo_OpFactory_t opFactory;
  uint32_t numInputParams;
  Udo_TensorParam_t *InputParams;
  uint32_t numOutputParams;
  Udo_TensorParam_t *outputParams;
  Udo_HexNNv2OpInfra_t opInfra;
} OpParams_t;

typedef struct ReluOpInfo {
  uint8_t *input;
  uint32_t inputsLen;
  float inputMin;
  float inputMax;
  float outputMin;
  float outputMax;
  uint8_t *output;
} ReluOpInfo_t;

typedef struct ReluOpFactory {
  Udo_String_t opType;
  uint32_t numOfStaticParams;
  Udo_Param_t *staticParams;
} ReluOpFactory_t;

UdoDspShared *new_relu(QnnOpPackage_GlobalInfrastructure_t sg_globalInfra);

Udo_ErrorType_t free_relu(QnnOpPackage_GlobalInfrastructure_t sg_globalInfra, UdoDspShared *opInfo);
