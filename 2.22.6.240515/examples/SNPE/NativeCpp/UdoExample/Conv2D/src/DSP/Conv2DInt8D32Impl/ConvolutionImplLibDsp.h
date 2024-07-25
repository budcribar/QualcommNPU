//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
/*
 * structs and enums used in UDO implementation libraries
 */

#include "SnpeUdo/UdoImplDsp.h"
#include "utils/UdoDspShared.h"

// structs of op factories

typedef struct Convolution_OpFactory_t {
    SnpeUdo_String_t opType;
    uint32_t numOfStaticParams;
    SnpeUdo_Param_t* staticParams;
} ConvolutionOpFactory;

UdoDspShared* new_Convolution(SnpeUdo_HexNNv2GlobalInfra_t* infra);    //constructor

SnpeUdo_ErrorType_t Convolution_QueryOperation(SnpeUdo_String_t operationType,
                                               uint32_t numOfStaticParams,
                                               const SnpeUdo_Param_t* staticParams,
                                               uint32_t* numOfInputs,
                                               SnpeUdo_QuantizationType_t** inputsQuantTypes,
                                               SnpeUdo_HexNNTensorLayout_t** inputsLayouts,
                                               uint32_t* numOfOutputs,
                                               SnpeUdo_QuantizationType_t** outputsQuantTypes,
                                               SnpeUdo_HexNNTensorLayout_t** outputsLayouts);

SnpeUdo_ErrorType_t Convolution_ValidateOperation(SnpeUdo_String_t operationType,
                                                  uint32_t numOfStaticParams,
                                                  const SnpeUdo_Param_t* staticParams);

SnpeUdo_ErrorType_t Convolution_CreateOpFactory(SnpeUdo_HexNNv2GlobalInfra_t* infra,
                                                SnpeUdo_CoreType_t udoCoreType,
                                                void* perFactoryInfrastructure,
                                                SnpeUdo_String_t operationType,
                                                uint32_t numOfStaticParams,
                                                SnpeUdo_Param_t* staticParams,
                                                SnpeUdo_OpFactory_t* opFactory);

SnpeUdo_ErrorType_t Convolution_ExecuteOp(SnpeUdo_HexNNv2GlobalInfra_t* infra,
                                          SnpeUdo_Operation_t operation,
                                          bool blocking,
                                          const uint32_t ID,
                                          SnpeUdo_ExternalNotify_t notifyFunc);


typedef struct{
    uint32_t B;
    uint32_t H;
    uint32_t W;
    uint32_t D;
    uint32_t heightPadBefore;
    uint32_t heightPadAfter;
    uint32_t widthPadBefore;
    uint32_t widthPadAfter;
    uint32_t depthPadBefore;
    uint32_t depthPadAfter;
} D32_Params;

typedef struct Conv2DScalarParams_t
{
    int32_t numFilters;
    int32_t biasTerm;
    int32_t padH;
    int32_t padW;
    int32_t strideH;
    int32_t strideW;
    int32_t kernelH;
    int32_t kernelW;
    int32_t groups;
} Conv2DScalarParams;

// op specific execution struct for storing inputs, outputs and multithreading info
typedef struct ConvolutionOpInfo_t {
    SnpeUdo_TensorParam_t* in_tensor;
    SnpeUdo_TensorParam_t* out_tensor;
    SnpeUdo_TensorParam_t* filt_tensor;
    SnpeUdo_TensorParam_t* bias_tensor;
    Conv2DScalarParams* scalarParams;
    D32_Params* in_d32prms;
    D32_Params* out_d32prms;
} ConvolutionOpInfo;
