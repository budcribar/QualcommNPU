<%doc>
# ==============================================================================
#
#  Copyright (c) 2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%page expression_filter="n" expression_filter="trim" />
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import get_hexnn_tensor_sig, get_hexnn_param_sig,_template_builder, is_valid_cpp_identifier, get_param_order_sig, build_scalar_param%>
<% is_valid_cpp_identifier(operator.type_name.lower()) %>
//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================

#include <string.h>
#include <stdlib.h>

#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"

// operations info
char g_${operator.type_name.lower()}OpType [] = "${operator.type_name}";
uint32_t g_${operator.type_name.lower()}StaticParamsNum = ${len(operator.param)};
uint32_t g_${operator.type_name.lower()}InputsNum = ${len(operator.input)};
uint32_t g_${operator.type_name.lower()}OutputsNum = ${len(operator.output)};
Udo_QuantizationType_t g_${operator.type_name.lower()}InputQuantizationTypes [] = {${','.join('UDO_QUANTIZATION_TF' for input in operator.input)}};
% if operator.type_name == "ArgMax":
<%   quantization_mode = "UDO_QUANTIZATION_NONE" %>
% else:
<%    quantization_mode = "UDO_QUANTIZATION_TF" %>
% endif

Udo_QuantizationType_t g_${operator.type_name.lower()}OutputQuantizationTypes [] =  {${','.join(quantization_mode for input in operator.output)}};
Udo_HexNNTensorLayout_t* g_${operator.type_name.lower()}Layout = NULL;

Udo_ErrorType_t
${operator.type_name.lower()}_createOpFactory (QnnOpPackage_GlobalInfrastructure_t globalInfra,
    Udo_CoreType_t udoCoreType, void *perFactoryInfrastructure,
    Udo_String_t operationType, uint32_t numOfStaticParams,
    Udo_Param_t *staticParams, Udo_OpFactory_t *opFactory)
{
    if(operationType == NULL || opFactory == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    if(strcmp(operationType, g_${operator.type_name.lower()}OpType) == 0) {
        ${operator.type_name.lower()}OpFactory_t* thisFactory = (${operator.type_name.lower()}OpFactory_t *)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(${operator.type_name.lower()}OpFactory_t));
        int size = strlen(operationType) + 1; // +1 to hold the '\0' character
        thisFactory->opType = (Udo_String_t)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
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

Udo_ErrorType_t
${operator.type_name.lower()}_releaseOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                              Udo_OpFactory_t opFactory)
{
    if(opFactory == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    ${operator.type_name.lower()}OpFactory_t* thisFactory = (${operator.type_name.lower()}OpFactory_t *)(opFactory);
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->opType));
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(thisFactory);
    /*
     * if this op has static params, add code here
     */
    return UDO_NO_ERROR;
}

Udo_ErrorType_t
${operator.type_name.lower()}_validateOperation (Udo_String_t operationType, uint32_t numOfStaticParams,
    const Udo_Param_t *staticParams) {
    if(strcmp(operationType, g_${operator.type_name.lower()}OpType) == 0) {
        if (numOfStaticParams != g_${operator.type_name.lower()}StaticParamsNum) {
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

Udo_ErrorType_t
${operator.type_name.lower()}_executeOp (QnnOpPackage_GlobalInfrastructure_t globalInfra,
    Udo_Operation_t operation, bool blocking, const uint32_t ID,
    Udo_ExternalNotify_t notifyFunc) {
    if(operation == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    OpParams_t* m_Operation = (OpParams_t*) operation;
    const char* opType = ((${operator.type_name.lower()}OpFactory_t*)(m_Operation->opFactory))->opType;
    if(opType == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    if(strcmp(opType, g_${operator.type_name.lower()}OpType) == 0) {
        /*
         * add code here
         */
        /*
         * To have good performance and stability, it is required to avoid heap
         * memory allocation in this function. The heap memory allocation
         * includes but not limited to calling malloc, operator new, constructing
         * STL container objects like std::vector with default allocator, and
         * adding items like calling std::vector::push_back to STL container
         * objects with default allocator.
         *
         * Please check in SDK documentation for more information.
         */
        return UDO_NO_ERROR;
    } else {
        return UDO_INVALID_ARGUMENT;
    }
}

Udo_ErrorType_t ${operator.type_name.lower()}_queryOperation (
    Udo_String_t operationType, uint32_t numOfStaticParams,
    const Udo_Param_t *staticParams, uint32_t *numOfInputs,
    Udo_QuantizationType_t **inputsQuantTypes,
    Udo_HexNNTensorLayout_t **inputsLayouts, uint32_t *numOfOutputs,
    Udo_QuantizationType_t **outputsQuantTypes,
    Udo_HexNNTensorLayout_t **outputsLayouts) {
    if(strcmp(operationType, g_${operator.type_name.lower()}OpType) == 0) {
        *numOfInputs = g_${operator.type_name.lower()}InputsNum;
        *inputsQuantTypes = g_${operator.type_name.lower()}InputQuantizationTypes;
        *inputsLayouts = g_${operator.type_name.lower()}Layout;
        *numOfOutputs = g_${operator.type_name.lower()}OutputsNum;
        *outputsQuantTypes = g_${operator.type_name.lower()}OutputQuantizationTypes;
        *outputsLayouts = g_${operator.type_name.lower()}Layout;
    } else {
        return UDO_WRONG_OPERATION;
    }
    return UDO_NO_ERROR;
}

UdoDspShared* new_${operator.type_name.lower()}(QnnOpPackage_GlobalInfrastructure_t globalInfra) {
    UdoDspShared* pOpObj = (UdoDspShared*)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(UdoDspShared));
    if (pOpObj == NULL) {
        return NULL;
    }
    pOpObj->opType = g_${operator.type_name.lower()}OpType;
    pOpObj->numOfStaticParams = g_${operator.type_name.lower()}StaticParamsNum;
    pOpObj->numOfInputs = g_${operator.type_name.lower()}InputsNum;
    pOpObj->numOfOutputs = g_${operator.type_name.lower()}OutputsNum;

    pOpObj->createOpFactory = ${operator.type_name.lower()}_createOpFactory;
    pOpObj->releaseOpFactory = ${operator.type_name.lower()}_releaseOpFactory;
    pOpObj->validateOp = ${operator.type_name.lower()}_validateOperation;
    pOpObj->executeOp = ${operator.type_name.lower()}_executeOp;
    pOpObj->queryOp = ${operator.type_name.lower()}_queryOperation;
    return pOpObj;
}

Udo_ErrorType_t free_${operator.type_name.lower()}(QnnOpPackage_GlobalInfrastructure_t globalInfra, UdoDspShared* opInfo) {
    if (opInfo == NULL) {
        return UDO_NO_ERROR;
    }
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(opInfo);
    return UDO_NO_ERROR;
}
