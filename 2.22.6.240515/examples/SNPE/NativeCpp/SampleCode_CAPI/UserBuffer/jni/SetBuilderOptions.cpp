//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.h"
#include "DlContainer/DlContainer.h"
#include "SNPE/SNPEBuilder.h"


Snpe_SNPE_Handle_t setBuilderOptions(Snpe_DlContainer_Handle_t& containerHandle,
                                                   Snpe_Runtime_t& runtime,
                                                   Snpe_RuntimeList_Handle_t& inputRuntimeListHandle,
                                                   bool useUserSuppliedBuffers,
                                                   Snpe_PlatformConfig_Handle_t& platformConfigHandle,
                                                   bool usingInitCache,
                                                   bool cpuFixedPointMode)
{
    Snpe_SNPE_Handle_t snpeHandle{};
    Snpe_SNPEBuilder_Handle_t snpeBuilderHandle = Snpe_SNPEBuilder_Create(containerHandle);
    if(Snpe_RuntimeList_Empty(inputRuntimeListHandle)){
        Snpe_RuntimeList_Add(inputRuntimeListHandle, runtime);
    }
    Snpe_SNPEBuilder_SetOutputLayers(snpeBuilderHandle, nullptr);
    Snpe_SNPEBuilder_SetRuntimeProcessorOrder(snpeBuilderHandle, inputRuntimeListHandle);
    Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(snpeBuilderHandle, useUserSuppliedBuffers);
    Snpe_SNPEBuilder_SetPlatformConfig(snpeBuilderHandle, platformConfigHandle);
    Snpe_SNPEBuilder_SetInitCacheMode(snpeBuilderHandle, usingInitCache);
    Snpe_SNPEBuilder_SetCpuFixedPointMode(snpeBuilderHandle, cpuFixedPointMode);
    snpeHandle = Snpe_SNPEBuilder_Build(snpeBuilderHandle);
    Snpe_SNPEBuilder_Delete(snpeBuilderHandle);

    return snpeHandle;
}
