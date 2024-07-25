//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "BuildPSNPE.hpp"

#include <iostream>
#include "SNPE/PSNPE.h"
#include "DlContainer/DlContainer.h"
#include "SNPE/RuntimeConfigList.h"
#include "DlSystem/DlEnums.h"


Snpe_PSNPE_Handle_t BuildPSNPE(Snpe_DlContainer_Handle_t dlcHandle,
                               const std::vector<Snpe_Runtime_t>& runtimes,
                               Snpe_PSNPE_InputOutputTransmissionMode_t executionMode,
                               Snpe_PerformanceProfile_t perfProfile,
                               bool usingInitCache,
                               bool cpuFixedPointMode,
                               size_t outputThreadNum,
                               void (*callbackFunc)(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t))
{

    // Prepare runtime list config handle
    Snpe_RuntimeConfigList_Handle_t runtimeConfigListHandle = Snpe_RuntimeConfigList_Create();
    for (Snpe_Runtime_t runtime : runtimes) {
        Snpe_RuntimeConfig_Handle_t runtimeConfigHandle = Snpe_RuntimeConfig_Create();
        Snpe_RuntimeConfig_SetRuntime(runtimeConfigHandle, runtime);
        Snpe_RuntimeConfig_SetEnableCPUFallback(runtimeConfigHandle, false);
        Snpe_RuntimeConfig_SetPerformanceProfile(runtimeConfigHandle, perfProfile);
        Snpe_RuntimeConfigList_PushBack(runtimeConfigListHandle, runtimeConfigHandle);

        //Snpe_RuntimeConfig_Delete(runtimeConfigHandle);
    }

    // Prepare PlatformOptions for some special setting
    static std::string platformOptions;
    if (cpuFixedPointMode) {
        platformOptions = "enableCpuFxpMode:ON";
    }

    // Create build config handle and set all parameters
    Snpe_BuildConfig_Handle_t psnpeConfigHandle = Snpe_BuildConfig_Create();
    Snpe_BuildConfig_SetContainer(psnpeConfigHandle, dlcHandle);
    Snpe_BuildConfig_SetRuntimeConfigList(psnpeConfigHandle, runtimeConfigListHandle);
    Snpe_BuildConfig_SetInputOutputTransmissionMode(psnpeConfigHandle, executionMode);
    Snpe_BuildConfig_SetEnableInitCache(psnpeConfigHandle, usingInitCache);
    Snpe_BuildConfig_SetPlatformOptions(psnpeConfigHandle, platformOptions.c_str());

    if (executionMode == SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_OUTPUTASYNC) {
        Snpe_BuildConfig_SetOutputThreadNumbers(psnpeConfigHandle, outputThreadNum);
        Snpe_BuildConfig_SetOutputCallback(psnpeConfigHandle, callbackFunc);
    }


    // Create and build PSNPE handle
    Snpe_PSNPE_Handle_t psnpeHandle = Snpe_PSNPE_Create();
    Snpe_ErrorCode_t status = Snpe_PSNPE_Build(psnpeHandle, psnpeConfigHandle);

    // Delete handle which are no longer needed
    Snpe_RuntimeConfigList_Delete(runtimeConfigListHandle);

    if (SNPE_SUCCESS != status) {
        std::cerr << "Error: PSNPE build failed. status:" << static_cast<typename std::underlying_type<decltype(status)>::type>(status) << std::endl;
        std::cerr << Snpe_ErrorCode_GetLastErrorString() << std::endl;
        Snpe_PSNPE_Delete(psnpeHandle);
        return nullptr;
    }

    return psnpeHandle;
}
