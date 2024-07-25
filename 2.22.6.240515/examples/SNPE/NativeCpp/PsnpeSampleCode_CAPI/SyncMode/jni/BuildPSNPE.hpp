//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef BUILD_PSNPE_H
#define BUILD_PSNPE_H

#include <string>
#include <vector>

#include "SNPE/PSNPE.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/PlatformConfig.h"

/**
 * @brief .
 *
 * Create and build PSNPE handle
 *
 * @param dlcHandle handle of contianer file
 *
 * @param runtimes  Selected runtimes
 *
 * @param executionMode PSNPE execution mode
 *
 * @param perfProfile execution performance mode
 *
 * @param usingInitCache Use init cache or not
 *
 * @param cpuFixedPointMode Enable the fixed point execution on CPU runtime
 *
 * @returns returns the handle of Snpe_PSNPE_Handle_t or nullptr
 */
Snpe_PSNPE_Handle_t BuildPSNPE(Snpe_DlContainer_Handle_t dlcHandle,
                               const std::vector<Snpe_Runtime_t>& runtimes,
                               Snpe_PSNPE_InputOutputTransmissionMode_t executionMode,
                               Snpe_PerformanceProfile_t perfProfile,
                               bool usingInitCache,
                               bool cpuFixedPointMode);

#endif //BUILD_PSNPE_H
