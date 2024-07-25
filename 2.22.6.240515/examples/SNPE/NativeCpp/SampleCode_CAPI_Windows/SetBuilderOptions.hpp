//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SETBUILDEROPTIONS_H
#define SETBUILDEROPTIONS_H

#include "DlSystem/RuntimeList.h"
#include "SNPE/SNPE.h"
#include "DlSystem/DlEnums.h"
#include "DlContainer/DlContainer.h"
#include "DlSystem/PlatformConfig.h"

/**
 * @brief .
 *
 * Sets all the SNPE builder options according to selected and available options
 *
 * @param[in] containerHandle Handle of the loaded container
 *
 * @param[in] runtime  Selected Runtime
 *
 * @param[in] inputRuntimeListHandle Handle to input runtime list
 *
 * @param[in] useUserSuppliedBuffers Whether to use user supplied buffers or not
 *
 * @param[in] platformConfigHandle Handle for setting platform config options
 *
 * @param[in] usingInitCache Use init cache or not
 *
 * @returns returns the snpe handle from the builder handle
 */
Snpe_SNPE_Handle_t setBuilderOptions(Snpe_DlContainer_Handle_t& containerHandle,
                                     Snpe_Runtime_t& runtime,
                                     Snpe_RuntimeList_Handle_t& inputRuntimeListHandle,
                                     bool useUserSuppliedBuffers,
                                     Snpe_PlatformConfig_Handle_t& platformConfigHandle,
                                     bool usingInitCache,
                                     bool cpuFixedPointMode);

#endif //SETBUILDEROPTIONS_H
