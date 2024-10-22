//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOADINPUTTENSOR_H
#define LOADINPUTTENSOR_H

#include <unordered_map>
#include <string>
#include <vector>

#include "SNPE/SNPE.h"
#include "DlSystem/TensorMap.h"

/**
 * @brief .
 *
 * @param[in] snpeHandle snpe Handle
 *
 * @param[in] fileLines Files names required in one batch
 *
 * @param[in] inputMapHandle  Handle of the input Buffer names
 *
 * @param[in] staticQuantization flag to specify whether to use static Quantization or not
 *
 * @param[in] bitWidth Bit Width for the selected user buffer
 *
 * @param[in] MemMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @returns boolean value according the loading of input TFN was successful or not
 */
bool LoadInputUserBufferTfN(Snpe_SNPE_Handle_t snpeHandle,
                            std::vector<std::string>& fileLines,
                            Snpe_UserBufferMap_Handle_t inputMapHandle,
                            bool staticQuantization,
                            int bitWidth, Snpe_UserMemoryMap_Handle_t MemMappedBufferMapHandle);

/**
 * @brief .
 *
 * @param snpeHandle snpe Handle
 *
 * @param fileLines Files names required in one batch
 *
 * @param MemMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @returns boolean value according the loading of input Float was successful or not
 */
bool LoadInputUserBufferFloat(Snpe_SNPE_Handle_t snpeHandle,
                              std::vector<std::string>& fileLines, Snpe_UserMemoryMap_Handle_t MemMappedBufferMapHandle);
#endif
