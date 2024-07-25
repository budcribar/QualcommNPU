//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SAVEOUTPUTTENSOR_H
#define SAVEOUTPUTTENSOR_H

#include <string>
#include <unordered_map>
#include <vector>

#include "SNPE/SNPE.h"
#include "DlSystem/ITensor.h"
#include "DlSystem/UserBufferMap.h"


/**
 * @brief .
 *
 * Saves the output of UserBuffer mode after successful execution
 *
 * @param outputMapHandle Handle of the output Buffer Map
 *
 * @param bufSizeMap Map of output buffer name and output buffer size
 *
 * @param outputDir Path to Output directory
 *
 * @param num Batch number
 *
 * @param batchSize current batch size
 *
 * @param isTfNBuffer Flag to know TfN mode is enabled or not
 *
 * @param bitWidth bit Width
 *
 * @param MemMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @returns Saving the output is successful or not
 */
bool SaveOutputUserBuffer(Snpe_UserBufferMap_Handle_t outputMapHandle,
                          std::unordered_map<std::string, size_t>& bufSizeMap,
                          const std::string& outputDir,
                          int num,
                          size_t batchSize,
                          bool isTfNBuffer,
                          int bitWidth, Snpe_UserMemoryMap_Handle_t MemMappedBufferMapHandle);

#endif
