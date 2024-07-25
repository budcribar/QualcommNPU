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
 * Saves the output of ITensor after successful execution
 *
 * @param[in] outputTensorMapHandle Handle of the output Tensor Map
 *
 * @param[in] outputDir  Path to Output directory
 *
 * @param[in] num Batch number
 *
 * @param[in] batchSize Size of batch
 *
 * @returns Saving of the output is success or not
 */
bool SaveOutputTensor(Snpe_TensorMap_Handle_t outputTensorMapHandle,
                      const std::string& outputDir,
                      int num,
                      size_t batchSize);



#endif
