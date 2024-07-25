//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef PREPROCESSINPUT_H
#define PREPROCESSINPUT_H

#include <string>
#include <vector>

/**
 * @brief .
 *
 * Preprocess the input list according to the batchSize of the network.
 *
 * @param[in] filePath path to input list
 *
 * @param[in] batchSize  Size of batch
 *
 * @returns vector of vector storing inputs that will fit one batch
 */
std::vector<std::vector<std::string>> PreprocessInput(const char* filePath, size_t batchSize);

#endif
