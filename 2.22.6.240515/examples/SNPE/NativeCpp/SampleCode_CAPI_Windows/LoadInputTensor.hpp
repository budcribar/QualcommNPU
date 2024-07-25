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
 * Loads all the inputs in the input Tensor Map
 *
 * @param[in] snpeHandle Snpe handle
 *
 * @param[in] fileLines  Files names required in one batch
 *
 * @param[in] inputTensorNamesHandle Handle of the input Tensor names
 *
 * @param[in] inputs  vector of Snpe_ITensor_Handle_t to store input Tensors
 *
 * @returns tuple of the Input Tensor Map and flag to check whether loading of Map successful or not
 */
std::tuple<Snpe_TensorMap_Handle_t, bool> LoadInputTensorMap(Snpe_SNPE_Handle_t snpeHandle,
                                                             std::vector<std::string>& fileLines,
                                                             Snpe_StringList_Handle_t inputTensorNamesHandle,
                                                             std::vector<Snpe_ITensor_Handle_t>& inputs);
#endif
