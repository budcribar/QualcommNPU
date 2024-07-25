//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CHECKRUNTIME_H
#define CHECKRUNTIME_H

#include "SNPE/SNPEUtil.h"

/**
 * @brief .
 *
 * Check whether selected runtime is supported or not. If not supported then Fall back to CPU
 *
 * @param[in] runtime  Supplied by the user
 *
 * @param[in] staticQuantization  Is supported by runtime or not
 *
 * @return Supported Runtime
 */
Snpe_Runtime_t CheckRuntime(Snpe_Runtime_t& runtime, bool& staticQuantization);

#endif
