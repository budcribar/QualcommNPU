//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOADCONTAINER_H
#define LOADCONTAINER_H

#include <string>

#include "DlContainer/DlContainer.h"

/**
 * @brief .
 *
 * Loads the Container from the Path
 *
 * @param[in] containerPath Location of container
 *
 * @returns the handle after loading the dlc
 */
Snpe_DlContainer_Handle_t LoadContainerFromPath(std::string& containerPath);

#endif
