//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <string>

#include "LoadContainer.hpp"

#include "DlContainer/DlContainer.h"

Snpe_DlContainer_Handle_t LoadContainerFromPath(std::string& containerPath)
{
    Snpe_DlContainer_Handle_t container;
    container = Snpe_DlContainer_Open(containerPath.c_str());
    return container;
}
