//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>

#include "CheckRuntime.hpp"

#include "SNPE/SNPEUtil.h"
#include "DlSystem/DlVersion.h"
#include "DlSystem/DlEnums.h"

// Command line settings
Snpe_Runtime_t CheckRuntime(Snpe_Runtime_t& runtime, bool& staticQuantization)
{
    auto libVersionHandle = Snpe_Util_GetLibraryVersion();
    std::cout << "SNPE v" << Snpe_DlVersion_ToString(libVersionHandle) << "\n";
    Snpe_DlVersion_Delete(libVersionHandle);

   if((runtime != SNPE_RUNTIME_DSP) && staticQuantization)
   {
      std::cerr << "ERROR: Cannot use static quantization with CPU/GPU runtimes. It is only designed for DSP/AIP runtimes.\n";
      std::cerr << "ERROR: Proceeding without static quantization on selected runtime."<< std::endl;
      staticQuantization = false;
   }

    if (!Snpe_Util_IsRuntimeAvailable(runtime))
    {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        runtime = SNPE_RUNTIME_CPU_FLOAT32;
    }

    return runtime;
}
