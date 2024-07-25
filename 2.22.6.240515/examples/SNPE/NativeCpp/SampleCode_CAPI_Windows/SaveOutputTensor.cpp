//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <algorithm>

#include "SaveOutputTensor.hpp"
#include "Util.hpp"

#include "DlSystem/ITensor.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorMap.h"
#include "DlSystem/TensorShape.h"

// Print the results to raw files
// ITensor
bool SaveOutputTensor(Snpe_TensorMap_Handle_t outputTensorMapHandle,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize)
{
    Snpe_StringList_Handle_t tensorNamesHandle = Snpe_TensorMap_GetTensorNames(outputTensorMapHandle);
    for (size_t i = 0; i< Snpe_StringList_Size(tensorNamesHandle); i++) {
        const char* name = Snpe_StringList_At(tensorNamesHandle, i);
        // Split batched outputs. Don't save padded inputs.
        for(size_t j=0; j<batchSize; j++) {
            std::ostringstream path;
            path << outputDir << "/"
                 << "Result_" << num + j << "/"
                 << name << ".raw";
            auto tensorHandle = Snpe_TensorMap_GetTensor_Ref(outputTensorMapHandle, name);
            size_t tensorStart = j * Snpe_ITensor_GetSize(tensorHandle) / batchSize;
            size_t tensorEnd = (j + 1) * Snpe_ITensor_GetSize(tensorHandle) / batchSize;
            if (!SaveITensor(path.str(), (float*)Snpe_ITensor_GetData(tensorHandle), tensorStart, tensorEnd))  {
                Snpe_StringList_Delete(tensorNamesHandle);
                return false;
            }
        }
    }
    Snpe_StringList_Delete(tensorNamesHandle);
    return true;
}
