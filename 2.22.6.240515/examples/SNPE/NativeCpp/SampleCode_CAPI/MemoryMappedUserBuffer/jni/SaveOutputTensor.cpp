//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "SaveOutputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorShape.h"

//Save reult in the raw files for User buffer case
bool SaveOutputUserBuffer(Snpe_UserBufferMap_Handle_t outputMapHandle,
                          std::unordered_map<std::string, size_t>& bufSizeMap,
                          const std::string& outputDir,
                          int num,
                          size_t batchSize,
                          bool isTfNBuffer,
                          int bitWidth, Snpe_UserMemoryMap_Handle_t MemMappedBufferMapHandle)
{
    // Get all output buffer names from the network
    Snpe_StringList_Handle_t outputNamesOpt = Snpe_UserBufferMap_GetUserBufferNames(outputMapHandle) ;
    int elementSize = bitWidth / 8;

    // Iterate through output buffers and print each output to a raw file
    for (size_t i = 0; i<Snpe_StringList_Size(outputNamesOpt); i++) {
        for(size_t j=0; j<batchSize; j++) {
            const char* name = Snpe_StringList_At(outputNamesOpt, i);
            std::ostringstream path;
            path << outputDir << "/"
                 << "Result_" << num + j << "/"
                 << name << ".raw";
            auto userbufferHandle = Snpe_UserBufferMap_GetUserBuffer_Ref(outputMapHandle,name);
            size_t bufferSize = Snpe_IUserBuffer_GetSize(userbufferHandle);
            size_t bufferOutputSize = Snpe_IUserBuffer_GetOutputSize(userbufferHandle);
            size_t batchChunk = bufferSize / batchSize;
            size_t dataChunk = bufferOutputSize / batchSize;
            if(batchChunk != dataChunk) {
                std::cout << "\tUserBuffer size is " << bufferSize << " bytes, but "
                          << bufferOutputSize << " bytes of data was found." << std::endl;
                if( dataChunk > batchChunk )
                    std::cout << "\tAssign a larger buffer using a bigger -z argument" << std::endl;
                batchChunk = std::min(batchChunk,dataChunk);
            }

            uint8_t *userBufferAddress = (uint8_t*)Snpe_UserMemoryMap_GetUserMemoryAddressAtIndex(MemMappedBufferMapHandle, name, 0);
            uint64_t userBufferOffset = Snpe_UserMemoryMap_GetUserMemoryOffsetAtIndex(MemMappedBufferMapHandle, name, 0);
            size_t bufSize = bufSizeMap.at(name);

            if (isTfNBuffer)
            {
                std::vector<uint8_t> output(bufSize * sizeof(float) / elementSize);
                auto userBufferEncoding = Snpe_IUserBuffer_GetEncoding_Ref(Snpe_UserBufferMap_GetUserBuffer_Ref(outputMapHandle,name));

                TfNToFloat(reinterpret_cast<float *>(&output[0]), userBufferAddress + userBufferOffset,
                           Snpe_UserBufferEncodingTfN_GetStepExactly0(userBufferEncoding), Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(userBufferEncoding),
                           bufSize / elementSize, bitWidth);
                if(!SaveUserBufferBatched(path.str(), output, j, batchChunk * sizeof(float) / elementSize))
                {
                    Snpe_StringList_Delete(outputNamesOpt);
                    return false;
                }
            }
            else
            {
                std::vector<uint8_t> output_data(userBufferAddress + userBufferOffset, userBufferAddress + userBufferOffset + bufSize);
                if(!SaveUserBufferBatched(path.str(), output_data, j, batchChunk))
                {
                    Snpe_StringList_Delete(outputNamesOpt);
                    return false;
                }
            }
        }
    }
    Snpe_StringList_Delete(outputNamesOpt);
    return true;
}
