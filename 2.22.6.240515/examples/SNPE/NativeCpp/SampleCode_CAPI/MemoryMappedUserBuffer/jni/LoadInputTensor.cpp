//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <assert.h>
#include <unordered_map>
#include <cstring>

#include "LoadInputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorShape.h"


bool LoadInputUserBufferTfN(Snpe_SNPE_Handle_t snpeHandle,
                            std::vector<std::string>& fileLines,
                            Snpe_UserBufferMap_Handle_t inputMapHandle,
                            bool staticQuantization,
                            int bitWidth, Snpe_UserMemoryMap_Handle_t MemMappedBufferMapHandle)
{
    // get input tensor names of the network that need to be populated
    Snpe_StringList_Handle_t inputNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle) ;
    if (!inputNamesHandle) throw std::runtime_error("Error obtaining input tensor names");
    auto inputNamesSize = Snpe_StringList_Size(inputNamesHandle);
    assert(inputNamesSize > 0);

    // Start processing the each Individual Input.
    if (inputNamesSize) std::cout << "Processing DNN Input: " << std::endl;
    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);

        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        // Parsing <inputname>:=<filepath> format
        std::unordered_map<std::string, std::string> nameToFilePathMap;
        if (fileLine.find(":=") != std::string::npos) {
            for (auto &line : filePaths) {
                if (line.empty())
                    continue;

                std::vector<std::string> lineContents;
                // Line is in format "<inputname>:=<filepath>"
                split(lineContents, line, '='); // Splits up to "<inputname>:", "<filepath>"

                std::string name = lineContents[0];
                name.erase(name.length()-1); // removes colon (i.e. "<inputname>:" -> "<inputname>")

                nameToFilePathMap.emplace(name, lineContents[1]);
            }
        }

        for (size_t j = 0; j < inputNamesSize; j++) {
            const char *name = Snpe_StringList_At(inputNamesHandle,j);

            // If nameToFilePathMap contains name then use the filepath associated with that name
            // Otherwise, use filepath at index j of filePaths
            std::string filePath(nameToFilePathMap.find(name) != nameToFilePathMap.end() ? nameToFilePathMap.at(name) : filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            void *rpcInput = Snpe_UserMemoryMap_GetUserMemoryAddressAtIndex(MemMappedBufferMapHandle, name, 0);
            uint64_t rpcOffset = Snpe_UserMemoryMap_GetUserMemoryOffsetAtIndex(MemMappedBufferMapHandle, name, 0);

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            if (staticQuantization) {
                // If static quantization is enabled then get the quantization parameters
                // from the user buffer and use them to load the file contents
                auto ubeTfNHandle = Snpe_UserBufferMap_GetUserBuffer_Ref(inputMapHandle,name);
                uint64_t stepEquivalentTo0 = Snpe_UserBufferEncodingTfN_GetStepExactly0(ubeTfNHandle);
                float quantizedStepSize = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(ubeTfNHandle);
                if(!loadByteDataFileBatchedTfN(filePath, rpcInput, rpcOffset, stepEquivalentTo0, quantizedStepSize, staticQuantization, bitWidth)){
                    Snpe_StringList_Delete(inputNamesHandle);
                    return false;
                }
            } else {
                // If static quantization is disabled then get the quantization parameters
                // dynamically from the inputs to load the file contents and set them to user buffer
                uint64_t stepEquivalentTo0;
                float quantizedStepSize;
                if(!loadByteDataFileBatchedTfN(filePath, rpcInput, rpcOffset, stepEquivalentTo0, quantizedStepSize, staticQuantization, bitWidth)){
                    Snpe_StringList_Delete(inputNamesHandle);
                    return false;
                }
                auto userBufferEncoding = Snpe_IUserBuffer_GetEncoding_Ref(Snpe_UserBufferMap_GetUserBuffer_Ref(inputMapHandle,name));
                Snpe_UserBufferEncodingTfN_SetStepExactly0(userBufferEncoding, stepEquivalentTo0);
                Snpe_UserBufferEncodingTfN_SetQuantizedStepSize(userBufferEncoding, quantizedStepSize);
            }
        }
    }

    // Delete the InputName Handle
    Snpe_StringList_Delete(inputNamesHandle);
    return true;
}

// Load multiple batched input user buffers
bool LoadInputUserBufferFloat(Snpe_SNPE_Handle_t snpeHandle,
                              std::vector<std::string>& fileLines, Snpe_UserMemoryMap_Handle_t MemMappedBufferMapHandle)
{
    // get input tensor names of the network that need to be populated
    Snpe_StringList_Handle_t inputNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle) ;
    if (!inputNamesHandle) throw std::runtime_error("Error obtaining input tensor names");
    auto inputNamesSize = Snpe_StringList_Size(inputNamesHandle);
    assert(inputNamesSize > 0);

    // Start processing the each Individual Input.
    if (inputNamesSize) std::cout << "Processing DNN Input: " << std::endl;
    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        // Parsing <inputname>:=<filepath> format
        std::unordered_map<std::string, std::string> nameToFilePathMap;
        if (fileLine.find(":=") != std::string::npos) {
            for (auto &line : filePaths) {
                if (line.empty())
                    continue;

                std::vector<std::string> lineContents;
                // Line is in format "<inputname>:=<filepath>"
                split(lineContents, line, '='); // Splits up to "<inputname>:", "<filepath>"

                std::string name = lineContents[0];
                name.erase(name.length()-1); // removes colon (i.e. "<inputname>:" -> "<inputname>")

                nameToFilePathMap.emplace(name, lineContents[1]);
            }
        }

        for (size_t j = 0; j < inputNamesSize; j++) {
            const char *name = Snpe_StringList_At(inputNamesHandle,j);

            // If nameToFilePathMap contains name then use the filepath associated with that name
            // Otherwise, use filepath at index j of filePaths
            std::string filePath(nameToFilePathMap.find(name) != nameToFilePathMap.end() ? nameToFilePathMap.at(name) : filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            void *rpcInput = Snpe_UserMemoryMap_GetUserMemoryAddressAtIndex(MemMappedBufferMapHandle, name, 0);
            uint64_t rpcOffset = Snpe_UserMemoryMap_GetUserMemoryOffsetAtIndex(MemMappedBufferMapHandle, name, 0);
            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            if(!loadByteDataFileBatchedFloat(filePath, rpcInput, rpcOffset))
            {
                Snpe_StringList_Delete(inputNamesHandle);
                return false;
            }
        }
    }

    // Delete the InputName Handle
    Snpe_StringList_Delete(inputNamesHandle);
    return true;
}
