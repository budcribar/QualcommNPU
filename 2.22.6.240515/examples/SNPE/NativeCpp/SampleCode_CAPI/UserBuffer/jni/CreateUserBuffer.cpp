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
#include <stdexcept>
#include <unordered_map>

#include "CreateUserBuffer.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/IUserBuffer.h"
#include "DlSystem/UserBufferMap.h"
#include "SNPE/SNPEUtil.h"

// Helper function to fill a single entry of the UserBufferMap with the given user-backed buffer
bool CreateUserBuffer(Snpe_UserBufferMap_Handle_t userBufferMap,
                      std::unordered_map<std::string, std::vector<uint8_t >>& applicationBuffers,
                      std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                      Snpe_SNPE_Handle_t snpeHandle,
                      const char* name,
                      const bool isTfNBuffer,
                      bool staticQuantization,
                      int bitWidth)
{
    // get attributes of buffer by name
    Snpe_IBufferAttributes_Handle_t bufferAttributesOpt = Snpe_SNPE_GetInputOutputBufferAttributes(snpeHandle,name) ;
    if (!bufferAttributesOpt) {
        std::cerr<<"Error obtaining attributes for tensor"<<name<<std::endl;
        Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
        return false;
    }

    // calculate the size of buffer required by the input tensor
    Snpe_TensorShape_Handle_t bufferShapeHandle = Snpe_IBufferAttributes_GetDims(bufferAttributesOpt);
    size_t bufferElementSize = 0;
    if (isTfNBuffer) {
        bufferElementSize = bitWidth / 8;
    }
    else {
        bufferElementSize = sizeof(float);
    }

    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    // Note: Buffer stride is usually known and does not need to be calculated.
    auto stridesHandle = calcStrides(bufferShapeHandle, bufferElementSize);
    size_t bufSize = calcSizeFromDims(Snpe_TensorShape_GetDimensions(bufferShapeHandle), Snpe_TensorShape_Rank(bufferShapeHandle), bufferElementSize);
    Snpe_TensorShape_Delete(bufferShapeHandle);

    // set the buffer encoding type
    Snpe_UserBufferEncoding_Handle_t userBufferEncodingHandle = nullptr;
    if (isTfNBuffer)
    {
        if((Snpe_IBufferAttributes_GetEncodingType(bufferAttributesOpt)) == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT && staticQuantization){
            std::cerr << "ERROR: Quantization parameters not present in model" << std::endl;
            Snpe_TensorShape_Delete(stridesHandle);
            Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
            return false;
        }
        Snpe_UserBufferEncoding_Handle_t ubeTfNHandle = Snpe_IBufferAttributes_GetEncoding_Ref(
                bufferAttributesOpt);
        if(Snpe_UserBufferEncoding_GetElementType(ubeTfNHandle) != SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 &&
           Snpe_UserBufferEncoding_GetElementType(ubeTfNHandle) != SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16){
            std::cerr << "Quantization encoding not found for tensor: " << name << "\n";
            Snpe_TensorShape_Delete(stridesHandle);
            Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
            return false;
        }
        uint64_t stepEquivalentTo0 = Snpe_UserBufferEncodingTfN_GetStepExactly0(ubeTfNHandle);
        float quantizedStepSize = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(ubeTfNHandle);
        userBufferEncodingHandle = Snpe_UserBufferEncodingTfN_Create(stepEquivalentTo0, quantizedStepSize, bitWidth);
    }
    else
    {
        userBufferEncodingHandle = Snpe_UserBufferEncodingFloat_Create();
    }

    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));

    // create SNPE user buffer from the user-backed buffer
    snpeUserBackendBufferHandles.push_back(Snpe_Util_CreateUserBuffer(applicationBuffers.at(name).data(), bufSize, stridesHandle,
                                                                      userBufferEncodingHandle));


    // Delete all the created handles for creating userBufferEncoding
    if(isTfNBuffer)
        Snpe_UserBufferEncodingTfN_Delete(userBufferEncodingHandle);
    else
        Snpe_UserBufferEncodingFloat_Delete(userBufferEncodingHandle);

    Snpe_TensorShape_Delete(stridesHandle);

    if (snpeUserBackendBufferHandles.back() == nullptr)
    {
        std::cerr << "Error while creating user buffer." << std::endl;
        Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
        return false;
    }

    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    Snpe_UserBufferMap_Add(userBufferMap, name, snpeUserBackendBufferHandles.back());

    // Delete the InputOutputBufferAttribute handle
    Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
    return true;
}

// Create a UserBufferMap of the SNPE network inputs
bool CreateInputBufferMap(Snpe_UserBufferMap_Handle_t inputMapHandle,
                          std::unordered_map<std::string, std::vector<uint8_t >>& applicationBuffers,
                          std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                          Snpe_SNPE_Handle_t snpeHandle,
                          bool isTfNBuffer,
                          bool staticQuantization,
                          int bitWidth)
{
    // get output tensor names of the network that need to be populated
    const Snpe_StringList_Handle_t inputBufferNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle);
    if (!inputBufferNamesHandle)
    {
        std::cerr<<"Error obtaining output tensor names"<<std::endl;
        return false;
    }
    size_t inputBufferSize = Snpe_StringList_Size(inputBufferNamesHandle);
    assert(inputBufferSize>0);

    // create SNPE user buffers for each application storage buffer
    for (size_t i = 0; i < inputBufferSize; ++i) {
        const char *name = Snpe_StringList_At(inputBufferNamesHandle, i);
        if(!CreateUserBuffer(inputMapHandle, applicationBuffers, snpeUserBackendBufferHandles, snpeHandle, name, isTfNBuffer, staticQuantization, bitWidth))
        {
            Snpe_StringList_Delete(inputBufferNamesHandle);
            return false;
        }
    }
    Snpe_StringList_Delete(inputBufferNamesHandle);
    return true;
}

// Create a UserBufferMap of the SNPE network outputs
bool CreateOutputBufferMap(Snpe_UserBufferMap_Handle_t outputMapHandle,
                           std::unordered_map<std::string, std::vector<uint8_t >>& applicationBuffers,
                           std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                           Snpe_SNPE_Handle_t snpeHandle,
                           bool isTfNBuffer,
                           int bitWidth)
{
    // get output tensor names of the network that need to be populated
    const Snpe_StringList_Handle_t outputBufferNamesHandle = Snpe_SNPE_GetOutputTensorNames(snpeHandle);
    if (!outputBufferNamesHandle){
        std::cerr<<"Error obtaining output tensor names"<<std::endl;
        return false;
    }
    size_t outputBufferSize = Snpe_StringList_Size(outputBufferNamesHandle);
    assert(outputBufferSize>0);

    // create SNPE user buffers for each application storage buffer
    for (size_t i = 0; i < outputBufferSize; ++i) {
        const char *name = Snpe_StringList_At(outputBufferNamesHandle, i);
        if(!CreateUserBuffer(outputMapHandle, applicationBuffers, snpeUserBackendBufferHandles, snpeHandle, name, isTfNBuffer, false, bitWidth)){
            Snpe_StringList_Delete(outputBufferNamesHandle);
            return false;
        }
    }
    Snpe_StringList_Delete(outputBufferNamesHandle);
    return true;
}