//==============================================================================
//
//  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <assert.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unordered_map>

#include "CreateUserBuffer.hpp"

#include "SNPE/SNPE.h"
#include "DlSystem/StringList.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/IUserBuffer.h"
#include "DlSystem/UserBufferMap.h"
#include "SNPE/SNPEUtil.h"

// Helper function to fill a single entry of the UserBufferMap with the given user-backed buffer
bool CreateUserBuffer(Snpe_UserBufferMap_Handle_t userBufferMap,
                      Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                      std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                      Snpe_SNPE_Handle_t snpeHandle,
                      const char* name,
                      const bool isTfNBuffer,
                      const bool useRpc,
                      bool staticQuantization,
                      int bitWidth,
                      void* bufferAllocator,
                      MemFnHandlesType_t& memFnHandles)
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

    std::pair<void*, int> addrFdPair = GetBufferAddrFd(bufSize, useRpc, bufferAllocator, memFnHandles);

    void* addr = addrFdPair.first;
    int fd = addrFdPair.second;

    if (!useRpc && fd <= 0) {
        std::cerr << "Failed to allocated memory via libdmabufheap.so" << std::endl;
        return false;
    }

    if (addr == nullptr || addr == MAP_FAILED) { // MAP_FAILED == (void*)-1 as returned by mmap() on failure
        std::cerr << "Failed to retrieve buffer address, fd: " << std::to_string(fd) << std::endl;
        return false;
    }

    // create SNPE user buffer from the user-backed buffer
    snpeUserBackendBufferHandles.push_back(Snpe_Util_CreateUserBuffer(addr, bufSize, stridesHandle, userBufferEncodingHandle));

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

    Snpe_UserMemoryMap_AddFdOffset(memMappedBufferMapHandle, name, addr, 0U, fd, 0U);
    // Delete the InputOutputBufferAttribute handle
    Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
    return true;
}

// Create a UserBufferMap of the SNPE network inputs
bool CreateInputBufferMap(Snpe_UserBufferMap_Handle_t inputMapHandle,
                          Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                          std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                          Snpe_SNPE_Handle_t snpeHandle,
                          bool isTfNBuffer,
                          bool useRpc,
                          bool staticQuantization,
                          int bitWidth,
                          void* bufferAllocator,
                          MemFnHandlesType_t& memFnHandles)
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
        if(!CreateUserBuffer(inputMapHandle, memMappedBufferMapHandle, snpeUserBackendBufferHandles, snpeHandle,
                             name, isTfNBuffer, useRpc, staticQuantization, bitWidth, bufferAllocator, memFnHandles))
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
                           Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                           std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                           Snpe_SNPE_Handle_t snpeHandle,
                           bool isTfNBuffer,
                           bool useRpc,
                           int bitWidth,
                           void* bufferAllocator,
                           MemFnHandlesType_t& memFnHandles)
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
        if(!CreateUserBuffer(outputMapHandle, memMappedBufferMapHandle, snpeUserBackendBufferHandles, snpeHandle,
                             name, isTfNBuffer, useRpc, false, bitWidth, bufferAllocator, memFnHandles))
        {
            Snpe_StringList_Delete(outputBufferNamesHandle);
            return false;
        }
    }
    Snpe_StringList_Delete(outputBufferNamesHandle);
    return true;
}

// Helper function to fill a single entry of the UserBufferMap with the given user-backed shared buffer
// Preconditions: baseAddr points to a pre-allocated rpc memory space
//                baseAddr + offset is within the allocated memory space
//                the remaining allocated memory space from baseAddr + offset is large enough for the
//                                                                               expected buffer size
//
// This function is different from CreateUserBuffer() in that it takes in a pointer to a pre-allocated
// memory space and uses that to create the user buffers as opposed to allocating a new memory space
// for each user buffer.
// The location of the user buffer in the memory space is based on baseAddr and offset, where offset
// is the number of bytes from baseAddr to create this specific user buffer
// It is very important that offset is updated between each user buffer created. Otherwise, the
// created user buffer will overwrite any existing user buffer previously allocated at the memory
// address specified by baseAddr + offset
bool CreateUserBufferShared(Snpe_UserBufferMap_Handle_t userBufferMapHandle,
                            Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                            std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                            Snpe_SNPE_Handle_t snpeHandle,
                            const char* name,
                            const bool isTfNBuffer,
                            bool staticQuantization,
                            const int bitWidth,
                            void* baseAddr,
                            int fd,
                            const size_t bufSize,
                            const size_t totalBufSize,
                            uint64_t& offset)
{
    // get attributes of buffer by name
    Snpe_IBufferAttributes_Handle_t bufferAttributesOpt = Snpe_SNPE_GetInputOutputBufferAttributes(snpeHandle,name) ;
    if (!bufferAttributesOpt) {
        std::cerr << "Error obtaining attributes for tensor "<< name <<std::endl;
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
        Snpe_UserBufferEncoding_Handle_t ubeTfNHandle = Snpe_IBufferAttributes_GetEncoding_Ref(bufferAttributesOpt);
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

    // Create/populate the buffer in the shared space at the right location using the
    // a pointer to the shared memory space and the appropriate offset
    snpeUserBackendBufferHandles.push_back(Snpe_Util_CreateUserBufferShared(static_cast<uint8_t**>(baseAddr), bufSize, offset, stridesHandle, userBufferEncodingHandle));

    // Add the new buffer information to both the user memory map and the user buffer map
    Snpe_UserMemoryMap_AddFdOffset(memMappedBufferMapHandle, name, baseAddr, totalBufSize, fd, offset);
    Snpe_UserBufferMap_Add(userBufferMapHandle, name, snpeUserBackendBufferHandles.back());

    // Update the offset
    // This offset must be continually updated by the size of the single buffer
    // This offset is passed in by reference in case this function is called for
    // multiple sets of buffer names (e.g. input then output).
    offset += bufSize;

    Snpe_TensorShape_Delete(stridesHandle);
    Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
    return true;
}

// Helper function to find the total buffer size for a specific set of buffer names
// e.g. for input names or output names
//
// bufSizeMap isn't necessarily required, but storing the calculated buffer size for
// a buffer name avoids re-calculating the buffer size later when creating the actual
// user buffer on the shared memory space
size_t GetTotalBufferSizeForHandle(Snpe_StringList_Handle_t bufferNamesHandle,
                                   bool isTfNBuffer,
                                   int bitWidth,
                                   Snpe_SNPE_Handle_t snpeHandle,
                                   std::unordered_map<std::string, size_t>& bufSizeMap)
{
    size_t bufferNamesSize = Snpe_StringList_Size(bufferNamesHandle);
    assert(bufferNamesSize > 0);

    size_t totalBufSize = 0U;

    // Get total buffer size for buffer memory allocation
    for (size_t i = 0; i < bufferNamesSize; ++i) {
        const char *name = Snpe_StringList_At(bufferNamesHandle, i);

        // get attributes of buffer by name
        Snpe_IBufferAttributes_Handle_t bufferAttributesOpt = Snpe_SNPE_GetInputOutputBufferAttributes(snpeHandle, name) ;
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

        // calculate the size of buffer required by the tensor
        size_t bufSize = calcSizeFromDims(Snpe_TensorShape_GetDimensions(bufferShapeHandle), Snpe_TensorShape_Rank(bufferShapeHandle), bufferElementSize);
        Snpe_TensorShape_Delete(bufferShapeHandle);

        // Update the total buffer size
        totalBufSize += bufSize;
        bufSizeMap.emplace(name, bufSize);

        Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
    }

    Snpe_StringList_Delete(bufferNamesHandle);
    return totalBufSize;
}

// Helper function to find the total buffer size for all input and output buffers
size_t GetTotalInputOutputBufferSize(Snpe_SNPE_Handle_t snpeHandle,
                                     bool isTfNBuffer,
                                     int bitWidth,
                                     std::unordered_map<std::string, size_t>& bufSizeMap)
{
    const Snpe_StringList_Handle_t inputBufferNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle);
    const Snpe_StringList_Handle_t outputBufferNamesHandle = Snpe_SNPE_GetOutputTensorNames(snpeHandle);

    return GetTotalBufferSizeForHandle(inputBufferNamesHandle, isTfNBuffer, bitWidth, snpeHandle, bufSizeMap)
           + GetTotalBufferSizeForHandle(outputBufferNamesHandle, isTfNBuffer, bitWidth, snpeHandle, bufSizeMap);
}

// Helper function to create/populate user buffer maps for all input buffers
bool CreateInputBufferMapShared(Snpe_UserBufferMap_Handle_t inputMapHandle,
                                Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                                std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                                const std::unordered_map<std::string, size_t>& bufSizeMap,
                                Snpe_SNPE_Handle_t snpeHandle,
                                bool isTfNBuffer,
                                bool staticQuantization,
                                int bitWidth,
                                void* baseAddr,
                                int fd,
                                const size_t totalBufSize,
                                uint64_t& offset)
{

    // get output tensor names of the network that need to be populated
    const Snpe_StringList_Handle_t inputBufferNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle);
    if (!inputBufferNamesHandle)
    {
        std::cerr<<"Error obtaining input tensor names"<<std::endl;
        return false;
    }

    // Populate the user buffer maps for all input tensors
    bool successful = CreateBufferMapShared(inputMapHandle, memMappedBufferMapHandle, inputBufferNamesHandle, snpeUserBackendBufferHandles,
                                            bufSizeMap, snpeHandle, isTfNBuffer, staticQuantization, bitWidth, baseAddr, fd, totalBufSize, offset);

    Snpe_StringList_Delete(inputBufferNamesHandle);
    return successful;
}

// Helper function to create/populate user buffer maps for all output buffers
bool CreateOutputBufferMapShared(Snpe_UserBufferMap_Handle_t outputMapHandle,
                                 Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                                 std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                                 const std::unordered_map<std::string, size_t>& bufSizeMap,
                                 Snpe_SNPE_Handle_t snpeHandle,
                                 bool isTfNBuffer,
                                 int bitWidth,
                                 void* baseAddr,
                                 int fd,
                                 const size_t totalBufSize,
                                 uint64_t& offset)
{

    // get output tensor names of the network that need to be populated
    const Snpe_StringList_Handle_t outputBufferNamesHandle = Snpe_SNPE_GetOutputTensorNames(snpeHandle);
    if (!outputBufferNamesHandle)
    {
        std::cerr<<"Error obtaining output tensor names"<<std::endl;
        return false;
    }

    // Populate the user buffer maps for all output tensors
    bool successful = CreateBufferMapShared(outputMapHandle, memMappedBufferMapHandle, outputBufferNamesHandle, snpeUserBackendBufferHandles,
                                            bufSizeMap, snpeHandle, isTfNBuffer, false, bitWidth, baseAddr, fd, totalBufSize, offset);

    Snpe_StringList_Delete(outputBufferNamesHandle);
    return successful;
}

// Helper function to create/populate user buffer maps for all buffer names specified in bufferNamesHandle
bool CreateBufferMapShared(Snpe_UserBufferMap_Handle_t bufferMapHandle,
                           Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                           Snpe_StringList_Handle_t bufferNamesHandle,
                           std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                           const std::unordered_map<std::string, size_t>& bufSizeMap,
                           Snpe_SNPE_Handle_t snpeHandle,
                           bool isTfNBuffer,
                           bool staticQuantization,
                           int bitWidth,
                           void* baseAddr,
                           int fd,
                           const size_t totalBufSize,
                           uint64_t& offset)
{
    size_t bufferNamesSize = Snpe_StringList_Size(bufferNamesHandle);
    assert(bufferNamesSize > 0);

    // For each buffer name, retrieve the buffer size from bufSizeMap for that buffer
    // Then create/populate the user buffer onto the shared memory space provided by
    // baseAddr and offset
    // offset is passed by reference in this function and in CreateuserBufferShared()
    // so it is contiunally updated between each user buffer
    // If offset is never updated (e.g. passed by value) then all user buffers will
    // be created in the same memory space, overwriting any existing buffers
    for (size_t i = 0; i < bufferNamesSize; ++i) {
        const char *name = Snpe_StringList_At(bufferNamesHandle, i);

        size_t bufSize = bufSizeMap.at(name);
        if(!CreateUserBufferShared(bufferMapHandle, memMappedBufferMapHandle, snpeUserBackendBufferHandles, snpeHandle,
                                   name, isTfNBuffer, staticQuantization, bitWidth, baseAddr, fd, bufSize, totalBufSize, offset))
        {
            return false;
        }
    }
    return true;
}

bool IsUserBufferConsistentWithTensorDataType(Snpe_SNPE_Handle_t snpeHandle, int bitWidth, bool isTfNBuffer)
{
    std::string ubtype = "";
    if(isTfNBuffer){
        ubtype = "fixedPoint" + std::to_string(bitWidth);
    }
    else
        ubtype = "float32";
    //Create UserBuffer to tensor data type map
    std::unordered_map<std::string, Snpe_UserBufferEncoding_ElementType_t> ubToTensorDataTypeMap = {
            {"float32",      SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT},
            {"fixedPoint8",  SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8},
            {"fixedPoint16", SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16}};

    const Snpe_StringList_Handle_t outputBufferNamesHandle = Snpe_SNPE_GetOutputTensorNames(snpeHandle);
    if (!outputBufferNamesHandle){
        std::cerr<<"Error obtaining output tensor names"<<std::endl;
        return false;
    }
    size_t outputBufferSize = Snpe_StringList_Size(outputBufferNamesHandle);
    assert(outputBufferSize>0);
    for (size_t i = 0; i < outputBufferSize; ++i) {
        const char *name = Snpe_StringList_At(outputBufferNamesHandle, i);
        Snpe_IBufferAttributes_Handle_t bufferAttributesOpt = Snpe_SNPE_GetInputOutputBufferAttributes(snpeHandle,name) ;
        if (!bufferAttributesOpt) {
            std::cerr<<"Error obtaining attributes for tensor"<<name<<std::endl;
            Snpe_StringList_Delete(outputBufferNamesHandle);
            return false;
        }
        Snpe_UserBufferEncoding_Handle_t ubeHandle = Snpe_IBufferAttributes_GetEncoding_Ref(bufferAttributesOpt);
        Snpe_UserBufferEncoding_ElementType_t tensorDataType = Snpe_UserBufferEncoding_GetElementType(ubeHandle);
        Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
        if(ubToTensorDataTypeMap[ubtype] != tensorDataType){
            Snpe_StringList_Delete(outputBufferNamesHandle);
            std::cerr << "Error: Memory Mapped User Buffer must have same userbuffer type and tensor data type for output tensor: "<<name<< std::endl;
            return false;
        }
    }
    Snpe_StringList_Delete(outputBufferNamesHandle);
    const Snpe_StringList_Handle_t inputBufferNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle);
    if (!inputBufferNamesHandle)
    {
        std::cerr<<"Error obtaining output tensor names"<<std::endl;
        return false;
    }
    size_t inputBufferSize = Snpe_StringList_Size(inputBufferNamesHandle);
    assert(inputBufferSize>0);

    // Checking for all the input buffers Having same Tensor data type as Userbuffer or not
    for (size_t i = 0; i < inputBufferSize; ++i) {
        const char *name = Snpe_StringList_At(inputBufferNamesHandle, i);
        Snpe_IBufferAttributes_Handle_t bufferAttributesOpt = Snpe_SNPE_GetInputOutputBufferAttributes(snpeHandle,name) ;
        if (!bufferAttributesOpt) {
            std::cerr<<"Error obtaining attributes for tensor"<<name<<std::endl;
            Snpe_StringList_Delete(inputBufferNamesHandle);
            return false;
        }
        Snpe_UserBufferEncoding_Handle_t ubeHandle = Snpe_IBufferAttributes_GetEncoding_Ref(bufferAttributesOpt);
        Snpe_UserBufferEncoding_ElementType_t tensorDataType = Snpe_UserBufferEncoding_GetElementType(ubeHandle);
        Snpe_IBufferAttributes_Delete(bufferAttributesOpt);
        if(ubToTensorDataTypeMap[ubtype] != tensorDataType){
            Snpe_StringList_Delete(inputBufferNamesHandle);
            std::cerr << "Error: Memory Mapped User Buffer must have same userbuffer type and tensor data type for input tensor: "<<name<< std::endl;
            return false;
        }
    }
    Snpe_StringList_Delete(inputBufferNamesHandle);
    return true;
}
