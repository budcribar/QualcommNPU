//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <fstream>

#include "ProcessUserBuffer.hpp"
#include "ProcessDataType.hpp"
#include "Util.hpp"

#include "DlSystem/TensorShape.h"
#include "DlSystem/StringList.h"


Snpe_IUserBuffer_Handle_t
CreateUserBuffer(const std::vector<size_t>& dims,
                 Snpe_UserBufferEncoding_ElementType_t userBufferType,
                 std::vector<uint8_t>& applicationBuffer,
                 uint64_t stepEquivalentTo0,
                 float quantizedStepSize)
{
    size_t bufferElementSize = 0;
    // Set the encoding type of user buffer
    Snpe_UserBufferEncoding_Handle_t userBufferEncodingHandle = nullptr;
    if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8
        || userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16) {
        //stepEquivalentTo0, quantizedStepSize are parameters for encode/decode tfN data
        bufferElementSize = userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 ? sizeof(uint8_t): sizeof(uint16_t);
        uint8_t bitWidth = userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 ? 8 : 16;
        userBufferEncodingHandle = Snpe_UserBufferEncodingTfN_Create(stepEquivalentTo0, quantizedStepSize, bitWidth);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) {
        bufferElementSize = sizeof(float);
        userBufferEncodingHandle = Snpe_UserBufferEncodingFloat_Create();
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32) {
        bufferElementSize = sizeof(int32_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingIntN_Create(32);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16) {
        bufferElementSize = sizeof(int16_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingIntN_Create(16);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8) {
        bufferElementSize = sizeof(int8_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingIntN_Create(8);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32) {
        bufferElementSize = sizeof(uint32_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingUintN_Create(32);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16) {
        bufferElementSize = sizeof(uint16_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingUintN_Create(16);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8) {
        bufferElementSize = sizeof(uint8_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingUintN_Create(8);
    }
    else if (userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8) {
        bufferElementSize = sizeof(uint8_t);
        userBufferEncodingHandle = Snpe_UserBufferEncodingBool_Create(8);
    }
    else {
        std::cerr << "Error: Unsupported data type: " << static_cast<typename std::underlying_type<decltype(userBufferType)>::type>(userBufferType) << std::endl;
        return nullptr;
    }

    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    std::vector<size_t> strides = CalcStrides(dims, bufferElementSize);
    Snpe_TensorShape_Handle_t stridesHandle = Snpe_TensorShape_CreateDimsSize(strides.data(), strides.size());
    // calculate the size of buffer required by the input tensor
    size_t bufSize = CalcSize(dims, bufferElementSize);

    // create user-backed storage to load input data onto it
    applicationBuffer = std::vector<uint8_t>(bufSize, 0);

    // create SNPE user buffer from the user-backed buffer
    Snpe_IUserBuffer_Handle_t userBufferHandle = nullptr;
    userBufferHandle = Snpe_Util_CreateUserBuffer(applicationBuffer.data(),
                                                  bufSize,
                                                  stridesHandle,
                                                  userBufferEncodingHandle);
    // Delete handles that are no longer needed
    Snpe_TensorShape_Delete(stridesHandle);
    if(userBufferType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8) {
        Snpe_UserBufferEncodingTfN_Delete(userBufferEncodingHandle);
    }
    else {
        Snpe_UserBufferEncodingFloat_Delete(userBufferEncodingHandle);
    }

    if (userBufferHandle == nullptr) {
        std::cerr << "Error: Snpe_Util_CreateUserBuffer fail." << std::endl;
        applicationBuffer.clear();
        return nullptr;
    }
    return userBufferHandle;
}

Snpe_UserBufferMap_Handle_t
CreateUserBufferMap(std::unordered_map<std::string, std::vector<size_t>>& bufferDims,
                    std::unordered_map<std::string, Snpe_UserBufferEncoding_ElementType_t>& userBufferTypes,
                    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBufferMap,
                    std::unordered_map<std::string, uint64_t>& stepEquivalentTo0,
                    std::unordered_map<std::string, float>& quantizedStepSize)
{
    Snpe_UserBufferMap_Handle_t userBufferMapHandle = Snpe_UserBufferMap_Create();
    for (auto& pair : bufferDims) {
        const std::string& name = pair.first;
        const std::vector<size_t>& dims = pair.second;
        Snpe_IUserBuffer_Handle_t userBufferHandle = nullptr;
        userBufferHandle = CreateUserBuffer(dims,
                                            userBufferTypes[name],
                                            applicationBufferMap[name],
                                            stepEquivalentTo0[name],
                                            quantizedStepSize[name]);
        if (userBufferHandle == nullptr) {
            std::cerr << "Error: Create user buffer fail. Name: " << name
                      << " dims: " << ArrayToStr(dims)
                      << " bufferType: " << static_cast<int>(userBufferTypes[name]) << std::endl;
            DeleteUserBufferMap(userBufferMapHandle);
            Snpe_UserBufferMap_Delete(userBufferMapHandle);
            applicationBufferMap.clear();
            return nullptr;
        }
        Snpe_UserBufferMap_Add(userBufferMapHandle, name.c_str(), userBufferHandle);
    }
    return userBufferMapHandle;
}

Snpe_UserBufferList_Handle_t
CreateUserBufferList(std::unordered_map<std::string, std::vector<size_t>>& bufferDims,
                     std::unordered_map<std::string, Snpe_UserBufferEncoding_ElementType_t>& userBufferTypes,
                     size_t bufferMapNumber,
                     std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>& applicationBufferList,
                     std::unordered_map<std::string, uint64_t>& stepEquivalentTo0,
                     std::unordered_map<std::string, float>& quantizedStepSize)
{
    Snpe_UserBufferList_Handle_t userBufferListHandle = Snpe_UserBufferList_Create();
    applicationBufferList = std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>(bufferMapNumber);
    for (size_t i = 0; i < bufferMapNumber; ++i) {
        Snpe_UserBufferMap_Handle_t userBufferMapHandle = nullptr;
        userBufferMapHandle = CreateUserBufferMap(bufferDims,
                                                  userBufferTypes,
                                                  applicationBufferList[i],
                                                  stepEquivalentTo0,
                                                  quantizedStepSize);
        if (userBufferMapHandle == nullptr) {
            std::cerr << "Error: Create userBufferMap fail. index: " << i << std::endl;
            DeleteUserBufferList(userBufferListHandle);
            Snpe_UserBufferMap_Delete(userBufferListHandle);
            applicationBufferList.clear();
            return nullptr;
        }
        Snpe_UserBufferList_PushBack(userBufferListHandle, userBufferMapHandle);
    }
    return userBufferListHandle;
}

void DeleteUserBufferMap(Snpe_UserBufferMap_Handle_t userBufferMapHandle)
{
    if (userBufferMapHandle == nullptr) {
        return;
    }
    Snpe_StringList_Handle_t namesHandle = Snpe_UserBufferMap_GetUserBufferNames(userBufferMapHandle);
    size_t size = Snpe_StringList_Size(namesHandle);
    for (size_t i = 0; i < size; i++) {
        const char* name = Snpe_StringList_At(namesHandle, i);
        Snpe_IUserBuffer_Handle_t userBufferRef = nullptr;
        userBufferRef = Snpe_UserBufferMap_GetUserBuffer_Ref(userBufferMapHandle, name);
        if (userBufferRef != nullptr) {
            Snpe_IUserBuffer_Delete(userBufferRef);
        }
    }
    Snpe_StringList_Delete(namesHandle);
}

void DeleteUserBufferList(Snpe_UserBufferList_Handle_t userBufferListHandle)
{
    if (userBufferListHandle == nullptr) {
        return;
    }
    size_t size = Snpe_UserBufferList_Size(userBufferListHandle);
    for (size_t i = 0; i < size; i ++) {
        Snpe_UserBufferMap_Handle_t userBufferMapRef = nullptr;
        userBufferMapRef = Snpe_UserBufferList_At_Ref(userBufferListHandle, i);
        DeleteUserBufferMap(userBufferMapRef);
    }
    Snpe_UserBufferList_Clear(userBufferListHandle);
    Snpe_UserBufferList_Delete(userBufferListHandle);
}

bool SaveUserBuffer(Snpe_IUserBuffer_Handle_t userBufferHandle,
                    std::vector<uint8_t>& applicationBuffer,
                    std::string& filePath,
                    bool splitBatch,
                    size_t batchSize,
                    size_t indexOffset,
                    size_t saveBatchChunk)
{
    if (userBufferHandle == nullptr) {
        std::cerr << "Error: userBufferHandle is null" << std::endl;
        return false;
    }
    size_t bufferSize = Snpe_IUserBuffer_GetSize(userBufferHandle);
    if (bufferSize != applicationBuffer.size()) {
        std::cerr << "Error: Unequaled size of UserBuffer(" << bufferSize << ") and "
                  << "applicationBuffer (" << applicationBuffer.size() << ")." << std::endl;
        return false;
    }

    // Convert data to float data for saving
    Snpe_UserBufferEncoding_Handle_t encodingRef = Snpe_IUserBuffer_GetEncoding_Ref(userBufferHandle);
    Snpe_UserBufferEncoding_ElementType_t dataType = Snpe_UserBufferEncoding_GetElementType(encodingRef);
    size_t elementSize = Snpe_UserBufferEncoding_GetElementSize(encodingRef);
    size_t elementNum = bufferSize / elementSize;
    std::vector<float> saveBuffer(elementNum, 0);
    float* dst = saveBuffer.data();
    uint8_t* src = applicationBuffer.data();
    switch (dataType) {
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
            dst = (float*)src;
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32:
            NativeToNative(dst, (int32_t*)src, elementNum);
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32:
            NativeToNative(dst, (uint32_t*)src, elementNum);
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8:
            NativeToNative(dst, (int8_t*)src, elementNum);
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8:
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8:
            NativeToNative(dst, (uint8_t*)src, elementNum);
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16:
            NativeToNative(dst, (int16_t*)src, elementNum);
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16:
            NativeToNative(dst, (uint16_t*)src, elementNum);
            break;
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16:
            {
                size_t bitWidth = dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 ? 8 : 16;
                uint64_t stepEquivalentTo0 = Snpe_UserBufferEncodingTfN_GetStepExactly0(encodingRef);
                float quantizedStepSize = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(encodingRef);
                TfNToFloat(dst, src, stepEquivalentTo0, quantizedStepSize, elementNum, bitWidth);
            }
            break;
        default:
            std::cerr << "Error: Unsupported data type: "<< static_cast<typename std::underlying_type<decltype(dataType)>::type>(dataType) << std::endl;
            return false;
        }

    // Process for splitBatch is false
    if (!splitBatch) {
        if (!SaveRawData(filePath, (const char*)dst, elementNum * sizeof(float))) {
            std::cerr << "Error: Save buffer fail. path: "<< filePath << std::endl;
            return false;
        }
        else {
            std::cout << "Save buffer to path: "<< filePath << std::endl;
            return true;
        }
    }

    // Process for splitBatch is true
    if (saveBatchChunk > batchSize) {
        std::cerr << "Error: saveBatchChunk(" << saveBatchChunk << ") should not bigger than batchSize("
                  << batchSize <<")." << std::endl;
        return false;
    }
    size_t chunkElementNum = elementNum / batchSize;
    std::string mark = "${index}";
    size_t pos = filePath.find(mark);
    std::string realFilePath;
    for(size_t i = 0; i < saveBatchChunk; i++) {
        realFilePath = filePath.substr(0, pos)
                     + std::to_string(indexOffset * batchSize + i)
                     + filePath.substr(pos + mark.size());
        if (!SaveRawData(realFilePath,
                         (const char*)dst + i * chunkElementNum * sizeof(float),
                         chunkElementNum * sizeof(float))) {
            std::cerr << "Error: Save buffer fail. path: "<< realFilePath << std::endl;
            return false;
        }
        else {
            std::cout << "Save buffer to path: "<< realFilePath << std::endl;
        }
    }
    return true;
}

bool SaveUserBufferMap(Snpe_UserBufferMap_Handle_t userBufferMapHandle,
                       std::unordered_map<std::string, std::vector<uint8_t>>& applicationBufferMap,
                       std::string& dirPath,
                       bool splitBatch,
                       size_t batchSize,
                       size_t indexOffset,
                       size_t saveBatchChunk)
{
    if (userBufferMapHandle == nullptr) {
        std::cerr << "Error: userBufferMapHandle is null" << std::endl;
        return false;
    }
    Snpe_StringList_Handle_t namesHandle = Snpe_UserBufferMap_GetUserBufferNames(userBufferMapHandle);
    size_t size = Snpe_StringList_Size(namesHandle);
    for (size_t i = 0; i < size; i++) {
        const char* name = Snpe_StringList_At(namesHandle, i);
        std::string filePath = dirPath + "/" + std::string(name) + ".raw";
        Snpe_IUserBuffer_Handle_t userBufferRef = nullptr;
        userBufferRef = Snpe_UserBufferMap_GetUserBuffer_Ref(userBufferMapHandle, name);
        if(!SaveUserBuffer(userBufferRef,
                           applicationBufferMap[name],
                           filePath,
                           splitBatch,
                           batchSize,
                           indexOffset,
                           saveBatchChunk)) {
            std::cerr << "Error: Save userBuffer("<< name <<") fail." << std::endl;
            return false;
        }
    }
    Snpe_StringList_Delete(namesHandle);
    return true;
}

bool SaveUserBufferList(Snpe_UserBufferList_Handle_t userBufferListHandle,
                        std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>& applicationBufferList,
                        std::string& dirPath,
                        bool splitBatch,
                        size_t batchSize,
                        size_t inputFileNum)
{
    if (userBufferListHandle == nullptr) {
        std::cerr << "Error: userBufferListHandle is null" << std::endl;
        return false;
    }
    std::string bufferMapDirPath;
    size_t size = Snpe_UserBufferList_Size(userBufferListHandle);
    bool saveSuccess = false;
    for (size_t i = 0; i < size; i ++) {
        Snpe_UserBufferMap_Handle_t userBufferMapRef = nullptr;
        userBufferMapRef = Snpe_UserBufferList_At_Ref(userBufferListHandle, i);
        if (splitBatch) {
            bufferMapDirPath = dirPath + "/Result_${index}";
            size_t saveBatchChunk = batchSize;
            if (i == size - 1 && inputFileNum % batchSize != 0) {
                // At the last batch, input files might not fully fill batch.
                // So we need to calculate valid chucks of batch
                saveBatchChunk = inputFileNum % batchSize;
            }
            saveSuccess = SaveUserBufferMap(userBufferMapRef,
                                            applicationBufferList[i],
                                            bufferMapDirPath,
                                            splitBatch,
                                            batchSize,
                                            i,
                                            saveBatchChunk);
        }
        else {
            bufferMapDirPath = dirPath + "/Result_" + std::to_string(i);
            saveSuccess = SaveUserBufferMap(userBufferMapRef,
                                            applicationBufferList[i],
                                            bufferMapDirPath);
        }
        if(!saveSuccess) {
            std::cerr << "Error: Save UserBufferMap(index:"<< i <<") fail." << std::endl;
            return false;
        }
    }
    return true;
}

bool
LoadFileToUserBuffer(Snpe_IUserBuffer_Handle_t userBufferHandle,
                     std::vector<std::string>& files,
                     std::vector<uint8_t>& applicationBuffer)
{
    if (userBufferHandle == nullptr) {
        std::cerr << "Error: userBufferHandle is null" << std::endl;
        return false;
    }
    size_t bufferSize = Snpe_IUserBuffer_GetSize(userBufferHandle);
    if (bufferSize != applicationBuffer.size()) {
        std::cerr << "Error: Unequaled size of UserBuffer(" << bufferSize << ") and "
                  << "applicationBuffer (" << applicationBuffer.size() << ")." << std::endl;
        return false;
    }

    Snpe_TensorShape_Handle_t stridesRef =  Snpe_IUserBuffer_GetStrides_Ref(userBufferHandle);
    size_t chunckSize = Snpe_TensorShape_At(stridesRef, 0);
    size_t batchSize = bufferSize / chunckSize;
    if (files.size() > batchSize) {
        std::cerr << "Error: Load file number(" << files.size() << ") should not bigger than batchSize("
                  << batchSize <<")." << std::endl;
        return false;
    }

    Snpe_UserBufferEncoding_Handle_t encodingRef = Snpe_IUserBuffer_GetEncoding_Ref(userBufferHandle);
    Snpe_UserBufferEncoding_ElementType_t dataType = Snpe_UserBufferEncoding_GetElementType(encodingRef);
    size_t elementSize = Snpe_UserBufferEncoding_GetElementSize(encodingRef);
    if (chunckSize % elementSize != 0) {
        std::cerr << "Error: Chunck size (" << chunckSize << ") should be divisible by element size ("
                  << elementSize <<")." << std::endl;
        return false;
    }

    //Read float raw data from file
    size_t elementNum = bufferSize / elementSize;
    size_t chunkElementNum = chunckSize / elementSize;
    std::vector<float> readBuf(elementNum, 0);
    float* src = readBuf.data();
    for (size_t i = 0; i < files.size(); i++) {
        bool readSuccess = false;
        if (!ReadRawData(files[i],
                         (char*)src + i * chunkElementNum * sizeof(float),
                         chunkElementNum * sizeof(float))) {
            std::cerr << "Error: Read buffer fail. path: "<< files[i] << std::endl;
            return false;
        }
    };

    //Conver data from float ot type of UserBuffer
    uint8_t* dst = applicationBuffer.data();
    switch (dataType) {
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
        NativeToNative((float*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32:
        NativeToNative((int32_t*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32:
        NativeToNative((uint32_t*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8:
        NativeToNative((int8_t*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8:
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8:
        NativeToNative((uint8_t*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16:
        NativeToNative((int16_t*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16:
        NativeToNative((uint16_t*)dst, src, elementNum);
        break;
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
    case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16:
        {
            uint64_t stepEquivalentTo0 = 0;
            float quantizedStepSize = 0;
            size_t bitWidth = dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 ? 8 : 16;
            if (!FloatToTfN(dst, stepEquivalentTo0, quantizedStepSize, src, elementNum, bitWidth)){
                std::cerr << "Error: Convert float to tf8 fail." << std::endl;
                return false;
            }
            Snpe_UserBufferEncodingTfN_SetStepExactly0(encodingRef, stepEquivalentTo0);
            Snpe_UserBufferEncodingTfN_SetQuantizedStepSize(encodingRef, quantizedStepSize);
        }
        break;
    default:
        std::cerr << "Error: Unsupported data type: "<< static_cast<typename std::underlying_type<decltype(dataType)>::type>(dataType) << std::endl;
        return false;
    }
    return true;
}

bool
LoadFileToUserBufferMap(Snpe_UserBufferMap_Handle_t userBufferMapHandle,
                        std::unordered_map<std::string, std::vector<std::string>>& fileMap,
                        std::unordered_map<std::string, std::vector<uint8_t>>& applicationBufferMap)
{
    if (userBufferMapHandle == nullptr) {
        std::cerr << "Error: userBufferMapHandle is null" << std::endl;
        return false;
    }
    Snpe_StringList_Handle_t namesHandle = Snpe_UserBufferMap_GetUserBufferNames(userBufferMapHandle);
    size_t size = Snpe_StringList_Size(namesHandle);
    for (size_t i = 0; i < size; i++) {
        const char* name = Snpe_StringList_At(namesHandle, i);
        Snpe_IUserBuffer_Handle_t userBufferRef = nullptr;
        userBufferRef = Snpe_UserBufferMap_GetUserBuffer_Ref(userBufferMapHandle, name);
        if(!LoadFileToUserBuffer(userBufferRef, fileMap[name], applicationBufferMap[name])) {
            std::cerr << "Error: Load file to userBuffer("<< name <<") fail." << std::endl;
            return false;
        }
    }
    Snpe_StringList_Delete(namesHandle);
    return true;
}

bool
LoadFileToUserBufferList(Snpe_UserBufferList_Handle_t userBufferListHandle,
                         std::vector<std::unordered_map<std::string, std::vector<std::string>>>& fileList,
                         std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>& applicationBufferList)
{
    if (userBufferListHandle == nullptr) {
        std::cerr << "Error: userBufferListHandle is null" << std::endl;
        return false;
    }
    size_t size = Snpe_UserBufferList_Size(userBufferListHandle);
    if (size != applicationBufferList.size() || size != fileList.size()) {
        std::cerr << "Error: Unmatched sizes of"
                  <<" applicationBufferList("<< applicationBufferList.size() <<"), "
                  <<" inputList("<< fileList.size() <<") and "
                  <<" userBufferList("<< size <<")."
                  << std::endl;
        return false;
    }

    for (size_t i = 0; i < size; i++) {
        Snpe_UserBufferMap_Handle_t userBufferMapRef = Snpe_UserBufferList_At_Ref(userBufferListHandle, i);
        if (!LoadFileToUserBufferMap(userBufferMapRef, fileList[i], applicationBufferList[i])) {
            std::cerr << "Error: Load file to UserBufferMap(index:"<< i <<") fail." << std::endl;
            return false;
        }
    }
    return true;
}

