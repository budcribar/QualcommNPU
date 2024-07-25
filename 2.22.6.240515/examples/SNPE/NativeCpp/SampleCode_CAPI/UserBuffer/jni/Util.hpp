//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <dlfcn.h>
#include "DlSystem/TensorShape.h"

size_t calcSizeFromDims(const size_t* dims, size_t rank, size_t elementSize );

Snpe_TensorShape_Handle_t calcStrides(Snpe_TensorShape_Handle_t dimsHandle, size_t elementSize);

template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
    result.clear();
    std::istringstream ss( s );
    while (!ss.eof())
    {
        typename Container::value_type field;
        getline( ss, field, delimiter );
        if (field.empty()) continue;
        result.push_back( field );
    }
    return result;
}

typedef void* (*rpcMemAllocFnHandleType_t)(int, uint32_t, int);
typedef void (*rpcMemFreeFnHandleType_t)(void*);

typedef void* (*dmaCreateBufferAllocatorFnHandleType_t)();
typedef void (*dmaFreeBufferAllocatorFnHandleType_t)(void*);
typedef int (*dmaMemAllocFnHandleType_t)(void*, const char*, size_t, unsigned int, size_t);
typedef int (*dmaMapHeapToIonFnHandleType_t)(void*, const char*, const char*, unsigned int, unsigned int, unsigned int);

struct MemFnHandlesType_t {
    rpcMemAllocFnHandleType_t rpcMemAllocFnHandle;
    rpcMemFreeFnHandleType_t rpcMemFreeFnHandle;

    dmaCreateBufferAllocatorFnHandleType_t dmaCreateBufAllocFnHandle;
    dmaMemAllocFnHandleType_t dmaMemAllocFnHandle;
    dmaFreeBufferAllocatorFnHandleType_t dmaFreeBufAllocFnHandle;
    dmaMapHeapToIonFnHandleType_t dmaMapHeapToIonFnHandle;
};

template <class T>
static inline T resolveSymbol(void* libHandle, const char* sym) {
    T ptr = (T)dlsym(libHandle, sym);
    if(ptr == nullptr) {
        std::cerr<<"Function is unavailable: "<<sym<<std::endl;
    }
    return ptr;
}

template<typename T>
bool loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (length % sizeof(T) != 0) {
        std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
        return false;
    }

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(T));
    } else if (loadVector.size() < length / sizeof(T)) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
        loadVector.resize(length / sizeof(T));
    }

    if (!in.read(reinterpret_cast<char*>(&loadVector[0]), length))
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
    return true;
}

template<typename T>
bool loadByteDataFileBatched(const std::string& inputFile, std::vector<T>& loadVector, size_t offset)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (length % sizeof(T) != 0) {
        std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
        return false;
    }

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(T));
    } else if (loadVector.size() < length / sizeof(T)) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
    }

    loadVector.resize( (offset+1) * length / sizeof(T) );

    if (!in.read( reinterpret_cast<char*> (&loadVector[offset * length/ sizeof(T) ]), length) )
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
    return true;
}

bool GetMemFnHandles(bool useRpc, MemFnHandlesType_t& memFnHandles);

std::pair<void*, int> GetBufferAddrFd(size_t buffSize, bool useRpc, void* bufferAllocator, MemFnHandlesType_t& memFnHandles);
int MapDmaHeapToIon(void* bufferAllocator, MemFnHandlesType_t& memFnHandles);

std::vector<float> loadFloatDataFile(const std::string& inputFile);
std::vector<unsigned char> loadByteDataFile(const std::string& inputFile);
std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile);
bool loadByteDataFileBatchedTf8(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset);
bool loadByteDataFileBatchedTfN(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset,
                                uint64_t& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, int bitWidth);
bool loadByteDataFileBatchedTfN(const std::string& inputFile, void* loadPtr, size_t offset, uint64_t& stepEquivalentTo0,
                                float& quantizedStepSize, bool staticQuantization, int bitWidth);
bool loadByteDataFileBatchedFloat(const std::string& inputFile, void* loadPtr, uint64_t offset);

bool SaveITensor(const std::string& path, float* data, size_t tensorStart=0, size_t tensorEnd=0);
bool SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex=0, size_t batchChunk=0);
bool EnsureDirectory(const std::string& dir);

void TfNToFloat(float *out, uint8_t *in, const uint64_t stepEquivalentTo0, const float quantizedStepSize, size_t numElement, int bitWidth);
bool FloatToTfN(uint8_t* out, uint64_t& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, float* in, size_t numElement, int bitWidth);
void setResizableDim(size_t resizableDim);
size_t getResizableDim();
#endif
