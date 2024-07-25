//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <dlfcn.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>

#include "Util.hpp"

#include "DlSystem/TensorShape.h"
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1
#define DMABUF_HEAP_NAME_SYSTEM "system"
#define DMABUF_DEFAULT_FLAGS  1

size_t resizable_dim;


size_t calcSizeFromDims(const size_t* dims, size_t rank, size_t elementSize )
{
    if (rank == 0) return 0;
    size_t size = elementSize;
    while (rank--) {
        (*dims == 0) ? size *= resizable_dim : size *= *dims;
        dims++;
    }
    return size;
}

Snpe_TensorShape_Handle_t calcStrides(Snpe_TensorShape_Handle_t dimsHandle, size_t elementSize){
    std::vector<size_t> strides(Snpe_TensorShape_Rank(dimsHandle));
    strides[strides.size() - 1] = elementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = Snpe_TensorShape_Rank(dimsHandle) - 1; i > 0; i--)
    {
        if(Snpe_TensorShape_At(dimsHandle, i) != 0)
            stride *= Snpe_TensorShape_At(dimsHandle, i);
        else
            stride *= resizable_dim;
        strides[i-1] = stride;
    }
    Snpe_TensorShape_Handle_t tensorShapeHandle = Snpe_TensorShape_CreateDimsSize(strides.data(), Snpe_TensorShape_Rank(dimsHandle));
    return tensorShapeHandle;
}

bool GetMemFnHandles(bool useRpc, MemFnHandlesType_t& memFnHandles) {
    // use libcdsprpc.so
    if (useRpc) {
        void* libRpcHandle = dlopen("libcdsprpc.so", RTLD_NOW|RTLD_GLOBAL);
        if(libRpcHandle == nullptr) {
            std::cerr << "Error: could not open libcdsprpc.so " << std::endl;
            return false;
        }
        memFnHandles.rpcMemAllocFnHandle = resolveSymbol<rpcMemAllocFnHandleType_t>(libRpcHandle, "rpcmem_alloc");
        if(memFnHandles.rpcMemAllocFnHandle == nullptr) {
            std::cerr << "Error: could not access rpcmem_alloc" << std::endl;
            return false;
        }
        memFnHandles.rpcMemFreeFnHandle = resolveSymbol<rpcMemFreeFnHandleType_t>(libRpcHandle, "rpcmem_free");
        if(memFnHandles.rpcMemFreeFnHandle == nullptr) {
            std::cerr << "Error: could not access rpcmem_free" << std::endl;
            return false;
        }
    // use libdmabufheap.so
    } else {
        void* libDmaHandle = dlopen("libdmabufheap.so", RTLD_NOW|RTLD_GLOBAL);
        if(libDmaHandle == nullptr) {
            std::cerr << "Error: could not open libdmabufheap.so " << std::endl;
            return false;
        }

        memFnHandles.dmaCreateBufAllocFnHandle = resolveSymbol<dmaCreateBufferAllocatorFnHandleType_t>(libDmaHandle, "CreateDmabufHeapBufferAllocator");
        if(memFnHandles.dmaCreateBufAllocFnHandle == nullptr) {
            std::cerr << "Error: could not access CreateDmabufHeapBufferAllocator" << std::endl;
            return false;
        }

        memFnHandles.dmaFreeBufAllocFnHandle = resolveSymbol<dmaFreeBufferAllocatorFnHandleType_t>(libDmaHandle, "FreeDmabufHeapBufferAllocator");
        if(memFnHandles.dmaFreeBufAllocFnHandle == nullptr) {
            std::cerr << "Error: could not access FreeDmabufHeapBufferAllocator" << std::endl;
            return false;
        }

        memFnHandles.dmaMemAllocFnHandle = resolveSymbol<dmaMemAllocFnHandleType_t>(libDmaHandle, "DmabufHeapAlloc");
        if(memFnHandles.dmaMemAllocFnHandle == nullptr) {
            std::cerr << "Error: could not access DmabufHeapAlloc" << std::endl;
            return false;
        }

        memFnHandles.dmaMapHeapToIonFnHandle = resolveSymbol<dmaMapHeapToIonFnHandleType_t>(libDmaHandle, "MapDmabufHeapNameToIonHeap");
        if(memFnHandles.dmaMapHeapToIonFnHandle == nullptr) {
            std::cerr << "Error: could not access MapDmabufHeapNameToIonHeap" << std::endl;
            return false;
        }
    }

    return true;
}

std::pair<void*, int> GetBufferAddrFd(size_t buffSize, bool useRpc, void* bufferAllocator, MemFnHandlesType_t& memFnHandles) {
    void* addr = nullptr;
    int fd = -1;

    if (useRpc) {
        addr = memFnHandles.rpcMemAllocFnHandle(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, buffSize);
    } else {
        fd = memFnHandles.dmaMemAllocFnHandle(bufferAllocator, DMABUF_HEAP_NAME_SYSTEM, buffSize, DMABUF_DEFAULT_FLAGS, 0);
        if (fd <= 0) {
            return {nullptr, -1};
        }
        addr = mmap(nullptr, buffSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    return {addr, fd};
}

int MapDmaHeapToIon(void* bufferAllocator, MemFnHandlesType_t& memFnHandles) {
    return memFnHandles.dmaMapHeapToIonFnHandle(bufferAllocator, DMABUF_HEAP_NAME_SYSTEM, "", 0, DMABUF_DEFAULT_FLAGS, 0);
}

bool EnsureDirectory(const std::string& dir)
{
    auto i = dir.find_last_of('/');
    std::string prefix = dir.substr(0, i);

    if (dir.empty() || dir == "." || dir == "..")
    {
        return true;
    }

    if (i != std::string::npos && !EnsureDirectory(prefix))
    {
        return false;
    }

    int rc = mkdir(dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    if (rc == -1 && errno != EEXIST)
    {
        return false;
    }
    else
    {
        struct stat st;
        if (stat(dir.c_str(), &st) == -1)
        {
            return false;
        }

        return S_ISDIR(st.st_mode);
    }
}

std::vector<float> loadFloatDataFile(const std::string& inputFile)
{
    std::vector<float> vec;
    loadByteDataFile(inputFile, vec);
    return vec;
}

std::vector<unsigned char> loadByteDataFile(const std::string& inputFile)
{
    std::vector<unsigned char> vec;
    loadByteDataFile(inputFile, vec);
    return vec;
}

std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile)
{
    std::vector<unsigned char> vec;
    size_t offset=0;
    loadByteDataFileBatched(inputFile, vec, offset);
    return vec;
}

void TfNToFloat(float *out,
                uint8_t *in,
                const uint64_t stepEquivalentTo0,
                const float quantizedStepSize,
                size_t numElement,
                int bitWidth)
{
    for (size_t i = 0; i < numElement; ++i) {
        if (bitWidth == 8) {
            double quantizedValue = static_cast <double> (in[i]);
            double stepEqTo0 = static_cast <double> (stepEquivalentTo0);
            out[i] = static_cast <double> ((quantizedValue - stepEqTo0) * quantizedStepSize);
        }
        else if (bitWidth == 16) {
            uint16_t *temp = (uint16_t *)in;
            double quantizedValue = static_cast <double> (temp[i]);
            double stepEqTo0 = static_cast <double> (stepEquivalentTo0);
            out[i] = static_cast <double> ((quantizedValue - stepEqTo0) * quantizedStepSize);
        }
    }
}

bool FloatToTfN(uint8_t* out,
                uint64_t& stepEquivalentTo0,
                float& quantizedStepSize,
                bool staticQuantization,
                float* in,
                size_t numElement,
                int bitWidth)
{
    double encodingMin;
    double encodingMax;
    double encodingRange;
    double trueBitWidthMax = pow(2, bitWidth) -1;

    if (!staticQuantization) {
        float trueMin = std::numeric_limits <float>::max();
        float trueMax = std::numeric_limits <float>::min();

        for (size_t i = 0; i < numElement; ++i) {
            trueMin = fmin(trueMin, in[i]);
            trueMax = fmax(trueMax, in[i]);
        }

        double stepCloseTo0;

        if (trueMin > 0.0f) {
            stepCloseTo0 = 0.0;
            encodingMin = 0.0;
            encodingMax = trueMax;
        } else if (trueMax < 0.0f) {
            stepCloseTo0 = trueBitWidthMax;
            encodingMin = trueMin;
            encodingMax = 0.0;
        } else {
            double trueStepSize = static_cast <double>(trueMax - trueMin) / trueBitWidthMax;
            stepCloseTo0 = -trueMin / trueStepSize;
            if (stepCloseTo0 == round(stepCloseTo0)) {
                // 0.0 is exactly representable
                encodingMin = trueMin;
                encodingMax = trueMax;
            } else {
                stepCloseTo0 = round(stepCloseTo0);
                encodingMin = (0.0 - stepCloseTo0) * trueStepSize;
                encodingMax = (trueBitWidthMax - stepCloseTo0) * trueStepSize;
            }
        }

        const double minEncodingRange = 0.01;
        encodingRange = encodingMax - encodingMin;
        quantizedStepSize = encodingRange / trueBitWidthMax;
        stepEquivalentTo0 = static_cast <uint64_t> (round(stepCloseTo0));

        if (encodingRange < minEncodingRange) {
            std::cerr << "Expect the encoding range to be larger than " << minEncodingRange << "\n"
                      << "Got: " << encodingRange << "\n";
            return false;
        }
    }
    else
    {
        if (bitWidth == 8) {
            encodingMin = (0 - static_cast <uint8_t> (stepEquivalentTo0)) * quantizedStepSize;
        } else if (bitWidth == 16) {
            encodingMin = (0 - static_cast <uint16_t> (stepEquivalentTo0)) * quantizedStepSize;
        } else {
            std::cerr << "Quantization bitWidth is invalid " << std::endl;
            return false;
        }
        encodingMax = (trueBitWidthMax - stepEquivalentTo0) * quantizedStepSize;
        encodingRange = encodingMax - encodingMin;
    }

    for (size_t i = 0; i < numElement; ++i) {
        int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);

        if (quantizedValue < 0)
            quantizedValue = 0;
        else if (quantizedValue > (int)trueBitWidthMax)
            quantizedValue = (int)trueBitWidthMax;

        if(bitWidth == 8){
            out[i] = static_cast <uint8_t> (quantizedValue);
        }
        else if(bitWidth == 16){
            uint16_t *temp = (uint16_t *)out;
            temp[i] = static_cast <uint16_t> (quantizedValue);
        }
    }
    return true;
}

size_t loadFileToBuffer(const std::string& inputFile, std::vector<float>& loadVector)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
        return 0U;
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    loadVector.resize(length / sizeof(float));
    if (!in.read( reinterpret_cast<char*> (&loadVector[0]), length) )
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
        return 0U;
    }

    return length;
}

bool loadByteDataFileBatchedTfN(const std::string& inputFile, void* loadPtr, size_t offset, uint64_t& stepEquivalentTo0,
                                float& quantizedStepSize, bool staticQuantization, int bitWidth) {
    std::vector<float> inVector;
    size_t inputFileLength = loadFileToBuffer(inputFile, inVector);

    if(inputFileLength == 0U || !FloatToTfN((uint8_t*)loadPtr + offset, stepEquivalentTo0, quantizedStepSize, staticQuantization, inVector.data(), inputFileLength/sizeof(float), bitWidth))
    {
        return false;
    }
    return true;
}

bool loadByteDataFileBatchedTfN(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset,
                                uint64_t& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, int bitWidth)
{
    std::vector<float> inVector;

    size_t inputFileLength = loadFileToBuffer(inputFile, inVector);

    if (inputFileLength == 0U) {
        return false;
    } else if (loadVector.size() == 0) {
        loadVector.resize(inputFileLength / sizeof(uint8_t));
    } else if (loadVector.size() < inputFileLength/sizeof(float)) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
        return false;
    }

    int elementSize = bitWidth / 8;
    size_t dataStartPos = (offset * inputFileLength * elementSize) / sizeof(float);
    if(!FloatToTfN(&loadVector[dataStartPos], stepEquivalentTo0, quantizedStepSize, staticQuantization, inVector.data(), inVector.size(), bitWidth))
    {
        return false;
    }
    return true;
}

bool loadByteDataFileBatchedTf8(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    std::vector<float> inVector;
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(uint8_t));
    } else if (loadVector.size() < length/sizeof(float)) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
    }

    inVector.resize((offset+1) * length / sizeof(uint8_t));
    if (!in.read( reinterpret_cast<char*> (&inVector[offset * length/ sizeof(uint8_t) ]), length) )
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }

    uint64_t stepEquivalentTo0;
    float quantizedStepSize;
    if(!FloatToTfN(loadVector.data(), stepEquivalentTo0, quantizedStepSize, false, inVector.data(), loadVector.size(), 8))
    {
        return false;
    }
    return true;
}

bool loadByteDataFileBatchedFloat(const std::string& inputFile, void* loadPtr, uint64_t offset)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (length % sizeof(float) != 0) {
        std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
        return false;
    }

    if (!in.read( (char*)loadPtr + offset, length) )
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
    return true;
}

bool SaveITensor(const std::string& path, float* data, size_t tensorStart, size_t tensorEnd)
{
    // Create the directory path if it does not exist
    auto idx = path.find_last_of('/');
    if (idx != std::string::npos)
    {
        std::string dir = path.substr(0, idx);
        if (!EnsureDirectory(dir))
        {
            std::cerr << "Failed to create output directory: " << dir << ": "
                      << std::strerror(errno) << "\n";
            return false;
        }
    }

    std::ofstream os(path, std::ofstream::binary);
    if (!os)
    {
        std::cerr << "Failed to open output file for writing: " << path << "\n";
        return false;
    }

    for ( auto it = data + tensorStart; it != data + tensorEnd; ++it )
    {
        float f = *it;
        if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
        {
            std::cerr << "Failed to write data to: " << path << "\n";
            return false;
        }
    }
    return true;
}

bool SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex, size_t batchChunk)
{
    if(batchChunk == 0)
        batchChunk = buffer.size();
    // Create the directory path if it does not exist
    auto idx = path.find_last_of('/');
    if (idx != std::string::npos)
    {
        std::string dir = path.substr(0, idx);
        if (!EnsureDirectory(dir))
        {
            std::cerr << "Failed to create output directory: " << dir << ": "
                      << std::strerror(errno) << "\n";
            return false;
        }
    }

    std::ofstream os(path, std::ofstream::binary);
    if (!os)
    {
        std::cerr << "Failed to open output file for writing: " << path << "\n";
        return false;
    }

    for ( auto it = buffer.begin() + batchIndex * batchChunk; it != buffer.begin() + (batchIndex+1) * batchChunk; ++it )
    {
        uint8_t u = *it;
        if(!os.write((char*)(&u), sizeof(uint8_t)))
        {
            std::cerr << "Failed to write data to: " << path << "\n";
            return false;
        }
    }
    return true;
}

void setResizableDim(size_t resizableDim)
{
    resizable_dim = resizableDim;
}

size_t getResizableDim()
{
    return resizable_dim;
}
