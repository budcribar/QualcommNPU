//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <vector>
#include <string>
#include <unordered_map>

#include "SNPE/SNPE.h"
#include "DlSystem/IUserBuffer.h"
#include "DlSystem/UserBufferMap.h"
#include "Util.hpp"

/**
 * @brief .
 *
 * @param userBufferMap handle for Input/Output Buffer
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param snpeUserBackendBufferHandles vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle snpe Handle
 *
 * @param name name of the tensor
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param useRpc Flag to know whether to use libcdsprpc.so or libdmabufheap.so
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth Bit width
 *
 * @param memFnHandles Struct containing functional pointers to Mem Alloc Function
 *
 * @return Successfully created or not
 *
 */
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
                      MemFnHandlesType_t& memFnHandles); // Defined in Common/Util.hpp

/**
 * @brief .
 *
 * @param inputMapHandle handle for Input user Buffer
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param snpeUserBackendBufferHandles vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle snpe Handle
 *
 * @param isTfNBuffer  Flag to know whether TfN mode is enabled or not
 *
 * @param useRpc Flag to know whether to use libcdsprpc.so or libdmabufheap.so
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth Bit width
 *
 * @param memFnHandles Struct containing functional pointers to Mem Alloc Function
 *
 * @return Successfully created or not
 */
bool CreateInputBufferMap(Snpe_UserBufferMap_Handle_t inputMapHandle,
                          Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                          std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                          Snpe_SNPE_Handle_t snpeHandle,
                          bool isTfNBuffer,
                          bool useRpc,
                          bool staticQuantization,
                          int bitWidth,
                          void* bufferAllocator,
                          MemFnHandlesType_t& memFnHandles); // Defined in Common/Util.hpp

/**
 * @brief .
 *
 * @param outputMapHandle handle for Output user Buffer
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param snpeUserBackendBufferHandles vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param useRpc Flag to know whether to use libcdsprpc.so or libdmabufheap.so
 *
 * @param bitWidth Bit width
 *
 * @param memFnHandles Struct containing functional pointers to Mem Alloc Function
 *
 * @return Successfully created or not
 *
 */
bool CreateOutputBufferMap(Snpe_UserBufferMap_Handle_t outputMapHandle,
                          Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                          std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                          Snpe_SNPE_Handle_t snpeHandle,
                          bool isTfNBuffer,
                          bool useRpc,
                          int bitWidth,
                          void* bufferAllocator,
                          MemFnHandlesType_t& memFnHandles); // Defined in Common/Util.hpp

/**
 * @brief .
 *
 * @param userBufferMap Handle for Input/Output Buffer
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param snpeUserBackendBufferHandles Vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle Snpe Handle
 *
 * @param name Name of the tensor
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth Bit width
 *
 * @param baseAddr Pointer pointing to start of shared memory space
 *
 * @param bufSize The size of the specific user buffer
 *
 * @param totalBufSize The total buffer size of the entire shared memory space
 *
 * @param offset The current offset from baseAddr pointing to where to create the next user buffer
 *
 * @return Successfully created or not
 *
 */
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
                            uint64_t& offset);



/**
 * @brief .
 *
 * @param bufferNamesHandle Handle for string list of buffer names
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param bitWidth Bit width
 *
 * @param snpeHandle Snpe Handle
 *
 * @param bufSizeMap A mapping of buffer names to their respective sizes
 *
 * @return total buffer size for the list of buffer names
 *
 */
size_t GetTotalBufferSizeForHandle(Snpe_StringList_Handle_t bufferNamesHandle,
                                   bool isTfNBuffer,
                                   int bitWidth,
                                   Snpe_SNPE_Handle_t snpeHandle,
                                   std::unordered_map<std::string, size_t>& bufSizeMap);


/**
 * @brief .
 *
 * @param snpeHandle Snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param bitWidth Bit width
 *
 * @param bufSizeMap A mapping of buffer names to their respective sizes
 *
 * @return total buffer size for the all input and output buffers
 *
 */
size_t GetTotalInputOutputBufferSize(Snpe_SNPE_Handle_t snpeHandle,
                                     bool isTfNBuffer,
                                     int bitWidth,
                                     std::unordered_map<std::string, size_t>& bufSizeMap);

/**
 * @brief .
 *
 * @param inputMapHandle Handle for Input Buffer Map
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param snpeUserBackendBufferHandles Vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param bufSizeMap A mapping of buffer names to their respective sizes
 *
 * @param snpeHandle Snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth Bit width
 *
 * @param baseAddr Pointer pointing to start of shared memory space
 *
 * @param fd The file descripter corresponding to baseAddr
 *
 * @param totalBufSize The total buffer size of the entire shared memory space
 *
 * @param offset The current offset from baseAddr pointing to where to create the next user buffer
 *
 * @return Successfully created or not
 *
 */
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
                                uint64_t& offset);

/**
 * @brief .
 *
 * @param outputMapHandle Handle for Output Buffer Map
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param snpeUserBackendBufferHandles Vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param bufSizeMap A mapping of buffer names to their respective sizes
 *
 * @param snpeHandle Snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param bitWidth Bit width
 *
 * @param baseAddr Pointer pointing to start of shared memory space
 *
 * @param fd The file descripter corresponding to baseAddr
 *
 * @param totalBufSize The total buffer size of the entire shared memory space
 *
 * @param offset The current offset from baseAddr pointing to where to create the next user buffer
 *
 * @return Successfully created or not
 *
 */
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
                                 uint64_t& offset);

/**
 * @brief .
 *
 * @param bufferMapHandle Handle for Buffer Map
 *
 * @param memMappedBufferMapHandle Handle to Memory Mapped User Buffer Map
 *
 * @param bufferNamesHandle Handle for string list of buffer names
 *
 * @param snpeUserBackendBufferHandles Vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param bufSizeMap A mapping of buffer names to their respective sizes
 *
 * @param snpeHandle Snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth Bit width
 *
 * @param baseAddr Pointer pointing to start of shared memory space
 *
 * @param fd The file descripter corresponding to baseAddr
 *
 * @param totalBufSize The total buffer size of the entire shared memory space
 *
 * @param offset The current offset from baseAddr pointing to where to create the next user buffer
 *
 * @return Successfully created or not
 *
 */
bool CreateBufferMapShared(Snpe_UserBufferMap_Handle_t bufferMapHandle,
                           Snpe_UserMemoryMap_Handle_t memMappedBufferMapHandle,
                           Snpe_StringList_Handle_t bufferNameshandle,
                           std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                           const std::unordered_map<std::string, size_t>& bufSizeMap,
                           Snpe_SNPE_Handle_t snpeHandle,
                           bool isTfNBuffer,
                           bool staticQuantization,
                           int bitWidth,
                           void* baseAddr,
                           int fd,
                           const size_t totalBufSize,
                           uint64_t& offset);


/**
 * @brief With memory Mapped UserBuffer Data type of Tensor must match the User Buffer Type
 *
 * @param snpeHandle snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param bitWidth bit width
 *
 * @return Userbuffer matches with Tensor Data type or not
 *
 */
bool IsUserBufferConsistentWithTensorDataType(Snpe_SNPE_Handle_t snpeHandle, int bitWidth, bool isTfNBuffer);
