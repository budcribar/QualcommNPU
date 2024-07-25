//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef PROCESS_USER_BUFFER_H
#define PROCESS_USER_BUFFER_H

#include <vector>
#include <string>
#include <unordered_map>

#include "SNPE/PSNPE.h"
#include "SNPE/SNPEUtil.h"
#include "SNPE/UserBufferList.h"
#include "DlSystem/IUserBuffer.h"
#include "DlSystem/UserBufferMap.h"


/**
 * @brief
 *
 * Create a UserBuffer and fill with a user-backed buffer
 *
 * @param dims Buffer dimensions required
 *
 * @param userBufferType Type of UserBuffer
 *
 * @param applicationBuffer Buffer storing tensor along with data
 *
 * @param stepEquivalentTo0 encoding param for data type tfN (only work under userBufferTypes is tf8 or tf16)
 *
 * @param quantizedStepSize encoding param for data type tfN (only work under userBufferTypes is tf8 or tf16)
 *
 * @return UserBuffer handle or nullptr
 *
 */
Snpe_IUserBuffer_Handle_t
CreateUserBuffer(const std::vector<size_t>& dims,
                 Snpe_UserBufferEncoding_ElementType_t userBufferType,
                 std::vector<uint8_t>& applicationBuffer,
                 uint64_t stepEquivalentTo0,
                 float quantizedStepSize);

/**
 * @brief
 *
 * Create a UserBufferMap of the SNPE network inputs
 *
 * @param bufferDims All dimensions of this buffer map, indexed by buffer name
 *
 * @param userBufferTypes Type of UserBuffer stored as map<BufferName, DataType>
 *
 * @param applicationBufferMap Map storing name of tensor along with data
 *
 * @param stepEquivalentTo0 encoding param for data type tfN (only work under userBufferTypes is tf8 or tf16)
 *
 * @param quantizedStepSize encoding param for data type tfN (only work under userBufferTypes is tf8 or tf16)
 *
 * @return UserBufferMap handle or nullptr
 */
Snpe_UserBufferMap_Handle_t
CreateUserBufferMap(std::unordered_map<std::string, std::vector<size_t>>& bufferDims,
                    std::unordered_map<std::string, Snpe_UserBufferEncoding_ElementType_t>& userBufferTypes,
                    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBufferMap,
                    std::unordered_map<std::string, uint64_t>& stepEquivalentTo0,
                    std::unordered_map<std::string, float>& quantizedStepSize);

/**
 * @brief
 *
 * Create a UserBufferList for PSNPE
 *
 * @param bufferDims All dimensions of this buffer map, indexed by buffer name
 *
 * @param userBufferTypes Type of UserBuffer stored as map<BufferName, DataType>
 *
 * @param bufferMapNumber Number of UserBufferMap will be loaded in UserBufferList
 *
 * @param applicationBufferList List of map storing name of tensor along with data
 *
 * @param stepEquivalentTo0 encoding param for data type tfN (only work under userBufferTypes is tf8 or tf16)
 *
 * @param quantizedStepSize encoding param for data type tfN (only work under userBufferTypes is tf8 or tf16)
 *
 * @return UserBufferList handle or nullptr
 *
 */
Snpe_UserBufferList_Handle_t
CreateUserBufferList(std::unordered_map<std::string, std::vector<size_t>>& bufferDims,
                     std::unordered_map<std::string, Snpe_UserBufferEncoding_ElementType_t>& userBufferTypes,
                     size_t bufferMapNumber,
                     std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>& applicationBufferList,
                     std::unordered_map<std::string, uint64_t>& stepEquivalentTo0,
                     std::unordered_map<std::string, float>& quantizedStepSize);


/**
 * @brief
 *
 * Totally clear and delete specified UserBufferMap
 *
 * @param userBufferMapHandle Handle of UserBufferMap need to be delete
 *
 * @return void
 */
void DeleteUserBufferMap(Snpe_UserBufferMap_Handle_t userBufferMapHandle);

/**
 * @brief
 *
 * Totally clear and delete specified UserBufferList
 *
 * @param userBufferListHandle Handle of UserBufferList need to be delete
 *
 * @return void
 */
void DeleteUserBufferList(Snpe_UserBufferList_Handle_t userBufferListHandle);



/**
 * @brief
 *
 * Save UserBuffer data as float raw data to file
 *
 * @param userBufferHandle Handle of the UserBuffer
 *
 * @param applicationBuffer Data buffer of UserBuffer
 *
 * @param filePath target saving file path
 *
 * @param splitBatch Determines whether to split and save according to batch
 *                   (default: false)
 *
 * @param batchSize  batch size of UserBuffer
 *                   (default: 0, working under splitBatch)
 *
 * @param indexOffset Set the starting value of the index which will be added
 *                    to the position marked with ${index} in the dirPath
 *                    (default: 0, working under splitBatch)
 *
 * @param saveBatchChunk Determines the number of part of batch will be save
 *                       (default: 0, working under splitBatch)
 *
 * @return boolean value of success or failure
 */
bool SaveUserBuffer(Snpe_IUserBuffer_Handle_t userBufferHandle,
                    std::vector<uint8_t>& applicationBuffer,
                    std::string& filePath,
                    bool splitBatch = false,
                    size_t batchSize = 0,
                    size_t indexOffset = 0,
                    size_t saveBatchChunk = 0);

/**
 * @brief
 *
 * Save UserBufferMap as float raw data to file. If UserBufferMap
 * has multiple userbuffer, this function will save each buffer as
 * a separate file named as buffer name.
 *
 * @param userBufferMapHandle Handle of the UserBufferMap
 *
 * @param applicationBufferMap Map storing name of tensor along with data
 *
 * @param dirPath target saving directory path
 *
 * @param splitBatch Determines whether to split and save according to batch
 *                   (default: false)
 *
 * @param batchSize  batch size of UserBuffer
 *                   (default: 0, working under splitBatch)
 *
 * @param indexOffset Set the starting value of the index which will be added
 *                    to the position marked with ${index} in the dirPath
 *                    (default: 0, working under splitBatch)
 *
 * @param saveBatchChunk Determines the number of part of batch will be save
 *                       (default: 0, working under splitBatch)
 *
 * @return boolean value of success or failure
 */
bool SaveUserBufferMap(Snpe_UserBufferMap_Handle_t userBufferMapHandle,
                       std::unordered_map<std::string, std::vector<uint8_t>>& applicationBufferMap,
                       std::string& dirPath,
                       bool splitBatch = false,
                       size_t batchSize = 0,
                       size_t indexOffset = 0,
                       size_t saveBatchChunk = 0);

/**
 * @brief
 *
 * Save all of UserBufferMaps in UserBufferList as float raw data to files
 *
 * @param userBufferListHandle Handle of the UserBufferList
 *
 * @param applicationBufferList List of map storing name of tensor along with data
 *
 * @param dirPath target saving directory path
 *
 * @param splitBatch Determines whether to split and save according to batch
 *                   (default: false)
 *
 * @param batchSize  batch size of UserBuffer
 *                   (default: 0, working under splitBatch)
 *
 * @param inputFileNum The number of loaded input files for calculating output number of last batch
 *                     (default: 0, working under splitBatch)
 *
 * @return boolean value of success or failure
 */
bool SaveUserBufferList(Snpe_UserBufferList_Handle_t userBufferListHandle,
                        std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>& applicationBufferList,
                        std::string& dirPath,
                        bool splitBatch,
                        size_t batchSize,
                        size_t inputFileNum);

/**
 * @brief Load data files to UserBufferMap
 *
 * @param userBufferHandle Handle of the UserBuffer
 *
 * @param files file pathes for loading into UserBuffer. Format: Map<BufferName,Batch<FilePath>>
 *
 * @param applicationBuffer Data buffer of UserBuffer
 *
 * @returns boolean value of success or failure
 */
bool
LoadFileToUserBuffer(Snpe_IUserBuffer_Handle_t userBufferHandle,
                     std::vector<std::string>& files,
                     std::vector<uint8_t>& applicationBuffer,
                     size_t batchSize);

/**
 * @brief Load data files to UserBufferMap
 *
 * @param userBufferMapHandle Handle of the UserBufferMap
 *
 * @param fileMap Sorted file pathes. Format: Map<BufferName,Batch<FilePath>>
 *
 * @param applicationBufferMap Buffer list storing name of tensor along with data.
 *                             Format: BufferMap<BufferName, BufferData>
 *
 * @returns boolean value of success or failure
 */
bool
LoadFileToUserBufferMap(Snpe_UserBufferMap_Handle_t userBufferMapHandle,
                        std::unordered_map<std::string, std::vector<std::string>>& fileMap,
                        std::unordered_map<std::string, std::vector<uint8_t>>& applicationBufferMap);

/**
 * @brief Load data files to UserBufferList
 *
 * @param userBufferListHandle Handle of the UserBufferList
 *
 * @param fileList Sorted file pathes. Format: Batches<Map<BufferName,Batch<FilePath>>>
 *
 * @param applicationBufferList Buffer list storing name of tensor along with data.
 *                              Format: Batches<BufferMap<BufferName, BufferData>>
 *
 * @returns boolean value of success or failure
 */
bool
LoadFileToUserBufferList(Snpe_UserBufferList_Handle_t userBufferListHandle,
                         std::vector<std::unordered_map<std::string, std::vector<std::string>>>& fileList,
                         std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>& applicationBufferList);

#endif //PROCESS_USER_BUFFER_H