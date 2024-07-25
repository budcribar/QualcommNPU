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


/**
 * @brief .
 *
 * @param userBufferMap handle for Input/Output Buffer
 *
 * @param applicationBuffers Map storing name of input/output tensor along with data
 *
 * @param snpeUserBackendBufferHandles vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle snpe Handle
 *
 * @param name name of the tensor
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth bit Width
 *
 * @return Successfully created or not
 *
 */
bool CreateUserBuffer(Snpe_UserBufferMap_Handle_t userBufferMap,
                      std::unordered_map<std::string, std::vector<uint8_t >>& applicationBuffers,
                      std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                      Snpe_SNPE_Handle_t snpeHandle,
                      const char* name,
                      const bool isTfNBuffer,
                      bool staticQuantization,
                      int bitWidth);

/**
 * @brief .
 *
 * @param inputMapHandle handle for Input user Buffer
 *
 * @param applicationBuffers Map storing name of input/output tensor along with data
 *
 * @param snpeUserBackendBufferHandles vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle snpe Handle
 *
 * @param isTfNBuffer  Flag to know whether TfN mode is enabled or not
 *
 * @param staticQuantization Flag to know whether static Quantization to be used or not
 *
 * @param bitWidth bit Width
 *
 * @return Successfully created or not
 */
bool CreateInputBufferMap(Snpe_UserBufferMap_Handle_t inputMapHandle,
                          std::unordered_map<std::string, std::vector<uint8_t >>& applicationBuffers,
                          std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                          Snpe_SNPE_Handle_t snpeHandle,
                          bool isTfNBuffer,
                          bool staticQuantization,
                          int bitWidth);

/**
 * @brief .
 *
 * @param outputMapHandle handle for Output user Buffer
 *
 * @param applicationBuffers Map storing name of input/output tensor along with data
 *
 * @param snpeUserBackendBufferHandles vector of Snpe_IUserBuffer_Handle_t for input/output buffer
 *
 * @param snpeHandle snpe Handle
 *
 * @param isTfNBuffer Flag to know whether TfN mode is enabled or not
 *
 * @param bitWidth bit Width
 *
 * @return Successfully created or not
 *
 */
bool CreateOutputBufferMap(Snpe_UserBufferMap_Handle_t outputMapHandle,
                           std::unordered_map<std::string, std::vector<uint8_t >>& applicationBuffers,
                           std::vector<Snpe_IUserBuffer_Handle_t>& snpeUserBackendBufferHandles,
                           Snpe_SNPE_Handle_t snpeHandle,
                           bool isTfNBuffer,
                           int bitWidth);