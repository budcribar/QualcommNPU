//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef PROCESS_INPUT_LIST_H
#define PROCESS_INPUT_LIST_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

/**
 * @brief
 *
 * Process the input list according to the batchSize of the network.
 *
 * @param[in] inputListPath path to input list
 *
 * @param[in] batchSize  Size of batch
 *
 * @param[in] requredInputNames  input names model required
 *
 * @param[out] inputFileNumber  Numbers of loaded input file
 *
 * @returns batches<inputMap<inputName, batch<inputPath>>>
 */
std::vector<std::unordered_map<std::string, std::vector<std::string>>>
ProcessInputList(const std::string& inputListPath,
                 size_t batchSize,
                 const std::set<std::string>& requredInputNames,
                 size_t& inputFileNumber);


/**
 * @brief
 *
 * Parse the one input line to a map combined with inputNames and inputPathes
 *
 * @param[in] line input line read from input list.
 * Exp: "inputName1:=inputPath1 inputName2:=inputPath2..."
 *
 * @param[in] requredInputNames input names model required
 *
 * @returns inputMap<inputName, inputPath>
 *
 */
std::unordered_map<std::string, std::string>
ParseInputLine(const std::string& line, const std::set<std::string>& requredInputNames);
#endif //PROCESS_INPUT_LIST_H
