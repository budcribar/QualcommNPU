//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>

std::string GetVersion();

size_t CalcSize(const std::vector<size_t>& dims, size_t elementSize);

std::vector<size_t> CalcStrides(const std::vector<size_t>& dims, size_t elementSize);

bool ReadRawData(const std::string& path, char* data, size_t length);

bool SaveRawData(const std::string& path, const char* data, size_t length);


bool EnsureDirectory(const std::string& dir);

std::string ArrayToStr(const std::vector<size_t>& array);


/**
 * @brief .
 *
 * Split a string line to string lit by separator.
 *
 * @param[in] str input string
 *
 * @param[in] separator  string separator for spliting
 *
 * @returns if find separator return splited strings, otherwise return empty vector
 */
std::vector<std::string>
Split(const std::string& str, const std::string& separator);

#endif //UTIL_H