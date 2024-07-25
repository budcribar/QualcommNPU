//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cstring>

#include "PreprocessInput.hpp"


std::vector<std::vector<std::string>> PreprocessInput(const char* filePath, size_t batchSize) {
    // Read lines from the input lists file
    // and store the paths to inputs in strings
    std::ifstream inputList(filePath);
    std::string fileLine;
    std::vector<std::string> lines;
    while (std::getline(inputList, fileLine)) {
        if (!fileLine.empty() && fileLine.back() == '\r') fileLine.pop_back();
        if (fileLine.empty())
            continue;
        lines.push_back(fileLine);
    }
    // Store batches of inputs into vectors of strings
    std::vector<std::vector<std::string>> result;
    std::vector<std::string> batch;
    for(size_t i=0; i<lines.size(); i++) {
        if(batch.size()==batchSize) {
            result.push_back(batch);
            batch.clear();
        }
        batch.push_back(lines[i]);
    }
    result.push_back(batch);
    return result;
}