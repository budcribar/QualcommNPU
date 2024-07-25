//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <fstream>

#include "ProcessInputList.hpp"
#include "Util.hpp"


std::vector<std::unordered_map<std::string, std::vector<std::string>>>
ProcessInputList(const std::string& inputListPath,
                 size_t batchSize,
                 const std::set<std::string>& requredInputNames,
                 size_t& inputFileNumber) {
    // Read lines from the input lists file
    // and store the paths to inputs in strings
    std::ifstream inputList(inputListPath);
    std::string line;
    std::vector<std::unordered_map<std::string, std::string>> inputMapList;
    while (std::getline(inputList, line)) {
        if (line.empty())
            continue;
        std::unordered_map<std::string, std::string> inputMap = ParseInputLine(line, requredInputNames);
        if (inputMap.empty()) {
            std::cerr << "Error: Parse line of input list fail." << std::endl;
            return std::vector<std::unordered_map<std::string, std::vector<std::string>>>();
        }
        inputMapList.push_back(inputMap);
    }
    // Store batches of inputs into vectors
    std::vector<std::unordered_map<std::string, std::vector<std::string>>> batches;
    bool creatNewBatch = true;
    size_t batchIdx = 0;
    for(size_t i = 0; i < inputMapList.size(); i++) {
        if (creatNewBatch) {
            batches.resize(batches.size() + 1);
            batchIdx = batches.size() - 1;
            creatNewBatch = false;
        }
        for (auto pair : inputMapList[i]) {
            std::string name = pair.first;
            std::string path = pair.second;
            if (batches[batchIdx].find(name) == batches[batchIdx].end()) {
                batches[batchIdx][name] = std::vector<std::string>();
            }
            batches[batchIdx][name].push_back(path);
            if (batches[batchIdx][name].size() == batchSize) {
                creatNewBatch = true;
            }
        }
    }
    inputFileNumber = inputMapList.size();
    return batches;
}

std::unordered_map<std::string, std::string>
ParseInputLine(const std::string& line, const std::set<std::string>& requredInputNames) {
    std::unordered_map<std::string, std::string> inputMap;
    std::string separator1 = " ";
    std::string separator2 = ":=";

    std::vector<std::string> allInputsInfo = Split(line, separator1);
    if (allInputsInfo.empty()) {
        allInputsInfo.push_back(line);
    }
    if (allInputsInfo.size() != requredInputNames.size()) {
        std::cerr << "Error: Numbers of input number are not match with model required, expected: "
                  << requredInputNames.size() <<" Got: "<< allInputsInfo.size() << std::endl;
        return std::unordered_map<std::string, std::string>();
    }

    std::string name;
    std::string path;
    // Special processing for single input model
    // In this case user does not need to specify input name in inputlist file.
    // Following code will set input name automatically.
    if (requredInputNames.size() == 1) {
        std::vector<std::string> inputPair = Split(allInputsInfo[0], separator2);
        if (inputPair.empty()) {
            name = *requredInputNames.begin();
            path = allInputsInfo[0];
        }
        else if (inputPair.size() == 2) {
            if (inputPair[0] != *requredInputNames.begin()) {
                std::cerr << "Error: Input Name specified is not match with model required, expected: "
                          << *requredInputNames.begin() <<" Got: "<< inputPair[0] << std::endl;
                return std::unordered_map<std::string, std::string>();
            }
            name = inputPair[0];
            path = inputPair[1];
        }
        else {
            std::cerr << "Error: Some thing wrong in input line: \"" << line <<"\"."<<std::endl;
            return std::unordered_map<std::string, std::string>();
        }
        inputMap[name] = path;
        return inputMap;
    }

    // Processing for multi-input model.
    // Every input file path show be specified a correct input name.
    for (std::string inputInfo : allInputsInfo) {
        // split inputName:=inputPath to [inputName, inputPath]
        std::vector<std::string> inputPair = Split(inputInfo, separator2);
        if (inputPair.size() != 2) {
            std::cerr << "Error: Some thing wrong in input line: \"" << line <<"\"."<<std::endl;
            return std::unordered_map<std::string, std::string>();
        }
        name = inputPair[0];
        path = inputPair[1];
        if (requredInputNames.find(name) == requredInputNames.end()) {
            std::cerr << "Error: Input Name \""<< name <<"\" specified is not match with model required." << std::endl;
            return std::unordered_map<std::string, std::string>();
        }
        else if (inputMap.find(name) != inputMap.end()) {
            std::cerr << "Error: Duplicate name \"" << name << "\" in input line: \"" << line <<"\"."<<std::endl;
            return std::unordered_map<std::string, std::string>();
        }
        inputMap[name] = path;
    }
    return inputMap;
}
