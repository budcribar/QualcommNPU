//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <sys/stat.h>
#include <cerrno>
#include <limits>

#include "Util.hpp"
#include "ProcessDataType.hpp"

#include "SNPE/SNPEUtil.h"
#include "DlSystem/DlVersion.h"

std::string GetVersion()
{
    Snpe_DlVersion_Handle_t versionHandle = Snpe_Util_GetLibraryVersion();
    return std::string(Snpe_DlVersion_ToString(versionHandle));
}

size_t CalcSize(const std::vector<size_t>& dims, size_t elementSize)
{
    if (dims.empty()) {
        return 0;
    }
    size_t size = elementSize;
    for (size_t dim : dims) {
        size *= dim;
    }
    return size;
}

std::vector<size_t> CalcStrides(const std::vector<size_t>& dims, size_t elementSize)
{
    std::vector<size_t> strides(dims.size());
    strides[strides.size() - 1] = elementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = dims.size() - 1; i > 0; i--) {
        stride *= dims[i];
        strides[i-1] = stride;
    }
    return strides;
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

bool ReadRawData(const std::string& path, char* data, size_t length)
{
    std::ifstream in(path, std::ifstream::binary);
    if (!in.is_open() || !in.good()) {
        std::cerr << "Error: Failed to open input file: " << path
                  << "errno: " << std::strerror(errno) << std::endl;
        return false;
    }

    in.seekg(0, in.end);
    size_t fileDataSize = in.tellg();
    in.seekg(0, in.beg);
    std::fstream f;

    if (fileDataSize != length) {
        std::cerr << "Error: file data size(" << fileDataSize << ") != request reading length("
                  << length << "). file path: " << path << std::endl;
        return false;
    }

    if (!in.read(data, length)) {
        std::cerr << "Error Failed to read the contents of: " << path
                  << "errno: " << std::strerror(errno) << std::endl;
        return false;
    }
    in.close();
    return true;
}

bool SaveRawData(const std::string& path, const char* data, size_t length)
{
    // Create the directory path if it does not exist
    auto idx = path.find_last_of('/');
    if (idx != std::string::npos) {
        std::string dir = path.substr(0, idx);
        if (!EnsureDirectory(dir)) {
            std::cerr << "Error: Failed to create output directory: " << dir
                      << ", errno: " << std::strerror(errno) << std::endl;;
            return false;
        }
    }

    std::ofstream os(path, std::ofstream::binary);
    if (!os) {
        std::cerr << "Error: Failed to open output file for writing: " << path
                  << ", errno: " << std::strerror(errno) << std::endl;
        return false;
    }

    if (!os.write(data, length)) {
        std::cerr << "Error: Failed to write data to: " << path
                  << ", errno: "<< std::strerror(errno) << std::endl;
        return false;
    }
    os.close();
    return true;
}

std::string ArrayToStr(const std::vector<size_t>& array)
{
    std::string str = "[";
    for (size_t i = 0; i < array.size(); ++i) {
        str += std::to_string(array[i]);
        str += i == array.size() - 1 ? "" : ", ";
    }
    str += "]";
    return str;
}


std::vector<std::string>
Split(const std::string& str, const std::string& separator)
{
    size_t pos = str.find(separator);
    if (pos == str.npos) {
        return std::vector<std::string>();
    }
    std::vector<std::string> strList;
    bool separatorInStrEnd = str.compare(str.size() - separator.size(), separator.size(), separator) == 0;
    std::string str2 = separatorInStrEnd ? str : str + separator;
    while (pos != str2.npos) {
        std::string sub = str2.substr(0, pos);
        strList.push_back(sub);
        str2 = str2.substr(pos + separator.size(), str2.size());
        pos = str2.find(separator);
    }
    return strList;
}