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
#include <sstream>

//#include "DlSystem/ITensorFactory.h"
#include "DlSystem/TensorShape.h"

template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
  result.clear();
  std::istringstream ss( s );
  while (!ss.eof())
  {
    typename Container::value_type field;
    getline( ss, field, delimiter );
    if (!field.empty() && field.back() == '\r') field.pop_back();
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}


std::vector<float> loadFloatDataFile(const std::string& inputFile);
std::vector<unsigned char> loadByteDataFile(const std::string& inputFile);
template<typename T> bool loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector);

std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile);
template<typename T> bool loadByteDataFileBatched(const std::string& inputFile, std::vector<T>& loadVector, size_t offset);
bool loadByteDataFileBatchedTf8(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset);
bool loadByteDataFileBatchedTfN(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset,
                                unsigned char& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, int bitWidth);

bool SaveITensor(const std::string& path, float* data, size_t tensorStart=0, size_t tensorEnd=0);
bool SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex=0, size_t batchChunk=0);
bool EnsureDirectory(const std::string& dir);

void TfNToFloat(float *out, uint8_t *in, const unsigned char stepEquivalentTo0, const float quantizedStepSize, size_t numElement, int bitWidth);
bool FloatToTfN(uint8_t* out, unsigned char& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, float* in, size_t numElement, int bitWidth);

void setResizableDim(size_t resizableDim);
size_t getResizableDim();

#endif

