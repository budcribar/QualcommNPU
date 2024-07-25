//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef PROCESS_DATA_TYPE_H
#define PROCESS_DATA_TYPE_H

#include <vector>
#include <string>

#include "DlSystem/IUserBuffer.h"


Snpe_UserBufferEncoding_ElementType_t
StrToDataType(std::string& dataTypeStr);

std::string
DataTypeToStr(Snpe_UserBufferEncoding_ElementType_t dataType);

void TfNToFloat(float *out,
                uint8_t *in,
                uint64_t stepEquivalentTo0,
                float quantizedStepSize,
                size_t numElement,
                uint8_t bitWidth);

bool FloatToTfN(uint8_t* out,
                uint64_t& stepEquivalentTo0,
                float& quantizedStepSize,
                float* in,
                size_t numElement,
                uint8_t bitWidth);

/**
 * @brief Data Convert between to Native type (T1 --> T2)
 *
 * @param out Output buffer of type T2
 *
 * @param in Input buffer of type T1
 *
 * @param numElement The number of elements need to be converted
 *
 * @returns boolean value of success or failure
 */
template <typename T1, typename T2>
void NativeToNative(T2* out, const T1* in, size_t numElement)
{
    for (size_t i = 0; i < numElement; ++i) {
        out[i] = static_cast<T1>(in[i]);
    }
}


#endif //PROCESS_DATA_TYPE_H