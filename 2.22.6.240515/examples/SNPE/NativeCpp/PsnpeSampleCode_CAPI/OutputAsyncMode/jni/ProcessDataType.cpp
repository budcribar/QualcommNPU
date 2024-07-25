//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <cmath>
#include <limits>

#include "ProcessDataType.hpp"

Snpe_UserBufferEncoding_ElementType_t
StrToDataType(std::string& dataTypeStr) {
    if (dataTypeStr == "float32") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT;
    }
    else if ("uint8") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8;
    }
    else if ("uint16") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16;
    }
    else if ("uint32") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32;
    }
    else if ("int8") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8;
    }
    else if ("int16") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16;
    }
    else if ("int32") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32;
    }
    else if ("bool8") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8;
    }
    else if ("tf8") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8;
    }
    else if ("tf16") {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16;
    }
    else {
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN;
    }
}

std::string
DataTypeToStr(Snpe_UserBufferEncoding_ElementType_t dataType) {
    if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) {
        return "float32";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8) {
        return "uint8";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16) {
        return "uint16";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32) {
        return "uint32";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8) {
        return "int8";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16) {
        return "int16";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32) {
        return "int32";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8) {
        return "bool8";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8) {
        return "tf8";
    }
    else if (dataType == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16) {
        return "tf16";
    }
    else {
        return std::string();
    }
}

void TfNToFloat(float *out,
                uint8_t *in,
                uint64_t stepEquivalentTo0,
                float quantizedStepSize,
                size_t numElement,
                uint8_t bitWidth)
{
    for (size_t i = 0; i < numElement; i++) {
        if (bitWidth == 8) {
            double quantizedValue = static_cast <double> (in[i]);
            double stepEqTo0 = static_cast <double> (stepEquivalentTo0);
            out[i] = static_cast <double> ((quantizedValue - stepEqTo0) * quantizedStepSize);
        }
        else if (bitWidth == 16) {
            uint16_t *temp = (uint16_t *)in;
            double quantizedValue = static_cast <double> (temp[i]);
            double stepEqTo0 = static_cast <double> (stepEquivalentTo0);
            out[i] = static_cast <double> ((quantizedValue - stepEqTo0) * quantizedStepSize);
        }
    }
}

bool FloatToTfN(uint8_t* out,
                uint64_t& stepEquivalentTo0,
                float& quantizedStepSize,
                float* in,
                size_t numElement,
                uint8_t bitWidth)
{
    double encodingMin;
    double encodingMax;
    double encodingRange;
    double trueBitWidthMax = pow(2, bitWidth) -1;

    float trueMin = std::numeric_limits <float>::max();
    float trueMax = std::numeric_limits <float>::min();

    for (size_t i = 0; i < numElement; ++i) {
        trueMin = fmin(trueMin, in[i]);
        trueMax = fmax(trueMax, in[i]);
    }

    double stepCloseTo0;

    if (trueMin > 0.0f) {
        stepCloseTo0 = 0.0;
        encodingMin = 0.0;
        encodingMax = trueMax;
    } else if (trueMax < 0.0f) {
        stepCloseTo0 = trueBitWidthMax;
        encodingMin = trueMin;
        encodingMax = 0.0;
    } else {
        double trueStepSize = static_cast <double>(trueMax - trueMin) / trueBitWidthMax;
        stepCloseTo0 = -trueMin / trueStepSize;
        if (stepCloseTo0 == round(stepCloseTo0)) {
            // 0.0 is exactly representable
            encodingMin = trueMin;
            encodingMax = trueMax;
        } else {
            stepCloseTo0 = round(stepCloseTo0);
            encodingMin = (0.0 - stepCloseTo0) * trueStepSize;
            encodingMax = (trueBitWidthMax - stepCloseTo0) * trueStepSize;
        }
    }

    const double minEncodingRange = 0.01;
    encodingRange = encodingMax - encodingMin;
    quantizedStepSize = encodingRange / trueBitWidthMax;
    stepEquivalentTo0 = static_cast <uint64_t> (round(stepCloseTo0));

    if (encodingRange < minEncodingRange) {
        std::cerr << "Expect the encoding range to be larger than " << minEncodingRange << "\n"
                  << "Got: " << encodingRange << "\n";
        return false;
    }

    for (size_t i = 0; i < numElement; ++i) {
        int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);

        if (quantizedValue < 0)
            quantizedValue = 0;
        else if (quantizedValue > (int)trueBitWidthMax)
            quantizedValue = (int)trueBitWidthMax;

        if (bitWidth == 8) {
            out[i] = static_cast <uint8_t> (quantizedValue);
        }
        else if (bitWidth == 16) {
            uint16_t *temp = (uint16_t *)out;
            temp[i] = static_cast <uint16_t> (quantizedValue);
        }
    }
    return true;
}
