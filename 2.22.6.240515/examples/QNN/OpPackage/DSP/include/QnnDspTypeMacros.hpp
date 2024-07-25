//==============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QNN_DSP_UTIL_HPP
#define QNN_DSP_UTIL_HPP

#include "QnnTypes.h"

// Below are macros duplicated from Core to support OpPackage Example
inline const char* getQnnDspOpConfigName(const Qnn_OpConfig_t& opConfig) {
  return opConfig.v1.name;
}

inline const char* getQnnDspOpConfigPackageName(const Qnn_OpConfig_t& opConfig) {
  return opConfig.v1.packageName;
}

inline const char* getQnnDspOpConfigTypeName(const Qnn_OpConfig_t& opConfig) {
  return opConfig.v1.typeName;
}

inline uint32_t getQnnDspOpConfigNumParams(const Qnn_OpConfig_t& opConfig) {
  return opConfig.v1.numOfParams;
}

inline uint32_t getQnnDspOpConfigNumInputs(const Qnn_OpConfig_t& opConfig) {
  return opConfig.v1.numOfInputs;
}

inline uint32_t getQnnDspOpConfigNumOutputs(const Qnn_OpConfig_t& opConfig) {
  return opConfig.v1.numOfOutputs;
}

// Accessors for QNN Op Config
#define QNN_DSP_OP_CFG_GET_NAME(opConfig)         getQnnDspOpConfigName(opConfig)
#define QNN_DSP_OP_CFG_GET_PACKAGE_NAME(opConfig) getQnnDspOpConfigPackageName(opConfig)
#define QNN_DSP_OP_CFG_GET_TYPE_NAME(opConfig)    getQnnDspOpConfigTypeName(opConfig)
#define QNN_DSP_OP_CFG_GET_NUM_PARAMS(opConfig)   getQnnDspOpConfigNumParams(opConfig)
#define QNN_DSP_OP_CFG_GET_NUM_INPUTS(opConfig)   getQnnDspOpConfigNumInputs(opConfig)
#define QNN_DSP_OP_CFG_GET_NUM_OUTPUTS(opConfig)  getQnnDspOpConfigNumOutputs(opConfig)

#endif  // QNN_DSP_UTIL_HPP
