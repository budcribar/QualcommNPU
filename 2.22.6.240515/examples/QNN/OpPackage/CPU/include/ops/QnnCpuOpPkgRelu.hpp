//=============================================================================
//
//  Copyright (c) 2020,2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once
#include <cmath>

#include "CPU/QnnCpuOpPackage.h"
#include "QnnCpuMacro.hpp"
#include "ops/QnnCpuOpPkgOpBase.hpp"

typedef struct {
  float beta;
} QnnCpuReluParams_t;

class QnnCpuOpPkgRelu final : public QnnCpuOpPkgOpBase {
 public:
  QnnCpuOpPkgRelu() {}
  QnnCpuOpPkgRelu(QnnCpuOpPackage_Node_t* node) : QnnCpuOpPkgOpBase(node->name, node->typeName) {}

  Qnn_ErrorHandle_t finalize();

  template <typename T_Ttype>
  void evaluateActivation(const T_Ttype* in, T_Ttype* out, T_Ttype limiter);

  void executeFloat(QnnCpuOpPackage_Tensor_t* in, QnnCpuOpPackage_Tensor_t* out);

  void executeQuantized(QnnCpuOpPackage_Tensor_t* in, QnnCpuOpPackage_Tensor_t* out);

  Qnn_ErrorHandle_t execute();

  Qnn_ErrorHandle_t setOpNode(QnnCpuOpPackage_Node_t* node);
};
