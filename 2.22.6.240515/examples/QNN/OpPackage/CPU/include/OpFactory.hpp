//==============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>

#include "CPU/QnnCpuOpPackage.h"
#include "QnnTypes.h"
#include "ops/QnnCpuOpPkgOpBase.hpp"

class OpFactory {
 public:
  OpFactory() {}

  Qnn_ErrorHandle_t getOp(QnnCpuOpPackage_Node_t* node, std::shared_ptr<QnnCpuOpPkgOpBase>& op);

  Qnn_ErrorHandle_t isOpValid(QnnCpuOpPackage_Node_t* node);

  template <typename T>
  Qnn_ErrorHandle_t construct(QnnCpuOpPackage_Node_t* node,
                              std::shared_ptr<QnnCpuOpPkgOpBase>& op) {
    Qnn_ErrorHandle_t status = QNN_SUCCESS;

    // Create object of op package node
    op = std::make_shared<T>(node);

    // set op node
    status = op->setOpNode(node);
    if (status != QNN_SUCCESS) {
      return status;
    }

    return status;
  }

 private:
};