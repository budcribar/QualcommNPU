//==============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <string>

#include "OpFactory.hpp"
#include "ops/QnnCpuOpPkgOpBase.hpp"
#include "ops/QnnCpuOpPkgRelu.hpp"

Qnn_ErrorHandle_t OpFactory::getOp(QnnCpuOpPackage_Node_t* node,
                                   std::shared_ptr<QnnCpuOpPkgOpBase>& op) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;
  std::string nodeType(node->typeName);

  if (!nodeType.compare("Relu")) {
    status = construct<QnnCpuOpPkgRelu>(node, op);
  }

  if (status != QNN_SUCCESS) {
    return status;
  }

  // Return value
  return status;
}

Qnn_ErrorHandle_t OpFactory::isOpValid(QnnCpuOpPackage_Node_t* node) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;
  std::string nodeType(node->typeName);
  if (!nodeType.compare("Relu")) {
    status = QNN_SUCCESS;
  } else {
    status = QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }

  return status;
}