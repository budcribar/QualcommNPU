//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <GPU/QnnGpuOpPackage.h>

#include <memory>

#include "GpuCustomOpPackage.hpp"
#include "Operation.hpp"
#include "QnnSdkBuildId.h"

std::unique_ptr<OpPackage> OpPackage::create(const std::string& packageName,
                                             const QnnGpu_DeviceProperties_t* deviceProperties) {
  // TODO [AISW-45591]: Add logging
  auto opPkg = new (std::nothrow) OpPackage(packageName, deviceProperties);
  if (!opPkg) {
    return nullptr;
  }
  return std::unique_ptr<OpPackage>(std::move(opPkg));
}

OpPackage::OpPackage(const std::string& packageName,
                     const QnnGpu_DeviceProperties_t* deviceProperties)
    : m_packageName(packageName), m_deviceProperties(deviceProperties) {
  m_sdkApiVersion = QNN_GPU_API_VERSION_INIT;

  m_packageInfo = QNN_OP_PACKAGE_INFO_INIT;

  m_packageInfo.packageName    = m_packageName.c_str();
  m_packageInfo.operationNames = m_opNames.data();
  m_packageInfo.numOperations  = (uint32_t)m_opNames.size();
  m_packageInfo.sdkBuildId     = QNN_SDK_BUILD_ID;
  m_packageInfo.sdkApiVersion  = &m_sdkApiVersion;
}

Qnn_ErrorHandle_t OpPackage::registerOperation(const std::string& opName,
                                               OpCreateFunc_t opCreateFunction) {
  m_opBuilders[opName] = opCreateFunction;
  m_opNames.push_back(opName.c_str());
  m_packageInfo.operationNames = m_opNames.data();
  m_packageInfo.numOperations  = (uint32_t)m_opNames.size();
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t OpPackage::getPackageInfo(const QnnOpPackage_Info_t** info) {
  *info = &m_packageInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t OpPackage::operationExists(std::string opName) {
  return (m_opBuilders.find(opName) == m_opBuilders.end())
             ? (QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE)
             : (QNN_SUCCESS);
}

Qnn_ErrorHandle_t OpPackage::createOperation(
    QnnGpuOpPackage_GraphInfrastructure_t* graphInfrastructure,
    const QnnGpuOpPackage_Node_t* node,
    QnnGpu_Operation_t** operation) {
  (void)graphInfrastructure;
  auto it = m_opBuilders.find(std::string(node->configs[0]->v1.typeName));
  if (it == m_opBuilders.end()) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }

  Qnn_ErrorHandle_t status;
  auto gpuOperation = (it->second)(node, &status);
  if (!gpuOperation) {
    return QNN_OP_PACKAGE_ERROR_GENERAL;
  } else if (status != QNN_SUCCESS) {
    return status;
  }

  auto tmpOperation                  = gpuOperation->getOperationInfo();
  m_compiledOperations[tmpOperation] = gpuOperation;
  *operation                         = tmpOperation;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t OpPackage::freeOperation(QnnGpu_Operation_t* operation) {
  auto pos = m_compiledOperations.find(operation);
  if (pos == m_compiledOperations.end()) {
    return QNN_OP_PACKAGE_ERROR_GENERAL;
  }
  m_compiledOperations.erase(pos);
  return QNN_SUCCESS;
}
