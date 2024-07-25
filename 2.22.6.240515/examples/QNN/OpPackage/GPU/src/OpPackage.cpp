//=============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <memory>

#include "Common.hpp"
#include "GPU/QnnGpuOpPackage.h"
#include "OpPackage.hpp"
#include "QnnSdkBuildId.h"
#include "ReluOperation.hpp"

std::unique_ptr<OpPackage> OpPackage::create(const std::string& packageName,
                                             const QnnGpu_DeviceProperties_t* deviceProperties) {
  log(QNN_LOG_LEVEL_VERBOSE, "Creating operation package: %s", packageName.c_str());
  auto opPkg = new (std::nothrow) OpPackage(packageName, deviceProperties);
  if (!opPkg) {
    log(QNN_LOG_LEVEL_ERROR, "Error creating operation package");
    return nullptr;
  }
  return std::unique_ptr<OpPackage>(std::move(opPkg));
}

OpPackage::OpPackage(const std::string& packageName,
                     const QnnGpu_DeviceProperties_t* deviceProperties)
    : m_packageName(packageName), m_deviceProperties(deviceProperties) {
  m_opBuilders[ReluOperation::s_operationType] = ReluOperation::create;
  m_opNames.push_back(ReluOperation::s_operationType.c_str());

  m_sdkApiVersion = QNN_GPU_API_VERSION_INIT;
  m_packageInfo   = {m_packageName.c_str(),       // packageName
                   m_opNames.data(),            // operationNames
                   nullptr,                     // operationInfo
                   (uint32_t)m_opNames.size(),  // numOperations
                   nullptr,                     // optimizations
                   0,                           // numOptimizations
                   QNN_SDK_BUILD_ID,            // sdkBuildId
                   &m_sdkApiVersion,            // sdkApiVersion
                   nullptr,                     // packageInfo
                   {0}};                        // reserved
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
  auto it = m_opBuilders.find(std::string(QNN_GPU_OP_CFG_GET_TYPE_NAME(*(node->configs[0]))));
  if (it == m_opBuilders.end()) {
    log(QNN_LOG_LEVEL_ERROR, "Operation does not exist");
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }

  Qnn_ErrorHandle_t status;
  auto gpuOperation = (it->second)(node, &status);
  if (!gpuOperation || (status != QNN_SUCCESS)) {
    log(QNN_LOG_LEVEL_ERROR, "Error creating operation");
    return QNN_OP_PACKAGE_ERROR_GENERAL;
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
