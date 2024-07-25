//=============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#include <cstring>
#include <unordered_map>

#include "CPU/QnnCpuOpPackage.h"
#include "OpFactory.hpp"
#include "QnnCpuOpPkg.hpp"
#include "QnnSdkBuildId.h"

#ifdef _MSC_VER
// Disable warning 4996 (deprecated function).
#pragma warning(disable : 4996)
#endif

static Qnn_ErrorHandle_t QnnOpPackage_execute(void* opPkgNodeData) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;
  auto opPkg               = QnnCpuOpPkg::getInstance();
  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_GENERAL;
  }

  status = opPkg->executeNode(opPkgNodeData);
  if (status != QNN_SUCCESS) {
    return status;
  }

  return status;
}

std::mutex QnnCpuOpPkg::s_mtx;
std::shared_ptr<QnnCpuOpPkg> QnnCpuOpPkg::s_opPackage;
bool QnnCpuOpPkg::s_isInitialized;

bool QnnCpuOpPkg::getIsInitialized() {
  std::lock_guard<std::mutex> locker(s_mtx);
  return s_isInitialized;
}

void QnnCpuOpPkg::destroyInstance() {
  setIsInitialized(false);
  s_opPackage.reset();
}

void QnnCpuOpPkg::setIsInitialized(bool isInitialized) {
  std::lock_guard<std::mutex> locker(s_mtx);
  s_isInitialized = isInitialized;
}

std::shared_ptr<QnnCpuOpPkg> QnnCpuOpPkg::getInstance() {
  std::lock_guard<std::mutex> locker(s_mtx);
  if (!s_opPackage) {
    s_opPackage.reset(new (std::nothrow) QnnCpuOpPkg());
  }
  return s_opPackage;
}

Qnn_ErrorHandle_t QnnCpuOpPkg::setPackageInfo(const char* packageName,
                                              std::vector<std::string> operations,
                                              uint32_t numOperations) {
  m_packageName.assign(packageName);
  m_numOps  = numOperations;
  m_opsList = static_cast<const char**>(malloc(numOperations * sizeof(char*)));
  for (uint32_t i = 0; i < m_numOps; ++i) {
    m_opsList[i] = static_cast<const char*>(malloc((operations[i].length() + 1) * sizeof(char)));
    std::strcpy(const_cast<char*>(m_opsList[i]), operations[i].c_str());
  }
  m_sdkApiVersion = QNN_CPU_API_VERSION_INIT;

  m_packageInfo                = QNN_OP_PACKAGE_INFO_INIT;
  m_packageInfo.packageName    = m_packageName.c_str();
  m_packageInfo.operationNames = m_opsList;
  m_packageInfo.numOperations  = m_numOps;
  m_packageInfo.sdkBuildId     = QNN_SDK_BUILD_ID;
  m_packageInfo.sdkApiVersion  = &m_sdkApiVersion;

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnCpuOpPkg::getPackageInfo(const QnnOpPackage_Info_t** info) {
  *info = &m_packageInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnCpuOpPkg::createOpImpl(
    QnnCpuOpPackage_GraphInfrastructure_t* graphInfrastructure,
    QnnCpuOpPackage_Node_t* node,
    QnnCpuOpPackage_OpImpl_t** opImplPtr) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;
  std::shared_ptr<QnnCpuOpPkgOpBase> op;

  // Get op from op factory
  status = m_opFactory.getOp(node, op);
  if (status != QNN_SUCCESS) {
    return status;
  }

  // Finalize
  status = op->finalize();
  if (status != QNN_SUCCESS) {
    return status;
  }

  // Update op reference
  auto opImpl      = std::make_shared<QnnCpuOpPackage_OpImpl_t>();
  opImpl->opImplFn = QnnOpPackage_execute;
  opImpl->userData = (void*)getHandle(op);

  // update out opImpl param
  *opImplPtr = opImpl.get();

  // update opImpl list
  m_OpImplList.emplace_back(opImpl);

  return status;
}

Qnn_ErrorHandle_t QnnCpuOpPkg::executeNode(void* kernelHandle) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;

  auto op = getObject((opPkgOpHandle)kernelHandle);

  status = op->execute();
  if (status != QNN_SUCCESS) {
    return status;
  }
  return status;
}

Qnn_ErrorHandle_t QnnCpuOpPkg::freeOpImpl(QnnCpuOpPackage_OpImpl_t* opImpl) {
  return (removeObject((opPkgOpHandle)opImpl->userData));
}
