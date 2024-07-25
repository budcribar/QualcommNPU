//=============================================================================
//
//  Copyright (c) 2020,2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "CPU/QnnCpuOpPackage.h"
#include "OpFactory.hpp"
#include "QnnTypes.h"
#include "ops/QnnCpuOpPkgOpBase.hpp"

using opPkgOpHandle = std::size_t;

class QnnCpuOpPkg {
 public:
  static std::shared_ptr<QnnCpuOpPkg> getInstance();
  static void destroyInstance();
  Qnn_ErrorHandle_t setPackageInfo(const char* packageName,
                                   std::vector<std::string> operations,
                                   uint32_t numOperations);

  Qnn_ErrorHandle_t getPackageInfo(const QnnOpPackage_Info_t** info);

  std::shared_ptr<QnnCpuOpPkgOpBase> getObject(opPkgOpHandle handle) {
    std::shared_ptr<QnnCpuOpPkgOpBase> op;
    std::lock_guard<std::mutex> locker(s_mtx);

    if (m_opMap.find(handle) == m_opMap.end()) {
      return op;
    }

    op = m_opMap.find(handle)->second;

    return op;
  }

  Qnn_ErrorHandle_t removeObject(opPkgOpHandle handle) {
    std::lock_guard<std::mutex> locker(s_mtx);
    if (m_opMap.find(handle) != m_opMap.end()) {
      m_opMap.erase(handle);
    }
    return QNN_SUCCESS;
  }

  opPkgOpHandle getHandle(std::shared_ptr<QnnCpuOpPkgOpBase> op) {
    std::lock_guard<std::mutex> locker(s_mtx);
    opPkgOpHandle handle = opPkgOpHandle(op.get());
    m_opMap[handle]      = op;
    return handle;
  }

  Qnn_ErrorHandle_t createOpImpl(QnnCpuOpPackage_GraphInfrastructure_t* graphInfrastructure,
                                 QnnCpuOpPackage_Node_t* node,
                                 QnnCpuOpPackage_OpImpl_t** opImplPtr);

  Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
    (void)opConfig;

    return QNN_SUCCESS;
  }

  Qnn_ErrorHandle_t executeNode(void* kernelHandle);

  Qnn_ErrorHandle_t freeOpImpl(QnnCpuOpPackage_OpImpl_t* opImpl);

  static bool getIsInitialized();

  static void setIsInitialized(bool isInitialized);

  ~QnnCpuOpPkg() {
    if (m_opsList) {
      for (uint32_t i = 0; i < m_numOps; ++i) {
        free(static_cast<void*>(const_cast<char*>(m_opsList[i])));
        m_opsList[i] = nullptr;
      }
      free(static_cast<void*>(m_opsList));
      m_opsList = nullptr;
    }
  }

 private:
  QnnCpuOpPkg() : m_opsList(nullptr){};
  std::string m_packageName;
  QnnOpPackage_Info_t m_packageInfo;
  static std::mutex s_mtx;
  static std::shared_ptr<QnnCpuOpPkg> s_opPackage;
  static bool s_isInitialized;
  std::unordered_map<opPkgOpHandle, std::shared_ptr<QnnCpuOpPkgOpBase>> m_opMap;
  std::list<std::shared_ptr<QnnCpuOpPackage_OpImpl_t>> m_OpImplList;
  OpFactory m_opFactory;
  const char** m_opsList;
  uint32_t m_numOps;
  Qnn_ApiVersion_t m_sdkApiVersion;
};