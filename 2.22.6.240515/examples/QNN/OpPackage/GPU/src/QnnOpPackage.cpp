//=============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <memory>
#include <mutex>

#include "GPU/QnnGpuOpPackage.h"
#include "OpPackage.hpp"
#include "QnnOpDef.h"

static std::unique_ptr<OpPackage> sg_opPackage;
static std::mutex sg_mtx;
QnnLog_Callback_t g_callback;
QnnLog_Level_t g_maxLogLevel;

void log(QnnLog_Level_t level, const char* fmt, ...) {
  if (!g_callback) {
    return;
  }
  if (level > g_maxLogLevel) {
    return;
  }
  va_list argp;
  va_start(argp, fmt);
  (*g_callback)(fmt, level, 0, argp);
  va_end(argp);
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_initialize(
    QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {
  std::lock_guard<std::mutex> locker(sg_mtx);

  if (sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  log(QNN_LOG_LEVEL_INFO, "QnnOpPackage_initialize");

  if (!globalInfrastructure) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  auto opPkg = OpPackage::create("examples.OpPackage", globalInfrastructure->deviceProperties);
  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  sg_opPackage = std::move(opPkg);
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_getInfo(
    const QnnOpPackage_Info_t** info) {
  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }
  log(QNN_LOG_LEVEL_INFO, "QnnOpPackage_getInfo");
  return sg_opPackage->getPackageInfo(info);
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_validateOpConfig(
    Qnn_OpConfig_t opConfig) {
  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }
  log(QNN_LOG_LEVEL_INFO, "QnnOpPackage_validateOpConfig");
  return sg_opPackage->operationExists(QNN_GPU_OP_CFG_GET_TYPE_NAME(opConfig));
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_createOpImpl(
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* operation) {
  if (!graphInfrastructure || !node || !operation) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }

  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  log(QNN_LOG_LEVEL_INFO, "QnnOpPackage_createKernels");

  return sg_opPackage->createOperation(graphInfrastructure, node, operation);
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_freeOpImpl(
    QnnOpPackage_OpImpl_t operation) {
  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }
  return sg_opPackage->freeOperation(operation);
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_terminate() {
  sg_opPackage.reset();
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_logInitialize(
    QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
  g_callback    = callback;
  g_maxLogLevel = maxLogLevel;
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_logSetLevel(
    QnnLog_Level_t maxLogLevel) {
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t QnnOpPackage_logTerminate(void) {
  return QNN_SUCCESS;
}

extern "C" QNN_API Qnn_ErrorHandle_t
QnnOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init              = QnnOpPackage_initialize;
  interface->v1_4.terminate         = QnnOpPackage_terminate;
  interface->v1_4.getInfo           = QnnOpPackage_getInfo;
  interface->v1_4.validateOpConfig  = QnnOpPackage_validateOpConfig;
  interface->v1_4.createOpImpl      = QnnOpPackage_createOpImpl;
  interface->v1_4.freeOpImpl        = QnnOpPackage_freeOpImpl;
  interface->v1_4.logInitialize     = QnnOpPackage_logInitialize;
  interface->v1_4.logSetLevel       = QnnOpPackage_logSetLevel;
  interface->v1_4.logTerminate      = QnnOpPackage_logTerminate;
  return QNN_SUCCESS;
}

extern "C" QNN_API Qnn_ErrorHandle_t QnnGpuOpPackage_getKernelBinary(const char* name,
                                                                     const uint8_t** binary,
                                                                     uint32_t* numBytes) {
  (void)name;
  (void)binary;
  (void)numBytes;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}
