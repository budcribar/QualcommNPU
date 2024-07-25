//=============================================================================
//
//  Copyright (c) 2020,2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "CPU/QnnCpuOpPackage.h"
#include "QnnCpuOpPkg.hpp"

static Qnn_ErrorHandle_t QnnOpPackage_initialize(
    QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {
  if (QnnCpuOpPkg::getIsInitialized()) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  auto opPkg = QnnCpuOpPkg::getInstance();

  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  opPkg->setPackageInfo("examples.OpPackage", {"Relu"}, 1);

  QnnCpuOpPkg::setIsInitialized(true);

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t QnnOpPackage_getInfo(const QnnOpPackage_Info_t** info) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;

  auto opPkg = QnnCpuOpPkg::getInstance();

  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  status = opPkg->getPackageInfo(info);
  if (status != QNN_SUCCESS) {
    return status;
  }

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t QnnOpPackage_validateOpConfig(Qnn_OpConfig_t opConfig) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;

  auto opPkg = QnnCpuOpPkg::getInstance();

  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  status = opPkg->validateOpConfig(opConfig);
  if (status != QNN_SUCCESS) {
    return status;
  }

  return status;
}

static Qnn_ErrorHandle_t QnnOpPackage_createOpImpl(
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* opImplPtr) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;

  auto opPkg = QnnCpuOpPkg::getInstance();

  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  auto opNode = reinterpret_cast<QnnCpuOpPackage_Node_t*>(node);
  auto ops    = reinterpret_cast<QnnCpuOpPackage_OpImpl_t**>(opImplPtr);

  status = opPkg->createOpImpl(graphInfrastructure, opNode, ops);
  if (status != QNN_SUCCESS) {
    return status;
  }

  return status;
}

static Qnn_ErrorHandle_t QnnOpPackage_freeOpImpl(QnnOpPackage_OpImpl_t opImpl) {
  Qnn_ErrorHandle_t status = QNN_SUCCESS;

  auto opPkg = QnnCpuOpPkg::getInstance();

  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  auto ops = reinterpret_cast<QnnCpuOpPackage_OpImpl_t*>(opImpl);

  status = opPkg->freeOpImpl(ops);
  if (status != QNN_SUCCESS) {
    return status;
  }

  return status;
}

static Qnn_ErrorHandle_t QnnOpPackage_terminate() {
  QnnCpuOpPkg::destroyInstance();

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t QnnOpPackage_logInitialize(QnnLog_Callback_t callback,
                                                    QnnLog_Level_t maxLogLevel) {
  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t QnnOpPackage_logSetLevel(QnnLog_Level_t maxLogLevel) {
  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t QnnOpPackage_logTerminate() { return QNN_SUCCESS; }

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
