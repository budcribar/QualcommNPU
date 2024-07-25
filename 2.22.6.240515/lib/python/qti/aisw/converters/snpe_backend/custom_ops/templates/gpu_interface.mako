<%doc>
//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================</%doc>
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import is_valid_cpp_identifier, log_info %>
<% is_valid_cpp_identifier(package_info.name) %>

//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================

#include <memory>
#include <mutex>

#include "GPU/QnnGpuOpPackage.h"
#include "GpuCustomOpPackage.hpp"
#include "QnnOpDef.h"
#include "Operation.hpp"

static std::unique_ptr<OpPackage> sg_opPackage;
static std::mutex sg_mutex;
QnnLog_Callback_t g_callback;
QnnLog_Level_t g_maxLogLevel;

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_initialize(
    QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {
  std::lock_guard<std::mutex> locker(sg_mutex);

  if (sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  }

  if (!globalInfrastructure) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  auto opPkg = OpPackage::create("${package_info.name}", globalInfrastructure->deviceProperties);
  if (!opPkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  %for operator in package_info.operators:
  opPkg->registerOperation(${operator.type_name}Operation::s_operationType, ${operator.type_name}Operation::create);
  %endfor

  sg_opPackage = std::move(opPkg);
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_getInfo(
    const QnnOpPackage_Info_t** info) {
  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  return sg_opPackage->getPackageInfo(info);
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_validateOpConfig(
    Qnn_OpConfig_t opConfig) {
  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }
  return sg_opPackage->operationExists(opConfig.v1.typeName);
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_createOpImpl(
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* operation) {
  if (!graphInfrastructure || !node || !operation) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }

  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  return sg_opPackage->createOperation(graphInfrastructure, node, operation);
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_freeOpImpl(
    QnnOpPackage_OpImpl_t operation) {
  if (!sg_opPackage) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  return sg_opPackage->freeOperation(operation);
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_terminate() {
  sg_opPackage.reset();
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_logInitialize(
    QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
  g_callback    = callback;
  g_maxLogLevel = maxLogLevel;
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_logSetLevel(
    QnnLog_Level_t maxLogLevel) {
  return QNN_SUCCESS;
}

__attribute__((unused)) static Qnn_ErrorHandle_t ${package_info.name}_logTerminate(void) {
  return QNN_SUCCESS;
}

extern "C" QNN_API Qnn_ErrorHandle_t
<% log_info("Note: Interface provider function will be named: {}".format(package_info.name + "InterfaceProvider")) %>
${package_info.name}InterfaceProvider(QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init              = ${package_info.name}_initialize;
  interface->v1_4.terminate         = ${package_info.name}_terminate;
  interface->v1_4.getInfo           = ${package_info.name}_getInfo;
  interface->v1_4.validateOpConfig  = ${package_info.name}_validateOpConfig;
  interface->v1_4.createOpImpl      = ${package_info.name}_createOpImpl;
  interface->v1_4.freeOpImpl        = ${package_info.name}_freeOpImpl;
  interface->v1_4.logInitialize     = ${package_info.name}_logInitialize;
  interface->v1_4.logSetLevel       = ${package_info.name}_logSetLevel;
  interface->v1_4.logTerminate      = ${package_info.name}_logTerminate;
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
