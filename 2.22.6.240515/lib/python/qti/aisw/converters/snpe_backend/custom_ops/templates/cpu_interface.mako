<%doc>
//=============================================================================
//
//  Copyright (c) 2020 - 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================</%doc>
<%page expression_filter="n" expression_filter="trim" />
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import is_valid_cpp_identifier, log_info %>
<% is_valid_cpp_identifier(package_info.name) %>

//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================
#include "QnnCpuOpPackage.h"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::macros;

static Qnn_ErrorHandle_t ${package_info.name}Initialize(
  QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {

  QNN_CUSTOM_BE_ENSURE(!(CustomOpPackage::getIsInitialized()),QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED);

  INIT_BE_OP_PACKAGE(${package_info.name})

  %for operator in package_info.operators:
  REGISTER_PACKAGE_OP(${operator.type_name})
  %endfor

  // INIT_BE_PACKAGE_OPTIMIZATIONS();

  CustomOpPackage::setIsInitialized(true);

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}GetInfo(const QnnOpPackage_Info_t** info) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(opPkg->getPackageInfo(info));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}ValidateOpConfig(Qnn_OpConfig_t opConfig) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  auto opRegistration = opPkg->getOpRegistration(opConfig.v1.typeName);

  QNN_CUSTOM_BE_ENSURE(opRegistration, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  QNN_CUSTOM_BE_ENSURE_STATUS(opRegistration->validateOpConfig(opConfig));

return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}CreateOpImpl(
   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
   QnnOpPackage_Node_t node,
   QnnOpPackage_OpImpl_t* opImpl) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(
    opPkg->createOpImpl(graphInfrastructure, node, opImpl));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}FreeOpImpl(
   QnnCpuOpPackage_OpImpl_t* opImpl) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(opPkg->freeOpImpl(opImpl));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}Terminate() {
  auto opPkg = CustomOpPackage::getInstance();

  CustomOpPackage::destroyInstance();
  opPkg->freeResolver();

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}LogInitialize(
QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
// function should be used if at least two backends support it
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY BE

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}LogSetLevel(
QnnLog_Level_t maxLogLevel) {
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY CPU BE

return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t ${package_info.name}LogTerminate() {
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY CPU BE

  return QNN_SUCCESS;
}

<% log_info("Note: Interface provider function will be named: {}".format(package_info.name + "InterfaceProvider")) %>
extern "C" QNN_API Qnn_ErrorHandle_t ${package_info.name}InterfaceProvider(
   QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init              = ${package_info.name}Initialize;
  interface->v1_4.terminate         = ${package_info.name}Terminate;
  interface->v1_4.getInfo           = ${package_info.name}GetInfo;
  interface->v1_4.validateOpConfig  = ${package_info.name}ValidateOpConfig;
  interface->v1_4.createOpImpl     =  ${package_info.name}CreateOpImpl;
  interface->v1_4.freeOpImpl        = ${package_info.name}FreeOpImpl;
  interface->v1_4.logInitialize     = ${package_info.name}LogInitialize;
  interface->v1_4.logSetLevel       = ${package_info.name}LogSetLevel;
  interface->v1_4.logTerminate      = ${package_info.name}LogTerminate;
  return QNN_SUCCESS;
}

