<%doc>
//=============================================================================
//
//  Copyright (c) 2020 - 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================</%doc>
<%page expression_filter="n" expression_filter="trim" />
//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace ${operator.type_name.lower()} {

Qnn_ErrorHandle_t execute(CustomOp* operation) {

  /**
   * Add code here
   **/
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), ${len(operator.input)}, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), ${len(operator.output)}, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

%if operator.param:
  // Add params
   // The getParam function returns a pair -> hasParam, paramValue
   // Check that parameter has be retrieved. Pair.first is false if it was not found and the paramValue is nullptr
 %for param in operator.param:

   auto ${param.name}Pair = getParam(node, "${param.name}");

   QNN_CUSTOM_BE_ENSURE(${param.name}Pair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("${param.name}", ${param.name}Pair.second);

 %endfor
%endif

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "${operator.type_name}"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, ${len(operator.input)}, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, ${len(operator.output)}, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace ${operator.type_name.lower()}

CustomOpRegistration_t* register_${operator.type_name.title()}CustomOp() {
  using namespace ${operator.type_name.lower()};
  static CustomOpRegistration_t ${operator.type_name.title()}Register = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &${operator.type_name.title()}Register;
}

REGISTER_OP(${operator.type_name}, register_${operator.type_name.title()}CustomOp);
