<%doc> define all relevant variables</%doc>
<%doc>
# ==============================================================================
#
#  Copyright (c) 2021 - 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import is_valid_cpp_identifier, log_info %>
<% is_valid_cpp_identifier(package_info.name) %>

//==============================================================================
// Auto Generated Code for ${package_info.name} v66
//==============================================================================

#include <array>

#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"

typedef struct Udo_OpTypesList {
  char *opType;
  UdoDspShared_t *opFunctionPtr;
  struct Udo_OpTypesList *next;
} Udo_OpTypesList_t;

// op package info
static char sg_packageName[] = "${package_info.name}";

static std::array<const char*, ${len(package_info.operators)}> sg_opNames{${_format_list_to_cpp_brace([operator.type_name for operator in package_info.operators])}};
static QnnOpPackage_OperationInfo_t sg_opInfos[sg_opNames.size()];

static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;

// global data
QnnOpPackage_GlobalInfrastructure_t sg_globalInfra;
static bool sg_packageInitialized = false;

Udo_OpTypesList_t* g_head = NULL;

/*
 * user provided logging call back function
 * currently only supported on linux x86-64 and nonrpc versions
 * typedef void (*QnnLog_Callback_t)(const char* fmt,
 *                                   QnnLog_Level_t level,
 *                                   uint64_t timestamp,
 *                                   va_list args);
 * usage: if(sg_logInitialized && level <= sg_maxLogLevel)
 *            sg_logCallback(fmt, level, timestamp, args);
 *
 * for cross rpc versions, skel side user provided logging call back function
 * can be defined as part of op packages. maximal log level sg_maxLogLevel
 * can be set by Qnn_ErrorHandle_t ${package_info.name}LogSetLevel(QnnLog_Level_t maxLogLevel)
 */
static QnnLog_Callback_t sg_logCallback =
    nullptr;  // user provided call back function pointer for logging
static QnnLog_Level_t sg_maxLogLevel =
    (QnnLog_Level_t)0;  // maximal log level used in user provided logging
static bool sg_logInitialized =
    false;  // tracks whether user provided logging method has been initialized

static UdoDspShared_t *getImplementOpPtr(const char *opType) {
    Udo_OpTypesList_t *current = g_head;
    while (current != NULL) {
        if (!strcmp(current->opType, opType))
            return current->opFunctionPtr;
        else
            current = current->next;
    }
    return NULL;
}

static void registerOpToOpList(const char *opType, UdoDspShared_t *obj)
{
    Udo_OpTypesList_t* newOpNode = (Udo_OpTypesList_t*)(*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc)) (sizeof(Udo_OpTypesList_t));
    int size = strlen(opType) + 1;
    newOpNode->opType = (char*)(*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size); // +1 to hold the '\0' character
    strlcpy(newOpNode->opType , opType, size);
    newOpNode->opFunctionPtr = obj;
    newOpNode->next = g_head;
    g_head  = newOpNode;
}

Udo_ErrorType_t CreateOpFactory(Udo_CoreType_t udoCoreType,
                                void *perFactoryInfrastructure,
                                Udo_String_t operationType,
                                uint32_t numOfStaticParams,
                                Udo_Param_t *staticParams,
                                Udo_OpFactory_t *opFactory) {
  if (sg_globalInfra == NULL) {
    return UDO_UNSUPPORTED_FEATURE;
  }
  if (operationType == NULL || operationType == 0) {
    return UDO_INVALID_ARGUMENT;
  }

  UdoDspShared_t *op_ptr = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return UDO_WRONG_OPERATION;
  } else {
    return op_ptr->createOpFactory(sg_globalInfra, udoCoreType, perFactoryInfrastructure,
                            operationType, numOfStaticParams, staticParams, opFactory);
  }
  return UDO_WRONG_OPERATION;
}

Udo_ErrorType_t CreateOperation(
    Udo_OpFactory_t opFactory, void *perOpInfrastructure,
    uint32_t numOfInputs, Udo_TensorParam_t *inputs, uint32_t numOfOutputs,
    Udo_TensorParam_t *outputs, Udo_Operation_t *operation) {
  if (opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if ((numOfInputs == 0 || inputs == NULL) &&
      (numOfOutputs == 0 || outputs == NULL)) {
    return UDO_INVALID_ARGUMENT;
  }
  if (sg_globalInfra == NULL) {
    return UDO_UNSUPPORTED_FEATURE;
  }
  OpParams_t *m_OpParams = (OpParams_t *)(*(
      sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(OpParams_t));
  m_OpParams->opInfra = (Udo_HexNNv2OpInfra_t)perOpInfrastructure;
  m_OpParams->opFactory = opFactory;
  m_OpParams->numInputParams = numOfInputs;
  // no inputs
  if (numOfInputs == 0 || inputs == NULL) {
    m_OpParams->numInputParams = 0;
    m_OpParams->InputParams = NULL;
  } else {
    m_OpParams->InputParams = inputs;
  }
  // no outputs
  m_OpParams->numOutputParams = numOfOutputs;
  if (numOfOutputs == 0 || outputs == NULL) {
    m_OpParams->numOutputParams = 0;
    m_OpParams->outputParams = NULL;
  } else {
    m_OpParams->outputParams = outputs;
  }
  *operation = (Udo_Operation_t)m_OpParams;

  return UDO_NO_ERROR;
}

Udo_ErrorType_t ExecuteOp(Udo_Operation_t operation,
                          bool blocking, const uint32_t ID,
                          Udo_ExternalNotify_t notifyFunc) {
  if (operation == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  OpParams_t *m_Operation = (OpParams_t *)operation;
  char *operationType = ((OpFactory_t *)(m_Operation->opFactory))->opType;
  if (operationType == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if (sg_globalInfra == NULL) {
    return UDO_INVALID_ARGUMENT;
  }

  UdoDspShared_t *op_ptr = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return UDO_WRONG_OPERATION;
  } else {
    return op_ptr->executeOp(sg_globalInfra, operation, blocking,
                            ID, notifyFunc);
  }
  return UDO_WRONG_OPERATION;
}

Udo_ErrorType_t ReleaseOp(Udo_Operation_t operation) {
  if (operation == NULL) {
    return UDO_NO_ERROR;
  }
  if (sg_globalInfra == NULL) {
    return UDO_UNSUPPORTED_FEATURE;
  }
  OpParams_t *m_Operation = (OpParams_t *)operation;
  (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(m_Operation);
  return UDO_NO_ERROR;
}

Udo_ErrorType_t ReleaseOpFactory(Udo_OpFactory_t opFactory) {
  if (sg_globalInfra == NULL) {
    return UDO_UNSUPPORTED_FEATURE;
  }
  if (opFactory == NULL) {
    return UDO_NO_ERROR;
  }

  OpFactory_t *factory = (OpFactory_t *)opFactory;
  UdoDspShared_t *op_ptr = getImplementOpPtr(factory->opType);
  if (op_ptr == NULL) {
    return UDO_WRONG_OPERATION;
  } else {
    return op_ptr->releaseOpFactory(sg_globalInfra, opFactory);
  }
  return UDO_NO_ERROR;
}

Udo_ErrorType_t ValidateOperation(Udo_String_t operationType,
                                  uint32_t numOfStaticParams,
                                  const Udo_Param_t *staticParams) {
  UdoDspShared_t *op_ptr = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return UDO_WRONG_OPERATION;
  } else {
    return op_ptr->validateOp(operationType, numOfStaticParams, staticParams);
  }
  return UDO_WRONG_OPERATION;
}

Udo_ErrorType_t QueryOperation(
    Udo_String_t operationType, uint32_t numOfStaticParams,
    const Udo_Param_t *staticParams, uint32_t *numOfInputs,
    Udo_QuantizationType_t **inputsQuantTypes,
    Udo_HexNNTensorLayout_t **inputsLayouts, uint32_t *numOfOutputs,
    Udo_QuantizationType_t **outputsQuantTypes,
    Udo_HexNNTensorLayout_t **outputsLayouts) {
  UdoDspShared_t *op_ptr = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return UDO_WRONG_OPERATION;
  } else {
    return op_ptr->queryOp(operationType, numOfStaticParams, staticParams, numOfInputs,
          inputsQuantTypes, inputsLayouts, numOfOutputs, outputsQuantTypes,
          outputsLayouts);
  }
  return UDO_WRONG_OPERATION;
}

/* op package API's */
Qnn_ErrorHandle_t ${package_info.name}Init(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  if (infrastructure == NULL) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }
  sg_globalInfra        = infrastructure;

%for i in range(len(package_info.operators)):
  UdoDspShared_t* ${package_info.operators[i].type_name.lower()}Op  = new_${package_info.operators[i].type_name.lower()}(sg_globalInfra);
  if (${package_info.operators[i].type_name.lower()}Op == NULL) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }
  registerOpToOpList("${package_info.operators[i].type_name}", ${package_info.operators[i].type_name.lower()}Op);
  sg_opInfos[${i}] = (QnnOpPackage_OperationInfo_t)(*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc)) (sizeof(_QnnOpPackage_OperationInfo_t));
  sg_opInfos[${i}]->opType = ${package_info.operators[i].type_name.lower()}Op->opType;
  sg_opInfos[${i}]->numOfStaticParams = ${package_info.operators[i].type_name.lower()}Op->numOfStaticParams;
  sg_opInfos[${i}]->numOfInputs = ${package_info.operators[i].type_name.lower()}Op->numOfInputs;
  sg_opInfos[${i}]->numOfOutputs = ${package_info.operators[i].type_name.lower()}Op->numOfOutputs;
  sg_opInfos[${i}]->createOpFactory = CreateOpFactory;
  sg_opInfos[${i}]->createOperation = CreateOperation;
  sg_opInfos[${i}]->executeOp = ExecuteOp;
  sg_opInfos[${i}]->releaseOp = ReleaseOp;
  sg_opInfos[${i}]->releaseOpFactory = ReleaseOpFactory;
  sg_opInfos[${i}]->validateOp = ValidateOperation;
  sg_opInfos[${i}]->queryOp = QueryOperation;
%endfor

  sg_packageInitialized = true;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ${package_info.name}GetInfo(const QnnOpPackage_Info_t** info) {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

  sg_packageInfo                = QNN_OP_PACKAGE_INFO_INIT;
  sg_packageInfo.packageName    = sg_packageName;
  sg_packageInfo.operationNames = sg_opNames.data();
  sg_packageInfo.operationInfo  = sg_opInfos;
  sg_packageInfo.numOperations  = sg_opNames.size();

  *info = &sg_packageInfo;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ${package_info.name}ValidateOpConfig(Qnn_OpConfig_t opConfig) {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (strcmp(opConfig.v1.packageName, sg_packageName)) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const char *operationType = opConfig.v1.typeName;
  UdoDspShared_t *op_ptr = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (opConfig.v1.numOfInputs != op_ptr->numOfInputs) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (opConfig.v1.numOfOutputs != op_ptr->numOfOutputs) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (opConfig.v1.numOfParams != op_ptr->numOfStaticParams) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  /*
   * If this op need to validate others, add code here
   */
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ${package_info.name}LogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
  if (sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_logCallback    = callback;
  sg_maxLogLevel    = maxLogLevel;
  sg_logInitialized = true;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ${package_info.name}LogSetLevel(QnnLog_Level_t maxLogLevel) {
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_maxLogLevel = maxLogLevel;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ${package_info.name}LogTerminate() {
  if (!sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  sg_logCallback    = nullptr;
  sg_maxLogLevel    = (QnnLog_Level_t)0;
  sg_logInitialized = false;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ${package_info.name}Terminate() {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  Udo_OpTypesList_t *current = g_head;
  while (current != NULL) {
    (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(current->opType);
    auto next = current->next;
    (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(current);
    current = next;
  }
%for i in range(len(package_info.operators)):
  (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(sg_opInfos[${i}]);
%endfor

  sg_globalInfra        = nullptr;
  sg_packageInitialized = false;
  return QNN_OP_PACKAGE_NO_ERROR;
}

#ifdef __cplusplus
extern "C" {
#endif

<% log_info("Note: Interface provider function will be named: {}".format(package_info.name + "InterfaceProvider")) %>
/* latest version */
Qnn_ErrorHandle_t ${package_info.name}InterfaceProvider(QnnOpPackage_Interface_t* interface) {
  if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion      = {1, 4, 0};
  interface->v1_4.init             = ${package_info.name}Init;
  interface->v1_4.terminate        = ${package_info.name}Terminate;
  interface->v1_4.getInfo          = ${package_info.name}GetInfo;
  interface->v1_4.validateOpConfig = ${package_info.name}ValidateOpConfig;
  interface->v1_4.createOpImpl     = nullptr;
  interface->v1_4.freeOpImpl       = nullptr;
  interface->v1_4.logInitialize    = ${package_info.name}LogInitialize;
  interface->v1_4.logSetLevel      = ${package_info.name}LogSetLevel;
  interface->v1_4.logTerminate     = ${package_info.name}LogTerminate;
  return QNN_OP_PACKAGE_NO_ERROR;
}

#ifdef __cplusplus
}
#endif

<%def name="_format_list_to_cpp_brace(list_object)" filter="trim">
<% string_list_type = str(list_object) %>
${string_list_type.replace('[', '{').replace(']', '}').replace('\'', '\"')}
</%def>

