//=============================================================================
//
//  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#include <array>

#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"
#include "QnnDspTypeMacros.hpp"

typedef struct Udo_OpTypesList {
  char *opType;
  UdoDspShared_t *opFunctionPtr;
  struct Udo_OpTypesList *next;
} Udo_OpTypesList_t;

// op package info
static char sg_packageName[] = "examples.OpPackage";

static std::array<const char *, 1> sg_opNames{{"Relu"}};
static QnnOpPackage_OperationInfo_t sg_opInfos[sg_opNames.size()];

static QnnOpPackage_Info_t sg_packageInfo = {
    sg_packageName, sg_opNames.data(), sg_opInfos, sg_opNames.size(), 0, 0};

// global data
QnnOpPackage_GlobalInfrastructure_t sg_globalInfra;
static bool sg_packageInitialized = false;

Udo_OpTypesList_t *g_head = NULL;

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
 * can be set by Qnn_ErrorHandle_t ExampleReluPackageLogSetLevel(QnnLog_Level_t maxLogLevel)
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

static void registerOpToOpList(const char *opType, UdoDspShared_t *obj) {
  Udo_OpTypesList_t *newOpNode = (Udo_OpTypesList_t *)(*(
      sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(Udo_OpTypesList_t));
  int size                     = strlen(opType) + 1;
  newOpNode->opType = (char *)(*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(
      size);  // +1 to hold the '\0' character
  strlcpy(newOpNode->opType, opType, size);
  newOpNode->opFunctionPtr = obj;
  newOpNode->next          = g_head;
  g_head                   = newOpNode;
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
    return op_ptr->createOpFactory(sg_globalInfra,
                                   udoCoreType,
                                   perFactoryInfrastructure,
                                   operationType,
                                   numOfStaticParams,
                                   staticParams,
                                   opFactory);
  }
  return UDO_WRONG_OPERATION;
}

Udo_ErrorType_t CreateOperation(Udo_OpFactory_t opFactory,
                                void *perOpInfrastructure,
                                uint32_t numOfInputs,
                                Udo_TensorParam_t *inputs,
                                uint32_t numOfOutputs,
                                Udo_TensorParam_t *outputs,
                                Udo_Operation_t *operation) {
  if (opFactory == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  if ((numOfInputs == 0 || inputs == NULL) && (numOfOutputs == 0 || outputs == NULL)) {
    return UDO_INVALID_ARGUMENT;
  }
  if (sg_globalInfra == NULL) {
    return UDO_UNSUPPORTED_FEATURE;
  }
  OpParams_t *m_OpParams =
      (OpParams_t *)(*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(OpParams_t));
  m_OpParams->opInfra        = (Udo_HexNNv2OpInfra_t)perOpInfrastructure;
  m_OpParams->opFactory      = opFactory;
  m_OpParams->numInputParams = numOfInputs;
  // no inputs
  if (numOfInputs == 0 || inputs == NULL) {
    m_OpParams->numInputParams = 0;
    m_OpParams->InputParams    = NULL;
  } else {
    m_OpParams->InputParams = inputs;
  }
  // no outputs
  m_OpParams->numOutputParams = numOfOutputs;
  if (numOfOutputs == 0 || outputs == NULL) {
    m_OpParams->numOutputParams = 0;
    m_OpParams->outputParams    = NULL;
  } else {
    m_OpParams->outputParams = outputs;
  }
  *operation = (Udo_Operation_t)m_OpParams;

  return UDO_NO_ERROR;
}

Udo_ErrorType_t ExecuteOp(Udo_Operation_t operation,
                          bool blocking,
                          const uint32_t ID,
                          Udo_ExternalNotify_t notifyFunc) {
  if (operation == NULL) {
    return UDO_INVALID_ARGUMENT;
  }
  OpParams_t *m_Operation = (OpParams_t *)operation;
  char *operationType     = ((OpFactory_t *)(m_Operation->opFactory))->opType;
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
    return op_ptr->executeOp(sg_globalInfra, operation, blocking, ID, notifyFunc);
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

  OpFactory_t *factory   = (OpFactory_t *)opFactory;
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

Udo_ErrorType_t QueryOperation(Udo_String_t operationType,
                               uint32_t numOfStaticParams,
                               const Udo_Param_t *staticParams,
                               uint32_t *numOfInputs,
                               Udo_QuantizationType_t **inputsQuantTypes,
                               Udo_HexNNTensorLayout_t **inputsLayouts,
                               uint32_t *numOfOutputs,
                               Udo_QuantizationType_t **outputsQuantTypes,
                               Udo_HexNNTensorLayout_t **outputsLayouts) {
  UdoDspShared_t *op_ptr = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return UDO_WRONG_OPERATION;
  } else {
    return op_ptr->queryOp(operationType,
                           numOfStaticParams,
                           staticParams,
                           numOfInputs,
                           inputsQuantTypes,
                           inputsLayouts,
                           numOfOutputs,
                           outputsQuantTypes,
                           outputsLayouts);
  }
  return UDO_WRONG_OPERATION;
}

/* op package API's */
Qnn_ErrorHandle_t ExampleReluPackageInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  if (infrastructure == NULL) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }
  sg_globalInfra = infrastructure;

  UdoDspShared_t *reluOp = new_relu(sg_globalInfra);
  if (reluOp == NULL) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }
  registerOpToOpList("Relu", reluOp);
  sg_opInfos[0] =
      (QnnOpPackage_OperationInfo_t)(*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(
          sizeof(QnnDspOpPackage_OperationInfo_t));
  sg_opInfos[0]->opType            = reluOp->opType;
  sg_opInfos[0]->numOfStaticParams = reluOp->numOfStaticParams;
  sg_opInfos[0]->numOfInputs       = reluOp->numOfInputs;
  sg_opInfos[0]->numOfOutputs      = reluOp->numOfOutputs;
  sg_opInfos[0]->createOpFactory   = CreateOpFactory;
  sg_opInfos[0]->createOperation   = CreateOperation;
  sg_opInfos[0]->executeOp         = ExecuteOp;
  sg_opInfos[0]->releaseOp         = ReleaseOp;
  sg_opInfos[0]->releaseOpFactory  = ReleaseOpFactory;
  sg_opInfos[0]->validateOp        = ValidateOperation;
  sg_opInfos[0]->queryOp           = QueryOperation;

  sg_packageInitialized = true;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ExampleReluPackageGetInfo(const QnnOpPackage_Info_t **info) {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

  *info = &sg_packageInfo;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ExampleReluPackageValidateOpConfig(Qnn_OpConfig_t opConfig) {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (strcmp(QNN_DSP_OP_CFG_GET_PACKAGE_NAME(opConfig), sg_packageName)) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const char *operationType = QNN_DSP_OP_CFG_GET_TYPE_NAME(opConfig);
  UdoDspShared_t *op_ptr    = getImplementOpPtr(operationType);
  if (op_ptr == NULL) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (QNN_DSP_OP_CFG_GET_NUM_INPUTS(opConfig) != op_ptr->numOfInputs) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (QNN_DSP_OP_CFG_GET_NUM_OUTPUTS(opConfig) != op_ptr->numOfOutputs) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (QNN_DSP_OP_CFG_GET_NUM_PARAMS(opConfig) != op_ptr->numOfStaticParams) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  /*
   * If this op need to validate others, add code here
   */
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ExampleReluPackageLogInitialize(QnnLog_Callback_t callback,
                                                  QnnLog_Level_t maxLogLevel) {
  if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_logCallback    = callback;
  sg_maxLogLevel    = maxLogLevel;
  sg_logInitialized = true;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ExampleReluPackageLogSetLevel(QnnLog_Level_t maxLogLevel) {
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_maxLogLevel = maxLogLevel;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ExampleReluPackageLogTerminate() {
  sg_logCallback    = nullptr;
  sg_maxLogLevel    = (QnnLog_Level_t)0;
  sg_logInitialized = false;
  return QNN_OP_PACKAGE_NO_ERROR;
}

Qnn_ErrorHandle_t ExampleReluPackageTerminate() {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  Udo_OpTypesList_t *current = g_head;
  while (current != NULL) {
    (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(current->opType);
    auto next = current->next;
    (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(current);
    current = next;
  }
  (*(sg_globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(sg_opInfos[0]);

  sg_globalInfra        = nullptr;
  sg_packageInitialized = false;
  return QNN_OP_PACKAGE_NO_ERROR;
}

#ifdef __cplusplus
extern "C" {
#endif

/* latest version */
Qnn_ErrorHandle_t ExampleReluPackageInterfaceProvider(QnnOpPackage_Interface_t *interface) {
  if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion      = {1, 4, 0};
  interface->v1_4.init             = ExampleReluPackageInit;
  interface->v1_4.terminate        = ExampleReluPackageTerminate;
  interface->v1_4.getInfo          = ExampleReluPackageGetInfo;
  interface->v1_4.validateOpConfig = ExampleReluPackageValidateOpConfig;
  interface->v1_4.createOpImpl     = nullptr;
  interface->v1_4.freeOpImpl       = nullptr;
  interface->v1_4.logInitialize    = ExampleReluPackageLogInitialize;
  interface->v1_4.logSetLevel      = ExampleReluPackageLogSetLevel;
  interface->v1_4.logTerminate     = ExampleReluPackageLogTerminate;
  return QNN_OP_PACKAGE_NO_ERROR;
}

#ifdef __cplusplus
}
#endif
