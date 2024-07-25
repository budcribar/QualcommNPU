<%doc> define all relevant variables</%doc>
<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import is_valid_cpp_identifier, log_info %>
<% is_valid_cpp_identifier(package_info.name) %>

//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()

/** Note that the order of declarations given here defines the order in which ops and graph optimizations are
 *  registered to the HTP Core.
 *  Append the latest OpName at the bottom
 */
% for operator in package_info.operators:
DECLARE_PKG_OPS_OPTS_LIST(PKG_${operator.type_name})
%endfor

END_PKG_OPS_OPTS_LIST()

// op package info
static constexpr auto sg_packageName = THIS_PKG_NAME_STR;  // package name passed in as compile flag

static std::array<const char*, ${len(package_info.operators)}> sg_opNames{${_format_list_to_cpp_brace([operator.type_name for operator in package_info.operators])}};

static Qnn_ApiVersion_t sg_sdkApiVersion  = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;

// global data
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra =
nullptr;  // global infrastructure not in use for now
static bool sg_packageInitialized = false;

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
/*
 * for alternative logging method provided by HTP core, please refer to log.h
 */
static QnnLog_Callback_t sg_logCallback =
    nullptr;  // user provided call back function pointer for logging
static QnnLog_Level_t sg_maxLogLevel =
    (QnnLog_Level_t)0;  // maximal log level used in user provided logging
static bool sg_logInitialized =
    false;  // tracks whether user provided logging method has been initialized


/*
* op initialization
* needs to be global in the package
* one initialization per package before any op definitions
* syntax: INIT_PACKAGE_OP_DEF()
*/
INIT_PACKAGE_OP_DEF()

/*
* optimization initialization
* needs to be global in the package
* one initialization per package before any optimization definitions
* syntax: INIT_PACKAGE_OPTIMIZATION_DEF()
*/
INIT_PACKAGE_OPTIMIZATION_DEF()

/*
 * op parameter order initialization
 * needs to be global in the package
 * one initialization per package before any op parameter order definitions
 * syntax: INIT_PACKAGE_PARAM_ORDER_DEF()
 */
INIT_PACKAGE_PARAM_ORDER_DEF()

/*
 * axis parameter name list
 * optional
 * needs to be global in the package
 * one list per package
 * for listing axis parameter names passed into Qnn_AddNode API
 * HTP backend auto-adjusts values in axis parameters based on HTP backfilling
 * note: HTP backend backfills tensor dimensions to 4 dimensions
 * syntax: LIST_PACKAGE_AXIS_PARAMS(...)
 * e.g. LIST_PACKAGE_AXIS_PARAMS("Axis", "AXIS", "axis")
 */
// LIST_PACKAGE_AXIS_PARAMS()

/*
 * per-channel quantized op name list
 * optional
 * needs to be global in the package
 * one list per package
 * for listing op names which support per-channel quantization
 * per-axis quantization info of an op is embeded in axisScaleOffsetEncoding
 *   inside Qnn_Tensor_t types
 * HTP backend only supports per-channel scale ops
 *   i.e. along last dimension, offset is always zero
 * if an op name is marked as having per-channel scale support, and in
 *   QNN_AddNode, at least one input, parameter, or output has
 *   QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET type:
 * then:
 *   HTP backend will pass to op implementation function the following:
 *     output(s), input(s), parameter(s),
 *     outputPerChannelScale(s), inputPerChannelScale(s), paramPerChannelScale(s)
 *
 * optimization rules can be used to remove extra perChannelScale tensors
 *
 * syntax: LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(...)
 * e.g. LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(sg_op1Name, sg_op2Name)
 */

// LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()

/*
* Declare and define the special intialize function for HTP Backend to load
*/
INIT_PKG_CORE_INIT_FUNC()

/* op package API's */

Qnn_ErrorHandle_t ${package_info.name}Init(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;

    /*
    * op parameter order registration
    * registers all defined op parameter orders in the package
    * syntax: REGISTER_PACKAGE_PARAM_ORDERS()
    */
    REGISTER_PACKAGE_PARAM_ORDERS()

    /*
    * op axis parameter name registration
    * registers all axis parameter names in the package
    * used with LIST_PACKAGE_AXIS_PARAMS(...)
    * syntax: REGISTER_PACKAGE_AXIS_PARAMS()
    */
    REGISTER_PACKAGE_AXIS_PARAMS()

    /*
    * per-channel scale op name registration
    * registers all per-channel scale op names in the package
    * used with LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(...)
    * syntax: REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()
    */
    REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()

    sg_globalInfra        = infrastructure;
    sg_packageInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}GetInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

    sg_packageInfo                = QNN_OP_PACKAGE_INFO_INIT;
    sg_packageInfo.packageName    = sg_packageName;
    sg_packageInfo.operationNames = sg_opNames.data();
    sg_packageInfo.numOperations  = sg_opNames.size();
    sg_packageInfo.sdkBuildId     = QNN_SDK_BUILD_ID;
    sg_packageInfo.sdkApiVersion  = &sg_sdkApiVersion;

    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}LogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
    if (sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback    = callback;
    sg_maxLogLevel    = maxLogLevel;
    sg_logInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}LogSetLevel(QnnLog_Level_t maxLogLevel) {
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_maxLogLevel = maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}LogTerminate() {
    if (!sg_logInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_logCallback    = nullptr;
    sg_maxLogLevel    = (QnnLog_Level_t)0;
    sg_logInitialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_info.name}ValidateOpConfig (Qnn_OpConfig_t opConfig){
    if (std::string(sg_packageName) != opConfig.v1.packageName) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    %if len(package_info.operators) > 0:
    /* auto-generated validation code below
     * Check if op config type matches any registered ops
     * If a match is found, check number of inputs, outputs and params
     */
    if (std::string(opConfig.v1.typeName) == "${package_info.operators[0].type_name}"){
        if (opConfig.v1.numOfParams != ${len(package_info.operators[0].param)} || opConfig.v1.numOfInputs != ${len(package_info.operators[0].input)} || opConfig.v1.numOfOutputs != ${len(package_info.operators[0].output)}){
          return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
        }
    }
    % for operator in package_info.operators[1:]:
    else if (std::string(opConfig.v1.typeName) == "${operator.type_name}"){
        if (opConfig.v1.numOfParams != ${len(operator.param)} || opConfig.v1.numOfInputs != ${len(operator.input)} || opConfig.v1.numOfOutputs != ${len(operator.output)}){
          return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
        }
    }
    %endfor
    else{
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }

    /*
    * additional validation code here
    * */

    %else:
    /*
    * add code here
    * */
    %endif
    return QNN_SUCCESS;
}

/* The following three functions in this comment are not called by HTP backend for now,
 * no auto-generated implementations are created. Users should see example for full function signatures.
 * (version 1.3.0) Qnn_ErrorHandle_t ${package_info.name}CreateKernels (QnnOpPackage_GraphInfrastructure_t
 * graphInfrastructure, QnnOpPackage_Node_t node, QnnOpPackage_Kernel_t** kernels, uint32_t*
 * numKernels)
 * (version 1.3.0) Qnn_ErrorHandle_t ${package_info.name}FreeKernels (QnnOpPackage_Kernel_t* kernels)
 *
 * (version 1.4.0) Qnn_ErrorHandle_t ${package_info.name}CreateOpImpl (QnnOpPackage_GraphInfrastructure_t
 * graphInfrastructure, QnnOpPackage_Node_t node, QnnOpPackage_OpImpl_t* opImpl)
 *(version 1.4.0) Qnn_ErrorHandle_t ${package_info.name}FreeOpImpl (QnnOpPackage_OpImpl_t opImpl)
 */

Qnn_ErrorHandle_t ${package_info.name}Terminate() {
if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

sg_globalInfra        = nullptr;
sg_packageInitialized = false;
return QNN_SUCCESS;
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
  return QNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif

<%def name="_format_list_to_cpp_brace(list_object)" filter="trim">
<% string_list_type = str(list_object) %>
${string_list_type.replace('[', '{').replace(']', '}').replace('\'', '\"')}
</%def>
