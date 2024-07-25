<%doc>
//=============================================================================
//
//  Copyright (c) 2021, 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================</%doc>

<%page expression_filter="n" expression_filter="trim" />
<%!
from qti.aisw.op_package_generator.helpers.template_helpers import get_hexnn_tensor_sig, get_hexnn_param_sig,_template_builder, is_valid_cpp_identifier, get_param_order_sig, build_scalar_param%>
<% is_valid_cpp_identifier(operator.type_name.lower()) %>
<%namespace name='helper' file="/htp_source.mako" import="*" />
//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_${operator.type_name});

%for i, param in enumerate(operator.param):
    %if param.default_value is not None and not isinstance(param.default_value, (list, tuple, str)):
${build_scalar_param(param)}
    %endif
%endfor

// op execute function declarations
${_template_builder(operator) | n}
<%helper:_get_op_impl_input_output funcname='${operator.type_name.lower()}Impl'>
</%helper:_get_op_impl_input_output>;

// forward declaration of sample cost function
static float ${operator.type_name.lower()}CostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((${operator.type_name.lower()}Impl${helper._get_template_tensors("F16CroutonTensor")}), "${operator.type_name}")
 */
DEF_PACKAGE_OP((${operator.type_name.lower()}Impl${helper._get_template_tensors("F16CroutonTensor")}), "${operator.type_name}")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((${operator.type_name.lower()}Impl${helper._get_template_tensors("F16CroutonTensor")}), "${operator.type_name}", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((${operator.type_name.lower()}Impl${helper._get_template_tensors("F16CroutonTensor")}),
 * "${operator.type_name}", ${operator.type_name.lower()}CostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax: DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
     * order of parameters listed determines the order of parameters passed into op execution functions
 * if an op does not have a parameter order definition, parameter order passed into Qnn_addNode
 *   will be passed into op execution functions
 * if an op has a parameter order definition, any parameter passed into Qnn_addNode with unlisted
     *   name will be abandoned
 * if two or more op packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at Qnn_addNode
 * DEFAULT is used when MANDATORY is false
 *     if provided as Qnn_Param_t*,
 *       DEFAULT will be used for graph construction when this parameter is not provided at
 *       Qnn_addNode
 *     if provided as nullptr,
 *       graph construction will skip this parameter when this parameter is not provided at
 *       Qnn_addNode
 */
%if operator.param:
DEF_PACKAGE_PARAM_ORDER("${operator.type_name}", ${get_param_order_sig(operator.param, len("DEF_PACKAGE_PARAM_ORDER("))|n}
%endif


/* execute functions for ops */

${_template_builder(operator) | n}
<%helper:_get_op_impl_input_output funcname='${operator.type_name.lower()}Impl'>
{
  /*
   * add code here
   * */
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */
  return GraphStatus::Success;
}
</%helper:_get_op_impl_input_output>

__attribute__((unused)) static float ${operator.type_name.lower()}CostFunc(const Op *op)
{
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_${operator.type_name});