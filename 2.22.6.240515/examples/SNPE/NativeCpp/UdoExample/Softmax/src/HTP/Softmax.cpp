//<!--
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//-->
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_Softmax);

// op execute function declarations
template <typename TensorType>
GraphStatus softmaxImpl(TensorType& out_0, const TensorType& in_0, const Tensor& axis);

// forward declaration of sample cost function
static float softmaxCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag
 * (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g. DEF_PACKAGE_OP((softmaxImpl<Tensor>),
 * "Softmax")
 */
DEF_PACKAGE_OP((softmaxImpl<Tensor>), "Softmax")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxImpl<PlainFloatTensor>), "Softmax", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((softmaxImpl<PlainFloatTensor>),
 * "Softmax", softmaxCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("Softmax", "axis", true, nullptr)

/* execute functions for ops */

template <typename TensorType>
GraphStatus softmaxImpl(TensorType& out_0, const TensorType& in_0, const Tensor& axis)

{
  out_0.set_dims(in_0);
  // NHWC
  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // Get maximum element
        float max = in_0(b, h, w, 0);
        for (Idx d = 0; d < d_in; d++) {
          float inval = in_0(b, h, w, d);
          max         = fmaxf(inval, max);
        }
        // Sum
        float sum = 0;
        for (Idx d = 0; d < d_in; d++) {
          float inval = in_0(b, h, w, d);
          sum += expf(inval - max);
        }
        // Normalization
        float sum_recip = 1.0f / sum;
        for (Idx d = 0; d < d_in; d++) {
          float inval       = in_0(b, h, w, d);
          float outval      = expf(inval - max);
          out_0(b, h, w, d) = outval * sum_recip;
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float softmaxCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_Softmax);