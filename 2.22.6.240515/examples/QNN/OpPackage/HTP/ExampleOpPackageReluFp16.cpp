//=============================================================================
//
//  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

/*
 * Relevant information on writing HTP op packages can be found in
 * "Op Writing Guidelines" section in QNN SDK docs/general/backend.html
 */

/* Add BEGIN_PKG_OP_DEFINITION(<name>), where <name> is a C++ identifier that
uniquely IDs the source file
NOTE: You must also append DECLARE_OPS_OPTS_LIST(<name>) to the list
defined in ExampleOpPackageInterface.cpp
*/
BEGIN_PKG_OP_DEFINITION(PKG_ReluFp16);

// op execute function declarations
// op 1
template <typename T_Ttype>
int reluImplFp(T_Ttype &out, const T_Ttype &in);

// op 2
template <typename T_TtypeI, typename T_TtypeX>
int reluXImplFp(T_TtypeI &out, const T_TtypeI &in, const T_TtypeX &inX);

// op 3
template <typename T_Ttype>
int relu1ImplFp(T_Ttype &out, const T_Ttype &in);

DEF_TENSOR_PROPERTIES(Op("Relu.fp16", "in0"), MainMemory("*", "in0"))

DEF_TENSOR_PROPERTIES(Op("ReluX", "in0", "in1"),
                      Crouton("*", "in0", "in1"),
                      MainMemory("*", "in0", "in1"))

DEF_TENSOR_PROPERTIES(Op("Relu1", "in0"), Crouton("*", "in0"), MainMemory("*", "in0"))

/*
 * op definitions
 * need to be global in the package
 * one definition per op
 */

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag
 * (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((reluImplFp<F16CroutonTensor>), "Relu")
 */

/*
 * method 2 for defining op with specified cost value (one of GLACIAL,
 * SNAIL, FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or
 * more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX,
 * RESOURCE_HMX(not supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImplFp<F16CroutonTensor>), "Relu",
 * "SNAIL")
 */

// method 2 is used in this example, refer to other examples for the use of other methods
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImplFp<F16CroutonTensor>),
                                  "Relu.fp16",
                                  FAST,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImplFp<PlainFloat16Tensor>),
                                  "Relu.fp16",
                                  FAST,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluXImplFp<F16CroutonTensor, F16CroutonTensor>),
                                  "ReluX",
                                  FAST,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((relu1ImplFp<F16CroutonTensor>),
                                  "Relu1",
                                  FAST,
                                  Flags::RESOURCE_HVX)

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((reluImplFp<F16CroutonTensor>),
 * "Relu", reluCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax:
 * DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include GRAPH_CLEANUP(0), EARLY(2000),
 * MIDDLE(3000), LATE(4000) HTP core provides some replacement functions for op
 * package to use for more information about optimization rules, please refer to
 * documentation located at QNN SDK docs/HTP/optimization_grammar.html
 */

#define RELAXED_PRECISION_RELU(OP)                                                \
  DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(                                            \
      GRAPH_CLEANUP,                                                              \
      relaxed_precision_flag,                                                     \
      Op(OP, "In"),                                                               \
      AND(EQ(DTYPE_OF("In"), DType::Float32), EQ(DTYPE_OF("*"), DType::Float32)), \
      WITH_OUTPUT_TYPE(                                                           \
          DType::Float32,                                                         \
          0,                                                                      \
          1.0f,                                                                   \
          Op(FROM_DEFAULT_PACKAGE("Cast"),                                        \
             WITH_SIZE("*",                                                       \
                       WITH_OUTPUT_TYPE(                                          \
                           DType::Float16,                                        \
                           0,                                                     \
                           1.0f,                                                  \
                           Op(OP, WITH_SIZE("In", Op(FROM_DEFAULT_PACKAGE("Cast"), "In"))))))))

RELAXED_PRECISION_RELU("Relu")
RELAXED_PRECISION_RELU("Relu1")

DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
    GRAPH_CLEANUP,
    relaxed_precision_flag,
    Op("ReluX", "In", "X"),
    AND(EQ(DTYPE_OF("In"), DType::Float32),
        EQ(DTYPE_OF("X"), DType::Float32),
        EQ(DTYPE_OF("*"), DType::Float32)),
    WITH_OUTPUT_TYPE(
        DType::Float32,
        0,
        1.0f,
        Op(FROM_DEFAULT_PACKAGE("Cast"),
           WITH_SIZE("*",
                     WITH_OUTPUT_TYPE(DType::Float16,
                                      0,
                                      1.0f,
                                      Op("ReluX",
                                         WITH_SIZE("In", Op(FROM_DEFAULT_PACKAGE("Cast"), "In")),
                                         WITH_SIZE("X", Op(FROM_DEFAULT_PACKAGE("Cast"), "X"))))))))

DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(GRAPH_CLEANUP + 1,
                                    relaxed_precision_flag,
                                    Op("Relu", "In"),
                                    EQ(DTYPE_OF("In"), DType::Float16),
                                    Op("Relu.fp16", "In"))

// Split on Height
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
    EARLY,
    relaxed_precision_flag,
    Op("Relu.fp16", "X"),
    GT(DIM_HEIGHT("*"), TILE_HEIGHT),
    AUTOSPLIT(1, "I", TILE_HEIGHT, Op("Relu.fp16", TYPICAL_SLICE("X", "I"))))

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax:
 * DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
 * order of parameters listed determines the order of parameters passed into op
 * execution functions if an op does not have a parameter order definition,
 * parameter order passed into Qnn_addNode will be passed into op execution
 * functions if an op has a parameter order definition, any parameter passed
 * into Qnn_addNode with unlisted name will be abandoned if two or more op
 * packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANATORY refers to whether this parameter is required to be provided at
 * Qnn_addNode DEFAULT is used when MANATORY is false if provided as
 * Qnn_Param_t*, DEFAULT will be used for graph construction when this parameter
 * is not provided at Qnn_addNode if provided as nullptr, graph construction
 * will skip this parameter when this parameter is not provided at Qnn_addNode
 * eg. DEF_PACKAGE_PARAM_ORDER("ReluX","X_VAL",true,nullptr)
 */

/* execute functions for ops */
// op 1 Relu fp
template <typename T_Ttype>
int reluImplFp(T_Ttype &out, const T_Ttype &in) {
  debuglog("relu fp execute... dims=(%zdx%zdx%zdx%zd)", in.dim(0), in.dim(1), in.dim(2), in.dim(3));
  debuglog("in=%p out=%p", &in, &out);
  out.set_dims(in);

  size_t inBlocks  = in.blocktab_len();
  auto inBlocktab  = in.blocktab_ptr();
  auto outBlocktab = out.blocktab_ptr();

  HVX_Vector vminval = Q6_Vh_vsplat_R(0x0);
  for (uint32_t i = 0; i < inBlocks; ++i) {
    auto inVptr  = (const HVX_Vector *)(inBlocktab[i]);
    auto outVptr = (HVX_Vector *)(outBlocktab[i]);
    for (uint32_t j = 0; j < 16; ++j) {
      HVX_Vector vin = inVptr[j];
      vin            = Q6_Vhf_vmax_VhfVhf(vin, vminval);
      outVptr[j]     = vin;
    }
  }
  return GraphStatus::Success;
}

// op 2 ReluX fp
template <typename T_TtypeI, typename T_TtypeX>
int reluXImplFp(T_TtypeI &out, const T_TtypeI &in, const T_TtypeX &inX) {
  debuglog(
      "relux fp execute... dims=(%zdx%zdx%zdx%zd)", in.dim(0), in.dim(1), in.dim(2), in.dim(3));
  debuglog("in=%p out=%p", &in, &out);
  out.set_dims(in);

  size_t inBlocks  = in.blocktab_len();
  auto inBlocktab  = in.blocktab_ptr();
  auto outBlocktab = out.blocktab_ptr();

  float x = inX(0, 0, 0, 0);

  if (!(x > 0.0f)) {
    errlog("reluX limit %f not > 0", x);
    return GraphStatus::ErrorFatal;
  }

  Float16 minval(0.0f);
  Float16 maxval(x);
  HVX_Vector vminval = Q6_Vh_vsplat_R(minval.raw());
  HVX_Vector vmaxval = Q6_Vh_vsplat_R(maxval.raw());

  for (uint32_t i = 0; i < inBlocks; ++i) {
    auto inVptr  = (const HVX_Vector *)(inBlocktab[i]);
    auto outVptr = (HVX_Vector *)(outBlocktab[i]);
    for (uint32_t j = 0; j < 16; ++j) {
      HVX_Vector vin = inVptr[j];
      vin            = Q6_Vhf_vmax_VhfVhf(vin, vminval);
      vin            = Q6_Vhf_vmin_VhfVhf(vin, vmaxval);
      outVptr[j]     = vin;
    }
  }
  return GraphStatus::Success;
}

// op 3 Relu1 fp
template <typename T_Ttype>
int relu1ImplFp(T_Ttype &out, const T_Ttype &in) {
  debuglog(
      "relu1 fp execute... dims=(%zdx%zdx%zdx%zd)", in.dim(0), in.dim(1), in.dim(2), in.dim(3));
  debuglog("in=%p out=%p", &in, &out);
  out.set_dims(in);

  size_t inBlocks  = in.blocktab_len();
  auto inBlocktab  = in.blocktab_ptr();
  auto outBlocktab = out.blocktab_ptr();

  HVX_Vector vminval = Q6_Vh_vsplat_R(0xbc00);  //-1.0
  HVX_Vector vmaxval = Q6_Vh_vsplat_R(0x3c00);  // 1.0

  for (uint32_t i = 0; i < inBlocks; ++i) {
    auto inVptr  = (const HVX_Vector *)(inBlocktab[i]);
    auto outVptr = (HVX_Vector *)(outBlocktab[i]);
    for (uint32_t j = 0; j < 16; ++j) {
      HVX_Vector vin = inVptr[j];
      vin            = Q6_Vhf_vmax_VhfVhf(vin, vminval);
      vin            = Q6_Vhf_vmin_VhfVhf(vin, vmaxval);
      outVptr[j]     = vin;
    }
  }
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_ReluFp16);
