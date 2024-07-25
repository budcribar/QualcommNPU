//=============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
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
BEGIN_PKG_OP_DEFINITION(PKG_Relu);

// op execute function declarations
// op 1
template <typename T_Ttype>
int reluImpl(T_Ttype &out, const T_Ttype &in);

// op 2
template <typename T_TtypeI, typename T_TtypeX>
int reluMinMaxImpl(T_TtypeI &out, const T_TtypeI &in, const T_TtypeX &inX, const T_TtypeX &inY);

// op 3
GraphStatus reluTablegenImpl(TensorContiguous<Tdefs::QuantUint8> &out,
                             const Tensor &inStepsize,
                             const Tensor &inOffset,
                             const Tensor &min,
                             const Tensor &max);

/*
 * op definitions
 * need to be global in the package
 * one definition per op
 */

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag
 * (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g. DEF_PACKAGE_OP(reluImpl<Tensor>, "Relu")
 */
DEF_PACKAGE_OP(reluImpl<Tensor>, "Relu")
DEF_PACKAGE_OP((reluMinMaxImpl<Tensor, Tensor>), "ReluMinMax")
DEF_PACKAGE_OP(reluTablegenImpl, "ReluTableGen")

DEF_TENSOR_PROPERTIES(Op("ReluTableGen", "in0", "in1", "in2", "in3"),
                      Outputs("out0", "out1"),
                      Flat("out0", "out1"),
                      MainMemory("out0", "out1"))

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or
 * more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX,
 * RESOURCE_HMX(not supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImpl<PlainFloatTensor>), "Relu",
 * SNAIL)
 */
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImpl<PlainFloatTensor>), "Relu", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("Relu", "in0"), Flat("*"), MainMemory("*"))

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g.
 * DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((reluMinMaxImpl<QUint16CroutonTensor,
 * Tensor >), "ReluMinMax", SNAIL, Flags::RESOURCE_HVX)
 */
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<QUint16CroutonTensor, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<QUint16CroutonTensor_TCM, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<QUint8CroutonTensor, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<QUint8CroutonTensor_TCM, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<QuantUint8Tensor, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<QuantUint8Tensor_TCM, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluMinMaxImpl<PlainFloatTensor, Tensor>),
                                  "ReluMinMax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("ReluMinMax", "in0", "in1", "in2"),
                      Flat("in1", "in2"),
                      MainMemory("in1", "in2"))

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax:
 * DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to documentation
 *   located at QNN SDK docs/HTP/optimization_grammar.html
 */
DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("Relu", "X"),
                         IS_QUANT_TYPE("X"),
                         Op("ReluMinMax", "X", gen_ConstScalar_f32(0.0f), gen_ConstScalar_f32(INF)))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("Relu6", "X"),
    OK,
    Op("ReluMinMax", "X", gen_ConstScalar_f32(0.0f), gen_ConstScalar_f32(6.0f)))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("Relu1", "X"),
    OK,
    Op("ReluMinMax", "X", gen_ConstScalar_f32(-1.0f), gen_ConstScalar_f32(1.0f)))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("ReluX", "X", "Max"),
    OK,
    Op("ReluMinMax", "X", gen_ConstScalar_f32(0.0f), gen_ConstScalar_f32(CONSTVAL_FLOAT("Max", 0))))

// Find min of range defined by a scale/offset for a qu8 tensor
#define MIN_QU8_CHECK(X) MUL(STEPSIZE_OF(X), MUL(-1.0f, ZERO_OFFSET_OF(X)))

// Find max of range defined by a scale/offset for a qu8 tensor
#define MAX_QU8_CHECK(X) MUL(STEPSIZE_OF(X), SUB(255.0f, ZERO_OFFSET_OF(X)))

// Drop relu if quant parms if input and output quant params indicate a range of
// min...max

DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("ReluMinMax", "In", "Min", "Max"),
                         AND(IS_QUINT8("In"),
                             IS_QUINT8("*"),
                             EQ(STEPSIZE_OF("In"), STEPSIZE_OF("*")),
                             EQ(ZERO_OFFSET_OF("In"), ZERO_OFFSET_OF("*")),
                             GE(MIN_QU8_CHECK("In"), CONSTVAL_FLOAT("Min", 0)),
                             LE(MAX_QU8_CHECK("In"), CONSTVAL_FLOAT("Max", 0))),
                         "In")

DEF_PACKAGE_OPTIMIZATION(EARLY + 1,
                         Op("ReluMinMax", "X", "Min", "Max"),
                         AND(IS_QUINT8("X"), IS_QUINT8("*"), NOT(SAME_QUANT("X", "*"))),
                         Op(FROM_DEFAULT_PACKAGE("TableLookup"),
                            "X",
                            WITH_SIZE(gen_Shape(1, 1, 1, 256),
                                      Op("ReluTableGen",
                                         gen_ConstScalar_f32(STEPSIZE_OF("X")),
                                         gen_ConstScalar_i32(ZERO_OFFSET_OF("X")),
                                         "Min",
                                         "Max"))))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 2,
    Op("ReluMinMax", "X", "Min", "Max"),
    GT(DIM_BATCHES("*"), 1),
    AUTOSPLIT(0, "I", 1, Op("ReluMinMax", TYPICAL_SLICE("X", "I"), "Min", "Max")))

// Split on depth
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 3,
    Op("ReluMinMax", "X", "Min", "Max"),
    GT(DIM_DEPTH("*"), CHANNEL_SPLIT_SIZE),
    AUTOSPLIT(3, "I", CHANNEL_SPLIT_SIZE, Op("ReluMinMax", TYPICAL_SLICE("X", "I"), "Min", "Max")))

DEF_PACKAGE_OPTIMIZATION(EARLY + 3,
                         Op("Relu", "X"),
                         GT(DIM_DEPTH("*"), CHANNEL_SPLIT_SIZE),
                         AUTOSPLIT(3, "I", CHANNEL_SPLIT_SIZE, Op("Relu", TYPICAL_SLICE("X", "I"))))

// Split on Height
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 4,
    Op("ReluMinMax", "X", "Min", "Max"),
    GT(DIM_HEIGHT("*"), TILE_HEIGHT),
    AUTOSPLIT(1, "I", TILE_HEIGHT, Op("ReluMinMax", TYPICAL_SLICE("X", "I"), "Min", "Max")))

DEF_PACKAGE_OPTIMIZATION(EARLY + 4,
                         Op("Relu", "X"),
                         GT(DIM_HEIGHT("*"), TILE_HEIGHT),
                         AUTOSPLIT(1, "I", TILE_HEIGHT, Op("Relu", TYPICAL_SLICE("X", "I"))))

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
 * eg. DEF_PACKAGE_PARAM_ORDER(sg_opNameReluX,"X_VAL",true,nullptr)
 */

/* execute functions for ops */
template <typename T_Ttype>
int reluImpl(T_Ttype &out, const T_Ttype &in) {
#ifdef OP_PERF_TEST
  auto pcyclepoint = PcyclePoint(true);
#endif
  debuglog("relu execute... dims=(%zdx%zdx%zdx%zd)", in.dim(0), in.dim(1), in.dim(2), in.dim(3));
  debuglog("in=%p out=%p", &in, &out);
  out.set_dims(in);
  for (Idx b = 0; b < in.dim(0); b++) {
    for (Idx h = 0; h < in.dim(1); h++) {
      for (Idx w = 0; w < in.dim(2); w++) {
        for (Idx d = 0; d < in.dim(3); d++) {
          float inval     = in(b, h, w, d);
          out(b, h, w, d) = fmaxf(inval, 0.0f);
        }
      }
    }
  }
#ifdef OP_PERF_TEST
  pcyclepoint.stop();
  infolog("hnn_relu_impl_cycle: %lu in(h x w x d)=%zdx%zdx%zd\n",
          (unsigned long)pcyclepoint.get_total(),
          in.dim(1),
          in.dim(2),
          in.dim(3));
#endif
  return GraphStatus::Success;
}

template <typename T_TtypeI, typename T_TtypeX>
int reluMinMaxImpl(T_TtypeI &out, const T_TtypeI &in, const T_TtypeX &inX, const T_TtypeX &inY) {
#ifdef OP_PERF_TEST
  auto pcyclepoint = PcyclePoint(true);
#endif
  debuglog("reluxy execute... dims=(%zdx%zdx%zdx%zd)", in.dim(0), in.dim(1), in.dim(2), in.dim(3));
  debuglog("in=%p out=%p", &in, &out);

  float x = inX(0, 0, 0, 0);
  float y = inY(0, 0, 0, 0);

  if (!(y > x)) {
    errlog("reluXY limit %f not > %f", x, y);
    return GraphStatus::ErrorFatal;
  }
  out.set_dims(in);

  bool noScaling = false;

  const auto [bIn, hIn, wIn, dIn] = in.dims();

  if constexpr (!(std::is_same<Tensor, T_TtypeI>::value)) {
    if (in.interface_scale() == out.interface_scale() &&
        in.interface_offset() == out.interface_offset()) {
      noScaling = true;
    }
  }

  if (noScaling) {
    static const float s_inf  = std::numeric_limits<float>::infinity();
    static const int s_minInt = std::numeric_limits<int>::min();
    static const int s_maxInt = std::numeric_limits<int>::max();

    if constexpr (std::is_base_of<LayoutCrouton_8, T_TtypeI>::value) {
      const float outStep     = out.interface_scale();
      const int outZeroOffset = out.interface_offset();
      size_t inBlocks         = in.blocktab_len();
      auto inBlockTab         = in.blocktab_ptr();
      auto outBlockTab        = out.blocktab_ptr();
      int minOutput           = s_minInt;
      int maxOutput           = s_maxInt;
      if (x > -s_inf) {
        minOutput = saturate_round<int>(x / outStep + outZeroOffset);
      }
      if (y < s_inf) {
        maxOutput = saturate_round<int>(y / outStep + outZeroOffset);
      }
      const int minClip = std::max((int)minOutput, 0);
      const int maxClip = std::min((int)maxOutput, 255);
      HVX_Vector vOmin  = Q6_Vb_vsplat_R(minClip);
      HVX_Vector vOmax  = Q6_Vb_vsplat_R(maxClip);
      for (uint32_t i = 0; i < inBlocks; ++i) {
        auto in_vptr  = (const HVX_Vector *)(inBlockTab[i]);
        auto out_vptr = (HVX_Vector *)(outBlockTab[i]);
        for (uint32_t j = 0; j < 16; ++j) {
          out_vptr[j] = Q6_Vub_vmin_VubVub(Q6_Vub_vmax_VubVub(in_vptr[j], vOmin), vOmax);
        }
      }
      return GraphStatus::Success;
    } else if constexpr (std::is_base_of<LayoutFlat_8, T_TtypeI>::value) {
      uint8_t *outptr      = &out.get_raw(0, 0, 0, 0);
      const uint8_t *inptr = &in.get_raw(0, 0, 0, 0);
      int32_t length       = bIn * hIn * wIn * dIn;

      const float outStep     = out.interface_scale();
      const int outZeroOffset = out.interface_offset();
      int minOutput           = s_minInt;
      int maxOutput           = s_maxInt;
      if (x > -s_inf) {
        minOutput = saturate_round<int>(x / outStep + outZeroOffset);
      }
      if (y < s_inf) {
        maxOutput = saturate_round<int>(y / outStep + outZeroOffset);
      }
      const int minClip = std::max((int)minOutput, 0);
      const int maxClip = std::min((int)maxOutput, 255);
      HVX_Vector vOmin  = Q6_Vb_vsplat_R(minClip);
      HVX_Vector vOmax  = Q6_Vb_vsplat_R(maxClip);

      int nvecs    = length >> 7;
      int leftover = length & 127;

      bool useUnalign = (((size_t)inptr) & 0x7f) != 0 || (((size_t)outptr) & 0x7f) != 0;

      if (useUnalign) {
        for (int n = 0; n < nvecs; n++) {
          q6op_vstu_AV(outptr, Q6_Vub_vmin_VubVub(Q6_Vub_vmax_VubVub(vmemu(inptr), vOmin), vOmax));
          inptr += 128;
          outptr += 128;
        }
      } else {
        const HVX_Vector *vinptr = (const HVX_Vector *)inptr;
        HVX_Vector *voptr        = (HVX_Vector *)outptr;
        for (int n = 0; n < nvecs; n++) {
          *voptr++ = Q6_Vub_vmin_VubVub(Q6_Vub_vmax_VubVub(*vinptr++, vOmin), vOmax);
        }
        inptr  = (const uint8_t *)vinptr;
        outptr = (uint8_t *)voptr;
      }
      if (leftover) {
        q6op_vstu_variable_ARV(
            outptr, leftover, Q6_Vub_vmin_VubVub(Q6_Vub_vmax_VubVub(vmemu(inptr), vOmin), vOmax));
      }

      return GraphStatus::Success;
    }
  }

  warnlog("Reluxy using reference.... in: (%ld, %ld, %ld %ld) %s ",
          bIn,
          hIn,
          wIn,
          dIn,
          __PRETTY_FUNCTION__);

  // float fyiMax = 0.0f;
  for (Idx b = 0; b < bIn; b++) {
    for (Idx h = 0; h < hIn; h++) {
      for (Idx w = 0; w < wIn; w++) {
        for (Idx d = 0; d < dIn; d++) {
          float inval = in(b, h, w, d);
          // debuglog("@(%zdx%zdx%zdx%zd): %f",b,h,w,d,inval);
          // fyiMax = fmaxf(fyiMax,inval);
          out(b, h, w, d) = fminf(fmaxf(inval, x), y);
        }
      }
    }
  }
  // debuglog("fyi: max=%f",fyiMax);
#ifdef OP_PERF_TEST
  pcyclepoint.stop();
  infolog("hnn_reluXY_impl_cycle: %lu in(h x w x d)=%zdx%zdx%zd\n",
          (unsigned long)pcyclepoint.get_total(),
          in.dim(1),
          in.dim(2),
          in.dim(3));
#endif
  return GraphStatus::Success;
}

inline size_t flatToVlut(size_t index) {
  return ((index & 63) << 1) | ((index >> 6) & 1) | (index & -128);
}

void tablegenFImpl(TensorContiguous<Tdefs::QuantUint8> &out,
                   const Tensor &inStepsize,
                   const Tensor &inOffset,
                   std::function<float(float)> const &f) {
  const float inStepsizeVal = inStepsize(0, 0, 0, 0);
  const float inOffsetVal   = inOffset(0, 0, 0, 0);
  for (int i = 0; i < 256; i++) {
    /* Calculate what input equal to i would mean */
    float inVal                 = (i - inOffsetVal) * inStepsizeVal;
    out(0, 0, 0, flatToVlut(i)) = f(inVal);
  }
}

GraphStatus reluTablegenImpl(TensorContiguous<Tdefs::QuantUint8> &out,
                             const Tensor &inStepsize,
                             const Tensor &inOffset,
                             const Tensor &min,
                             const Tensor &max) {
  float inMin = min(0, 0, 0, 0);
  float inMax = max(0, 0, 0, 0);
  tablegenFImpl(
      out, inStepsize, inOffset, [inMin, inMax](float x) { return fmaxf(inMin, fminf(x, inMax)); });
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_Relu);