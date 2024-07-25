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
BEGIN_PKG_OP_DEFINITION(PKG_Softmax);

DEF_PACKAGE_PARAM_ORDER("Softmax", "beta", false, nullptr, "axis", false, nullptr)

// op execute function declarations
template <typename T_Ttype>
int softmaxWithbetaWrapper(T_Ttype &out, const T_Ttype &in, const Tensor &beta);

template <typename OutTtype, typename InTtype>
int softmax_fp_impl(OutTtype &out, const InTtype &in, const Tensor &beta);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag
 * (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((softmaxWithbetaWrapper<Tensor>), "Softmax")
 */
DEF_PACKAGE_OP(softmaxWithbetaWrapper<Tensor>, "Softmax")
DEF_PACKAGE_OP((softmaxWithbetaWrapper<QuantUint8Tensor>), "Softmax")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or
 * more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX,
 * RESOURCE_HMX(not supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxWithbetaWrapper<PlainFloatTensor>),
 * "Softmax", SNAIL, Flags::IS_CONST)
 */

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxWithbetaWrapper<PlainFloatTensor>),
                                  "Softmax",
                                  SNAIL,
                                  Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("Softmax", "in", "Beta"), Flat("*"), MainMemory("*"))

// Softmax float registration
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmax_fp_impl<PlainFloat16Tensor, PlainFloat16Tensor>),
                                  "Softmax_fp",
                                  FAST,
                                  Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmax_fp_impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>),
                                  "Softmax_fp",
                                  FAST,
                                  Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("Softmax_fp", "in", "Beta"), Flat("*", "in", "Beta"))

// Only need for float version
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
    GRAPH_CLEANUP,
    relaxed_precision_flag,
    Op("Softmax", "In", "Beta", "axis"),
    AND(EQ(DTYPE_OF("In"), DType::Float32), EQ(DTYPE_OF("*"), DType::Float32)),
    WITH_OUTPUT_TYPE(
        DType::Float32,
        0,
        1.0f,
        Op(FROM_DEFAULT_PACKAGE("Cast"),
           WITH_SIZE("*",
                     WITH_OUTPUT_TYPE(DType::Float16,
                                      0,
                                      1.0f,
                                      Op("Softmax_fp",
                                         WITH_SIZE("In", Op(FROM_DEFAULT_PACKAGE("Cast"), "In")),
                                         "Beta"))))))

DEF_PACKAGE_OPTIMIZATION(
    GRAPH_CLEANUP + 150,
    Op("Softmax_fp",
       // Apply for fp16 conv + activation op
       Op(FROM_DEFAULT_PACKAGE("QNN_OP_ELEMENT_WISE_MULTIPLY"),
          "In",
          Op(FROM_DEFAULT_PACKAGE("QNN_OP_CAST"), OPCONST("MaybeScalarVal"))),
       "Beta"),
    AND(IS_SCALAR("MaybeScalarVal"),
        IS_FLOAT32("MaybeScalarVal"),
        GT(CONSTVAL_FLOAT("MaybeScalarVal", 0), 0)),
    Op("Softmax_fp",
       "In",
       WITH_SAME_OUTPUT("Beta",
                        gen_ConstScalar_f32(MUL(CONSTVAL_FLOAT("Beta", 0),
                                                CONSTVAL_FLOAT("MaybeScalarVal", 0))))))

DEF_PACKAGE_OPTIMIZATION(QNN,
                         Op("Softmax", "In"),
                         OK,
                         Op("Softmax", "In", gen_ConstScalar_f32(1.0f)))

DEF_PACKAGE_OPTIMIZATION(QNN, Op("Softmax", "In", "Beta", "axis"), OK, Op("Softmax", "In", "Beta"))

void softmax_hf_approx(Float16 *pout, const Float16 *pin, float scale, int length) {
  union {
    float f;
    int32_t i;
  } scaleu, c0, c1, c2, c3, sum, sum_recip;
  scale /= float(log(2.0));
  scaleu.f = scale;
  // c0 + c1*x + c2*x^2 + c3*x^3
  c0.f = 1.f;
  c1.f = 0.692850309695840;
  c2.f = 0.237504551482093;
  c3.f = 0.046751431261525;

  HVX_Vector *iptr = (HVX_Vector *)pin;
  HVX_Vector xmax  = Q6_Vh_vsplat_R(0xFC00);  // -Infinity half

  // find max
  for (int d = length; d > 63; d -= 64) {
    HVX_Vector xinval = vmemu(iptr);
    iptr++;
    xmax = Q6_Vhf_vmax_VhfVhf(xmax, xinval);
  }
  if ((length & 63) != 0) {
    HVX_Vector xinval         = vmemu(iptr);
    HVX_VectorPred qfinalmask = Q6_Q_vsetq2_R(length * 2);
    xinval                    = Q6_V_vmux_QVV(qfinalmask, xinval, xmax);
    xmax                      = Q6_Vhf_vmax_VhfVhf(xmax, xinval);
  }
  int nshift = 2;
  for (int i = 0; i < 6; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(xmax, xmax, nshift);
    xmax                 = Q6_Vhf_vmax_VhfVhf(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }

  // calculate sum
  HVX_Vector vzero     = Q6_V_vzero();
  HVX_Vector voneh     = Q6_Vh_vsplat_R(0x3c00);
  HVX_Vector f0        = Q6_V_vsplat_R(c0.i);
  HVX_Vector f1        = Q6_V_vsplat_R(c1.i);
  HVX_Vector f2        = Q6_V_vsplat_R(c2.i);
  HVX_Vector f3        = Q6_V_vsplat_R(c3.i);
  HVX_Vector c7f800000 = Q6_V_vsplat_R(0x7f800000);
  HVX_Vector c807fffff = Q6_V_vsplat_R(0x807fffff);
  HVX_Vector vbeta     = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(scaleu.i), vzero);
  HVX_Vector c126      = Q6_V_vsplat_R(126 << 23);
  HVX_Vector c1w       = Q6_V_vsplat_R(1 << 23);
  HVX_Vector c2w       = Q6_Vw_vadd_VwVw(c1w, c1w);
  HVX_Vector c3w       = Q6_Vw_vadd_VwVw(c2w, c1w);
  HVX_Vector c4w       = Q6_Vw_vadd_VwVw(c3w, c1w);
  HVX_Vector c5w       = Q6_Vw_vadd_VwVw(c4w, c1w);
  HVX_Vector vsumf     = Q6_V_vzero();
  HVX_Vector *optr     = (HVX_Vector *)pout;
  iptr                 = (HVX_Vector *)pin;

  for (int d = length; d > 0; d -= 64) {
    HVX_Vector x, x0, x1, p0, p1;
    HVX_VectorPair xdiff, p10;
    HVX_VectorPred q0, q1;

    x = vmemu(iptr);
    iptr++;
    HVX_Vector xd = Q6_Vqf16_vsub_VhfVhf(x, xmax);
    xdiff         = Q6_Wqf32_vmpy_Vqf16Vhf(xd, voneh);

    x0                    = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(xdiff), vbeta);
    x0                    = Q6_Vsf_equals_Vqf32(x0);
    HVX_Vector x0exp      = Q6_V_vand_VV(x0, c7f800000);
    HVX_Vector x0explimit = Q6_Vw_vmin_VwVw(x0exp, c126);
    x0exp                 = Q6_Vw_vsub_VwVw(x0exp, x0explimit);
    HVX_Vector x0norm     = Q6_V_vor_VV(Q6_V_vand_VV(c807fffff, x0), x0explimit);

    x1                    = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(xdiff), vbeta);
    x1                    = Q6_Vsf_equals_Vqf32(x1);
    HVX_Vector x1exp      = Q6_V_vand_VV(x1, c7f800000);
    HVX_Vector x1explimit = Q6_Vw_vmin_VwVw(x1exp, c126);
    x1exp                 = Q6_Vw_vsub_VwVw(x1exp, x1explimit);
    HVX_Vector x1norm     = Q6_V_vor_VV(Q6_V_vand_VV(c807fffff, x1), x1explimit);

    p0     = Q6_Vqf32_vmpy_VsfVsf(x0norm, f3);
    p0     = Q6_Vqf32_vadd_Vqf32Vsf(p0, f2);
    x0norm = Q6_Vqf32_vadd_VsfVsf(x0norm, vzero);
    p0     = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, x0norm);
    p0     = Q6_Vqf32_vadd_Vqf32Vsf(p0, f1);
    p0     = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, x0norm);
    p0     = Q6_Vqf32_vadd_Vqf32Vsf(p0, f0);

    HVX_Vector p0_2  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, p0);
    p0_2             = Q6_Vqf32_vadd_Vqf32Vsf(p0_2, vzero);
    HVX_Vector p0_4  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_2, p0_2);
    p0_4             = Q6_Vqf32_vadd_Vqf32Vsf(p0_4, vzero);
    HVX_Vector p0_8  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_4, p0_4);
    p0_8             = Q6_Vqf32_vadd_Vqf32Vsf(p0_8, vzero);
    HVX_Vector p0_16 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_8, p0_8);
    p0_16            = Q6_Vqf32_vadd_Vqf32Vsf(p0_16, vzero);
    HVX_Vector p0_32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_16, p0_16);
    p0_32            = Q6_Vqf32_vadd_Vqf32Vsf(p0_32, vzero);
    HVX_Vector p0_64 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_32, p0_32);

    p1     = Q6_Vqf32_vmpy_VsfVsf(x1norm, f3);
    p1     = Q6_Vqf32_vadd_Vqf32Vsf(p1, f2);
    x1norm = Q6_Vqf32_vadd_VsfVsf(x1norm, vzero);
    p1     = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, x1norm);
    p1     = Q6_Vqf32_vadd_Vqf32Vsf(p1, f1);
    p1     = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, x1norm);
    p1     = Q6_Vqf32_vadd_Vqf32Vsf(p1, f0);

    HVX_Vector p1_2  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, p1);
    p1_2             = Q6_Vqf32_vadd_Vqf32Vsf(p1_2, vzero);
    HVX_Vector p1_4  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_2, p1_2);
    p1_4             = Q6_Vqf32_vadd_Vqf32Vsf(p1_4, vzero);
    HVX_Vector p1_8  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_4, p1_4);
    p1_8             = Q6_Vqf32_vadd_Vqf32Vsf(p1_8, vzero);
    HVX_Vector p1_16 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_8, p1_8);
    p1_16            = Q6_Vqf32_vadd_Vqf32Vsf(p1_16, vzero);
    HVX_Vector p1_32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_16, p1_16);
    p1_32            = Q6_Vqf32_vadd_Vqf32Vsf(p1_32, vzero);
    HVX_Vector p1_64 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_32, p1_32);

    q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c1w);
    q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c1w);
    p0 = Q6_V_vmux_QVV(q0, p0_2, p0);
    p1 = Q6_V_vmux_QVV(q1, p1_2, p1);

    q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c2w);
    q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c2w);
    p0 = Q6_V_vmux_QVV(q0, p0_4, p0);
    p1 = Q6_V_vmux_QVV(q1, p1_4, p1);

    q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c3w);
    q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c3w);
    p0 = Q6_V_vmux_QVV(q0, p0_8, p0);
    p1 = Q6_V_vmux_QVV(q1, p1_8, p1);

    q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c4w);
    q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c4w);
    p0 = Q6_V_vmux_QVV(q0, p0_16, p0);
    p1 = Q6_V_vmux_QVV(q1, p1_16, p1);

    q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c5w);
    q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c5w);
    p0 = Q6_V_vmux_QVV(q0, p0_32, p0);
    p1 = Q6_V_vmux_QVV(q1, p1_32, p1);

    q0 = Q6_Q_vcmp_gt_VwVw(x0exp, c5w);
    q1 = Q6_Q_vcmp_gt_VwVw(x1exp, c5w);
    p0 = Q6_V_vmux_QVV(q0, p0_64, p0);
    p1 = Q6_V_vmux_QVV(q1, p1_64, p1);

    if (d >= 64) {
      p10 = Q6_W_vcombine_VV(p1, p0);
      q6op_vstu_AV(optr, Q6_Vhf_equals_Wqf32(p10));
      optr++;
      vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p0);
      vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p1);
    } else {
      HVX_VectorPred Q0 = Q6_Q_vsetq2_R(4 * (((d & 63) + 1) / 2));
      HVX_VectorPred Q1 = Q6_Q_vsetq2_R(4 * (((d & 63) + 0) / 2));
      p0                = Q6_V_vmux_QVV(Q0, p0, vzero);
      p1                = Q6_V_vmux_QVV(Q1, p1, vzero);
      vsumf             = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p0);
      vsumf             = Q6_Vqf32_vadd_Vqf32Vqf32(vsumf, p1);
      p10               = Q6_W_vcombine_VV(p1, p0);
      q6op_vstu_variable_ARV(optr, 2 * (d & 63), Q6_Vhf_equals_Wqf32(p10));
      optr++;
    }
  }

  for (int i = 0, nshift = 4; i < 5; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(vsumf, vsumf, nshift);
    vsumf                = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }
  vsumf = Q6_Vsf_equals_Vqf32(vsumf);

  // scale output
  sum.i             = Q6_R_vextract_VR(vsumf, 0);
  sum_recip.f       = 1.0f / sum.f;
  HVX_Vector vrecip = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(sum_recip.i), vzero);
  HVX_Vector *ioptr = (HVX_Vector *)pout;

  for (int d = length; d > 63; d -= 64) {
    HVX_VectorPair xx = Q6_Wqf32_vmpy_VhfVhf(vmemu(&ioptr[0]), voneh);
    HVX_Vector xl = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xx), vzero), vrecip);
    HVX_Vector xh = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xx), vzero), vrecip);
    xx            = Q6_W_vcombine_VV(xh, xl);
    q6op_vstu_AV(ioptr, Q6_Vhf_equals_Wqf32(xx));
    ioptr++;
  }
  if ((length & 63) != 0) {
    HVX_VectorPair xx = Q6_Wqf32_vmpy_VhfVhf(vmemu(&ioptr[0]), voneh);
    HVX_Vector xl = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xx), vzero), vrecip);
    HVX_Vector xh = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xx), vzero), vrecip);
    xx            = Q6_W_vcombine_VV(xh, xl);
    q6op_vstu_variable_ARV(ioptr, (length & 63) * 2, Q6_Vhf_equals_Wqf32(xx));
  }
}

template <typename OutTtype, typename InTtype>
int softmax_fp_impl(OutTtype &out, const InTtype &in, const Tensor &beta) {
  // debuglog("fast softmax (%s)", __PRETTY_FUNCTION__);
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();
  float scale                   = in.interface_scale() * beta(0, 0, 0, 0);
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        using T                               = typename InTtype::element_type;
        const T *pin                          = &in.get_raw(b, h, w, 0);
        typename OutTtype::element_type *pout = &out.get_raw(b, h, w, 0);
        softmax_hf_approx(pout, pin, scale, d_in);
      }
    }
  }
  return GraphStatus::Success;
}

template <typename Ttype>
int softmax_impl(Ttype &out, const Ttype &in, const float beta) {
  // debuglog("reference softmax (%s)", __PRETTY_FUNCTION__);
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        float max = in(b, h, w, 0);
        for (Idx d = 0; d < d_in; d++) {
          float const inval = in(b, h, w, d);
          max               = fmaxf(inval, max);
        }
        float sum = 0;
        for (Idx d = 0; d < d_in; d++) {
          float const inval = in(b, h, w, d);
          sum += (out(b, h, w, d) = expf(beta * (inval - max)));
        }
        float const sum_recip = 1.0f / sum;
        for (Idx d = 0; d < d_in; d++) {
          float const outval = out(b, h, w, d);
          out(b, h, w, d)    = outval * sum_recip;
        }
      }
    }
  }
  return GraphStatus::Success;
}

template <typename Ttype>
int softmaxWithbetaWrapper(Ttype &out, const Ttype &in, const Tensor &beta) {
  return softmax_impl<Ttype>(out, in, beta(0, 0, 0, 0));
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_Softmax);
