//=============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

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
BEGIN_PKG_OP_DEFINITION(PKG_MaxPool);

// op execute function declarations
template <typename T>
int maxpoolOpt(T &out,
               const T &in,
               const TensorShape<4> &windowShape,
               const TensorShape<4> &strideShape);

template <bool sameShape, typename T_Ttype>
int maxpoolImpl(T_Ttype &out,
                const T_Ttype &in,
                const TensorShape<4> &windowShape,
                const TensorShape<4> &strideShape);

// cost functions
static float maxpoolCost(const Op *op);

// op declarations
DEF_PACKAGE_OP((maxpoolImpl<false, Tensor>), "MaxPool_valid.ref")

DEF_TENSOR_PROPERTIES(Op("MaxPool_valid.ref", "in0", "in1", "in2"),
                      Flat("*", "in0"),
                      MainMemory("*", "in0", "in1", "in2"))

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((maxpoolImpl<false, PlainFloatTensor>),
                                    "MaxPool_valid.fp",
                                    maxpoolCost,
                                    Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("MaxPool_valid.fp", "in0", "in1", "in2"),
                      Flat("*", "in0"),
                      MainMemory("*", "in0", "in1", "in2"))

DEF_PACKAGE_OP((maxpoolOpt<QUint16CroutonTensor>), "MaxPool_valid")

DEF_PACKAGE_OP((maxpoolOpt<QUint16CroutonTensor_TCM>), "MaxPool_valid")

DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((maxpoolOpt<QUint8CroutonTensor>),
                                    "MaxPool_valid",
                                    maxpoolCost,
                                    Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((maxpoolOpt<QUint8CroutonTensor_TCM>),
                                    "MaxPool_valid",
                                    maxpoolCost,
                                    Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("MaxPool_valid", "in0", "in1", "in2"),
                      Flat("in1", "in2"),
                      Crouton("*", "in0"),
                      MainMemory("in1", "in2"))

/* optimization declarations */
DEF_PACKAGE_OPTIMIZATION(
    QNN,
    Op("PoolMax2d", "Act", "filter_size", "stride", "pad_amount"),
    AND(NOT(IS_FLOAT16("Act")), NOT(IS_FLOAT16("*"))),
    Op("MaxPool_valid",
       WITH_SIZE(SHAPEFN_APPLY(pad_total_for_qnn, "Act", "pad_amount"),
                 WITH_TYPE("Act",
                           Op(FROM_DEFAULT_PACKAGE("SlicePad_shape"),
                              "Act",
                              SHAPEFN_APPLY(pad_before_for_qnn, "Act", "pad_amount"),
                              gen_Shape(0, 0, 0, 0),
                              SHAPEFN_APPLY(pad_total_for_qnn, "Act", "pad_amount"),
                              gen_ConstScalar_i32(0)))),
       SHAPEFN_APPLY(reshape_hw_to_4d, "filter_size"),
       SHAPEFN_APPLY(reshape_hw_to_4d, "stride")))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("MaxPool_valid", "Act", "Window", "Stride"),
    OK,
    Op("MaxPool_valid.tmp", "Act", gen_ShapeOf("Window"), gen_ShapeOf("Stride")))

DEF_PACKAGE_OPTIMIZATION(EARLY + 1,
                         Op("MaxPool_valid.tmp", "A", "W", "S"),
                         OK,
                         Op("MaxPool_valid", "A", "W", "S"))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 1,
    Op("MaxPool_valid", "In", "Window", "Stride"),
    AND(OR(AND(IS_QUINT8("*"), IS_QUINT8("In")), AND(IS_QUINT16("*"), IS_QUINT16("In"))),
        OR(LT(DIV(STEPSIZE_OF("In"), STEPSIZE_OF("*")), 0.00390625f),
           GT(DIV(STEPSIZE_OF("In"), STEPSIZE_OF("*")), 63.99f))),
    Op(FROM_DEFAULT_PACKAGE("Requantize"),
       WITH_SIZE(gen_ShapeOf("*"), WITH_TYPE("In", Op("MaxPool_valid", "In", "Window", "Stride")))))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 3,
    Op("MaxPool_valid", "A", "W", "S"),
    GT(DIM_BATCHES("*"), 1),
    AUTOSPLIT(0, "I", 1, Op("MaxPool_valid", TYPICAL_SLICE("A", "I"), "W", "S")))

// Slice 32 or 64 channels at a time, as a workaround
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 4,
    Op("MaxPool_valid", "Act", "W", "S"),
    AND(NOT(IS_FLOAT16("Act")),
        NOT(IS_FLOAT16("*")),
        GT(DIM_DEPTH("*"), size_t(CHANNEL_SPLIT_SIZE))),
    AUTOSPLIT(3, "I", CHANNEL_SPLIT_SIZE, Op("MaxPool_valid", TYPICAL_SLICE("Act", "I"), "W", "S")))

DEF_PACKAGE_OPTIMIZATION(EARLY + 5,
                         Op("MaxPool_valid", "Act", "W", "S"),
                         GT(DIM_HEIGHT("*"), TILE_HEIGHT),
                         AUTOSPLIT(1,
                                   "I",
                                   TILE_HEIGHT,
                                   Op("MaxPool_valid_tmp",
                                      TYPICAL_CONV_SLICE("Act", "I", "S", DIM_HEIGHT("W")),
                                      "W",
                                      "S")))
DEF_PACKAGE_OPTIMIZATION(EARLY + 6,
                         Op("MaxPool_valid_tmp", "A", "W", "S"),
                         OK,
                         Op("MaxPool_valid", "A", "W", "S"))

DEF_PACKAGE_OPTIMIZATION(LATE,
                         Op("MaxPool_valid", "In", "Window", "Stride"),
                         AND(EQ(DIM_WIDTH("Window"), 1),
                             EQ(DIM_HEIGHT("Window"), 1),
                             OR(GT(DIM_WIDTH("Stride"), 1), GT(DIM_HEIGHT("Stride"), 1))),
                         Op(FROM_DEFAULT_PACKAGE("Decimate"), "In", "Stride"))

DEF_PACKAGE_OPTIMIZATION(
    LATE,
    Op("MaxPool_valid", "In", "Window", "Stride"),

    AND(EQ(DIM_WIDTH("Window"), DIM_HEIGHT("Window")),
        EQ(DIM_WIDTH("Stride"), 1),
        EQ(DIM_HEIGHT("Stride"), 2)),
    Op(FROM_DEFAULT_PACKAGE("Decimate"),  // input: "In" pass through maxpool
                                          // with stride 1x1 win 2x2
       WITH_SIZE(                         // genereate proper output size
           gen_Shape(DIM_BATCHES("In"),   // batch doesn't change
                     ADD(SUB(DIM_HEIGHT("In"), DIM_HEIGHT("Window")),
                         1),  //(H-F+2*P )/S)+1
                     ADD(SUB(DIM_WIDTH("In"), DIM_WIDTH("Window")),
                         1),  //(W-F+2*P )/S)+1
                     DIM_DEPTH("In")),
           Op("MaxPool_valid", "In", "Window", gen_Shape(1, 1, 1, 1))  // change the stride 2x1 to
                                                                       // 1x1
           ),
       "Stride"))

DEF_PACKAGE_OPTIMIZATION(LATE + 10,
                         Op("MaxPool_valid", "In", "W", "S"),
                         IS_FLOAT("In"),
                         Op("MaxPool_valid.fp", "In", "W", "S"))

DEF_PACKAGE_OPTIMIZATION(LATE + 11,
                         Op("MaxPool_valid", "In", "W", "S"),
                         NOT(OR(IS_FLOAT("In"), IS_QUINT8("In"), IS_QUINT16("In"))),
                         Op("MaxPool_valid.ref", "In", "W", "S"))

/* execute functions for ops */
inline HVX_VectorPair __attribute__((always_inline))
verticalMax3(HVX_Vector *iptr0, HVX_Vector *iptr1, HVX_Vector *iptr2) {
  HVX_Vector maxT = Q6_Vub_vmax_VubVub(iptr1[0], iptr1[2]);
  HVX_Vector max0 = Q6_Vub_vmax_VubVub(maxT, iptr0[0]);
  HVX_Vector max1 = Q6_Vub_vmax_VubVub(maxT, iptr2[0]);
  return Q6_W_vcombine_VV(max1, max0);
}

struct TileData {
  uint8_t **addr;
  uint32_t offsetTCol;
  uint32_t offsetTRow;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
};

#define TILEDATASETUP(adrtab, nextTabCol, nextTabRow, h, w, d)                          \
  {                                                                                     \
    .addr = (uint8_t **)(adrtab), .offsetTCol = static_cast<uint32_t>(nextTabCol),      \
    .offsetTRow = static_cast<uint32_t>(nextTabRow), .width = static_cast<uint32_t>(w), \
    .height = static_cast<uint32_t>(h), .depth = static_cast<uint32_t>(d)               \
  }

void maxpoolb3x3wtSliceStride2N00(struct TileData *outdata,
                                  struct TileData *indata,
                                  uint32_t offsets,
                                  uint32_t scalingParams,
                                  const unsigned tH) {
  const uint32_t bTH = Q6_R_ct0_R(tH);  // log2(tH)
  const uint32_t tW  = 64 >> bTH;
  const uint32_t vTW = 16 >> bTH;

  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;
  size_t wIn  = indata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e = (-1 << (shifte + 8)) | (1 << shifte);
  c2e         = Q6_R_combine_RlRl(c2e, c2e);
  scale       = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  HVX_Vector *optr;

  int hRemain = hOut;

  uint8_t **iAdrTab = indata->addr;
  uint8_t **oAdrTab = outdata->addr;

  for (int h = 0; h < hOut; h += 2, hRemain -= 2) {
    int inH0        = 2 * h + 0;
    int inH1        = 2 * h + 2;
    int inH2        = 2 * h + 4;
    uint8_t **ppin0 = iAdrTab + (inH0 >> bTH) * iTNextRow;
    uint8_t **ppin1 = iAdrTab + (inH1 >> bTH) * iTNextRow;
    uint8_t **ppin2 = iAdrTab + (inH2 >> bTH) * iTNextRow;
    uint8_t **ppout = oAdrTab + (h >> bTH) * oTNextRow;
    if (hRemain < 2) ppin2 = ppin1;

    int iTileOffset0 = (inH0 & (tH - 1)) << (11 - bTH);
    int iTileOffset1 = (inH1 & (tH - 1)) << (11 - bTH);
    int iTileOffset2 = (inH2 & (tH - 1)) << (11 - bTH);
    int oTileOffset  = (h & (tH - 1)) << (11 - bTH);

    HVX_Vector *iptr0 = (HVX_Vector *)(ppin0[0] + iTileOffset0);
    HVX_Vector *iptr1 = (HVX_Vector *)(ppin1[0] + iTileOffset1);
    HVX_Vector *iptr2 = (HVX_Vector *)(ppin2[0] + iTileOffset2);

    if (wIn > tW) {
      ppin0 += iTNextCol;
      ppin1 += iTNextCol;
      ppin2 += iTNextCol;
    }

    HVX_Vector in0v1 = iptr0[vTW];
    HVX_Vector in0v0 = *iptr0++;
    HVX_Vector in0v3 = iptr1[vTW];
    HVX_Vector in0v2 = *iptr1++;
    HVX_Vector in0v4 = *iptr2++;

    HVX_Vector max0lo = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(in0v0, in0v1), in0v2);
    HVX_Vector max0hi = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(in0v2, in0v3), in0v4);

    int iw = 0;
    for (int w = 0; w < wOut; w += 4) {
      HVX_Vector in1v1 = iptr0[vTW];
      HVX_Vector in1v0 = *iptr0++;
      HVX_Vector in1v3 = iptr1[vTW];
      HVX_Vector in1v2 = *iptr1++;
      HVX_Vector in1_4 = *iptr2++;
      iw += 8;

      HVX_Vector max1lo = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(in1v0, in1v1), in1v2);
      HVX_Vector max1hi = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(in1v2, in1v3), in1_4);

      HVX_VectorPair W_Odds_Evens0 = Q6_W_vdeal_VVR(max1lo, max0lo, -32);
      HVX_VectorPair W_Odds_Evens1 = Q6_W_vdeal_VVR(max1hi, max0hi, -32);

      HVX_Vector out0 = Q6_Vub_vmax_VubVub(Q6_V_hi_W(W_Odds_Evens0), Q6_V_lo_W(W_Odds_Evens0));
      HVX_Vector out1 = Q6_Vub_vmax_VubVub(Q6_V_hi_W(W_Odds_Evens1), Q6_V_lo_W(W_Odds_Evens1));

      if ((iw & (tW - 1)) == 0) {
        iptr0 = (HVX_Vector *)(ppin0[0] + iTileOffset0);
        iptr1 = (HVX_Vector *)(ppin1[0] + iTileOffset1);
        iptr2 = (HVX_Vector *)(ppin2[0] + iTileOffset2);
        if (iw < (int)(wIn - tW)) {
          ppin0 += iTNextCol;
          ppin1 += iTNextCol;
          ppin2 += iTNextCol;
        }
      }

      HVX_Vector in2_1 = iptr0[vTW];
      HVX_Vector in2_0 = *iptr0++;
      HVX_Vector in2_3 = iptr1[vTW];
      HVX_Vector in2_2 = *iptr1++;
      HVX_Vector in2_4 = *iptr2++;

      HVX_Vector nextMax0v0 = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(in2_0, in2_1), in2_2);
      HVX_Vector nextMax0v1 = Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub(in2_2, in2_3), in2_4);

      HVX_Vector horiz0 = Q6_V_valign_VVR(nextMax0v0, Q6_V_lo_W(W_Odds_Evens0), 32);
      HVX_Vector horiz1 = Q6_V_valign_VVR(nextMax0v1, Q6_V_lo_W(W_Odds_Evens1), 32);

      out0 = Q6_Vub_vmax_VubVub(out0, horiz0);
      out1 = Q6_Vub_vmax_VubVub(out1, horiz1);

      HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1), c2e);

      HVX_Vector a0E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
      HVX_Vector a0O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
      HVX_Vector a1E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
      HVX_Vector a1O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);

      a0 = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
      a1 = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);

      if ((w & (tW - 1)) == 0) {
        optr = (HVX_Vector *)(ppout[0] + oTileOffset);
        ppout += oTNextCol;
      }

      optr[0]   = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
      optr[vTW] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
      optr++;

      max0lo = nextMax0v0;
      max0hi = nextMax0v1;
    }
  }
}

#define VERTICAL_MAX3S2_U8(max0, max1, iptr0, iptr1, iptr2) \
  x2   = iptr1[vTW];                                        \
  x4   = iptr2[vTW];                                        \
  x0   = *iptr0++;                                          \
  x1   = *iptr1++;                                          \
  x3   = *iptr2++;                                          \
  max0 = Q6_Vub_vmax_VubVub(x0, x1);                        \
  max1 = Q6_Vub_vmax_VubVub(x3, x4);                        \
  max0 = Q6_Vub_vmax_VubVub(max0, x2);                      \
  max1 = Q6_Vub_vmax_VubVub(max1, x2);

void maxpoolb3x3wtSliceStride2N11(struct TileData *outdata,
                                  struct TileData *indata,
                                  uint32_t offsets,
                                  uint32_t scalingParams,
                                  const unsigned tH) {
  const size_t bTH = Q6_R_ct0_R(tH);  // log2(tH)
  const size_t tW  = 64 >> bTH;
  const size_t vTW = 16 >> bTH;

  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e = (-1 << (shifte + 8)) | (1 << shifte);
  c2e         = Q6_R_combine_RlRl(c2e, c2e);
  scale       = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  uint8_t **iAdrTab = (uint8_t **)indata->addr + iTNextRow;
  uint8_t **oAdrTab = (uint8_t **)outdata->addr;

  HVX_Vector x0, x1, x2, x3, x4;
  HVX_Vector max0lo, max0hi, max0_2, max0_3;
  HVX_Vector max1lo, max1hi, max1_2, max1_3;
  HVX_Vector *optr;

  int iTOffset0 = ((-1) & (tH - 1)) << (11 - bTH);
  int iTOffset1 = 0;
  int iTOffset2 = ((2) & (tH - 1)) << (11 - bTH);
  int oTOffset  = 0;

  for (int h = 0; h < hOut; h += 2) {
    uint8_t **ppin     = iAdrTab + ((2 * h) >> bTH) * iTNextRow;
    uint8_t **ppout    = oAdrTab + ((1 * h) >> bTH) * oTNextRow;
    uint8_t **ppinLast = ppin + (iTNextRow - iTNextCol);

    int above = iTOffset1 > iTOffset0 ? 0 : -iTNextRow;
    int below = iTOffset2 > iTOffset1 || (hOut - h) == 1 ? 0 : iTNextRow;

    HVX_Vector *iptr0 = (HVX_Vector *)(ppin[above] + iTOffset0 + tW * 32 - 128);
    HVX_Vector *iptr1 = (HVX_Vector *)(ppin[0] + iTOffset1 + tW * 32 - 128);
    HVX_Vector *iptr2 = (HVX_Vector *)(ppin[below] + iTOffset2 + tW * 32 - 128);
    ppin += iTNextCol;

    VERTICAL_MAX3S2_U8(max0lo, max1lo, iptr0, iptr1, iptr2);

    iptr0 = (HVX_Vector *)(ppin[above] + iTOffset0);
    iptr1 = (HVX_Vector *)(ppin[0] + iTOffset1);
    iptr2 = (HVX_Vector *)(ppin[below] + iTOffset2);
    ppin += iTNextCol;
    ppin = std::min(ppin, ppinLast);

    VERTICAL_MAX3S2_U8(max0hi, max1hi, iptr0, iptr1, iptr2);

    HVX_VectorPair maxoe0vp0 = Q6_W_vdeal_VVR(max0hi, max0lo, -32);
    HVX_VectorPair maxoe0vp1 = Q6_W_vdeal_VVR(max1hi, max1lo, -32);

    for (int w = 0; w < wOut; w += 4) {
      if ((w & (tW - 1)) == 0) {
        optr = (HVX_Vector *)(ppout[0] + oTOffset);
        ppout += oTNextCol;
      }

      VERTICAL_MAX3S2_U8(max0_2, max1_2, iptr0, iptr1, iptr2);

      if (((2 * w + 8) & (tW - 1)) == 0) {
        iptr0 = (HVX_Vector *)(ppin[above] + iTOffset0);
        iptr1 = (HVX_Vector *)(ppin[0] + iTOffset1);
        iptr2 = (HVX_Vector *)(ppin[below] + iTOffset2);
        ppin += iTNextCol;
        ppin = std::min(ppin, ppinLast);
      }

      VERTICAL_MAX3S2_U8(max0_3, max1_3, iptr0, iptr1, iptr2);

      HVX_VectorPair maxoe1vp0 = Q6_W_vdeal_VVR(max0_3, max0_2, -32);
      HVX_VectorPair maxoe1vp1 = Q6_W_vdeal_VVR(max1_3, max1_2, -32);

      HVX_Vector out0 = Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 32);
      HVX_Vector out1 = Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 32);
      out0 =
          Q6_Vub_vmax_VubVub(out0, Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp0), Q6_V_lo_W(maxoe0vp0), 64));
      out0 =
          Q6_Vub_vmax_VubVub(out0, Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 64));
      out1 =
          Q6_Vub_vmax_VubVub(out1, Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp1), Q6_V_lo_W(maxoe0vp1), 64));
      out1 =
          Q6_Vub_vmax_VubVub(out1, Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 64));

      maxoe0vp0 = maxoe1vp0;
      maxoe0vp1 = maxoe1vp1;

      HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1), c2e);
      HVX_Vector a0E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
      HVX_Vector a0O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
      HVX_Vector a1E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
      HVX_Vector a1O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);
      a0                = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
      a1                = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);
      optr[0]           = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
      optr[vTW]         = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
      optr++;
    }
    iTOffset0 = (iTOffset0 + (4 << (11 - bTH))) & 0x7ff;
    iTOffset1 = (iTOffset1 + (4 << (11 - bTH))) & 0x7ff;
    iTOffset2 = (iTOffset2 + (4 << (11 - bTH))) & 0x7ff;
    oTOffset  = (oTOffset + (2 << (11 - bTH))) & 0x7ff;
  }
}

void maxpoolb3x3tSliceStride2x2N00(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  maxpoolb3x3wtSliceStride2N00(outdata, indata, offsets, scalingParams, 8);
}

void maxpoolh3x3tSliceStride2x2N00(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e       = (-1 << (shifte + 8)) | (1 << shifte);
  c2e               = Q6_R_combine_RlRl(c2e, c2e);
  HVX_Vector vscale = Q6_Vh_vsplat_R(scale);

  HVX_Vector inoff  = Q6_Vh_vsplat_R(inOffset);
  HVX_Vector outoff = Q6_Vh_vsplat_R(outOffset);
  outoff            = Q6_Vh_vasl_VhR(outoff, shift);
  const HVX_Vector *blockPtr;
  const HVX_Vector *blockBelowPtr;
  HVX_Vector *outPtr;
  int hRemain = hOut;

  for (int h = 0; h < hOut; h += 4, hRemain -= 4) {
    uint8_t **ppin  = indata->addr + ((h * 2) >> 3) * iTNextRow;
    uint8_t **ppout = outdata->addr + (h >> 3) * oTNextRow;

    size_t blockBelow = hRemain >= 4 ? iTNextRow : 0;

    blockPtr      = (const HVX_Vector *)ppin[0];
    blockBelowPtr = (const HVX_Vector *)ppin[blockBelow];

    HVX_Vector in0v0 = blockPtr[0];
    HVX_Vector in0v1 = blockPtr[2];
    HVX_Vector in0v2 = blockPtr[4];
    HVX_Vector in0v3 = blockPtr[6];
    HVX_Vector in0v4 = blockPtr[8];
    HVX_Vector in0v5 = blockPtr[10];
    HVX_Vector in0v6 = blockPtr[12];
    HVX_Vector in0v7 = blockPtr[14];
    HVX_Vector in0v8 = blockBelowPtr[0];

    HVX_Vector max0lo = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in0v0, in0v1), in0v2);
    HVX_Vector max0hi = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in0v2, in0v3), in0v4);
    HVX_Vector max0_2 = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in0v4, in0v5), in0v6);
    HVX_Vector max0_3 = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in0v6, in0v7), in0v8);

    for (int w = 0; w < wOut; w += 2) {
      HVX_Vector in1v0 = blockPtr[1];
      HVX_Vector in1v1 = blockPtr[3];
      HVX_Vector in1v2 = blockPtr[5];
      HVX_Vector in1v3 = blockPtr[7];
      HVX_Vector in1_4 = blockPtr[9];
      HVX_Vector in1_5 = blockPtr[11];
      HVX_Vector in1_6 = blockPtr[13];
      HVX_Vector in1_7 = blockPtr[15];
      HVX_Vector in1_8 = blockBelowPtr[1];

      HVX_Vector max1lo = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in1v0, in1v1), in1v2);
      HVX_Vector max1hi = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in1v2, in1v3), in1_4);
      HVX_Vector max1_2 = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in1_4, in1_5), in1_6);
      HVX_Vector max1_3 = Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(in1_6, in1_7), in1_8);

      HVX_VectorPair W_Odds_Evens0 = Q6_Wh_vshuffoe_VhVh(max1lo, max0lo);
      HVX_VectorPair W_Odds_Evens1 = Q6_Wh_vshuffoe_VhVh(max1hi, max0hi);
      HVX_VectorPair W_Odds_Evens2 = Q6_Wh_vshuffoe_VhVh(max1_2, max0_2);
      HVX_VectorPair W_Odds_Evens3 = Q6_Wh_vshuffoe_VhVh(max1_3, max0_3);

      HVX_Vector out0 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(W_Odds_Evens0), Q6_V_lo_W(W_Odds_Evens0));
      HVX_Vector out1 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(W_Odds_Evens1), Q6_V_lo_W(W_Odds_Evens1));
      HVX_Vector out2 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(W_Odds_Evens2), Q6_V_lo_W(W_Odds_Evens2));
      HVX_Vector out3 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(W_Odds_Evens3), Q6_V_lo_W(W_Odds_Evens3));

      if ((wOut - w) >= 2) ppin += iTNextCol;
      blockPtr      = (const HVX_Vector *)ppin[0];
      blockBelowPtr = (const HVX_Vector *)ppin[blockBelow];

      HVX_Vector next_in0v0 = blockPtr[0];
      HVX_Vector next_in0v1 = blockPtr[2];
      HVX_Vector next_in0v2 = blockPtr[4];
      HVX_Vector next_in0v3 = blockPtr[6];
      HVX_Vector next_in0v4 = blockPtr[8];
      HVX_Vector next_in0v5 = blockPtr[10];
      HVX_Vector next_in0v6 = blockPtr[12];
      HVX_Vector next_in0v7 = blockPtr[14];
      HVX_Vector next_in0v8 = blockBelowPtr[0];

      HVX_Vector nextMax0v0 =
          Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(next_in0v0, next_in0v1), next_in0v2);
      HVX_Vector nextMax0v1 =
          Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(next_in0v2, next_in0v3), next_in0v4);
      HVX_Vector nextMax0v2 =
          Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(next_in0v4, next_in0v5), next_in0v6);
      HVX_Vector nextMax0v3 =
          Q6_Vuh_vmax_VuhVuh(Q6_Vuh_vmax_VuhVuh(next_in0v6, next_in0v7), next_in0v8);

      HVX_Vector horiz0 = Q6_Vh_vshuffe_VhVh(nextMax0v0, max1lo);
      HVX_Vector horiz1 = Q6_Vh_vshuffe_VhVh(nextMax0v1, max1hi);
      HVX_Vector horiz2 = Q6_Vh_vshuffe_VhVh(nextMax0v2, max1_2);
      HVX_Vector horiz3 = Q6_Vh_vshuffe_VhVh(nextMax0v3, max1_3);

      out0 = Q6_Vuh_vmax_VuhVuh(out0, horiz0);
      out1 = Q6_Vuh_vmax_VuhVuh(out1, horiz1);
      out2 = Q6_Vuh_vmax_VuhVuh(out2, horiz2);
      out3 = Q6_Vuh_vmax_VuhVuh(out3, horiz3);

      HVX_VectorPair a0 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_VectorPair a1 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out1), c2e);
      HVX_VectorPair a2 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out2), c2e);
      HVX_VectorPair a3 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out3), c2e);

      HVX_Vector a0E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a0), vscale);
      HVX_Vector a0O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a0), vscale);
      HVX_Vector a1E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a1), vscale);
      HVX_Vector a1O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a1), vscale);
      HVX_Vector a2E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a2), vscale);
      HVX_Vector a2O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a2), vscale);
      HVX_Vector a3E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a3), vscale);
      HVX_Vector a3O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a3), vscale);

      a0E = Q6_Vw_vadd_VwVw_sat(a0E, outoff);
      a0O = Q6_Vw_vadd_VwVw_sat(a0O, outoff);
      a1E = Q6_Vw_vadd_VwVw_sat(a1E, outoff);
      a1O = Q6_Vw_vadd_VwVw_sat(a1O, outoff);
      a2E = Q6_Vw_vadd_VwVw_sat(a2E, outoff);
      a2O = Q6_Vw_vadd_VwVw_sat(a2O, outoff);
      a3E = Q6_Vw_vadd_VwVw_sat(a3E, outoff);
      a3O = Q6_Vw_vadd_VwVw_sat(a3O, outoff);

      if (w % 4 == 0) {
        outPtr = (HVX_Vector *)ppout[0] + (h % 8) * 2;
        ppout += oTNextCol;
      } else {
        outPtr++;
      }
      outPtr[0] = Q6_Vuh_vasr_VwVwR_rnd_sat(a0O, a0E, shift);
      outPtr[2] = Q6_Vuh_vasr_VwVwR_rnd_sat(a1O, a1E, shift);
      outPtr[4] = Q6_Vuh_vasr_VwVwR_rnd_sat(a2O, a2E, shift);
      outPtr[6] = Q6_Vuh_vasr_VwVwR_rnd_sat(a3O, a3E, shift);

      max0lo = nextMax0v0;
      max0hi = nextMax0v1;
      max0_2 = nextMax0v2;
      max0_3 = nextMax0v3;
    }
  }
}

void maxpoolb2x2wtSliceStride2N00(struct TileData *outdata,
                                  struct TileData *indata,
                                  uint32_t offsets,
                                  uint32_t scalingParams,
                                  const unsigned tH) {
  const size_t bTH = Q6_R_ct0_R(tH);  // log2(tH)
  const size_t tW  = 64 >> bTH;
  const size_t vTW = 16 >> bTH;

  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextRow = outdata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e = (-1 << (shifte + 8)) | (1 << shifte);
  c2e         = Q6_R_combine_RlRl(c2e, c2e);
  scale       = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  for (int h = 0; h < hOut; h += 2) {
    uint8_t **ppin  = indata->addr + ((2 * h) >> bTH) * iTNextRow;
    uint8_t **ppout = outdata->addr + (h >> bTH) * oTNextRow;

    int iTOffset0 = ((2 * h + 0) & (tH - 1)) << (11 - bTH);
    int iTOffset1 = ((2 * h + 2) & (tH - 1)) << (11 - bTH);
    int oTOffset  = ((1 * h + 0) & (tH - 1)) << (11 - bTH);
    int below     = iTOffset1 > iTOffset0 || (hOut - h) == 1 ? 0 : iTNextRow;

    HVX_Vector *iptr0, *iptr1, *optr;

    for (int w = 0; w < wOut; w += 4) {
      if (((2 * w) & (tW - 1)) == 0) {
        iptr0 = (HVX_Vector *)(ppin[0] + iTOffset0);
        iptr1 = (HVX_Vector *)(ppin[below] + iTOffset1);
        ppin += iTNextCol;
      }
      HVX_Vector in0v1 = iptr0[vTW];  // h1 w0
      HVX_Vector in0v3 = iptr1[vTW];  // h3 w0
      HVX_Vector in0v0 = *iptr0++;    // h0 w0
      HVX_Vector in0v2 = *iptr1++;    // h2 w0

      HVX_Vector max0lo = Q6_Vub_vmax_VubVub(in0v0, in0v1);  // max bt h0 h1 [h0 w0] [h1 w0]
      HVX_Vector max0hi = Q6_Vub_vmax_VubVub(in0v2, in0v3);  // max bt h2 h3 [h2 w0] [h3 w0]

      HVX_Vector in1v1 = iptr0[vTW];  // h1 w1
      HVX_Vector in1v3 = iptr1[vTW];  // h3 w1
      HVX_Vector in1v0 = *iptr0++;    // h0 w1
      HVX_Vector in1v2 = *iptr1++;    // h2 w1

      HVX_Vector max1lo = Q6_Vub_vmax_VubVub(in1v0, in1v1);  // max bt h0 h1 [h0 w1] [h1 w1]
      HVX_Vector max1hi = Q6_Vub_vmax_VubVub(in1v2, in1v3);  // max bt h2 h3 [h2 w0] [h3 w0]

      HVX_VectorPair max0 =
          Q6_W_vdeal_VVR(max1lo, max0lo, -32);  // [m04 m06 m00 m02 | m05 m07 m01 m03]
      HVX_VectorPair max1 =
          Q6_W_vdeal_VVR(max1hi, max0hi, -32);  // [m14 m16 m10 m12 | m15 m17 m11 m13]

      HVX_Vector out0 = Q6_Vub_vmax_VubVub(Q6_V_hi_W(max0), Q6_V_lo_W(max0));  // [M02 M03 M00 M01]
      HVX_Vector out1 = Q6_Vub_vmax_VubVub(Q6_V_hi_W(max1), Q6_V_lo_W(max1));  // [M12 M13 M10 M11]

      HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0),
                                           c2e);  // shift to [M00 M01 M02 M03]
      HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1),
                                           c2e);  // shift to [M10 M11 M12 M13]

      if ((w & (tW - 1)) == 0) {
        optr = (HVX_Vector *)(ppout[0] + oTOffset);
        ppout += oTNextCol;
      }

      HVX_Vector a0E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
      HVX_Vector a0O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
      HVX_Vector a1E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
      HVX_Vector a1O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);

      a0 = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
      a1 = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);

      optr[0]   = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
      optr[vTW] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
      optr++;
    }
  }
}

void maxpoolb2x2tSliceStride2x2N00(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  maxpoolb2x2wtSliceStride2N00(outdata, indata, offsets, scalingParams, 8);
}

void maxpoolh2x2tSliceStride2x2N00(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextRow = outdata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  const HVX_Vector *blockPtr;
  HVX_Vector *outPtr;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e       = (-1 << (shifte + 8)) | (1 << shifte);
  c2e               = Q6_R_combine_RlRl(c2e, c2e);
  HVX_Vector vscale = Q6_Vh_vsplat_R(scale);

  HVX_Vector inoff  = Q6_Vh_vsplat_R(inOffset);
  HVX_Vector outoff = Q6_V_vsplat_R(outOffset);
  outoff            = Q6_Vw_vasl_VwR(outoff, shift);

  for (int h = 0; h < hOut; h += 4) {
    uint8_t **ppin  = indata->addr + ((h * 2) >> 3) * iTNextRow;
    uint8_t **ppout = outdata->addr + (h >> 3) * oTNextRow;

    for (int w = 0; w < wOut; w += 2) {
      blockPtr = (const HVX_Vector *)ppin[0];

      HVX_Vector in0v0 = blockPtr[0];
      HVX_Vector in0v1 = blockPtr[2];
      HVX_Vector in0v2 = blockPtr[4];
      HVX_Vector in0v3 = blockPtr[6];
      HVX_Vector in0v4 = blockPtr[8];
      HVX_Vector in0v5 = blockPtr[10];
      HVX_Vector in0v6 = blockPtr[12];
      HVX_Vector in0v7 = blockPtr[14];

      HVX_Vector max0lo = Q6_Vuh_vmax_VuhVuh(in0v0, in0v1);
      HVX_Vector max0hi = Q6_Vuh_vmax_VuhVuh(in0v2, in0v3);
      HVX_Vector max0_2 = Q6_Vuh_vmax_VuhVuh(in0v4, in0v5);
      HVX_Vector max0_3 = Q6_Vuh_vmax_VuhVuh(in0v6, in0v7);

      HVX_Vector in1v0 = blockPtr[1];
      HVX_Vector in1v1 = blockPtr[3];
      HVX_Vector in1v2 = blockPtr[5];
      HVX_Vector in1v3 = blockPtr[7];
      HVX_Vector in1_4 = blockPtr[9];
      HVX_Vector in1_5 = blockPtr[11];
      HVX_Vector in1_6 = blockPtr[13];
      HVX_Vector in1_7 = blockPtr[15];

      HVX_Vector max1lo = Q6_Vuh_vmax_VuhVuh(in1v0, in1v1);
      HVX_Vector max1hi = Q6_Vuh_vmax_VuhVuh(in1v2, in1v3);
      HVX_Vector max1_2 = Q6_Vuh_vmax_VuhVuh(in1_4, in1_5);
      HVX_Vector max1_3 = Q6_Vuh_vmax_VuhVuh(in1_6, in1_7);

      HVX_VectorPair max0 = Q6_Wh_vshuffoe_VhVh(max1lo, max0lo);
      HVX_VectorPair max1 = Q6_Wh_vshuffoe_VhVh(max1hi, max0hi);
      HVX_VectorPair max2 = Q6_Wh_vshuffoe_VhVh(max1_2, max0_2);
      HVX_VectorPair max3 = Q6_Wh_vshuffoe_VhVh(max1_3, max0_3);

      HVX_Vector out0 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(max0), Q6_V_lo_W(max0));
      HVX_Vector out1 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(max1), Q6_V_lo_W(max1));
      HVX_Vector out2 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(max2), Q6_V_lo_W(max2));
      HVX_Vector out3 = Q6_Vuh_vmax_VuhVuh(Q6_V_hi_W(max3), Q6_V_lo_W(max3));

      HVX_VectorPair a0 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_VectorPair a1 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out1), c2e);
      HVX_VectorPair a2 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out2), c2e);
      HVX_VectorPair a3 = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(inoff, out3), c2e);

      HVX_Vector a0E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a0), vscale);
      HVX_Vector a0O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a0), vscale);
      HVX_Vector a1E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a1), vscale);
      HVX_Vector a1O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a1), vscale);
      HVX_Vector a2E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a2), vscale);
      HVX_Vector a2O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a2), vscale);
      HVX_Vector a3E = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_lo_W(a3), vscale);
      HVX_Vector a3O = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_V_hi_W(a3), vscale);

      a0E = Q6_Vw_vadd_VwVw(a0E, outoff);
      a0O = Q6_Vw_vadd_VwVw(a0O, outoff);
      a1E = Q6_Vw_vadd_VwVw(a1E, outoff);
      a1O = Q6_Vw_vadd_VwVw(a1O, outoff);
      a2E = Q6_Vw_vadd_VwVw(a2E, outoff);
      a2O = Q6_Vw_vadd_VwVw(a2O, outoff);
      a3E = Q6_Vw_vadd_VwVw(a3E, outoff);
      a3O = Q6_Vw_vadd_VwVw(a3O, outoff);

      // move to next crouton
      if (w % 4 == 0) {
        outPtr = (HVX_Vector *)ppout[0] + (h % 8) * 2;
        ppout += oTNextCol;
      } else {
        outPtr++;
      }
      out0 = Q6_Vuh_vasr_VwVwR_rnd_sat(a0O, a0E, shift);
      out1 = Q6_Vuh_vasr_VwVwR_rnd_sat(a1O, a1E, shift);
      out2 = Q6_Vuh_vasr_VwVwR_rnd_sat(a2O, a2E, shift);
      out3 = Q6_Vuh_vasr_VwVwR_rnd_sat(a3O, a3E, shift);

      outPtr[0] = out0;
      outPtr[2] = out1;
      outPtr[4] = out2;
      outPtr[6] = out3;

      ppin += iTNextCol;
    }
  }
}

void maxpoolb3x3tSliceStride1x1N11(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e = (-1 << (shifte + 8)) | (1 << shifte);
  c2e         = Q6_R_combine_RlRl(c2e, c2e);
  scale       = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  HVX_Vector **iAdrTab = (HVX_Vector **)indata->addr + iTNextRow;
  HVX_Vector **oAdrTab = (HVX_Vector **)outdata->addr;

  int iOffset0 = 7 * 2;
  int iOffset1 = 0 * 2;
  int iOffset2 = 2 * 2;

  HVX_Vector *outPtr;

  for (int h = 0; h < hOut; h += 2) {
    HVX_Vector **ppin     = iAdrTab + (h >> 3) * iTNextRow;
    HVX_Vector **ppout    = oAdrTab + (h >> 3) * oTNextRow;
    HVX_Vector **ppinLast = ppin + (iTNextRow - iTNextCol);

    size_t above = iOffset0 > iOffset1 ? -iTNextRow : 0;
    size_t below = iOffset2 < iOffset1 && (h + 1) < hOut ? iTNextRow : 0;

    HVX_Vector *iptr0 = ppin[above] + iOffset0;
    HVX_Vector *iptr1 = ppin[0] + iOffset1;
    HVX_Vector *iptr2 = ppin[below] + iOffset2;
    ppin += iTNextCol;

    HVX_VectorPair max0s = verticalMax3(iptr0 + 1, iptr1 + 1, iptr2 + 1);
    HVX_Vector max0lo    = Q6_V_lo_W(max0s);
    HVX_Vector max0hi    = Q6_V_hi_W(max0s);

    iptr0 = ppin[above] + iOffset0;
    iptr1 = ppin[0] + iOffset1;
    iptr2 = ppin[below] + iOffset2;
    ppin += iTNextCol;
    ppin = std::min(ppin, ppinLast);

    HVX_VectorPair max1s = verticalMax3(iptr0, iptr1, iptr2);
    HVX_Vector max1lo    = Q6_V_lo_W(max1s);
    HVX_Vector max1hi    = Q6_V_hi_W(max1s);

    for (int w = 0; w < wOut; w += 4) {
      if ((w & 4) == 0) {
        iptr0++;
        iptr1++;
        iptr2++;
        outPtr = (HVX_Vector *)ppout[0] + (h & 7) * 2;
        ppout += oTNextCol;
      } else {
        iptr0 = ppin[above] + iOffset0;
        iptr1 = ppin[0] + iOffset1;
        iptr2 = ppin[below] + iOffset2;
        ppin += iTNextCol;
        ppin = std::min(ppin, ppinLast);
        outPtr++;
      }

      HVX_VectorPair max2s = verticalMax3(iptr0, iptr1, iptr2);
      HVX_Vector max2lo    = Q6_V_lo_W(max2s);
      HVX_Vector max2hi    = Q6_V_hi_W(max2s);

      HVX_Vector out0 = Q6_Vub_vmax_VubVub(max1lo, Q6_V_vlalign_VVR(max1lo, max0lo, 1 * 32));
      HVX_Vector out1 = Q6_Vub_vmax_VubVub(max1hi, Q6_V_vlalign_VVR(max1hi, max0hi, 1 * 32));
      out0            = Q6_Vub_vmax_VubVub(out0, Q6_V_valign_VVR(max2lo, max1lo, 1 * 32));
      out1            = Q6_Vub_vmax_VubVub(out1, Q6_V_valign_VVR(max2hi, max1hi, 1 * 32));
      max0lo          = max1lo;
      max0hi          = max1hi;
      max1lo          = max2lo;
      max1hi          = max2hi;

      HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1), c2e);
      HVX_Vector a0E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
      HVX_Vector a0O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
      HVX_Vector a1E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
      HVX_Vector a1O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);
      a0                = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
      a1                = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);
      outPtr[0]         = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
      outPtr[2]         = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
    }
    iOffset0 = iOffset1 + 1 * 2;
    iOffset1 = iOffset2;
    iOffset2 = (iOffset2 + 2 * 2) & 15;
  }
}

inline HVX_VectorPair __attribute__((always_inline))
verticalMax5(HVX_Vector *iptr0, HVX_Vector *iptr1, HVX_Vector *iptr2) {
  HVX_Vector max0 = iptr0[0];
  HVX_Vector maxT = iptr0[2];
  maxT            = Q6_Vub_vmax_VubVub(maxT, iptr1[0]);
  maxT            = Q6_Vub_vmax_VubVub(maxT, iptr1[2]);
  maxT            = Q6_Vub_vmax_VubVub(maxT, iptr2[0]);
  HVX_Vector max1 = iptr2[2];
  max0            = Q6_Vub_vmax_VubVub(max0, maxT);
  max1            = Q6_Vub_vmax_VubVub(max1, maxT);
  return Q6_W_vcombine_VV(max1, max0);
}

void maxpoolb5x5tSliceStride1x1N22(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;
  size_t wIn  = indata->width + 6;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;
  int32_t scale     = (scalingParams & 0x0FFFF);
  int shifte        = (scalingParams >> 24) & 0xFF;
  int shift         = (scalingParams >> 16) & 0xFF;

  int32_t c2e = (-1 << (shifte + 8)) | (1 << shifte);
  c2e         = Q6_R_combine_RlRl(c2e, c2e);
  scale       = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  int iOffset0 = 6 * 2;
  int iOffset1 = 0 * 2;
  int iOffset2 = 2 * 2;

  HVX_Vector *outPtr;

  for (int h = 0; h < hOut; h += 2) {
    int w_remain       = wIn - 8;
    HVX_Vector **ppin  = (HVX_Vector **)indata->addr + ((h + 6) >> 3) * iTNextRow;
    HVX_Vector **ppout = (HVX_Vector **)outdata->addr + (h >> 3) * oTNextRow;

    size_t below1 = iOffset1 < iOffset0 ? iTNextRow : 0;
    size_t below2 = iOffset2 < iOffset0 ? iTNextRow : 0;

    HVX_Vector *iptr0 = ppin[0] + iOffset0;
    HVX_Vector *iptr1 = ppin[below1] + iOffset1;
    HVX_Vector *iptr2 = ppin[below2] + iOffset2;
    ppin += iTNextCol;

    HVX_VectorPair max0s = verticalMax5(iptr0 + 1, iptr1 + 1, iptr2 + 1);
    HVX_Vector max0lo    = Q6_V_lo_W(max0s);
    HVX_Vector max0hi    = Q6_V_hi_W(max0s);

    iptr0 = ppin[0] + iOffset0;
    iptr1 = ppin[below1] + iOffset1;
    iptr2 = ppin[below2] + iOffset2;
    if (w_remain > 8) {
      ppin += iTNextCol;
      w_remain -= 8;
    }

    HVX_VectorPair max1s = verticalMax5(iptr0, iptr1, iptr2);
    HVX_Vector max1lo    = Q6_V_lo_W(max1s);
    HVX_Vector max1hi    = Q6_V_hi_W(max1s);

    for (int w = 0; w < wOut; w += 4) {
      if ((w & 4) == 0) {
        iptr0++;
        iptr1++;
        iptr2++;
        outPtr = (HVX_Vector *)ppout[0] + (h & 7) * 2;
        ppout += oTNextCol;
      } else {
        iptr0 = ppin[0] + iOffset0;
        iptr1 = ppin[below1] + iOffset1;
        iptr2 = ppin[below2] + iOffset2;
        if (w_remain > 8) {
          ppin += iTNextCol;
          w_remain -= 8;
        }
        outPtr++;
      }

      HVX_VectorPair max2s = verticalMax5(iptr0, iptr1, iptr2);
      HVX_Vector max2lo    = Q6_V_lo_W(max2s);
      HVX_Vector max2hi    = Q6_V_hi_W(max2s);

      HVX_Vector out0 = max1lo;
      HVX_Vector out1 = max1hi;
      out0            = Q6_Vub_vmax_VubVub(out0, Q6_V_vlalign_VVR(max1lo, max0lo, 2 * 32));
      out1            = Q6_Vub_vmax_VubVub(out1, Q6_V_vlalign_VVR(max1hi, max0hi, 2 * 32));
      out0            = Q6_Vub_vmax_VubVub(out0, Q6_V_vlalign_VVR(max1lo, max0lo, 1 * 32));
      out1            = Q6_Vub_vmax_VubVub(out1, Q6_V_vlalign_VVR(max1hi, max0hi, 1 * 32));
      max0lo          = max1lo;
      max0hi          = max1hi;

      out0   = Q6_Vub_vmax_VubVub(out0, Q6_V_valign_VVR(max2lo, max1lo, 1 * 32));
      out1   = Q6_Vub_vmax_VubVub(out1, Q6_V_valign_VVR(max2hi, max1hi, 1 * 32));
      out0   = Q6_Vub_vmax_VubVub(out0, Q6_V_valign_VVR(max2lo, max1lo, 2 * 32));
      out1   = Q6_Vub_vmax_VubVub(out1, Q6_V_valign_VVR(max2hi, max1hi, 2 * 32));
      max1lo = max2lo;
      max1hi = max2hi;

      HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1), c2e);

      HVX_Vector a0E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
      HVX_Vector a0O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
      HVX_Vector a1E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
      HVX_Vector a1O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);
      a0             = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
      a1             = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);
      outPtr[0]      = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
      outPtr[2]      = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
    }
    iOffset0 = iOffset1;
    iOffset1 = iOffset2;
    iOffset2 = (iOffset2 + 2 * 2) & 15;
  }
}

inline HVX_VectorPair __attribute__((always_inline)) verticalMax5s2(HVX_Vector v0,
                                                                    HVX_Vector v1,
                                                                    HVX_Vector v2,
                                                                    HVX_Vector v3,
                                                                    HVX_Vector v4,
                                                                    HVX_Vector v5,
                                                                    HVX_Vector v6) {
  HVX_Vector maxT = Q6_Vub_vmax_VubVub(v2, v3);
  HVX_Vector max0 = Q6_Vub_vmax_VubVub(v0, v1);
  HVX_Vector max1 = Q6_Vub_vmax_VubVub(v5, v6);

  maxT = Q6_Vub_vmax_VubVub(maxT, v4);
  max0 = Q6_Vub_vmax_VubVub(max0, maxT);
  max1 = Q6_Vub_vmax_VubVub(max1, maxT);
  return Q6_W_vcombine_VV(max1, max0);
}

void maxpoolb5x5tSliceStride2x2(struct TileData *outdata,
                                struct TileData *indata,
                                uint32_t offsets,
                                uint32_t scalingParams,
                                uint32_t tileStarts) {
  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;
  int32_t scale     = (scalingParams & 0x0FFFF);
  int shifte        = (scalingParams >> 24) & 0xFF;
  int shift         = (scalingParams >> 16) & 0xFF;
  int32_t c2e       = (-1 << (shifte + 8)) | (1 << shifte);
  c2e               = Q6_R_combine_RlRl(c2e, c2e);
  scale             = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  HVX_Vector **iAdrTab = (HVX_Vector **)indata->addr + iTNextRow;
  HVX_Vector **oAdrTab = (HVX_Vector **)outdata->addr;

  HVX_Vector *iptr0, *iptr1, *iptr2, *outPtr;
  int iOffset0, iOffset1, iOffset2;

  if (tileStarts == 0x36) {
    iOffset0 = 6 * 2;
    iOffset1 = 0 * 2;
    iOffset2 = 4 * 2;

    for (int h = 0; h < hOut; h += 2) {
      HVX_Vector **ppin     = iAdrTab + (h >> 2) * iTNextRow;
      HVX_Vector **ppout    = oAdrTab + (h >> 3) * oTNextRow;
      HVX_Vector **ppinLast = ppin + (iTNextRow - iTNextCol);

      int above = iOffset1 < iOffset0 ? -iTNextRow : 0;
      int below = iOffset2 < iOffset1 && (h + 1) < hOut ? iTNextRow : 0;

      iptr0 = ppin[above] + iOffset0;
      iptr1 = ppin[0] + iOffset1;
      iptr2 = ppin[below] + iOffset2;
      ppin += iTNextCol;

      HVX_VectorPair max0s =
          verticalMax5s2(iptr0[1], iptr0[3], iptr1[1], iptr1[3], iptr1[5], iptr1[7], iptr2[1]);

      iptr0 = ppin[above] + iOffset0;
      iptr1 = ppin[0] + iOffset1;
      iptr2 = ppin[below] + iOffset2;
      ppin += iTNextCol;
      ppin = std::min(ppin, ppinLast);

      HVX_VectorPair max1s =
          verticalMax5s2(iptr0[0], iptr0[2], iptr1[0], iptr1[2], iptr1[4], iptr1[6], iptr2[0]);

      HVX_VectorPair maxoe0vp0 = Q6_W_vdeal_VVR(Q6_V_lo_W(max1s), Q6_V_lo_W(max0s), -32);
      HVX_VectorPair maxoe0vp1 = Q6_W_vdeal_VVR(Q6_V_hi_W(max1s), Q6_V_hi_W(max0s), -32);

      for (int w = 0; w < wOut; w += 4) {
        HVX_VectorPair max2s =
            verticalMax5s2(iptr0[1], iptr0[3], iptr1[1], iptr1[3], iptr1[5], iptr1[7], iptr2[1]);

        iptr0 = ppin[above] + iOffset0;
        iptr1 = ppin[0] + iOffset1;
        iptr2 = ppin[below] + iOffset2;
        ppin += iTNextCol;
        ppin = std::min(ppin, ppinLast);

        HVX_VectorPair max3s =
            verticalMax5s2(iptr0[0], iptr0[2], iptr1[0], iptr1[2], iptr1[4], iptr1[6], iptr2[0]);

        HVX_VectorPair maxoe1vp0 = Q6_W_vdeal_VVR(Q6_V_lo_W(max3s), Q6_V_lo_W(max2s), -32);
        HVX_VectorPair maxoe1vp1 = Q6_W_vdeal_VVR(Q6_V_hi_W(max3s), Q6_V_hi_W(max2s), -32);

        HVX_Vector out0 = Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp0), Q6_V_lo_W(maxoe0vp0), 32);
        HVX_Vector out1 = Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp1), Q6_V_lo_W(maxoe0vp1), 32);
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 32));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 32));
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp0), Q6_V_lo_W(maxoe0vp0), 64));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp1), Q6_V_lo_W(maxoe0vp1), 64));
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 64));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 64));
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp0), Q6_V_lo_W(maxoe0vp0), 96));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp1), Q6_V_lo_W(maxoe0vp1), 96));

        maxoe0vp0 = maxoe1vp0;
        maxoe0vp1 = maxoe1vp1;

        if ((w & 4) == 0) {
          outPtr = (HVX_Vector *)ppout[0] + (h & 7) * 2;
          ppout += oTNextCol;
        }

        HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
        HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1), c2e);

        HVX_Vector a0E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
        HVX_Vector a0O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
        HVX_Vector a1E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
        HVX_Vector a1O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);
        a0             = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
        a1             = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);
        outPtr[0]      = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
        outPtr[2]      = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
        outPtr++;
      }
      iOffset0 ^= 4 * 2;
      iOffset1 ^= 4 * 2;
      iOffset2 ^= 4 * 2;
    }

  } else if (tileStarts == 0x3f) {
    iOffset0 = 7 * 2;
    iOffset1 = 0 * 2;
    iOffset2 = 4 * 2;

    for (int h = 0; h < hOut; h += 2) {
      HVX_Vector **ppin     = iAdrTab + (h >> 2) * iTNextRow;
      HVX_Vector **ppout    = oAdrTab + (h >> 3) * oTNextRow;
      HVX_Vector **ppinLast = ppin + (iTNextRow - iTNextCol);

      int above = iOffset1 < iOffset0 ? -iTNextRow : 0;
      int below = iOffset2 < iOffset1 && (h + 1) < hOut ? iTNextRow : 0;

      iptr0 = ppin[above] + iOffset0;
      iptr1 = ppin[0] + iOffset1;
      iptr2 = ppin[below] + iOffset2;
      ppin += iTNextCol;

      HVX_VectorPair max0s =
          verticalMax5s2(iptr0[1], iptr1[1], iptr1[3], iptr1[5], iptr1[7], iptr2[1], iptr2[3]);

      iptr0 = ppin[above] + iOffset0;
      iptr1 = ppin[0] + iOffset1;
      iptr2 = ppin[below] + iOffset2;
      ppin += iTNextCol;
      ppin = std::min(ppin, ppinLast);

      HVX_VectorPair max1s =
          verticalMax5s2(iptr0[0], iptr1[0], iptr1[2], iptr1[4], iptr1[6], iptr2[0], iptr2[2]);

      HVX_VectorPair maxoe0vp0 = Q6_W_vdeal_VVR(Q6_V_lo_W(max1s), Q6_V_lo_W(max0s), -32);
      HVX_VectorPair maxoe0vp1 = Q6_W_vdeal_VVR(Q6_V_hi_W(max1s), Q6_V_hi_W(max0s), -32);

      for (int w = 0; w < wOut; w += 4) {
        HVX_VectorPair max2s =
            verticalMax5s2(iptr0[1], iptr1[1], iptr1[3], iptr1[5], iptr1[7], iptr2[1], iptr2[3]);

        iptr0 = ppin[above] + iOffset0;
        iptr1 = ppin[0] + iOffset1;
        iptr2 = ppin[below] + iOffset2;
        ppin += iTNextCol;
        ppin = std::min(ppin, ppinLast);

        HVX_VectorPair max3s =
            verticalMax5s2(iptr0[0], iptr1[0], iptr1[2], iptr1[4], iptr1[6], iptr2[0], iptr2[2]);

        HVX_VectorPair maxoe1vp0 = Q6_W_vdeal_VVR(Q6_V_lo_W(max3s), Q6_V_lo_W(max2s), -32);
        HVX_VectorPair maxoe1vp1 = Q6_W_vdeal_VVR(Q6_V_hi_W(max3s), Q6_V_hi_W(max2s), -32);

        HVX_Vector out0 = Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 32);
        HVX_Vector out1 = Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 32);
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp0), Q6_V_lo_W(maxoe0vp0), 64));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp1), Q6_V_lo_W(maxoe0vp1), 64));
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 64));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 64));
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp0), Q6_V_lo_W(maxoe0vp0), 96));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_lo_W(maxoe1vp1), Q6_V_lo_W(maxoe0vp1), 96));
        out0            = Q6_Vub_vmax_VubVub(out0,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp0), Q6_V_hi_W(maxoe0vp0), 96));
        out1            = Q6_Vub_vmax_VubVub(out1,
                                  Q6_V_valign_VVR(Q6_V_hi_W(maxoe1vp1), Q6_V_hi_W(maxoe0vp1), 96));

        maxoe0vp0 = maxoe1vp0;
        maxoe0vp1 = maxoe1vp1;

        if ((w & 4) == 0) {
          outPtr = (HVX_Vector *)ppout[0] + (h & 7) * 2;
          ppout += oTNextCol;
        }

        HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
        HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out1), c2e);

        HVX_Vector a0E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
        HVX_Vector a0O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
        HVX_Vector a1E = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
        HVX_Vector a1O = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);
        a0             = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
        a1             = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);
        outPtr[0]      = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
        outPtr[2]      = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
        outPtr++;
      }
      iOffset0 ^= 4 * 2;
      iOffset1 ^= 4 * 2;
      iOffset2 ^= 4 * 2;
    }
  }
}

inline HVX_Vector __attribute__((always_inline))
verticalMax7p2(HVX_Vector *iptr0, HVX_Vector *iptr1, HVX_Vector *iptr2) {
  HVX_Vector v0 = iptr0[0];
  HVX_Vector v1 = iptr0[2];
  HVX_Vector v2 = iptr1[0];
  HVX_Vector v3 = iptr1[2];
  HVX_Vector v4 = iptr1[4];
  HVX_Vector v5 = iptr1[6];
  HVX_Vector v6 = iptr2[0];

  HVX_Vector vmaxT = Q6_Vub_vmax_VubVub(v0, v1);
  vmaxT            = Q6_Vub_vmax_VubVub(vmaxT, v2);
  vmaxT            = Q6_Vub_vmax_VubVub(vmaxT, v3);
  vmaxT            = Q6_Vub_vmax_VubVub(vmaxT, v4);
  vmaxT            = Q6_Vub_vmax_VubVub(vmaxT, v5);
  vmaxT            = Q6_Vub_vmax_VubVub(vmaxT, v6);
  return vmaxT;
}

void maxpoolb7x7tSliceStride4x4N22(struct TileData *outdata,
                                   struct TileData *indata,
                                   uint32_t offsets,
                                   uint32_t scalingParams) {
  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;
  int32_t scale     = (scalingParams & 0x0FFFF);
  int shifte        = (scalingParams >> 24) & 0xFF;
  int shift         = (scalingParams >> 16) & 0xFF;
  int32_t c2e       = (-1 << (shifte + 8)) | (1 << shifte);
  c2e               = Q6_R_combine_RlRl(c2e, c2e);
  scale             = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  uint8_t **iAdrTab = (uint8_t **)indata->addr + iTNextRow;
  uint8_t **oAdrTab = (uint8_t **)outdata->addr;

  HVX_Vector *iptr0, *iptr1, *iptr2, *outPtr;

  int iOffset0 = 6 * 2 * 128;
  int iOffset1 = 0 * 2 * 128;
  int iOffset2 = 4 * 2 * 128;

  for (int h = 0; h < hOut; h++) {
    uint8_t **ppin     = iAdrTab + (h >> 1) * iTNextRow;
    uint8_t **ppout    = oAdrTab + (h >> 3) * oTNextRow;
    uint8_t **ppinLast = ppin + (iTNextRow - iTNextCol);

    int above = iOffset1 < iOffset0 ? -iTNextRow : 0;
    int below = iOffset2 < iOffset1 ? iTNextRow : 0;

    iptr0 = (HVX_Vector *)(ppin[above] + iOffset0);
    iptr1 = (HVX_Vector *)(ppin[0] + iOffset1);
    iptr2 = (HVX_Vector *)(ppin[below] + iOffset2);
    ppin += iTNextCol;

    HVX_Vector vmax0  = verticalMax7p2(iptr0 + 1, iptr1 + 1, iptr2 + 1);
    HVX_Vector maxT1T = Q6_Vub_vmax_VubVub(vmax0, Q6_V_vror_VR(vmax0, -32));

    iptr0 = (HVX_Vector *)(ppin[above] + iOffset0);
    iptr1 = (HVX_Vector *)(ppin[0] + iOffset1);
    iptr2 = (HVX_Vector *)(ppin[below] + iOffset2);
    ppin += iTNextCol;
    ppin = std::min(ppin, ppinLast);

    vmax0 = verticalMax7p2(iptr0, iptr1, iptr2);
    iptr0++;
    iptr1++;
    iptr2++;

    for (int w = 0; w < wOut; w += 4) {
      HVX_Vector vmax1 = verticalMax7p2(iptr0, iptr1, iptr2);

      iptr0 = (HVX_Vector *)(ppin[above] + iOffset0);
      iptr1 = (HVX_Vector *)(ppin[0] + iOffset1);
      iptr2 = (HVX_Vector *)(ppin[below] + iOffset2);
      ppin += iTNextCol;
      ppin             = std::min(ppin, ppinLast);
      HVX_Vector vmax2 = verticalMax7p2(iptr0, iptr1, iptr2);
      iptr0++;
      iptr1++;
      iptr2++;

      HVX_Vector vmax3 = verticalMax7p2(iptr0, iptr1, iptr2);

      iptr0 = (HVX_Vector *)(ppin[above] + iOffset0);
      iptr1 = (HVX_Vector *)(ppin[0] + iOffset1);
      iptr2 = (HVX_Vector *)(ppin[below] + iOffset2);
      ppin += iTNextCol;
      ppin             = std::min(ppin, ppinLast);
      HVX_Vector vmax4 = verticalMax7p2(iptr0, iptr1, iptr2);
      iptr0++;
      iptr1++;
      iptr2++;

      HVX_VectorPair vmax0426n1537 = Q6_W_vshuff_VVR(vmax1, vmax0, 32);
      HVX_VectorPair vmax8caen9dbf = Q6_W_vshuff_VVR(vmax3, vmax2, 32);

      HVX_VectorPair vmax048cn26ae =
          Q6_W_vshuff_VVR(Q6_V_lo_W(vmax8caen9dbf), Q6_V_lo_W(vmax0426n1537), 64);
      HVX_VectorPair vmax159dn37bf =
          Q6_W_vshuff_VVR(Q6_V_hi_W(vmax8caen9dbf), Q6_V_hi_W(vmax0426n1537), 64);

      HVX_Vector vmax048c = Q6_V_lo_W(vmax048cn26ae);
      HVX_Vector vmax26ae = Q6_V_hi_W(vmax048cn26ae);
      HVX_Vector vmax159d = Q6_V_lo_W(vmax159dn37bf);
      HVX_Vector vmax37bf = Q6_V_hi_W(vmax159dn37bf);
      HVX_Vector vmax48cZ = Q6_V_valign_VVR(vmax4, vmax048c, 32);

      HVX_Vector maxT0 = Q6_Vub_vmax_VubVub(vmax048c, vmax159d);
      HVX_Vector maxT1 = Q6_Vub_vmax_VubVub(vmax26ae, vmax37bf);

      HVX_Vector out0 = Q6_Vub_vmax_VubVub(maxT0, maxT1);
      out0            = Q6_Vub_vmax_VubVub(out0, vmax48cZ);
      out0            = Q6_Vub_vmax_VubVub(out0, Q6_V_vlalign_VVR(maxT1, maxT1T, 32));

      vmax0  = vmax4;
      maxT1T = maxT1;

      if ((w & 4) == 0) {
        outPtr = (HVX_Vector *)ppout[0] + (h & 7) * 2;
        ppout += oTNextCol;
      }

      HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, out0), c2e);
      HVX_Vector a0E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
      HVX_Vector a0O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);

      a0        = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
      outPtr[0] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
      outPtr++;
    }
    iOffset0 ^= 4 * 2 * 128;
    iOffset1 ^= 4 * 2 * 128;
    iOffset2 ^= 4 * 2 * 128;
  }
}

void maxpoolBSpecial(struct TileData *outdata,
                     struct TileData *indata,
                     uint32_t windowSize,
                     uint32_t stride,
                     uint32_t offsets,
                     uint32_t scalingparams,
                     uint32_t tileStarts) {
  int dOut = outdata->depth;
  for (int d = 0; d < dOut; d += 32) {
    if (windowSize == 3 && stride == 2 && tileStarts == 0)
      maxpoolb3x3tSliceStride2x2N00(outdata, indata, offsets, scalingparams);
    else if (windowSize == 3 && stride == 2 && tileStarts == 0x3f)
      maxpoolb3x3wtSliceStride2N11(outdata, indata, offsets, scalingparams, 8);
    else if (windowSize == 2 && stride == 2)
      maxpoolb2x2tSliceStride2x2N00(outdata, indata, offsets, scalingparams);
    else if (windowSize == 3 && stride == 1)
      maxpoolb3x3tSliceStride1x1N11(outdata, indata, offsets, scalingparams);
    else if (windowSize == 5 && stride == 1)
      maxpoolb5x5tSliceStride1x1N22(outdata, indata, offsets, scalingparams);
    else if (windowSize == 5 && stride == 2)
      maxpoolb5x5tSliceStride2x2(outdata, indata, offsets, scalingparams, tileStarts);
    else if (windowSize == 7 && stride == 4)
      maxpoolb7x7tSliceStride4x4N22(outdata, indata, offsets, scalingparams);

    indata->addr++;
    outdata->addr++;
  }
}

void maxpoolHSpecial(struct TileData *outdata,
                     struct TileData *indata,
                     uint32_t windowSize,
                     uint32_t stride,
                     uint32_t offsets,
                     uint32_t scalingparams,
                     uint32_t tileStarts) {
  int dOut = outdata->depth;
  for (int d = 0; d < dOut; d += 32) {
    if (windowSize == 3 && stride == 2)
      maxpoolh3x3tSliceStride2x2N00(outdata, indata, offsets, scalingparams);
    else if (windowSize == 2 && stride == 2)
      maxpoolh2x2tSliceStride2x2N00(outdata, indata, offsets, scalingparams);

    indata->addr++;
    outdata->addr++;
  }
}

static void verticalMaxU8(HVX_Vector *optr,
                          uint8_t **ppin,
                          size_t iTNextCol,
                          size_t iTNextRow,
                          int hWindow,
                          int hstart,
                          int wstart,
                          int hOut,  // 2, or 1
                          int wOut,
                          const unsigned tH,
                          const unsigned bTH,
                          const unsigned tW,
                          const unsigned vTW) {
  int wTotal = wOut + (wstart & 3);
  int woff   = (wstart & (tW - 1) & ~3) << 5;
  HVX_Vector max0tPrev, max1tPrev, max0t, max1t;

  for (int w = 0; w < wTotal; w += 4) {
    HVX_Vector *iptr = (HVX_Vector *)(ppin[0] + ((hstart & (tH - 1)) << (11 - bTH)) + woff);
    uint8_t **ppinT  = ppin + iTNextRow;

    max1t         = Q6_V_vzero();
    HVX_Vector x0 = *iptr;
    iptr += vTW;

    for (int hh = 1; hh < hWindow; hh++) {
      if (((hh + hstart) & (tH - 1)) == 0) {
        iptr = (HVX_Vector *)(ppinT[0] + woff);
        ppinT += iTNextRow;
      }
      max1t = Q6_Vub_vmax_VubVub(max1t, *iptr);
      iptr += vTW;
    }

    max0t = Q6_Vub_vmax_VubVub(x0, max1t);

    if (hOut > 1) {
      if (((hWindow + hstart) & (tH - 1)) == 0) {
        iptr = (HVX_Vector *)(ppinT[0] + woff);
      }
      max1t = Q6_Vub_vmax_VubVub(max1t, *iptr);
    }

    woff = (woff + 128) & (tW * 32 - 1);
    if (woff == 0) ppin += iTNextCol;

    HVX_Vector vmax0 = Q6_V_vlalign_VVR(max0t, max0tPrev, -wstart * 32);
    HVX_Vector vmax1 = Q6_V_vlalign_VVR(max1t, max1tPrev, -wstart * 32);

    if ((wstart & 3) == 0 || w > 0) {
      *optr++ = vmax0;
      *optr++ = vmax1;
      wOut -= 4;
    }
    max0tPrev = max0t;
    max1tPrev = max1t;
  }
  if (wOut > 0) {
    optr[0] = Q6_V_vlalign_VVR(max0t, max0tPrev, -wstart * 32);
    optr[1] = Q6_V_vlalign_VVR(max1t, max1tPrev, -wstart * 32);
  }
}

void maxpoolNxMBStride1(struct TileData *outdata,
                        struct TileData *indata,
                        const unsigned tH,
                        uint32_t hWindow,
                        uint32_t wWindow,
                        uint32_t offsets,
                        uint32_t scalingParams,
                        uint32_t starts,
                        void *interm) {
  const unsigned bTH = Q6_R_ct0_R(tH);
  const unsigned tW  = 64 >> bTH;
  const unsigned vTW = 16 >> bTH;

  size_t iTNextCol = indata->offsetTCol;
  size_t iTNextRow = indata->offsetTRow;
  size_t oTNextCol = outdata->offsetTCol;
  size_t oTNextRow = outdata->offsetTRow;

  size_t hOut = outdata->height;
  size_t wOut = outdata->width;
  size_t dOut = outdata->depth;

  int32_t inOffset  = offsets & 0x0FFFF;
  int32_t outOffset = offsets >> 16;

  int32_t scale = (scalingParams & 0x0FFFF);
  int shifte    = (scalingParams >> 24) & 0xFF;
  int shift     = (scalingParams >> 16) & 0xFF;

  int32_t c2e = (-1 << (shifte + 8)) | (1 << shifte);
  c2e         = Q6_R_combine_RlRl(c2e, c2e);
  scale       = Q6_R_combine_RlRl(scale, scale);

  HVX_Vector inoff       = Q6_Vb_vsplat_R(inOffset);
  HVX_Vector outoff      = Q6_Vh_vsplat_R(outOffset);
  outoff                 = Q6_Vh_vasl_VhR(outoff, shift);
  HVX_VectorPair outoffs = Q6_W_vcombine_VV(outoff, outoff);

  uint8_t **iAdrTab  = (uint8_t **)indata->addr;
  uint8_t **oAdrTab  = (uint8_t **)outdata->addr;
  HVX_Vector *ptrMax = (HVX_Vector *)interm;

  int xstart = starts & (tW - 1);
  int ystart = (starts >> (6 - bTH)) & (tH - 1);

  for (int d = 0; d < dOut; d += 32)
    for (int h = 0; h < hOut; h += 2) {
      int nrows       = Q6_R_min_RR(hOut - h, 2);
      uint8_t **ppin  = iAdrTab + (d >> 5) + ((h + ystart) >> bTH) * iTNextRow;
      uint8_t **ppout = oAdrTab + (d >> 5) + (h >> bTH) * oTNextRow;
      int oTOffset    = (h & (tH - 1)) << (11 - bTH);

      for (int w = 0; w < wOut; w += 32) {
        int ncols       = Q6_R_min_RR(wOut - w, 32);
        uint8_t **ppin0 = ppin + ((w + xstart) >> (6 - bTH)) * iTNextCol;
        //
        // vertical filtering
        verticalMaxU8(ptrMax,
                      ppin0,
                      iTNextCol,
                      iTNextRow,
                      hWindow,
                      h + ystart,
                      xstart,
                      nrows,
                      ncols + wWindow - 1,
                      tH,
                      bTH,
                      tW,
                      vTW);

        HVX_Vector *iptr = ptrMax;
        HVX_Vector *optr;

        for (int wk = 0; wk < ncols; wk += 4) {
          if ((wk & (tW - 1)) == 0) {
            optr = (HVX_Vector *)(ppout[0] + oTOffset);
            ppout += oTNextCol;
          }
          //
          // horizontal filtering
          HVX_Vector max0 = Q6_V_vzero();
          HVX_Vector max1 = Q6_V_vzero();

          HVX_Vector x0v0   = *iptr++;
          HVX_Vector x1v0   = *iptr++;
          HVX_Vector *iptrT = iptr;

          for (int ww = wWindow; ww > 0; ww -= 4) {
            int wwk         = Q6_R_min_RR(4 * 32, ww * 32);
            HVX_Vector x0v1 = *iptrT++;
            HVX_Vector x1v1 = *iptrT++;

            for (int k = 0; k < wwk; k += 32) {
              HVX_Vector x0k = Q6_V_valign_VVR(x0v1, x0v0, k);
              HVX_Vector x1k = Q6_V_valign_VVR(x1v1, x1v0, k);
              max0           = Q6_Vub_vmax_VubVub(max0, x0k);
              max1           = Q6_Vub_vmax_VubVub(max1, x1k);
            }
            x0v0 = x0v1;
            x1v0 = x1v1;
          }
          //
          // requantize if required
          if (scalingParams != 0) {
            HVX_VectorPair a0 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, max0), c2e);
            HVX_VectorPair a1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(inoff, max1), c2e);
            HVX_Vector a0E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a0), scale);
            HVX_Vector a0O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a0), scale);
            HVX_Vector a1E    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(a1), scale);
            HVX_Vector a1O    = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(a1), scale);
            a0                = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a0O, a0E), outoffs);
            a1                = Q6_Wh_vadd_WhWh_sat(Q6_W_vcombine_VV(a1O, a1E), outoffs);
            max0              = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a0), Q6_V_lo_W(a0), shift);
            max1              = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(a1), Q6_V_lo_W(a1), shift);
          }
          optr[0]   = max0;
          optr[vTW] = max1;
          optr++;
        }
      }
    }
}

template <bool sameShape, typename T_Ttype>
int maxpoolImpl(T_Ttype &out,
                const T_Ttype &in,
                const TensorShape<4> &windowShape,
                const TensorShape<4> &strideShape) {
  auto [bIn, hIn, wIn, dIn]                 = in.dims();
  auto [bStride, hStride, wStride, dStride] = strideShape.dims();
  auto [bWindow, hWindow, wWindow, dWindow] = windowShape.dims();
  debuglog("maxpool execute... dims=%zdx%zdx%zdx%zd", bIn, hIn, wIn, dIn);
  debuglog("maxpool execute... dims=%zdx%zdx%zdx%zd", bWindow, hWindow, wWindow, dWindow);
  warnlog("maxpool using reference.... %s", __PRETTY_FUNCTION__);
  // debuglog("in=%p out=%p window=%p stride=%p
  // sameShape=%d",&in,&out,&windowShape,&strideShape,sameShape);

  const auto [bOut, hOut, wOut, dOut] = out.dims();
  size_t hAdj                         = sameShape ? (hWindow - 1) / 2 : 0;
  size_t wAdj                         = sameShape ? (wWindow - 1) / 2 : 0;

  for (Idx b = 0; b < bOut; b++) {
    for (Idx h = 0; h < hOut; h++) {
      for (Idx w = 0; w < wOut; w++) {
        for (Idx d = 0; d < dOut; d++) {
          const int inHOrigin  = (h * hStride) - hAdj;
          const int inWOrigin  = (w * wStride) - wAdj;
          float maxval         = std::numeric_limits<float>::lowest();
          const int filtHStart = std::max(0, -inHOrigin);
          const int filtWStart = std::max(0, -inWOrigin);
          const int filtHEnd   = std::min(hWindow, hIn - inHOrigin);
          const int filtWEnd   = std::min(wWindow, wIn - inWOrigin);
          for (int hh = filtHStart; hh < filtHEnd; hh++) {
            for (int ww = filtWStart; ww < filtWEnd; ww++) {
              const int inH = inHOrigin + hh;
              const int inW = inWOrigin + ww;
              float inval   = in(b, inH, inW, d);
              maxval        = std::max(maxval, inval);
            }
          }
          out(b, h, w, d) = maxval;
        }
      }
    }
  }
  return GraphStatus::Success;
}

template <typename T>
int maxpoolOpt(T &out,
               const T &in,
               const TensorShape<4> &windowShape,
               const TensorShape<4> &strideShape) {
  auto [bIn, hIn, wIn, dIn]                   = in.dims();
  auto [windowB, windowH, windowW, window_d]  = windowShape.dims();
  auto [stride_b, strideH, strideW, stride_d] = strideShape.dims();

  size_t bOut      = bIn;
  size_t hOut      = stride_divide(hIn - windowH + 1, strideH);
  size_t wOut      = stride_divide(wIn - windowW + 1, strideW);
  size_t dOut      = dIn;
  size_t outDims[] = {bOut, hOut, wOut, dOut};
  out.set_dims(outDims);

  const float scaleF = in.interface_scale() * out.interface_scale_recip();
  //    const float scaleF = in.interface_scale()/out.interface_scale();

  bool maxpoolDone = false;

  if (scaleF < 64.0f && scaleF >= 0.00390625f &&  // scaleF in [2^-8,2^6); otherwise need 32-bit
                                                  // HVX implementation
      strideH == strideW) {
    const auto [inTNextBat, inTNextRow, inTNextCol, inTNextDepth]     = in.tile_strides();
    const auto [outTNextBat, outTNextRow, outTNextCol, outTNextDepth] = out.tile_strides();
    const int32_t inOffset                                            = in.interface_offset();
    const int32_t outOffset                                           = out.interface_offset();

    uint32_t start = uint32_t(in.get_raw_addr(0, 0, 0, 0) - in.block_ptr(0, 0, 0, 0));

    if constexpr (sizeof(typename T::element_type) == 2) {
      start |= (start & 1) << 5;
    }
    start >>= 5;

    struct TileData tin = TILEDATASETUP(in.blocktab_ptr(), inTNextCol, inTNextRow, hIn, wIn, dIn);
    dcfetch_block(tin.addr, inTNextBat * sizeof(tin.addr[0]));

    struct TileData tout =
        TILEDATASETUP(out.blocktab_ptr(), outTNextCol, outTNextRow, hOut, wOut, dOut);
    dcfetch_block(tout.addr, outTNextBat * sizeof(tout.addr[0]));

    uint32_t scalingparams = get_scaling_params(scaleF, 6, 7);
    uint32_t offsets       = (outOffset << 16) | (inOffset & 0x0FFFF);

    if constexpr (std::is_base_of<LayoutCrouton_8, T>::value) {
      if ((windowH == 3 && windowW == 3 && strideW == 2 && (start == 0 || start == 0x3f)) ||
          (windowH == 2 && windowW == 2 && strideW == 2 && start == 0) ||
          (windowH == 3 && windowW == 3 && strideW == 1 && start == 0x3f) ||
          (windowH == 5 && windowW == 5 && strideW == 1 && start == 0x36) ||
          (windowH == 5 && windowW == 5 && strideW == 2 && (start == 0x36 || start == 0x3f)) ||
          (windowH == 7 && windowW == 7 && strideW == 4 && start == 0x36)) {
        maxpoolBSpecial(&tout, &tin, windowW, strideW, offsets, scalingparams, start);
        maxpoolDone = true;
      }
    } else if constexpr (std::is_base_of<LayoutCrouton_16, T>::value &&
                         std::is_same<uint16_t, typename T::element_type>::value) {
      if ((windowH == 3 && windowW == 3 && strideW == 2 && start == 0) ||
          (windowH == 2 && windowW == 2 && strideW == 2 && start == 0)) {
        maxpoolHSpecial(&tout, &tin, windowW, strideW, offsets, scalingparams, start);
        maxpoolDone = true;
      }
    }

    if constexpr (std::is_base_of<LayoutCrouton_8, T>::value) {
      constexpr unsigned tH = T::layout.ChunkSizes[1];
      if (!maxpoolDone && windowW < 32 && strideW == 1) {
        // buffer for processing data of (h,w,d)=2x32x32 with windowW up to 32
        tileExt::tile_buffers<1 + 1> tile_bufs;
        void *buf = tile_bufs.buf(0);
        maxpoolNxMBStride1(&tout, &tin, tH, windowH, windowW, offsets, scalingparams, start, buf);
        maxpoolDone = true;
      }
    }
  }

  if (!maxpoolDone) maxpoolImpl<false, T>(out, in, windowShape, strideShape);

  return GraphStatus::Success;
}

/* cost functions */

static float maxpoolCost(const Op *op) {
  auto [winB, winH, winW, winD] = op->get_input(1)->dims();
  auto [outB, outH, outW, outD] = op->get_output(0)->dims();

  float cost = float(outB * outH * outW * outD * winH * winW);
  logmsg(2, "Calculating cost=%f", cost);
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_MaxPool);