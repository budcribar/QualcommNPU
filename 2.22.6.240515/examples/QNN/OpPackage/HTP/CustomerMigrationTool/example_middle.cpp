//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(ExampleOpPackage)

// This is a demonstration OpPackage to exercise the behavior
// of the customer migration tool!
// NOTE: This is the MIDDLE file. It contains FIXES to
// example_start.cpp based SOLELY on the Documentation's suggestions
// ALL FIXES ARE NOTED BY "FIX" comments

// We'll illustrate the tool through 2 common kernels you might implement:
// refImpl - a reference implementation of a new kernel that works on
// all QUint8/QUint16/Int32 type tensors in any format
//
// fastImpl - a specialized QUint8 version of a kernel that expects
//   certain inputs to be in Crouton format and produces an output
//   in Flat format (CAN be either TCM or MainMemory)
//
// NOTE: Both implementations accept 2 input tensors AND a 3rd
// scalar value (exec_mode) that (though not shown) we assume
// affects the control flow of the kernel internally and will
// be a single int32
template <typename OutType, typename InType>
GraphStatus refImpl(OutType& out,
                    const InType& in_0,
                    const InType& in_1,
                    // FIX: We know it'll be Int32, so remove ultra-generic Tensor in favor of
                    // Int32, flat+main memory
                    const Int32Tensor& exec_mode);

template <typename OutType, typename InTypeA, typename InTypeB>
GraphStatus fastImpl(OutType& out,
                     const InTypeA& in_0,
                     const InTypeB& in_1,
                     // FIX: We know it'll be Int32, so remove ultra-generic Tensor in favor of
                     // Int32, flat+main memory
                     const Int32Tensor& exec_mode);

// REGISTRATION
// We'll register both sorts of impls to different OpStrs

// Slow implementation maps to RefExampleOp
// FIX: Avoid usage of the ultra-generic `Tensor` unless we're quite sure we can support that
// ALSO -> `Tensor`-type outputs are treated as Flat+MM in our centralized layout and placement
// pass, so it's critical you enumerate all the supported tensor types if you support Crouton
// or TCM-type tensors in your output!
DEF_PACKAGE_OP((refImpl<QUint8CroutonTensor, QUint8CroutonTensor>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<QUint8CroutonTensor_TCM, QUint8CroutonTensor_TCM>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<QUint16CroutonTensor, QUint16CroutonTensor>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<QUint16CroutonTensor_TCM, QUint16CroutonTensor_TCM>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<Int32CroutonTensor, Int32CroutonTensor>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<Int32CroutonTensor_TCM, Int32CroutonTensor_TCM>), "RefExampleOp")

DEF_PACKAGE_OP((refImpl<QuantUint8Tensor, QuantUint8Tensor>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<QuantUint8Tensor_TCM, QuantUint8Tensor_TCM>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<QuantUint16Tensor, QuantUint16Tensor>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<QuantUint16Tensor_TCM, QuantUint16Tensor_TCM>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<Int32Tensor, Int32Tensor>), "RefExampleOp")
DEF_PACKAGE_OP((refImpl<Int32Tensor_TCM, Int32Tensor_TCM>), "RefExampleOp")

// Fast Implementation maps to "FastExampleOp"
DEF_PACKAGE_OP((fastImpl<QuantInt8Tensor_TCM, QUint8CroutonTensor_TCM, QuantUint8Tensor_TCM>),
               "FastExampleOp")
// FIX: nothing about our fastImpl's implementation (based on the description) prevents
// it from running out of MainMemory. Our centralized layout and placement pass is biased
// toward running out of TCM *WHEN POSSIBLE*, but if your memory footprint is too large, or
// we expect a lot of other ops to be running concurrently and impacting our TCM memory usage,
// it's important we have a FALLBACK implementation that is still quite fast, but runs on
// tensors out of MainMemory.
// --> We generally believe it's better to experience a slowdown due to different tensor placement
//     than to fail because of a lack of TCM!
DEF_PACKAGE_OP((fastImpl<QuantInt8Tensor, QUint8CroutonTensor, QuantUint8Tensor>), "FastExampleOp")

// OPTIMIZATION
// "If the dtypes of the inputs look right, and the output is QUINT8, let's
// pick the specialized option that we expect to have TCM outputs
DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("ExampleOp", "in_0", "in_1", "exec_mode"),
                         AND(IS_QUINT8("in_0"), IS_QUINT8("in_1"), IS_QUINT8("*")),
                         Op("FastExampleOp", "in_0", "in_1", "exec_mode"))

// "If we don't have the earlier DTYPES that allowed a fast impl, then
// let's transform all Op("ExampleOp") instances to the reference one"
DEF_PACKAGE_OPTIMIZATION(EARLY + 1,
                         Op("ExampleOp", "in_0", "in_1", "exec_mode"),
                         OK,
                         Op("RefExampleOp", "in_0", "in_1", "exec_mode"))

#define ESTIMATE_SIZE_OF(OPREF) \
  MUL(DIM_OF(OPREF, 0), DIM_OF(OPREF, 1), DIM_OF(OPREF, 2), DIM_OF(OPREF, 3))

// FIX: Since we know that we now support FastExampleOp both in MainMemory AND in
// TCM, let's remove the below optimization.

// "Toward middle of optimization, if we have an instance of "FastExampleOp" that looks
// like it ought to fit into our memory constraints, then wrap it in a flat_from_vtcm to
// indicate the format of the output and the location of the tensor
// DEF_PACKAGE_OPTIMIZATION(MIDDLE,
//     Op("FastExampleOp", "in_0", "in_1", "exec_mode"),
//     LT(ADD(ESTIMATE_TENSOR_SIZE("*"),ESTIMATE_TENSOR_SIZE("in_0"), ESTIMATE_TENSOR_SIZE("in_1")),
//     TCM_MAXTENSOR_SIZE), Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
//         Op("FastExampleOp", "in_0", "in_1", "exec_mode"))
// )

// FIX: Again, since we have support in FastExampleOp for non-TCM tensors,
// we can safely keep the optimized version and can remove the below optimization.

// "OTHERWISE, any remaining FastExampleOps can go back to reference impl"
// DEF_PACKAGE_OPTIMIZATION(MIDDLE+1,
//     Op("FastExampleOp", "in_0", "in_1", "exec_mode"),
//     NOT(PRODUCER_FOR("*", FROM_DEFAULT_PACKAGE("flat_from_vtcm"))),
//     Op("RefExampleOp", "in_0", "in_1", "exec_mode")
// )

// There is a special case we want to handle toward the end of optimization
// for the ref impl that requires us to add another wrapper op around an input
// FIX: Do NOT use the DEF_OPT macro or internal-only pass names. Please stick to
// those exposed for customer use (GRAPH_CLEANUP, EARLY, MIDDLE, LATE, and LAYOUT_AND_PLACEMENT)
//  DEF_OPT(TCM_MIGRATION+50,
DEF_PACKAGE_OPTIMIZATION(
    LATE,
    Op("RefExampleOp", "in_0", "in_1", "exec_mode"),
    SPECIAL_CONDITION,
    Op("SpecialCaseExampleOp", "in_0", Op("AdditionalTransform", "in_1"), "exec_mode"))

// Later on, convert our Special case op back to "RefExampleOp"
// since we have an implementation for that and will have wrapped one of its inputs
DEF_PACKAGE_OPTIMIZATION(LATE + 200,
                         Op("SpecialCaseExampleOp", "in_0", "special_in", "exec_mode"),
                         OK,
                         Op("RefExampleOp", "in_0", "special_in", "exec_mode"))

// "As far back in the optimization as we can go, let's make sure we load in
// the inputs to FastExampleOp to their appropriate formats
DEF_PACKAGE_OPTIMIZATION(LATE + 900,
                         Op("FastExampleOp", "in_0", "in_1", "exec_mode"),
                         OK,
                         Op("FastExampleOp",
                            // FIX: Since we know that these inputs may actually be in MainMemory,
                            // let's just convert them to formatting ops, rather than include data
                            // movement ones Op(FROM_DEFAULT_PACKAGE("crouton_to_vtcm"),
                            // Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "in_0")),
                            // Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"),
                            // Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"),"in_1")),
                            Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "in_0"),
                            Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "in_1"),
                            "exec_mode"))

END_PKG_OP_DEFINITION(ExampleOpPackage)