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
// NOTE: This is the FINAL file. It contains fixes derived from
// the generated suggestions from running the customer_migration.py
// script on example_middle
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
                    const Int32Tensor& exec_mode);

template <typename OutType, typename InTypeA, typename InTypeB>
GraphStatus fastImpl(OutType& out,
                     const InTypeA& in_0,
                     const InTypeB& in_1,
                     const Int32Tensor& exec_mode);

// REGISTRATION
// We'll register both sorts of impls to different OpStrs

// Slow implementation maps to RefExampleOp
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

// There is a special case we want to handle toward the end of optimization
// for the ref impl that requires us to add another wrapper op around an input
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

// FIX: While we *could* move this rule to our LAYOUT_AND_PLACEMENT pass
// and let the centralized pass try to infer our tensor properties
// it's actually easier to just remove it altogether, since we expect
// information about the layout of our tensors to be accounted for
// by our DEF_TENSOR_PROPERTIES. Thus, we expect any needed ForceFormat*s
// or *to/*from_vtcm ops to already be inserted after LAYOUT_AND_PLACEMENT
// DEF_PACKAGE_OPTIMIZATION(LATE+900,
//     Op("FastExampleOp", "in_0", "in_1", "exec_mode"),
//     OK,
//     Op("FastExampleOp",
//         Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "in_0"),
//         Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "in_1"),
//         "exec_mode"
//     )
// )

// DEF_TENSOR_PROPERTIES
// FIX: These are the synthesized DEF_TENSOR_PROPERTIES based on the registrations
// available for each impl
DEF_TENSOR_PROPERTIES(Op("RefExampleOp", "in0", "in1", "in2"), Flat("in2"), MainMemory("in2"))

DEF_TENSOR_PROPERTIES(Op("FastExampleOp", "in0", "in1", "in2"),
                      Flat("*", "in1", "in2"),
                      Crouton("in0"),
                      MainMemory("in2"))

// TRANSIENT OPS
// FIX: Since "SpecialCaseExampleOp" doesn't have any registered implementations AND
// will be completely replaced by another op ("RefExampleOp") at the end of optimization BUT
// does require tensor information (since it can exist at the end of our
// LAYOUT_AND_PLACEMENT phase), we borrow the contents of the DEF_TENSOR_PROPERTIES
// of the op(s) it'll turn into. In this case, "RefExampleOp"
DEF_TENSOR_PROPERTIES(Op("SpecialCaseExampleOp", "in0", "in1", "in2"),
                      Flat("in2"),
                      MainMemory("in2"))

END_PKG_OP_DEFINITION(ExampleOpPackage)