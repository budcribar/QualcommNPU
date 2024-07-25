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
// NOTE: This is the BEFORE file. It is IMPERFECT ON PURPOSE.

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
GraphStatus refImpl(OutType& out, const InType& in_0, const InType& in_1, const Tensor& exec_mode);

template <typename OutType, typename InTypeA, typename InTypeB>
GraphStatus fastImpl(OutType& out,
                     const InTypeA& in_0,
                     const InTypeB& in_1,
                     const Tensor& exec_mode);

// REGISTRATION
// We'll register both sorts of impls to different OpStrs

// Slow implementation maps to RefExampleOp
DEF_PACKAGE_OP((refImpl<Tensor, Tensor>), "RefExampleOp")

// Fast Implementation maps to "FastExampleOp"
DEF_PACKAGE_OP((fastImpl<QuantInt8Tensor_TCM, QUint8CroutonTensor_TCM, QuantUint8Tensor_TCM>),
               "FastExampleOp")

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

// "Toward middle of optimization, if we have an instance of "FastExampleOp" that looks
// like it ought to fit into our memory constraints, then wrap it in a flat_from_vtcm to
// indicate the format of the output and the location of the tensor
DEF_PACKAGE_OPTIMIZATION(
    MIDDLE,
    Op("FastExampleOp", "in_0", "in_1", "exec_mode"),
    LT(ADD(ESTIMATE_TENSOR_SIZE("*"), ESTIMATE_TENSOR_SIZE("in_0"), ESTIMATE_TENSOR_SIZE("in_1")),
       TCM_MAXTENSOR_SIZE),
    Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"), Op("FastExampleOp", "in_0", "in_1", "exec_mode")))

// "OTHERWISE, any remaining FastExampleOps can go back to reference impl"
DEF_PACKAGE_OPTIMIZATION(MIDDLE + 1,
                         Op("FastExampleOp", "in_0", "in_1", "exec_mode"),
                         NOT(PRODUCER_FOR("*", FROM_DEFAULT_PACKAGE("flat_from_vtcm"))),
                         Op("RefExampleOp", "in_0", "in_1", "exec_mode"))

// There is a special case we want to handle toward the end of optimization
// for the ref impl that requires us to add another wrapper op around an input
DEF_OPT(TCM_MIGRATION + 50,
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
                            Op(FROM_DEFAULT_PACKAGE("crouton_to_vtcm"),
                               Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "in_0")),
                            Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"),
                               Op(FROM_DEFAULT_PACKAGE("ForceFormat_Crouton"), "in_1")),
                            "exec_mode"))

END_PKG_OP_DEFINITION(ExampleOpPackage)