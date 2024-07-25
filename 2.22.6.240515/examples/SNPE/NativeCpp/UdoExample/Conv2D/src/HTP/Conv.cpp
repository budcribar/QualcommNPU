//==============================================================================
// Auto Generated Code for Conv2DPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_Conv);

static Qnn_Scalar_t sg_opDefaultGroupScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                              .int32Value = 1};
static Qnn_Param_t sg_opDefaultGroup = {.paramType = QNN_PARAMTYPE_SCALAR,
                                       .scalarParam = sg_opDefaultGroupScalar};

// op execute function declarations
template<typename TensorType>
GraphStatus convImpl(TensorType& out_0,
                     const TensorType& in_0,
                     const TensorType& weight,
                     const TensorType& bias,
                     const Int32Tensor& group,
                     const Int32Tensor& pads,
                     const Int32Tensor& strides,
                     const Int32Tensor& dilations,
                     const Int32Tensor& kernel_shape);

// forward declaration of sample cost function
static float convCostFunc(const Op *op);

/*
* method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
* syntax: DEF_PACKAGE_OP(F,OP)
* e.g. DEF_PACKAGE_OP((convImpl<Tensor>), "Conv")
*/
DEF_PACKAGE_OP((convImpl<Tensor>), "Conv")

/*
* method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
* and provided flags
* syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
* can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
* RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
* e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((convImpl<PlainFloatTensor>), "Conv", SNAIL)
*/

/*
* method 3 for defining op with cost function pointer and provided flags
* cost function pointer type: typedef float (*cost_function) (const Op * op);
* syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
* e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((convImpl<PlainFloatTensor>),
* "Conv", convCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("Conv",
                        "group",
                        false,
                        &sg_opDefaultGroup,
                        "pads",
                        true,
                        nullptr,
                        "strides",
                        true,
                        nullptr,
                        "dilations",
                        true,
                        nullptr,
                        "kernel_shape",
                        true,
                        nullptr)


/* execute functions for ops */

template<typename TensorType>
GraphStatus convImpl(TensorType& out_0,
                     const TensorType& in_0,
                     const TensorType& weight,
                     const TensorType& bias,
                     const Int32Tensor& group,
                     const Int32Tensor& pads,
                     const Int32Tensor& strides,
                     const Int32Tensor& dilations,
                     const Int32Tensor& kernel_shape)

{
    //Initialise params
    int32_t groups = group(0, 0, 0, 0);
    int32_t padH = pads(0, 0, 0, 0);
    int32_t padW = pads(0, 0, 0, 1);
    int32_t strideH = strides(0, 0, 0, 0);
    int32_t strideW = strides(0, 0, 0, 1);

    auto [b_out, h_out, w_out, d_out] = out_0.dims();

    int32_t d_filter = weight.dim(2);
    int32_t h_filter = weight.dim(0);
    int32_t w_filter = weight.dim(1);

    int32_t h_in = in_0.dim(1);
    int32_t w_in = in_0.dim(2);

    Idx outputGroupDepth = d_out / groups;

    for (int32_t ob = 0; ob < b_out; ob++)
    {
        for (int32_t oh = 0; oh < h_out; oh++)
        {
            for (int32_t ow = 0; ow < w_out; ow++)
            {
                for (int32_t g = 0; g < groups; g++)
                {
                    for (int32_t d = 0; d < outputGroupDepth; d++)
                    {
                        int32_t inputOriginH = (int32_t) oh * strideH - padH;
                        int32_t inputOriginW = (int32_t) ow * strideW - padW;
                        int32_t depth = d + g * outputGroupDepth;
                        float sum = bias(0, 0, 0, depth);
                        for (int32_t fh = 0; fh < h_filter; fh++)
                        {
                            for (int32_t fw = 0; fw < w_filter; fw++)
                            {
                                int32_t inputH  = inputOriginH + (int32_t) fh;
                                int32_t inputW  = inputOriginW + (int32_t) fw;
                                for (int32_t fd = 0; fd < d_filter; fd++)
                                {
                                    if (inputH >= 0 && inputH < (int32_t)(h_in) && inputW >= 0 &&
                                        inputW < (int32_t)(w_in))
                                    {
                                        float inval = in_0(0, inputH, inputW, fd);
                                        float filtval = weight(fh, fw, fd, depth);
                                        sum += inval * filtval;
                                    }
                                }
                            }
                        }
                        out_0(ob, oh, ow, depth) = sum;
                    }
                }
            }
        }
    }

return GraphStatus::Success;
}

__attribute__((unused)) static float convCostFunc(const Op *op)
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
END_PKG_OP_DEFINITION(PKG_Conv);
