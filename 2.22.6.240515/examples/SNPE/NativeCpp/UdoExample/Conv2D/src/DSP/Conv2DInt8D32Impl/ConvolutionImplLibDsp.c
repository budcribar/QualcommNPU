//=============================================================================
//
//  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include "ConvolutionImplLibDsp.h"
#include "SnpeUdo/UdoImplDsp.h"

// operations info
char ConvolutionOpType [] = "Convolution";
uint32_t ConvolutionInputsNum = 1;
uint32_t ConvolutionOutputsNum = 1;
SnpeUdo_QuantizationType_t ConvolutionInputQuantizationTypes [] = {SNPE_UDO_QUANTIZATION_TF};
SnpeUdo_QuantizationType_t ConvolutionOutputQuantizationTypes [] =  {SNPE_UDO_QUANTIZATION_TF};
SnpeUdo_HexNNTensorLayout_t ConvolutionLayout [] = {SNPE_UDO_DSP_TENSOR_LAYOUT_D32};

UdoDspShared* new_Convolution(SnpeUdo_HexNNv2GlobalInfra_t* infra)
{
    UdoDspShared *pOpObj = (*(infra->udoMalloc))(sizeof(UdoDspShared));
    if (pOpObj == NULL)
    {
        return NULL;
    }

    pOpObj->QueryOp = Convolution_QueryOperation;
    pOpObj->ValidateOp = Convolution_ValidateOperation;
    pOpObj->CreateOp = Convolution_CreateOpFactory;
    pOpObj->ExecuteOp = Convolution_ExecuteOp;

    return pOpObj;
}

SnpeUdo_ErrorType_t
Convolution_QueryOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                            const SnpeUdo_Param_t* staticParams, uint32_t* numOfInputs,
                            SnpeUdo_QuantizationType_t** inputsQuantTypes,
                            SnpeUdo_HexNNTensorLayout_t** inputsLayouts, uint32_t* numOfOutputs,
                            SnpeUdo_QuantizationType_t** outputsQuantTypes,
                            SnpeUdo_HexNNTensorLayout_t** outputsLayouts)
{
    if(strcmp(operationType, ConvolutionOpType) == 0)
    {
        *numOfInputs = ConvolutionInputsNum;
        *inputsQuantTypes = ConvolutionInputQuantizationTypes;
        *inputsLayouts = ConvolutionLayout;
        *numOfOutputs = ConvolutionOutputsNum;
        *outputsQuantTypes = ConvolutionOutputQuantizationTypes;
        *outputsLayouts = ConvolutionLayout;
    } else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
Convolution_ValidateOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                               const SnpeUdo_Param_t* staticParams)
{
    if(strcmp(operationType, ConvolutionOpType) == 0)
    {
        /*
         * This check is not valid for some models  as param count differ for different layers

         if (numOfStaticParams != ConvolutionStaticParamsNum)
         {
             //return SNPE_UDO_UNSUPPORTED_FEATURE;
         }
        */
    } else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
Convolution_CreateOpFactory (SnpeUdo_HexNNv2GlobalInfra_t* infra, SnpeUdo_CoreType_t udoCoreType,
                             void* perFactoryInfrastructure,
                             SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                             SnpeUdo_Param_t* staticParams, SnpeUdo_OpFactory_t* opFactory)
{
    if(operationType == NULL || operationType == 0)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    if(strcmp(operationType, ConvolutionOpType) == 0)
    {
        ConvolutionOpFactory* this_factory = (*(infra->udoMalloc))(sizeof(ConvolutionOpFactory));
        int size = strlen(operationType) + 1; // +1 to hold the '\0' character
        this_factory->opType = (*(infra->udoMalloc))(size);
        strlcpy((this_factory->opType), operationType, size);

        // Add static params to OpFactory
        // Note that we need to deep-copy values, not just pointers of parameters.
        this_factory->numOfStaticParams = numOfStaticParams;
        this_factory->staticParams = (*(infra->udoMalloc))(sizeof(SnpeUdo_Param_t) * numOfStaticParams);

        SnpeUdo_Param_t* prms = this_factory->staticParams;
        for (int i = 0; i < numOfStaticParams; i++)
        {
            size = strlen(staticParams[i].paramName) + 1; // +1 to hold the '\0' character
            prms[i].paramName = (*(infra->udoMalloc))(size);
            strlcpy(prms[i].paramName, staticParams[i].paramName, size);
            prms[i].paramType = staticParams[i].paramType;
            if( staticParams[i].paramType == SNPE_UDO_PARAMTYPE_SCALAR  )
            {
                prms[i].scalarParam.dataType = staticParams[i].scalarParam.dataType;
                if( staticParams[i].scalarParam.dataType == SNPE_UDO_DATATYPE_INT_32 )
                {
                    prms[i].scalarParam.dataValue.int32Value = staticParams[i].scalarParam.dataValue.int32Value;
                }
            }
            else if ( staticParams[i].paramType == SNPE_UDO_PARAMTYPE_TENSOR )
            {
                prms[i].tensorParam.dataType = staticParams[i].tensorParam.dataType;
                if(staticParams[i].tensorParam.dataType == SNPE_UDO_DATATYPE_UINT_8 || staticParams[i].tensorParam.dataType == SNPE_UDO_DATATYPE_INT_32 )
                {
                    uint32_t rank = staticParams[i].tensorParam.tensorRank;
                    prms[i].tensorParam.tensorRank = rank;
                    prms[i].tensorParam.currDimensions = (*(infra->udoMalloc))(sizeof(uint32_t) * rank);

                    int tensorSize= 1;
                    for (int j = 0; j < rank; j++)
                    {
                        prms[i].tensorParam.currDimensions[j] = staticParams[i].tensorParam.currDimensions[j];
                        tensorSize *= prms[i].tensorParam.currDimensions[j];
                    }

                    uint32_t dtype = prms[i].tensorParam.dataType;
                    prms[i].tensorParam.layout = staticParams[i].tensorParam.layout;
                    prms[i].tensorParam.quantizeParams.quantizeType = staticParams[i].tensorParam.quantizeParams.quantizeType;
                    prms[i].tensorParam.quantizeParams.TFParams.minValue = staticParams[i].tensorParam.quantizeParams.TFParams.minValue;
                    prms[i].tensorParam.quantizeParams.TFParams.maxValue = staticParams[i].tensorParam.quantizeParams.TFParams.maxValue;
                    prms[i].tensorParam.tensorData = (dtype == SNPE_UDO_DATATYPE_UINT_8) ? (*(infra->udoMalloc))(sizeof(uint8_t) * tensorSize)
                                                                                         : (*(infra->udoMalloc))(sizeof(int32_t) * tensorSize);
                    if (dtype == SNPE_UDO_DATATYPE_UINT_8){
                        uint8_t* ptr1 = (uint8_t*)prms[i].tensorParam.tensorData;
                        uint8_t* ptr2 = (uint8_t*)staticParams[i].tensorParam.tensorData;
                        for(int j = 0; j < tensorSize; j++)
                        {
                            ptr1[j] = ptr2[j];
                        }
                    }
                    else
                    {
                        int32_t* ptr1 = (int32_t*)prms[i].tensorParam.tensorData;
                        int32_t* ptr2 = (int32_t*)staticParams[i].tensorParam.tensorData;
                        for(int j = 0; j < tensorSize; j++)
                        {
                            ptr1[j] = ptr2[j];
                        }
                    }
                }
            }
        }
        *opFactory = (SnpeUdo_OpFactory_t) this_factory;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

size_t tensor_offset_for_d32(
        int32_t b,
        int32_t h,
        int32_t w,
        int32_t d,
        D32_Params* p)
{
    uint32_t h_before = p->heightPadBefore;
    uint32_t w_before = p->widthPadBefore;
    uint32_t d_before = p->depthPadBefore;
    uint32_t h_total = h_before + p->H + p->heightPadAfter;
    uint32_t w_total = w_before + p->W + p->widthPadAfter;
    uint32_t d_total = d_before + p->D + p->depthPadAfter;
    uint32_t d_pos = d + d_before;

    return b*h_total*w_total*d_total
           + (h+h_before)*w_total*d_total
           + (w+w_before)*32
           + (d_pos/32)*(w_total*32)
           + d_pos%32;
}

uint32_t Get_D32_Dimensions(uint32_t* dim,D32_Params* params){

    int b = dim[0];
    int h = dim[1];
    int w = dim[2];
    int d = dim[3];

    params->B = b;
    params->H = h;
    params->heightPadBefore = 4;
    params->heightPadAfter = 4;

    params->W = w;
    params->widthPadBefore = 4;
    uint32_t  w_min = 0;
    uint32_t  w_total = (params->widthPadBefore + w + w_min + 3) & ~3;
    params->widthPadAfter = w_total - (params->widthPadBefore + w);
    params->D = d;
    params->depthPadBefore = 0;
    params->depthPadAfter = (-(d + params->depthPadBefore)) & 31;

    uint32_t h_total = params->H + params->heightPadBefore + params->heightPadAfter;
    uint32_t d_total = params->D + params->depthPadBefore + params->depthPadAfter;
    uint32_t total_size=b*h_total*w_total*d_total;

    return total_size;
};

static inline int32_t quantize_int(float val, float minval, float maxval)
{
    /* We want 0.0 -- 255.0 to resize to 0..255 */
    float range = fmaxf(1e-18f, maxval - minval);
    float resize_amt = 255.0f/(range);
    float value_f = (val - minval) * resize_amt;
    int32_t value_i = roundf(value_f);
    return (-1)*value_i;
}

void worker_thread_Conv2D (void* perOpInfrastructure, void* userData)
{
    ConvolutionOpInfo * data = (ConvolutionOpInfo *) userData;
    SnpeUdo_TensorParam_t* in_tensor = data->in_tensor;
    SnpeUdo_TensorParam_t* out_tensor = data->out_tensor;
    SnpeUdo_TensorParam_t* filt_tensor = data->filt_tensor;
    SnpeUdo_TensorParam_t* bias_tensor = data->bias_tensor;
    D32_Params* in_d32prms = data->in_d32prms;
    D32_Params* out_d32prms = data->out_d32prms;

    int32_t numFilters = data->scalarParams->numFilters;
    int32_t biasTerm = data->scalarParams->biasTerm;
    int32_t padH = data->scalarParams->padH;
    int32_t padW = data->scalarParams->padW;
    int32_t strideH = data->scalarParams->strideH;
    int32_t strideW = data->scalarParams->strideW;
    int32_t groups = data->scalarParams->groups;

    //input height, width and depth
    size_t inputHeight = in_tensor->currDimensions[1];
    size_t inputWidth = in_tensor->currDimensions[2];
    size_t inputDepth = in_tensor->currDimensions[3];

    in_d32prms->H=inputHeight;
    in_d32prms->W=inputWidth;
    in_d32prms->D=inputDepth;

    size_t filterHeight = filt_tensor->currDimensions[2];
    size_t filterWidth = filt_tensor->currDimensions[3];
    size_t filterDepth = filt_tensor->currDimensions[1];

    //output height, width and depth
    size_t outputBatch = out_tensor->currDimensions[0];
    size_t outputHeight = out_tensor->currDimensions[1];
    size_t outputWidth = out_tensor->currDimensions[2];
    size_t outputGroupDepth = numFilters / groups;

    uint8_t* in = in_tensor->tensorData;
    uint8_t* filter = filt_tensor->tensorData;
    uint8_t* bias = NULL;
    if(biasTerm) bias = bias_tensor->tensorData;
    uint8_t* out = out_tensor->tensorData;

    float min_in = in_tensor->quantizeParams.TFParams.minValue;
    float max_in = in_tensor->quantizeParams.TFParams.maxValue;
    float delta_in = (max_in - min_in) / 255;
    float offset_in = (float) quantize_int(0.0f,min_in,max_in);

    float min_filt = filt_tensor->quantizeParams.TFParams.minValue;
    float max_filt = filt_tensor->quantizeParams.TFParams.maxValue;
    float delta_filt = (max_filt - min_filt) / 255;
    float offset_filt = (float) quantize_int(0.0f,min_filt,max_filt);

    float min_bias = 0.0;
    float max_bias = 0.0;
    float delta_bias = 0.0;
    float offset_bias = 0.0;

    if (biasTerm){
        min_bias = bias_tensor->quantizeParams.TFParams.minValue;
        max_bias = bias_tensor->quantizeParams.TFParams.maxValue;
        delta_bias = (max_bias - min_bias) / 255;
        offset_bias = (float) quantize_int(0.0f,min_bias,max_bias);
    }

    float min_out = fmax(out_tensor->quantizeParams.TFParams.minValue,INT32_MIN);
    float max_out = fmin(out_tensor->quantizeParams.TFParams.maxValue,INT32_MAX);
    float delta_out = (max_out - min_out) / 255;
    float offset_out = (float) quantize_int(0.0f,min_out,max_out);

    uint8_t * inputBase = in;
    for (uint32_t ob = 0; ob < outputBatch; ob++)
    {
        for (uint32_t oh = 0; oh < outputHeight; oh++)
        {
            for (uint32_t ow = 0; ow < outputWidth; ow++)
            {
                const uint8_t* filterBase = filter;
                for (uint32_t g = 0; g < groups; g++) {
                    for (uint32_t d = 0; d < outputGroupDepth; d++) {
                        int32_t inputOriginH = (int32_t) oh * strideH - padH;
                        int32_t inputOriginW = (int32_t) ow * strideW - padW;
                        float sum = 0.0f;
                        for (uint32_t fd = 0; fd < filterDepth; fd++) {
                            for (uint32_t fh = 0; fh < filterHeight; fh++) {
                                for (uint32_t fw = 0; fw < filterWidth; fw++) {
                                    int32_t inputH = inputOriginH + (int32_t) fh;
                                    int32_t inputW = inputOriginW + (int32_t) fw;
                                    uint32_t inputD = filterDepth * g + fd;
                                    if (inputH >= 0 && inputH < (int32_t)(inputHeight) && inputW >= 0 &&
                                        inputW < (int32_t)(inputWidth)) {
                                        uint32_t filterIndex = fd * filterHeight * filterWidth + fh * filterWidth + fw;
                                        uint32_t inputIndex = tensor_offset_for_d32(ob, inputH, inputW, inputD,
                                                                                    in_d32prms);
                                        float filterElement = ((float) filterBase[filterIndex] + offset_filt) * delta_filt;
                                        float inputElement = ((float) inputBase[inputIndex] + offset_in) * delta_in;
                                        sum += inputElement * filterElement;
                                    }
                                }
                            }
                        }

                        uint32_t filterNo = g * outputGroupDepth + d;
                        if (biasTerm) {
                            float bias_element = ((float) bias[g * outputGroupDepth + d] + offset_bias) * delta_bias;
                            sum += bias_element;
                        }
                        uint32_t outputIndex = tensor_offset_for_d32(ob, oh, ow, filterNo, out_d32prms);
                        out[outputIndex] = (sum / delta_out) - offset_out;
                        filterBase += (filterHeight * filterWidth * filterDepth);
                    }
                }
            }
        }
        inputBase += inputDepth * inputWidth * inputHeight;
    }
}

SnpeUdo_ErrorType_t
Convolution_ExecuteOp (SnpeUdo_HexNNv2GlobalInfra_t* infra, SnpeUdo_Operation_t operation,
                       bool blocking, const uint32_t ID, SnpeUdo_ExternalNotify_t notifyFunc)
{
    if(operation == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    OpParams* m_Operation = (OpParams*) operation;
    char* m_OpType = ((ConvolutionOpFactory*) (m_Operation->opFactory))->opType;
    uint32_t numOfStaticParams = ((ConvolutionOpFactory*) (m_Operation->opFactory))->numOfStaticParams;
    SnpeUdo_Param_t* staticParams = ((ConvolutionOpFactory*) (m_Operation->opFactory))->staticParams;

    if(m_OpType == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    if(strcmp(m_OpType, ConvolutionOpType) == 0)
    {
        SnpeUdo_TensorParam_t* in_tensor = &(m_Operation->InputParams[0]);
        SnpeUdo_TensorParam_t* out_tensor = &(m_Operation->outputParams[0]);
        SnpeUdo_TensorParam_t* filt_tensor;
        SnpeUdo_TensorParam_t* bias_tensor;

        uint32_t inIdx=0;
        D32_Params in_d32prms;
        (*(infra->udoGetInputD32Paddings))(m_Operation->opInfra,inIdx, &(in_d32prms.heightPadBefore), &(in_d32prms.heightPadAfter),
                                           &(in_d32prms.widthPadBefore), &(in_d32prms.widthPadAfter),
                                           &(in_d32prms.depthPadBefore), &(in_d32prms.depthPadAfter));

        // scalar params
        int32_t numFilters = 0;
        int32_t biasTerm = 1;
        int32_t padH = 0;
        int32_t padW = 0;
        int32_t strideH = 1;
        int32_t strideW = 1;
        int32_t kernelH = 0;
        int32_t kernelW = 0;
        int32_t groups = 1;

        //tensor params
        int32_t* kernelSize = NULL;
        int32_t* pad = NULL;
        int32_t* stride = NULL;

        for(int i = 0; i < numOfStaticParams; i++){
            SnpeUdo_Param_t* param = &staticParams[i];
            if (strcmp(param->paramName, "weight_filler") == 0)
            {
                filt_tensor = &(param->tensorParam);
            }
            else if ( strcmp(param->paramName, "bias_filler") == 0 )
            {
                bias_tensor = &(param->tensorParam);
            }
            else if ( param->paramType == SNPE_UDO_PARAMTYPE_SCALAR )
            {
                if(strcmp(param->paramName, "bias_term") == 0){
                    biasTerm = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "group") == 0){
                    groups = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "kernel_h") == 0)
                {
                    kernelH = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "kernel_w") == 0)
                {
                    kernelW = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "pad_h") == 0)
                {
                    padH = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "pad_w") == 0)
                {
                    padW = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "stride_h") == 0)
                {
                    strideH = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "stride_w") == 0)
                {
                    strideW = param->scalarParam.dataValue.int32Value;
                }
                else if(strcmp(param->paramName, "num_output") == 0)
                {
                    numFilters = param->scalarParam.dataValue.int32Value;
                }
                else
                {
                    return SNPE_UDO_INVALID_ARGUMENT;
                }
            }
            else if ( param->paramType == SNPE_UDO_PARAMTYPE_TENSOR )
            {
                if(strcmp(param->paramName, "pad") == 0)
                {
                    pad = param->tensorParam.tensorData;
                }
                else if(strcmp(param->paramName, "kernel_size") == 0)
                {
                    kernelSize = param->tensorParam.tensorData;
                }
                else if(strcmp(param->paramName, "stride") == 0)
                {
                    stride = param->tensorParam.tensorData;
                }
                else
                {
                    return  SNPE_UDO_INVALID_ARGUMENT;
                }
            }
        }

        if (kernelSize != NULL)
        {
            kernelH = kernelSize[0];
            kernelW = kernelSize[0];
        }
        if (pad != NULL)
        {
            padH = pad[0];
            padW = pad[0];
        }
        if (stride != NULL)
        {
            strideH = stride[0];
            strideW = stride[0];
        }

        // set output height, width and depth to infra
        D32_Params out_d32prms;
        Get_D32_Dimensions(out_tensor->currDimensions, &out_d32prms);
        int out_idx = 0;
        (*(infra->udoSetOutputD32ShapeSizePaddings))(m_Operation->opInfra, out_idx, out_d32prms.B,
                                                     out_d32prms.H, out_d32prms.heightPadBefore, out_d32prms.heightPadAfter,
                                                     out_d32prms.W, out_d32prms.widthPadBefore, out_d32prms.widthPadAfter,
                                                     out_d32prms.D, out_d32prms.depthPadBefore, out_d32prms.depthPadAfter,
                                                     SNPE_UDO_DATATYPE_FIXED_8);

        Conv2DScalarParams scalarParams;
        scalarParams.numFilters = numFilters;
        scalarParams.biasTerm = biasTerm;
        scalarParams.padH = padH;
        scalarParams.padW = padW;
        scalarParams.strideH = strideH > 0 ? strideH : 1;
        scalarParams.strideW = strideW > 0 ? strideW : 1;
        scalarParams.kernelH = kernelH;
        scalarParams.kernelW = kernelW;
        scalarParams.groups = groups;

        ConvolutionOpInfo workerThreadIn = {in_tensor, out_tensor, filt_tensor, bias_tensor, &scalarParams, &in_d32prms, &out_d32prms};
        (*(infra->udoRunWorkerThreads))(m_Operation->opInfra, 1, worker_thread_Conv2D, &workerThreadIn);
        return SNPE_UDO_NO_ERROR;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}
