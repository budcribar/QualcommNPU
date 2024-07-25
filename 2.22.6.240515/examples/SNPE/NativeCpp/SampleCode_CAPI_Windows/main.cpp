//==============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>

#include "GetOpt.hpp"
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"

#include "DlSystem/DlError.h"
#include "DlSystem/RuntimeList.h"
#include "DlSystem/UserBufferMap.h"
#include "DlSystem/IUserBuffer.h"
#include "DlSystem/DlEnums.h"
#include "DlContainer/DlContainer.h"
#include "SNPE/SNPE.h"
#include "SNPE/SNPEUtil.h"
#include "DiagLog/IDiagLog.h"

const int FAILURE = 1;
const int SUCCESS = 0;

int main(int argc, char** argv)
{
    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string outputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    std::string userBufferSourceStr = "CPUBUFFER";
    std::string staticQuantizationStr = "false";
    std::string UdoPackagePath="";
    static Snpe_Runtime_t runtime = SNPE_RUNTIME_CPU_FLOAT32;
    static Snpe_RuntimeList_Handle_t inputRuntimeListHandle = nullptr;

    bool runtimeSpecified = false;
    bool usingInitCache = false;
    bool staticQuantization = false;
    bool cpuFixedPointMode = false;

#ifdef __ANDROID__
    // Initialize Logs with level LOG_ERROR.
    Snpe_Util_InitializeLogging(SNPE_LOG_LEVEL_ERROR);
#else
    // Initialize Logs with specified log level as LOG_ERROR and log path as "./Log".
    Snpe_Util_InitializeLoggingPath(SNPE_LOG_LEVEL_ERROR, "./Log");
#endif

    // Update Log Level to LOG_WARN.
    Snpe_Util_SetLogLevel(SNPE_LOG_LEVEL_WARN);
    // Process command line arguments
    int opt = 0;
    enum OPTIONS
    {
        OPT_HELP = 0,
        OPT_CONTAINER = 1,
        OPT_INPUT_LIST = 2,
        OPT_OUTPUT_DIR = 3,
        OPT_USERBUFFER = 4,
        OPT_RUNTIME = 5,
        OPT_RESIZABLE_DIM = 6,
        OPT_INITBLOBSCACHE = 7,
        OPT_RUNTIME_ORDER = 8,
        OPT_STATIC_QUANTIZATION = 9,
        OPT_FXP_CPU = 10,
        OPT_UDO_PACKAGE_PATH = 11
    };
    static struct WinOpt::option long_options[] = {
      {"help",                    WinOpt::no_argument,          NULL,  OPT_HELP},
      {"container",               WinOpt::required_argument,    NULL,  OPT_CONTAINER},
      {"input_list",              WinOpt::required_argument,    NULL,  OPT_INPUT_LIST},
      {"output_dir",              WinOpt::required_argument,    NULL,  OPT_OUTPUT_DIR},
      {"userbuffer",              WinOpt::required_argument,    NULL,  OPT_USERBUFFER},
      {"runtime",                 WinOpt::required_argument,    NULL,  OPT_RUNTIME},
      {"resizable_dim",           WinOpt::required_argument,    NULL,  OPT_RESIZABLE_DIM},
      {"enable_init_cache",       WinOpt::no_argument,          NULL,  OPT_INITBLOBSCACHE},
      {"runtime_order",           WinOpt::required_argument,    NULL,  OPT_RUNTIME_ORDER},
      {"static_quantization",     WinOpt::required_argument,    NULL,  OPT_STATIC_QUANTIZATION},
      {"fxp_cpu",     WinOpt::required_argument,    NULL,  OPT_FXP_CPU},
      {"udo_package_path",     WinOpt::required_argument,    NULL,  OPT_UDO_PACKAGE_PATH},
      {NULL,                      0,                  NULL,  0 }
    };
    int long_index = 0;
    while ((opt = WinOpt::GetOptLongOnly(argc, argv, "", long_options, &long_index)) != -1)
    {
        switch (opt)
        {
            case OPT_HELP:
                std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to load and execute a neural network\n"
                        << "using the SNPE C++ API.\n"
                        << "\n\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  --container  <FILE>   Path to the DL container containing the network.\n"
                        << "  --input_list  <FILE>   Path to a file listing the inputs for the network.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  --output_dir  <PATH>    Path to directory to store output results.\n"
                        << "  --userbuffer  <TYPE>    Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16] (" << bufferTypeStr << " is default).\n"
                        << "  --static_quantization  <BOOL>    Specifies to use static quantization parameters from the model instead of input specific quantization [true, false]. Used in conjunction with USERBUFFER_TF8. \n"
                        << "  --runtime  <RUNTIME> The runtime to be used [gpu, dsp, aip, cpu] (cpu is default). \n"
                        << "  --udo_package_path  <VAL,VAL> Path to UDO package with registration library for UDOs. \n"
                        << "                Optionally, user can provide multiple packages as a comma-separated list. \n"
                        << "  -resizable_dim  <NUMBER>  The maximum number that resizable dimensions can grow into. \n"
                        << "                Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer and is not applicable when using ITensor. \n"
                        << "  --enable_init_cache            Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.\n"
                        << "  --fxp_cpu            Enable the fixed point execution on CPU runtime for SNPE. Defaults to disable.\n"
                        << "  --runtime_order  <VAL,VAL,VAL> Specifies the order of precedence for runtime e.g  cpu_float32, dsp_fixed8_tf etc. Valid values are:- \n"
                        << "                    cpu_float32 (Snapdragon CPU)       = Data & Math: float 32bit \n"
                        << "                    gpu_float32_16_hybrid (Adreno GPU) = Data: float 16bit Math: float 32bit \n"
                        << "                    dsp_fixed8_tf (Hexagon DSP)        = Data & Math: 8bit fixed point Tensorflow style format \n"
                        << "                    gpu_float16 (Adreno GPU)           = Data: float 16bit Math: float 16bit \n"
                        #if DNN_RUNTIME_HAVE_AIP_RUNTIME
                        << "                    aip_fixed8_tf (Snapdragon HTA+HVX) = Data & Math: 8bit fixed point Tensorflow style format \n"
                        #endif
                        << "                    cpu (Snapdragon CPU)               = Same as cpu_float32 \n"
                        << "                    gpu (Adreno GPU)                   = Same as gpu_float32_16_hybrid \n"
                        << "                    dsp (Hexagon DSP)                  = Same as dsp_fixed8_tf \n"
                        #if DNN_RUNTIME_HAVE_AIP_RUNTIME
                        << "                    aip (Snapdragon HTA+HVX)           = Same as aip_fixed8_tf \n"
                        #endif
                        << std::endl;
                std::exit(SUCCESS);
            case OPT_INPUT_LIST:
                inputFile = WinOpt::optarg;
                break;
            case OPT_CONTAINER:
                dlc = WinOpt::optarg;
                break;
            case OPT_OUTPUT_DIR:
                outputDir = WinOpt::optarg;
                break;
            case OPT_USERBUFFER:
                bufferTypeStr = WinOpt::optarg;
                std::cout<<"Feature is not supported yet\n";
                break;
            case OPT_STATIC_QUANTIZATION:
                staticQuantizationStr = WinOpt::optarg;
                break;
            case OPT_RESIZABLE_DIM:
                setResizableDim(atoi(WinOpt::optarg));
                std::cout<<"Feature is not supported yet\n";
                break;
            case OPT_RUNTIME:
                runtimeSpecified = true;
                if (strcmp(WinOpt::optarg, "gpu") == 0)
                {
                    runtime = SNPE_RUNTIME_GPU;
                }
                else if (strcmp(WinOpt::optarg, "aip") == 0)
                {
                    runtime = SNPE_RUNTIME_AIP_FIXED8_TF;
                }
                else if (strcmp(WinOpt::optarg, "dsp") == 0)
                {
                    runtime = SNPE_RUNTIME_DSP;
                }
                else if (strcmp(WinOpt::optarg, "cpu") == 0)
                {
                    runtime = SNPE_RUNTIME_CPU_FLOAT32;
                }
                else
                {
                    std::cerr << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;

                }
                break;

            case OPT_RUNTIME_ORDER:
            {
                std::string inputString = WinOpt::optarg;
                std::vector<std::string> runtimeStrVector;
                split(runtimeStrVector, inputString, ',');

                //Check for duplicate
                for(auto it = runtimeStrVector.begin(); it != runtimeStrVector.end()-1; it++)
                {
                    auto found = std::find(it+1, runtimeStrVector.end(), *it);
                    if(found != runtimeStrVector.end())
                    {
                        std::cerr << "Error: Invalid values passed to the argument "<< argv[WinOpt::optind-2] << ". Duplicate entries in runtime order" << std::endl;
                        std::exit(FAILURE);
                    }
                }

                inputRuntimeListHandle = Snpe_RuntimeList_Create();
                for(auto& runtimeStr : runtimeStrVector)
                {

                    Snpe_Runtime_t runtime = Snpe_RuntimeList_StringToRuntime(runtimeStr.c_str());
                    if (runtime != SNPE_RUNTIME_UNSET)
                    {
                        auto ret = Snpe_RuntimeList_Add(inputRuntimeListHandle, runtime);
                        if (ret != SNPE_SUCCESS)
                        {
                            std::cerr << Snpe_ErrorCode_GetLastErrorString() << std::endl;
                            std::cerr << "Error: Invalid values passed to the argument "<< argv[WinOpt::optind-2] << ". Please provide comma seperated runtime order of precedence" << std::endl;
                            std::exit(FAILURE);
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Invalid values passed to the argument "<< argv[WinOpt::optind-2] << ". Please provide comma seperated runtime order of precedence" << std::endl;
                        std::exit(FAILURE);
                    }
                }
            }
                break;

            case OPT_INITBLOBSCACHE:
                usingInitCache = true;
                break;
            case OPT_FXP_CPU:
                cpuFixedPointMode = true;
                break;
            case OPT_UDO_PACKAGE_PATH:
                UdoPackagePath = WinOpt::optarg;
                std::cout<<"Feature is not supported yet\n";
                break;
            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the --help flag to see required arguments" << std::endl;
                std::exit(FAILURE);
        }
    }

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        return EXIT_FAILURE;
    }

    // Check if given buffer type is valid
    int bufferType = ITENSOR;

    if (staticQuantizationStr == "true")
    {
        staticQuantization = true;
    }
    else if (staticQuantizationStr == "false")
    {
        staticQuantization = false;
    }
    else
    {
        std::cout << "Static quantization value is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
        return EXIT_FAILURE;
    }

    if(!inputRuntimeListHandle)
        inputRuntimeListHandle = Snpe_RuntimeList_Create();
    //Check if both runtimelist and runtime are passed in
    if(runtimeSpecified && !Snpe_RuntimeList_Empty(inputRuntimeListHandle))
    {
        std::cout << "Invalid option cannot mix runtime order -l with runtime -r " << std::endl;
        std::exit(FAILURE);
    }

    // Open the DL container that contains the network to execute.
    // Create an instance of the SNPE network from the now opened container.
    // The factory functions provided by SNPE allow for the specification
    // of which layers of the network should be returned as output and also
    // if the network should be run on the CPU or GPU.
    // The runtime availability API allows for runtime support to be queried.
    // If a selected runtime is not available, we will issue a warning and continue,
    // expecting the invalid configuration to be caught at SNPE network creation.

    if (runtimeSpecified) {
        runtime = CheckRuntime(runtime, staticQuantization);
    }

    //Getting the Container Handle
    Snpe_DlContainer_Handle_t containerHandle = LoadContainerFromPath(dlc);
    if (!containerHandle) {
        std::cerr << "Error while opening the container file." << std::endl;
        return EXIT_FAILURE;
    }

    //Setting UserSuppliedBuffers to false as buffer mode is ITensor for now
    bool useUserSuppliedBuffers = false;

    // Setting nullptr to snpeHandle
    Snpe_SNPE_Handle_t snpeHandle{};

    Snpe_PlatformConfig_Handle_t platformConfigHandle = Snpe_PlatformConfig_Create();

    snpeHandle = setBuilderOptions(containerHandle, runtime, inputRuntimeListHandle, useUserSuppliedBuffers,
                                   platformConfigHandle, usingInitCache, cpuFixedPointMode);
    if (snpeHandle == nullptr) {
        std::cerr << "Error while building SNPE object." << std::endl;
        return EXIT_FAILURE;
    }

    //If Init Cache is set overwriting the dlc with the initCache
    if (usingInitCache) {
        if (Snpe_DlContainer_Save(containerHandle, dlc.c_str()) == SNPE_SUCCESS) {
            std::cout << "Saved container into archive successfully" << "\n";
        } else {
            std::cout << "Failed to save container into archive" << std::endl;
        }
    }

    //Deleting all the Handles which are no longer needed
    Snpe_PlatformConfig_Delete(platformConfigHandle);
    Snpe_RuntimeList_Delete(inputRuntimeListHandle);
    Snpe_DlContainer_Delete(containerHandle);


    // Configure logging output and start logging. The snpe-diagview
    // executable can be used to read the content of this diagnostics file
    auto diagLogHandle = Snpe_SNPE_GetDiagLogInterface_Ref(snpeHandle);
    if (diagLogHandle == nullptr) throw std::runtime_error("SNPE failed to obtain logging interface");

    auto optionsHandle = Snpe_IDiagLog_GetOptions(diagLogHandle);
    Snpe_Options_SetLogFileDirectory(optionsHandle, outputDir.c_str());
    if (Snpe_IDiagLog_SetOptions(diagLogHandle, optionsHandle) != SNPE_SUCCESS) {
        std::cerr << "Failed to set options" << std::endl;
        return EXIT_FAILURE;
    }
    if (Snpe_IDiagLog_Start(diagLogHandle) != SNPE_SUCCESS) {
        std::cerr << "Failed to start logger" << std::endl;
        return EXIT_FAILURE;
    }
    Snpe_Options_Delete(optionsHandle);

    // Check the batch size for the container
    // SNPE 2.x assumes the first dimension of the tensor shape
    // is the batch size.
    //Getting the Shape of the first input Tensor
    Snpe_TensorShape_Handle_t inputShapeHandle = Snpe_SNPE_GetInputDimensionsOfFirstTensor(snpeHandle);
    if (Snpe_TensorShape_Rank(inputShapeHandle) == 0)
        return EXIT_FAILURE;

    //Getting the first dimension of the input Shape
    const size_t *inputFirstDimenison = Snpe_TensorShape_GetDimensions(inputShapeHandle);
    size_t batchSize = *inputFirstDimenison;

    Snpe_TensorShape_Delete(inputShapeHandle);

    std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = PreprocessInput(inputFile, batchSize);

    // Load contents of input file batches into a SNPE tensor
    // execute the network with the input and save each of the returned output to a file.

    if (bufferType == ITENSOR) {
        // Output Tensor Map Handle
        Snpe_TensorMap_Handle_t outputTensorMapHandle = Snpe_TensorMap_Create();

        //Get input Tensor Names Handle from SNPE Handle
        Snpe_StringList_Handle_t networkInputTensorNamesHandle = Snpe_SNPE_GetInputTensorNames(snpeHandle);
        if (networkInputTensorNamesHandle == nullptr) {
            throw std::runtime_error("Error obtaining Input tensor names");
        }
        for (size_t i = 0; i < inputs.size(); i++) {

            // Printing for current batch Size being processed
            if (batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;

            // Loaading the input Tensors from InputTensorNamesHandle
            std::vector<Snpe_ITensor_Handle_t> inputTensors(Snpe_StringList_Size(networkInputTensorNamesHandle));
            Snpe_TensorMap_Handle_t inputTensorMapHandle;

            bool inputLoadStatus = false;
            std::tie(inputTensorMapHandle, inputLoadStatus) = LoadInputTensorMap(snpeHandle, inputs[i],
                                                                                 networkInputTensorNamesHandle,
                                                                                 inputTensors);
            if (!inputLoadStatus) {
                return EXIT_FAILURE;
            }

            // Execute and Save the execution results if execution successful
            if (Snpe_SNPE_ExecuteITensors(snpeHandle, inputTensorMapHandle, outputTensorMapHandle) != SNPE_SUCCESS) {
                std::cerr << "Error while executing the network." << std::endl;

            } else {
                if (!SaveOutputTensor(outputTensorMapHandle, outputDir, i * batchSize, batchSize)) {
                    return EXIT_FAILURE;
                }
            }
            for (size_t i = 0; i < inputTensors.size(); i++)
                Snpe_ITensor_Delete(inputTensors[i]);
            Snpe_TensorMap_Delete(inputTensorMapHandle);
        }
        Snpe_StringList_Delete(networkInputTensorNamesHandle);
        Snpe_TensorMap_Delete(outputTensorMapHandle);
    }

    std::cout << "Successfully executed!" << std::endl;
    // Freeing of snpe handle
    Snpe_SNPE_Delete(snpeHandle);

    return SUCCESS;
}
