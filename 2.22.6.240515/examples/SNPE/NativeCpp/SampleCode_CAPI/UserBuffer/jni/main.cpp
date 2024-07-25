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
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "CreateUserBuffer.hpp"
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

static Snpe_RuntimeList_Handle_t inputRuntimeListHandle = nullptr;
static Snpe_DlContainer_Handle_t containerHandle = nullptr;
static Snpe_SNPE_Handle_t snpeHandle = nullptr;
static Snpe_PlatformConfig_Handle_t platformConfigHandle = nullptr;
static Snpe_Options_Handle_t optionsHandle = nullptr;
static Snpe_TensorShape_Handle_t inputShapeHandle = nullptr;
static Snpe_UserBufferMap_Handle_t outputMapHandle = nullptr;
static Snpe_UserBufferMap_Handle_t inputMapHandle = nullptr;
std::vector<Snpe_IUserBuffer_Handle_t> snpeUserBackedBuffers;

void cleanup()
{
    if(inputRuntimeListHandle)
        Snpe_RuntimeList_Delete(inputRuntimeListHandle);
    if(platformConfigHandle)
        Snpe_PlatformConfig_Delete(platformConfigHandle);
    if(containerHandle)
        Snpe_DlContainer_Delete(containerHandle);
    if(optionsHandle)
        Snpe_Options_Delete(optionsHandle);
    if(inputShapeHandle)
        Snpe_TensorShape_Delete(inputShapeHandle);
    if(outputMapHandle)
        Snpe_UserBufferMap_Delete(outputMapHandle);
    if(inputMapHandle)
        Snpe_UserBufferMap_Delete(inputMapHandle);
    for(auto &snpeuserbackedbuffer : snpeUserBackedBuffers)
        if(snpeuserbackedbuffer)
            Snpe_IUserBuffer_Delete(snpeuserbackedbuffer);
    if(snpeHandle)
        Snpe_SNPE_Delete(snpeHandle);
}

int main(int argc, char** argv) {
    enum {
        UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, USERBUFFER_TF16
    };
    enum {
        CPUBUFFER, GLBUFFER
    };

    // Command line arguments
    static std::string dlc = "";
    static std::string outputDir = "./output/";
    const char *inputFile = "";
    std::string bufferTypeStr = "USERBUFFER_FLOAT";
    std::string userBufferSourceStr = "CPUBUFFER";
    std::string staticQuantizationStr = "false";
    std::string UdoPackagePath = "";
    static Snpe_Runtime_t runtime = SNPE_RUNTIME_CPU_FLOAT32;
    inputRuntimeListHandle = Snpe_RuntimeList_Create();

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
    while ((opt = getopt(argc, argv, "hi:d:o:b:q:s:z:r:l:u:c:x")) != -1) {
        switch (opt) {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to load and execute a neural network\n"
                        << "using the SNPE C API.\n"
                        << "\n\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -d  <FILE>   Path to the DL container containing the network.\n"
                        << "  -i  <FILE>   Path to a file listing the inputs for the network.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -o  <PATH>    Path to directory to store output results.\n"
                        << "  -b  <TYPE>    Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, USERBUFFER_TF16] ("
                        << bufferTypeStr << " is default).\n"
                        << "  -q  <BOOL>    Specifies to use static quantization parameters from the model instead of input specific quantization [true, false]. Used in conjunction with USERBUFFER_TF8. \n"
                        << "  -r  <RUNTIME> The runtime to be used [gpu, dsp, aip, cpu] (cpu is default). \n"
                        << "  -u  <VAL,VAL> Path to UDO package with registration library for UDOs. \n"
                        << "                Optionally, user can provide multiple packages as a comma-separated list. \n"
                        << "  -z  <NUMBER>  The maximum number that resizable dimensions can grow into. \n"
                        << "                Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer. \n"
                        #ifdef ENABLE_GL_BUFFER
                        << "  -s  <TYPE>    Source of user buffers to use [GLBUFFER, CPUBUFFER] (" << userBufferSourceStr << " is default).\n"
                        #endif
                        << "  -c            Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.\n"
                        << "  -x            Enable the fixed point execution on CPU runtime for SNPE. Defaults to disable.\n"
                        << "  -l  <VAL,VAL,VAL> Specifies the order of precedence for runtime e.g  cpu_float32, dsp_fixed8_tf etc. Valid values are:- \n"
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
                cleanup();
                std::exit(SUCCESS);
            case 'i':
                inputFile = optarg;
                break;
            case 'd':
                dlc = optarg;
                break;
            case 'o':
                outputDir = optarg;
                break;
            case 'b':
                bufferTypeStr = optarg;
                break;
            case 'q':
                staticQuantizationStr = optarg;
                break;
            case 's':
                userBufferSourceStr = optarg;
                break;
            case 'z':
                setResizableDim(atoi(optarg));
                break;
            case 'r':
                runtimeSpecified = true;
                if (strcmp(optarg, "gpu") == 0) {
                    runtime = SNPE_RUNTIME_GPU;
                } else if (strcmp(optarg, "aip") == 0) {
                    runtime = SNPE_RUNTIME_AIP_FIXED8_TF;
                } else if (strcmp(optarg, "dsp") == 0) {
                    runtime = SNPE_RUNTIME_DSP;
                } else if (strcmp(optarg, "cpu") == 0) {
                    runtime = SNPE_RUNTIME_CPU_FLOAT32;
                } else {
                    std::cerr << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;
                    cleanup();
                    std::exit(EXIT_FAILURE);
                }
                break;

            case 'l': {
                std::string inputString = optarg;
                std::vector<std::string> runtimeStrVector;
                split(runtimeStrVector, inputString, ',');

                //Check for duplicate
                for (auto it = runtimeStrVector.begin(); it != runtimeStrVector.end() - 1; it++) {
                    auto found = std::find(it + 1, runtimeStrVector.end(), *it);
                    if (found != runtimeStrVector.end()) {
                        std::cerr << "Error: Invalid values passed to the argument " << argv[optind - 2]
                                  << ". Duplicate entries in runtime order" << std::endl;
                        cleanup();
                        std::exit(FAILURE);
                    }
                }
                for (auto &runtimeStr: runtimeStrVector) {

                    Snpe_Runtime_t runtime = Snpe_RuntimeList_StringToRuntime(runtimeStr.c_str());
                    if (runtime != SNPE_RUNTIME_UNSET) {
                        auto ret = Snpe_RuntimeList_Add(inputRuntimeListHandle, runtime);
                        if (ret != SNPE_SUCCESS) {
                            std::cerr << Snpe_ErrorCode_GetLastErrorString() << std::endl;
                            std::cerr << "Error: Invalid values passed to the argument " << argv[optind - 2]
                                      << ". Please provide comma seperated runtime order of precedence" << std::endl;
                            cleanup();
                            std::exit(FAILURE);
                        }
                    } else {
                        std::cerr << "Error: Invalid values passed to the argument " << argv[optind - 2]
                                  << ". Please provide comma seperated runtime order of precedence" << std::endl;
                        cleanup();
                        std::exit(FAILURE);
                    }
                }
            }
                break;

            case 'c':
                usingInitCache = true;
                break;
            case 'x':
                cpuFixedPointMode = true;
                break;
            case 'u':
                UdoPackagePath = optarg;
                std::cout << "Feature is not supported yet\n";
                break;
            default:
                std::cout
                        << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments"
                        << std::endl;
                cleanup();
                std::exit(FAILURE);
        }
    }

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        std::cout
                << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details"
                << std::endl;
        cleanup();
        return EXIT_FAILURE;
    }

    // Check if given buffer type is valid
    int bufferType = UNKNOWN;
    int bitWidth = 0;
    if (bufferTypeStr == "USERBUFFER_FLOAT") {
        bufferType = USERBUFFER_FLOAT;
    } else if (bufferTypeStr == "USERBUFFER_TF8") {
        bufferType = USERBUFFER_TF8;
        bitWidth = 8;
    } else if (bufferTypeStr == "USERBUFFER_TF16") {
        bufferType = USERBUFFER_TF16;
        bitWidth = 16;
    } else {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
    }

    //Check if given user buffer source type is valid
    int userBufferSourceType;
    // CPUBUFFER supported only for USERBUFFER_FLOAT
    if (bufferType == USERBUFFER_FLOAT) {
        if (userBufferSourceStr == "CPUBUFFER") {
            userBufferSourceType = CPUBUFFER;
        } else {
            std::cout
                    << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details"
                    << std::endl;
            cleanup();
            std::exit(EXIT_FAILURE);
        }
    }


    if (staticQuantizationStr == "true") {
        staticQuantization = true;
    } else if (staticQuantizationStr == "false") {
        staticQuantization = false;
    } else {
        std::cout << "Static quantization value is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
    }


    //Check if both runtimelist and runtime are passed in
    if (runtimeSpecified && !Snpe_RuntimeList_Empty(inputRuntimeListHandle)) {
        std::cout << "Invalid option cannot mix runtime order -l with runtime -r " << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
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
    containerHandle = LoadContainerFromPath(dlc);
    if (!containerHandle) {
        std::cerr << "Error while opening the container file." << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
    }

    //Setting UserSuppliedBuffers to false as buffer mode is ITensor for now
    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8 ||
                                   bufferType == USERBUFFER_TF16);

    snpeHandle = nullptr;
    platformConfigHandle = Snpe_PlatformConfig_Create();

    snpeHandle = setBuilderOptions(containerHandle, runtime, inputRuntimeListHandle, useUserSuppliedBuffers,
                                   platformConfigHandle, usingInitCache, cpuFixedPointMode);
    if (snpeHandle == nullptr) {
        std::cerr << "Error while building SNPE object." << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
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
    if (diagLogHandle == nullptr) {
        std::cerr << "SNPE failed to obtain logging interface" << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
    }

    optionsHandle = Snpe_IDiagLog_GetOptions(diagLogHandle);
    Snpe_Options_SetLogFileDirectory(optionsHandle, outputDir.c_str());
    if (Snpe_IDiagLog_SetOptions(diagLogHandle, optionsHandle) != SNPE_SUCCESS) {
        std::cerr << "Failed to set options" << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
    }
    if (Snpe_IDiagLog_Start(diagLogHandle) != SNPE_SUCCESS) {
        std::cerr << "Failed to start logger" << std::endl;
        cleanup();
        std::exit(EXIT_FAILURE);
    }
    Snpe_Options_Delete(optionsHandle);

    // Check the batch size for the container
    // SNPE 2.x assumes the first dimension of the tensor shape
    // is the batch size.
    //Getting the Shape of the first input Tensor
    inputShapeHandle = Snpe_SNPE_GetInputDimensionsOfFirstTensor(snpeHandle);
    if (Snpe_TensorShape_Rank(inputShapeHandle) == 0) {
        cleanup();
        std::exit(EXIT_FAILURE);
    }

    //Getting the first dimension of the input Shape
    const size_t *inputFirstDimenison = Snpe_TensorShape_GetDimensions(inputShapeHandle);
    size_t batchSize = *inputFirstDimenison;

    Snpe_TensorShape_Delete(inputShapeHandle);

    std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = PreprocessInput(inputFile, batchSize);

    // Load contents of input file batches into a SNPE tensor
    // execute the network with the input and save each of the returned output to a file.
    // SNPE allows its input and output buffers that are fed to the network
    // to come from user-backed buffers. First, SNPE buffers are created from
    // user-backed storage. These SNPE buffers are then supplied to the network
    // and the results are stored in user-backed output buffers. This allows for
    // reusing the same buffers for multiple inputs and outputs.

    outputMapHandle = Snpe_UserBufferMap_Create();
    inputMapHandle = Snpe_UserBufferMap_Create();
    std::unordered_map<std::string, std::vector<uint8_t >> applicationOutputBuffers;
    if (bufferType == USERBUFFER_TF8 || bufferType == USERBUFFER_TF16) {
        if (!CreateOutputBufferMap(outputMapHandle, applicationOutputBuffers, snpeUserBackedBuffers,
                                   snpeHandle, true, bitWidth)) {
            std::cerr << "Error while creating output map" << std::endl;
            cleanup();
            std::exit(EXIT_FAILURE);
        }
        std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
        if (!CreateInputBufferMap(inputMapHandle, applicationInputBuffers, snpeUserBackedBuffers,
                                  snpeHandle, true, staticQuantization, bitWidth)) {
            std::cerr << "Error while creating input map" << std::endl;
            cleanup();
            std::exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < inputs.size(); i++) {
            // Load input user buffer(s) with values from file(s)
            if (batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            if (!LoadInputUserBufferTfN(applicationInputBuffers, snpeHandle, inputs[i], inputMapHandle,
                                        staticQuantization, bitWidth)) {
                std::cerr << "Error while loading the input." << std::endl;
                cleanup();
                std::exit(EXIT_FAILURE);
            }
            // Execute the input buffer map on the model with SNPE
            // Save the execution results only if successful
            if (Snpe_SNPE_ExecuteUserBuffers(snpeHandle, inputMapHandle, outputMapHandle) == SUCCESS) {
                if (!SaveOutputUserBuffer(outputMapHandle, applicationOutputBuffers, outputDir, i * batchSize,
                                          batchSize, true, bitWidth)) {
                    std::cerr << "Error while saving the results." << std::endl;
                    cleanup();
                    std::exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "Error while executing the network." << std::endl;
                cleanup();
                std::exit(EXIT_FAILURE);
            }
        }
    } else {
        if (!CreateOutputBufferMap(outputMapHandle, applicationOutputBuffers, snpeUserBackedBuffers, snpeHandle, false,
                                   bitWidth)) {
            std::cerr << "Error while creating output map" << std::endl;
            cleanup();
            std::exit(EXIT_FAILURE);
        }
        if (userBufferSourceType == CPUBUFFER) {
            std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
            if (!CreateInputBufferMap(inputMapHandle, applicationInputBuffers, snpeUserBackedBuffers,
                                      snpeHandle, false, staticQuantization, bitWidth)) {
                std::cerr << "Error while creating input map" << std::endl;
                cleanup();
                std::exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < inputs.size(); i++) {
                // Load input user buffer(s) with values from file(s)
                if (batchSize > 1)
                    std::cout << "Batch " << i << ":" << std::endl;
                if (!LoadInputUserBufferFloat(applicationInputBuffers, snpeHandle, inputs[i])) {
                    std::cerr << "Error while loading the input." << std::endl;
                    cleanup();
                    std::exit(EXIT_FAILURE);
                }
                // Execute the input buffer map on the model with SNPE
                // Save the execution results only if successful
                if (Snpe_SNPE_ExecuteUserBuffers(snpeHandle, inputMapHandle, outputMapHandle) == SUCCESS) {
                    if (!SaveOutputUserBuffer(outputMapHandle, applicationOutputBuffers, outputDir,
                                              i * batchSize, batchSize,
                                              false, bitWidth)) {
                        std::cerr << "Error while saving the results." << std::endl;
                        cleanup();
                        std::exit(EXIT_FAILURE);
                    }
                } else {
                    std::cerr << "Error while executing the network." << std::endl;
                    cleanup();
                    std::exit(EXIT_FAILURE);
                }
            }
        } else if (userBufferSourceType == GLBUFFER) {
            std::cout << "Feature is not supported yet" << std::endl;
            cleanup();
            std::exit(EXIT_FAILURE);
        } else {
            std::cerr << "Error while executing the network." << std::endl;
            cleanup();
            std::exit(EXIT_FAILURE);
        }
    }
    Snpe_UserBufferMap_Delete(outputMapHandle);
    Snpe_UserBufferMap_Delete(inputMapHandle);
    for (size_t i = 0; i < snpeUserBackedBuffers.size(); i++)
        Snpe_IUserBuffer_Delete(snpeUserBackedBuffers[i]);

    snpeUserBackedBuffers.clear();
    std::cout << "Successfully executed!" << std::endl;
    // Freeing of snpe handle
    Snpe_SNPE_Delete(snpeHandle);

    return SUCCESS;
}