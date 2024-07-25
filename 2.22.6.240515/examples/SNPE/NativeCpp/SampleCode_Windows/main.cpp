//==============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//

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
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"

/* Windows Modification
* Replace <getopt.h> to <GetOpt.hpp> and refactor the "Process command line arguments" part
*/

const int FAILURE = 1;
const int SUCCESS = 0;

int main(int argc, char** argv)
{
    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    std::string userBufferSourceStr = "CPUBUFFER";
    std::string staticQuantizationStr = "false";
    static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    static zdl::DlSystem::RuntimeList runtimeList;
    bool runtimeSpecified = false;
    bool execStatus = false;
    bool usingInitCaching = false;
    bool staticQuantization = false;

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
                        << "  --output_dir  <PATH>   Path to directory to store output results.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  --userbuffer  <TYPE>   Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16] (" << bufferTypeStr << " is default).\n"
                        << "  --static_quantization  <BOOL>    Specifies to use static quantization parameters from the model instead of input specific quantization [true, false]. Used in conjunction with USERBUFFER_TF8. \n"
                        << "  --runtime  <RUNTIME> The runtime to be used [gpu, dsp, aip, cpu] (cpu is default). \n"
                        << "  --resizable_dim  <NUMBER>  The maximum number that resizable dimensions can grow into. \n"
                        << "                Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer and is not applicable when using ITensor. \n"
                        << "  --enable_init_cache           Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.\n"
                        << "  --runtime_order  <VAL,VAL,VAL> Specifies the order of precedence for runtime e.g  cpu_float32, dsp_fixed8_tf etc. Valid values are:- \n"
                        << "                    cpu_float32 (Snapdragon CPU)       = Data & Math: float 32bit \n"
                        << "                    gpu_float32_16_hybrid (Adreno GPU) = Data: float 16bit Math: float 32bit \n"
                        << "                    dsp_fixed8_tf (Hexagon DSP)        = Data & Math: 8bit fixed point Tensorflow style format \n"
                        << "                    gpu_float16 (Adreno GPU)           = Data: float 16bit Math: float 16bit \n"
                        << "                    cpu (Snapdragon CPU)               = Same as cpu_float32 \n"
                        << "                    gpu (Adreno GPU)                   = Same as gpu_float32_16_hybrid \n"
                        << "                    dsp (Hexagon DSP)                  = Same as dsp_fixed8_tf \n"
                        << std::endl;

                std::exit(SUCCESS);
            case OPT_CONTAINER:
                dlc = WinOpt::optarg;
                break;
            case OPT_INPUT_LIST:
                inputFile = WinOpt::optarg;
                break;
            case OPT_OUTPUT_DIR:
                OutputDir = WinOpt::optarg;
                break;
            case OPT_USERBUFFER:
                bufferTypeStr = WinOpt::optarg;
                break;
            case OPT_RESIZABLE_DIM:
                setResizableDim(atoi(WinOpt::optarg));
                break;
            case OPT_RUNTIME:
                runtimeSpecified = true;
                if (strcmp(WinOpt::optarg, "gpu") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::GPU;
                }
                else if (strcmp(WinOpt::optarg, "aip") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
                }
                else if (strcmp(WinOpt::optarg, "dsp") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::DSP;
                }
                else if (strcmp(WinOpt::optarg, "cpu") == 0)
                {
                   runtime = zdl::DlSystem::Runtime_t::CPU;
                }
                else
                {
                   std::cerr << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;

                }
                break;

            case OPT_RUNTIME_ORDER:
                {
                   std::string inputString = WinOpt::optarg;
                   //std::cout<<"Input String: "<<inputString<<std::endl;
                   std::vector<std::string> runtimeStrVector;
                   split(runtimeStrVector, inputString, ',');

                   //Check for dups
                   for(auto it = runtimeStrVector.begin(); it != runtimeStrVector.end()-1; it++)
                   {
                      auto found = std::find(it+1, runtimeStrVector.end(), *it);
                      if(found != runtimeStrVector.end())
                      {
                         std::cerr << "Error: Invalid values passed to the argument "<< argv[WinOpt::optind-2] << ". Duplicate entries in runtime order" << std::endl;
                         std::exit(FAILURE);
                      }
                   }

                   runtimeList.clear();
                   for(auto& runtimeStr : runtimeStrVector)
                   {
                      //std::cout<<runtimeStr<<std::endl;
                      zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::RuntimeList::stringToRuntime(runtimeStr.c_str());
                      if(runtime != zdl::DlSystem::Runtime_t::UNSET)
                      {
                         bool ret = runtimeList.add(runtime);
                         if(ret == false)
                         {
                            std::cerr <<zdl::DlSystem::getLastErrorString()<<std::endl;
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
               usingInitCaching = true;
               break;

            case OPT_STATIC_QUANTIZATION:
               staticQuantizationStr = WinOpt::optarg;
               break;

            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
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
    int bufferType;
    int bitWidth = 0;
    if (bufferTypeStr == "USERBUFFER_FLOAT")
    {
        bufferType = USERBUFFER_FLOAT;
    }
    else if (bufferTypeStr == "USERBUFFER_TF8")
    {
        bufferType = USERBUFFER_TF8;
        bitWidth = 8;
    }
    else if (bufferTypeStr == "USERBUFFER_TF16")
    {
        bufferType = USERBUFFER_TF16;
        bitWidth = 16;
    }
    else if (bufferTypeStr == "ITENSOR")
    {
        bufferType = ITENSOR;
    }
    else
    {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        return EXIT_FAILURE;
    }

    //Check if given user buffer source type is valid
    int userBufferSourceType;
    // CPUBUFFER / GLBUFFER supported only for USERBUFFER_FLOAT
    if (bufferType == USERBUFFER_FLOAT)
    {
        if( userBufferSourceStr == "CPUBUFFER" )
        {
            userBufferSourceType = CPUBUFFER;
        }
        else if( userBufferSourceStr == "GLBUFFER" )
        {
            std::cout << "GLBUFFER mode is only supported on Android OS" << std::endl;
            return EXIT_FAILURE;
            userBufferSourceType = GLBUFFER;
        }
        else
        {
            std::cout
                  << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
            return EXIT_FAILURE;
        }
    }

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

    //Check if both runtimelist and runtime are passed in
    if(runtimeSpecified && runtimeList.empty() == false)
    {
        std::cout << "Invalid option cannot mix runtime order -l with runtime -r " << std::endl;
        std::exit(FAILURE);
    }

    if(runtimeSpecified)
    {
        runtime = checkRuntime(runtime, staticQuantization);
    }

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       return EXIT_FAILURE;
    }

    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT ||
                                   bufferType == USERBUFFER_TF8 ||
                                   bufferType == USERBUFFER_TF16);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;

    snpe = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, platformConfig, usingInitCaching);
    if (snpe == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       return EXIT_FAILURE;
    }
    if (usingInitCaching)
    {
       if (container->save(dlc))
       {
          std::cout << "Saved container into archive successfully" << std::endl;
       }
       else
       {
          std::cout << "Failed to save container into archive" << std::endl;
       }
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);

    // Load contents of input file batches ino a SNPE tensor or user buffer,
    // user buffer include cpu buffer and OpenGL buffer,
    // execute the network with the input and save each of the returned output to a file.
    if(useUserSuppliedBuffers)
    {
       // SNPE allows its input and output buffers that are fed to the network
       // to come from user-backed buffers. First, SNPE buffers are created from
       // user-backed storage. These SNPE buffers are then supplied to the network
       // and the results are stored in user-backed output buffers. This allows for
       // reusing the same buffers for multiple inputs and outputs.
       zdl::DlSystem::UserBufferMap inputMap, outputMap;
       std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;

       if( bufferType == USERBUFFER_TF8 || bufferType == USERBUFFER_TF16 )
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, true, bitWidth);

          std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
          createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, true, staticQuantization, bitWidth);

          for( size_t i = 0; i < inputs.size(); i++ )
          {
             // Load input user buffer(s) with values from file(s)
             if( batchSize > 1 )
                std::cout << "Batch " << i << ":" << std::endl;
             if(!loadInputUserBufferTfN(applicationInputBuffers, snpe, inputs[i], inputMap, staticQuantization, bitWidth))
             {
                 return EXIT_FAILURE;
             }
             // Execute the input buffer map on the model with SNPE
             execStatus = snpe->execute(inputMap, outputMap);
             // Save the execution results only if successful
             if (execStatus == true)
             {
                if(!saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, true, bitWidth))
                {
                    return EXIT_FAILURE;
                }

             }
             else
             {
                std::cerr << "Error while executing the network." << std::endl;
             }
          }
       }
       else if( bufferType == USERBUFFER_FLOAT )
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, false, bitWidth);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, false, false, bitWidth);

             for( size_t i = 0; i < inputs.size(); i++ )
             {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                   std::cout << "Batch " << i << ":" << std::endl;
                if(!loadInputUserBufferFloat(applicationInputBuffers, snpe, inputs[i]))
                {
                    return EXIT_FAILURE;
                }
                // Execute the input buffer map on the model with SNPE
                execStatus = snpe->execute(inputMap, outputMap);
                // Save the execution results only if successful
                if (execStatus == true)
                {
                   if(!saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, false, bitWidth))
                   {
                       return EXIT_FAILURE;
                   }
                }
                else
                {
                   std::cerr << "Error while executing the network." << std::endl;
                }
             }
          }
       }
    }
    else if(bufferType == ITENSOR)
    {
        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;
        //Get input names and number
        const auto& inputTensorNamesRef = snpe->getInputTensorNames();
        if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
        const auto &inputTensorNames = *inputTensorNamesRef;

        for (size_t i = 0; i < inputs.size(); i++) {
            // Load input/output buffers with ITensor
            if(batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            if (inputTensorNames.size() == 1)
            {
                // Load input/output buffers with ITensor
                std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputs[i], inputTensorNames);
                if(!inputTensor)
                {
                    return EXIT_FAILURE;
                }
                // Execute the input tensor on the model with SNPE
                execStatus = snpe->execute(inputTensor.get(), outputTensorMap);
            }
            else
            {
                std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputTensors(inputTensorNames.size());
                zdl::DlSystem::TensorMap inputTensorMap;
                bool inputLoadStatus = false;
                // Load input/output buffers with TensorMap
                std::tie(inputTensorMap, inputLoadStatus) = loadMultipleInput(snpe, inputs[i], inputTensorNames, inputTensors);
                if(!inputLoadStatus)
                {
                    return EXIT_FAILURE;
                }
                // Execute the multiple input tensorMap on the model with SNPE
                execStatus = snpe->execute(inputTensorMap, outputTensorMap);
            }
            // Save the execution results if execution successful
            if (execStatus == true)
            {
               if(!saveOutput(outputTensorMap, OutputDir, i * batchSize, batchSize))
               {
                   return EXIT_FAILURE;
               }
            }
            else
            {
               std::cerr << "Error while executing the network." << std::endl;
            }
        }
    }
    // Freeing of snpe object
    snpe.reset();
    return SUCCESS;
}
