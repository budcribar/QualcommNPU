//==============================================================================
//
//  Copyright (c) 2017-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

#include "LoadInputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"


// Load a batched single input tensor for a network which requires a single input
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                                                            std::vector<std::string>& fileLines,
                                                            const zdl::DlSystem::StringList& inputTensorNames)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    // Make sure the network requires only a single input
    assert (inputTensorNames.size() == 1);

    // If the network has a single input, each line represents the input file to be loaded for that input
    std::vector<float> inputVec;
    for(size_t i=0; i<fileLines.size(); i++) {
        std::string filePath(fileLines[i]);
        std::cout << "Processing DNN Input: " << filePath << "\n";
        std::vector<float> loadedFile = loadFloatDataFile(filePath);
        inputVec.insert(inputVec.end(), loadedFile.begin(), loadedFile.end());
    }

    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(inputTensorNames.at(0));
    const auto &inputShape = *inputDims_opt;

    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains the expected number of elements.
       With the input dimensions computed create a tensor to convey the input into the network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    //Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    zdl::DlSystem::TensorShape tensorShape= snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    if(fileLines.size()<batchSize) {
           for(size_t j=0; j<batchSize-fileLines.size(); j++) {
                std::vector<float> padding(input->getSize()/batchSize,0);
                inputVec.insert(inputVec.end(),padding.begin(),padding.end());
           }
    }

    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }

    /* Copy the loaded input file contents into the networks input tensor. SNPE's ITensor supports C++ STL functions like std::copy() */
    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}

// Load multiple input tensors for a network which require multiple inputs
std::tuple<zdl::DlSystem::TensorMap, bool> loadMultipleInput (std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                                                                std::vector<std::string>& fileLines,
                                                                const zdl::DlSystem::StringList& inputTensorNames,
                                                                std::vector<std::unique_ptr<zdl::DlSystem::ITensor>>& inputs)
{
    zdl::DlSystem::TensorMap dummy; // dummy map for returning on failure
    // Make sure the network requires multiple inputs
    assert (inputTensorNames.size() > 1);

    if (inputTensorNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    zdl::DlSystem::TensorMap  inputTensorMap;
    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // Treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        // Parsing <inputname>:=<filepath> format
        std::unordered_map<std::string, std::string> nameToFilePathMap;
        if (fileLine.find(":=") != std::string::npos) {
            for (auto &line : filePaths) {
                if (line.empty())
                    continue;

                std::vector<std::string> lineContents;
                // Line is in format "<inputname>:=<filepath>"
                split(lineContents, line, '='); // Splits up to "<inputname>:", "<filepath>"

                std::string name = lineContents[0];
                name.erase(name.length()-1); // removes colon (i.e. "<inputname>:" -> "<inputname>")

                nameToFilePathMap.emplace(name, lineContents[1]);
            }
        }

        for (size_t j = 0; j<inputTensorNames.size(); j++) {
            std::string inputName(inputTensorNames.at(j));

            // If nameToFilePathMap contains name then use the filepath associated with that name
            // Otherwise, use filepath at index j of filePaths
            std::string filePath(nameToFilePathMap.find(inputName) != nameToFilePathMap.end() ? nameToFilePathMap.at(inputName) : filePaths[j]);
            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            std::vector<float> inputVec = loadFloatDataFile(filePath);

            const auto &inputShape_opt = snpe->getInputDimensions(inputTensorNames.at(j));
            const auto &inputShape = *inputShape_opt;
            inputs[j] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

            if (inputs[j]->getSize() != inputVec.size()) {
                std::cerr << "Size of input does not match network.\n"
                          << "Expecting: " << inputs[j]->getSize() << "\n"
                          << "Got: " << inputVec.size() << "\n";
                return std::make_tuple(dummy, false);
            }

            std::copy(inputVec.begin(), inputVec.end(), inputs[j]->begin());
            inputTensorMap.add(inputName.c_str(), inputs[j].get());
        }
    }
    std::cout << "Finished processing inputs for current inference \n";
    return std::make_tuple(inputTensorMap, true);
}

bool loadInputUserBufferTfN(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines,
                         zdl::DlSystem::UserBufferMap& inputMap,
                         bool staticQuantization,
                         int bitWidth)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        // Parsing <inputname>:=<filepath> format
        std::unordered_map<std::string, std::string> nameToFilePathMap;
        if (fileLine.find(":=") != std::string::npos) {
            for (auto &line : filePaths) {
                if (line.empty())
                    continue;

                std::vector<std::string> lineContents;
                // Line is in format "<inputname>:=<filepath>"
                split(lineContents, line, '='); // Splits up to "<inputname>:", "<filepath>"

                std::string name = lineContents[0];
                name.erase(name.length()-1); // removes colon (i.e. "<inputname>:" -> "<inputname>")

                nameToFilePathMap.emplace(name, lineContents[1]);
            }
        }

        for (size_t j = 0; j < inputNames.size(); j++) {
            const char *name = inputNames.at(j);

            // If nameToFilePathMap contains name then use the filepath associated with that name
            // Otherwise, use filepath at index j of filePaths
            std::string filePath(nameToFilePathMap.find(name) != nameToFilePathMap.end() ? nameToFilePathMap.at(name) : filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            if (staticQuantization) {
                // If static quantization is enabled then get the quantization parameters
                // from the user buffer and use them to load the file contents
                auto userBufferEncoding = dynamic_cast<zdl::DlSystem::UserBufferEncodingTfN *>(&inputMap.getUserBuffer(name)->getEncoding());
                unsigned char stepEquivalentTo0 = userBufferEncoding->getStepExactly0();
                float quantizedStepSize = userBufferEncoding->getQuantizedStepSize();

                if(!loadByteDataFileBatchedTfN(filePath, applicationBuffers.at(name), i, stepEquivalentTo0, quantizedStepSize, staticQuantization, bitWidth))
                    return false;
            } else {
                // If static quantization is disabled then get the quantization parameters
                // dynamically from the inputs to load the file contents and set them to user buffer
                unsigned char stepEquivalentTo0;
                float quantizedStepSize;

                if(!loadByteDataFileBatchedTfN(filePath, applicationBuffers.at(name), i, stepEquivalentTo0, quantizedStepSize, staticQuantization, bitWidth))
                    return false;

                auto userBufferEncoding = dynamic_cast<zdl::DlSystem::UserBufferEncodingTfN *>(&inputMap.getUserBuffer(name)->getEncoding());
                userBufferEncoding->setStepExactly0(stepEquivalentTo0);
                userBufferEncoding->setQuantizedStepSize(quantizedStepSize);
            }
        }
    }
    return true;
}

// Load multiple batched input user buffers
bool loadInputUserBufferFloat(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        // Parsing <inputname>:=<filepath> format
        std::unordered_map<std::string, std::string> nameToFilePathMap;
        if (fileLine.find(":=") != std::string::npos) {
            for (auto &line : filePaths) {
                if (line.empty())
                    continue;

                std::vector<std::string> lineContents;
                // Line is in format "<inputname>:=<filepath>"
                split(lineContents, line, '='); // Splits up to "<inputname>:", "<filepath>"

                std::string name = lineContents[0];
                name.erase(name.length()-1); // removes colon (i.e. "<inputname>:" -> "<inputname>")

                nameToFilePathMap.emplace(name, lineContents[1]);
            }
        }

        for (size_t j = 0; j < inputNames.size(); j++) {
            const char *name = inputNames.at(j);

            // If nameToFilePathMap contains name then use the filepath associated with that name
            // Otherwise, use filepath at index j of filePaths
            std::string filePath(nameToFilePathMap.find(name) != nameToFilePathMap.end() ? nameToFilePathMap.at(name) : filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            if(!loadByteDataFileBatched(filePath, applicationBuffers.at(name), i))
            {
                return false;
            }
        }
    }
    return true;
}

void loadInputUserBuffer(std::unordered_map<std::string, GLuint>& applicationBuffers,
                               std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                               const GLuint inputglbuffer)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    for (size_t i = 0; i < inputNames.size(); i++) {
        const char* name = inputNames.at(i);
        applicationBuffers.at(name) = inputglbuffer;
    };
}
