//==============================================================================
//
//  Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "DataUtil.hpp"
#include "Logger.hpp"
#ifndef __hexagon__
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#endif
#include "PAL/StringOp.hpp"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"
#include "QnnWrapperUtils.hpp"

#include <chrono>


using namespace qnn;
using namespace qnn::tools;

// Default path where the outputs will be stored if outputPath is
// not supplied.
const std::string sample_app::QnnSampleApp::s_defaultOutputPath = "./output/";

sample_app::QnnSampleApp::QnnSampleApp(QnnFunctionPointers qnnFunctionPointers,
                                       std::string inputListPaths,
                                       std::string opPackagePaths,
                                       void* backendLibraryHandle,
                                       std::string outputPath,
                                       bool debug,
                                       iotensor::OutputDataType outputDataType,
                                       iotensor::InputDataType inputDataType,
                                       sample_app::ProfilingLevel profilingLevel,
                                       bool dumpOutputs,
                                       std::string cachedBinaryPath,
                                       std::string saveBinaryName)
    : m_qnnFunctionPointers(qnnFunctionPointers),
      m_outputPath(outputPath),
      m_saveBinaryName(saveBinaryName),
      m_cachedBinaryPath(cachedBinaryPath),
      m_debug(debug),
      m_outputDataType(outputDataType),
      m_inputDataType(inputDataType),
      m_profilingLevel(profilingLevel),
      m_dumpOutputs(dumpOutputs),
      m_backendLibraryHandle(backendLibraryHandle),
      m_isBackendInitialized(false),
      m_isContextCreated(false) {
  split(m_inputListPaths, inputListPaths, ',');
  split(m_opPackagePaths, opPackagePaths, ',');
  if (m_outputPath.empty()) {
    m_outputPath = s_defaultOutputPath;
  }
  return;
}

sample_app::QnnSampleApp::~QnnSampleApp() {
  // Free Profiling object if it was created
  if (nullptr != m_profileBackendHandle) {
    QNN_DEBUG("Freeing backend profile object.");
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileFree(m_profileBackendHandle)) {
      QNN_ERROR("Could not free backend profile handle.");
    }
  }
  // Free context if not already done
  if (m_isContextCreated) {
    QNN_DEBUG("Freeing context");
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
      QNN_ERROR("Could not free context");
    }
  }
  m_isContextCreated = false;
  // Terminate backend
  if (m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) {
    QNN_DEBUG("Freeing backend");
    if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
      QNN_ERROR("Could not free backend");
    }
  }
  m_isBackendInitialized = false;
  // Terminate logging in the backend
  if (nullptr != m_qnnFunctionPointers.qnnInterface.logFree && nullptr != m_logHandle) {
    if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.logFree(m_logHandle)) {
      QNN_WARN("Unable to terminate logging in the backend.");
    }
  }
  return;
}

std::string sample_app::QnnSampleApp::getBackendBuildId() {
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS !=
      m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    QNN_ERROR("Unable to get build Id from the backend.");
  }
  return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

// Initialize QnnSampleApp. Things it does:
//  1. Create output directory
//  2. Read all input list paths provided
//      during creation.
sample_app::StatusCode sample_app::QnnSampleApp::initialize() {
  // Create Output Directory
#ifndef __hexagon__
  if (m_dumpOutputs && !::pal::FileOp::checkFileExists(m_outputPath) &&
      !pal::Directory::makePath(m_outputPath)) {
    exitWithMessage("Could not create output directory: " + m_outputPath, EXIT_FAILURE);
  }
#endif
  // Read Input File List
  bool readSuccess;
  std::tie(m_inputFileLists, m_inputNameToIndex, readSuccess) = readInputLists(m_inputListPaths);
  if (!readSuccess) {
    exitWithMessage("Could not read input lists", EXIT_FAILURE);
  }
  // initialize logging in the backend
  if (log::isLogInitialized()) {
    auto logCallback = log::getLogCallback();
    auto logLevel    = log::getLogLevel();
    QNN_INFO("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
             logCallback,
             logLevel);
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
      QNN_WARN("Unable to initialize logging in the backend.");
    }
  } else {
    QNN_WARN("Logging not available in the backend.");
  }
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::initializeProfiling() {
  if (ProfilingLevel::OFF != m_profilingLevel) {
    QNN_INFO("Profiling turned on; level = %d", m_profilingLevel);
    if (ProfilingLevel::BASIC == m_profilingLevel) {
      QNN_INFO("Basic profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          m_qnnFunctionPointers.qnnInterface.profileCreate(
              m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
        QNN_WARN("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
      QNN_INFO("Detailed profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          m_qnnFunctionPointers.qnnInterface.profileCreate(
              m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
        QNN_ERROR("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    }
  }
  return StatusCode::SUCCESS;
}

// Simple method to report error from app to lib.
int32_t sample_app::QnnSampleApp::reportError(const std::string& err) {
  QNN_ERROR("%s", err.c_str());
  return EXIT_FAILURE;
}

// Initialize a QnnBackend.
sample_app::StatusCode sample_app::QnnSampleApp::initializeBackend() {
  auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
      m_logHandle, (const QnnBackend_Config_t**)m_backendConfig, &m_backendHandle);
  if (QNN_BACKEND_NO_ERROR != qnnStatus) {
    QNN_ERROR("Could not initialize backend due to error = %d", qnnStatus);
    return StatusCode::FAILURE;
  }
  QNN_INFO("Initialize Backend Returned Status = %d", qnnStatus);
  m_isBackendInitialized = true;
  return StatusCode::SUCCESS;
}

// Terminate the backend after done.
sample_app::StatusCode sample_app::QnnSampleApp::terminateBackend() {
  if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) &&
      QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
    QNN_ERROR("Could not terminate backend");
    return StatusCode::FAILURE;
  }
  m_isBackendInitialized = false;
  return StatusCode::SUCCESS;
}

// Register op packages and interface providers supplied during
// object creation. If there are multiple op packages, register
// them sequentially in the order provided.
sample_app::StatusCode sample_app::QnnSampleApp::registerOpPackages() {
  const size_t pathIdx              = 0;
  const size_t interfaceProviderIdx = 1;
  for (auto const& opPackagePath : m_opPackagePaths) {
    std::vector<std::string> opPackage;
    split(opPackage, opPackagePath, ':');
    QNN_DEBUG("opPackagePath: %s", opPackagePath.c_str());
    const char* target     = nullptr;
    const size_t targetIdx = 2;
    if (opPackage.size() != 2 && opPackage.size() != 3) {
      QNN_ERROR("Malformed opPackageString provided: %s", opPackagePath.c_str());
      return StatusCode::FAILURE;
    }
    if (opPackage.size() == 3) {
      target = (char*)opPackage[targetIdx].c_str();
    }
    if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
      QNN_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
      return StatusCode::FAILURE;
    }
    if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                                    m_backendHandle,
                                    (char*)opPackage[pathIdx].c_str(),
                                    (char*)opPackage[interfaceProviderIdx].c_str(),
                                    target)) {
      QNN_ERROR("Could not register Op Package: %s and interface provider: %s",
                opPackage[pathIdx].c_str(),
                opPackage[interfaceProviderIdx].c_str());
      return StatusCode::FAILURE;
    }
    QNN_INFO("Registered Op Package: %s and interface provider: %s",
             opPackage[pathIdx].c_str(),
             opPackage[interfaceProviderIdx].c_str());
  }
  return StatusCode::SUCCESS;
}

// Create a Context in a backend.
sample_app::StatusCode sample_app::QnnSampleApp::createContext() {
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextCreate(m_backendHandle,
                                                       m_deviceHandle,
                                                       (const QnnContext_Config_t**)m_contextConfig,
                                                       &m_context)) {
    QNN_ERROR("Could not create context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = true;
  return StatusCode::SUCCESS;
}

// Free context after done.
sample_app::StatusCode sample_app::QnnSampleApp::freeContext() {
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
    QNN_ERROR("Could not free context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = false;
  return StatusCode::SUCCESS;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in m_graphsInfo and m_graphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
sample_app::StatusCode sample_app::QnnSampleApp::composeGraphs() {
  auto returnStatus = StatusCode::SUCCESS;
  if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR !=
      m_qnnFunctionPointers.composeGraphsFnHandle(
          m_backendHandle,
          m_qnnFunctionPointers.qnnInterface,
          m_context,
          (const qnn_wrapper_api::GraphConfigInfo_t**)m_graphConfigsInfo,
          m_graphConfigsInfoCount,
          &m_graphsInfo,
          &m_graphsCount,
          m_debug,
          log::getLogCallback(),
          log::getLogLevel())) {
    QNN_ERROR("Failed in composeGraphs()");
    returnStatus = StatusCode::FAILURE;
  }
  return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::finalizeGraphs() {
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.graphFinalize(
            (*m_graphsInfo)[graphIdx].graph, m_profileBackendHandle, nullptr)) {
      return StatusCode::FAILURE;
    }
  }
  if (ProfilingLevel::OFF != m_profilingLevel) {
    extractBackendProfilingInfo(m_profileBackendHandle);
  }
  auto returnStatus = StatusCode::SUCCESS;
  if (!m_saveBinaryName.empty()) {
    QNN_INFO("Before saveBinary(): saving context and metadata.");
    returnStatus = saveBinary();
  } else {
    QNN_DEBUG("m_saveBinaryName is empty()");
  }
  return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::createFromBinary() {
  if (m_cachedBinaryPath.empty()) {
    QNN_ERROR("No name provided to read binary file from.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return StatusCode::FAILURE;
  }
  uint64_t bufferSize{0};
  std::shared_ptr<uint8_t> buffer{nullptr};
  // read serialized binary into a byte buffer
  tools::datautil::StatusCode status{tools::datautil::StatusCode::SUCCESS};
  std::tie(status, bufferSize) = tools::datautil::getFileSize(m_cachedBinaryPath);
  if (0 == bufferSize) {
    QNN_ERROR("Received path to an empty file. Nothing to deserialize.");
    return StatusCode::FAILURE;
  }
  buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], std::default_delete<uint8_t[]>());
  if (!buffer) {
    QNN_ERROR("Failed to allocate memory.");
    return StatusCode::FAILURE;
  }

  status = tools::datautil::readBinaryFromFile(
      m_cachedBinaryPath, reinterpret_cast<uint8_t*>(buffer.get()), bufferSize);
  if (status != tools::datautil::StatusCode::SUCCESS) {
    QNN_ERROR("Failed to read binary data.");
    return StatusCode::FAILURE;
  }

  // inspect binary info
  auto returnStatus = StatusCode::SUCCESS;
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle.");
    returnStatus = StatusCode::FAILURE;
  }
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (StatusCode::SUCCESS == returnStatus &&
      QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                         sysCtxHandle,
                         static_cast<void*>(buffer.get()),
                         bufferSize,
                         &binaryInfo,
                         &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info");
    returnStatus = StatusCode::FAILURE;
  }

  // fill GraphInfo_t based on binary info
  if (StatusCode::SUCCESS == returnStatus &&
      !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
    QNN_ERROR("Failed to copy metadata.");
    returnStatus = StatusCode::FAILURE;
  }
  m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
  sysCtxHandle = nullptr;

  if (StatusCode::SUCCESS == returnStatus &&
      nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
    QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr.");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS == returnStatus &&
      m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
          m_backendHandle,
          m_deviceHandle,
          (const QnnContext_Config_t**)m_contextConfig,
          static_cast<void*>(buffer.get()),
          bufferSize,
          &m_context,
          m_profileBackendHandle)) {
    QNN_ERROR("Could not create context from binary.");
    returnStatus = StatusCode::FAILURE;
  }
  if (ProfilingLevel::OFF != m_profilingLevel) {
    extractBackendProfilingInfo(m_profileBackendHandle);
  }
  m_isContextCreated = true;
  if (StatusCode::SUCCESS == returnStatus) {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
      if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
        QNN_ERROR("graphRetrieveFnHandle is nullptr.");
        returnStatus = StatusCode::FAILURE;
        break;
      }
      if (QNN_SUCCESS !=
          m_qnnFunctionPointers.qnnInterface.graphRetrieve(
              m_context, (*m_graphsInfo)[graphIdx].graphName, &((*m_graphsInfo)[graphIdx].graph))) {
        QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
      }
    }
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_DEBUG("Cleaning up graph Info structures.");
    qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  }
  return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::saveBinary() {
  if (m_saveBinaryName.empty()) {
    QNN_ERROR("No name provided to save binary file.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinarySize ||
      nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinary) {
    QNN_ERROR("contextGetBinarySizeFnHandle or contextGetBinaryFnHandle is nullptr.");
    return StatusCode::FAILURE;
  }
  uint64_t requiredBufferSize{0};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextGetBinarySize(m_context, &requiredBufferSize)) {
    QNN_ERROR("Could not get the required binary size.");
    return StatusCode::FAILURE;
  }
  std::unique_ptr<uint8_t[]> saveBuffer(new uint8_t[requiredBufferSize]);
  if (nullptr == saveBuffer) {
    QNN_ERROR("Could not allocate buffer to save binary.");
    return StatusCode::FAILURE;
  }
  uint64_t writtenBufferSize{0};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextGetBinary(m_context,
                                                          reinterpret_cast<void*>(saveBuffer.get()),
                                                          requiredBufferSize,
                                                          &writtenBufferSize)) {
    QNN_ERROR("Could not get binary.");
    return StatusCode::FAILURE;
  }
  if (requiredBufferSize < writtenBufferSize) {
    QNN_ERROR(
        "Illegal written buffer size [%d] bytes. Cannot exceed allocated memory of [%d] bytes",
        writtenBufferSize,
        requiredBufferSize);
    return StatusCode::FAILURE;
  }
#ifndef __hexagon__
  auto dataUtilStatus = tools::datautil::writeBinaryToFile(
      m_outputPath, m_saveBinaryName + ".bin", (uint8_t*)saveBuffer.get(), writtenBufferSize);
  if (tools::datautil::StatusCode::SUCCESS != dataUtilStatus) {
    QNN_ERROR("Error while writing binary to file.");
    return StatusCode::FAILURE;
  }
#endif
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("Backend Profile handle is nullptr; may not be initialized.");
    return StatusCode::FAILURE;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(
                                  profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in profile get events.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event));
    extractProfilingSubEvents(*(profileEvents + event));
  }
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractProfilingSubEvents(
    QnnProfile_EventId_t profileEventId) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
                                  profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in profile get sub events.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent));
    extractProfilingSubEvents(*(profileSubEvents + subEvent));
  }
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractProfilingEvent(
    QnnProfile_EventId_t profileEventId) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("Printing Event Info - Event Type: [%d], Event Value: [%" PRIu64
            "], Event Identifier: [%s], Event Unit: [%d]",
            eventData.type,
            eventData.value,
            eventData.identifier,
            eventData.unit);
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
  auto returnStatus = sample_app::StatusCode::FAILURE;
  switch (errCode) {
    case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
      returnStatus = sample_app::StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
      break;
    case QNN_COMMON_ERROR_SYSTEM:
      returnStatus = sample_app::StatusCode::FAILURE_SYSTEM_ERROR;
      break;
    case QNN_COMMON_ERROR_NOT_SUPPORTED:
      returnStatus = sample_app::StatusCode::QNN_FEATURE_UNSUPPORTED;
      break;
    default:
      break;
  }
  return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::isDevicePropertySupported() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.propertyHasCapability) {
    auto qnnStatus =
        m_qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
      QNN_WARN("Device property is not supported");
    }
    if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
      QNN_ERROR("Device property is not known to backend");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::createDevice() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceCreate) {
    auto qnnStatus =
        m_qnnFunctionPointers.qnnInterface.deviceCreate(m_logHandle, nullptr, &m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      QNN_ERROR("Failed to create device");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::freeDevice() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceFree) {
    auto qnnStatus = m_qnnFunctionPointers.qnnInterface.deviceFree(m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      QNN_ERROR("Failed to free device");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}


// executeGraphs() that is currently used by qnn-sample-app's main.cpp.
// This function runs all the graphs present in model.so by reading
// inputs from input_list based files and writes output to .raw files.
sample_app::StatusCode sample_app::QnnSampleApp::executeGraphs(int seconds) {
    printf("Offsets for Qnn_TensorV2_t:\n");
    printf("Offset of id: %zu\n", offsetof(Qnn_TensorV2_t, id));
    printf("Offset of name: %zu\n", offsetof(Qnn_TensorV2_t, name));
    printf("Offset of type: %zu\n", offsetof(Qnn_TensorV2_t, type));
    printf("Offset of dataFormat: %zu\n", offsetof(Qnn_TensorV2_t, dataFormat));
    printf("Offset of dataType: %zu\n", offsetof(Qnn_TensorV2_t, dataType));
    printf("Offset of quantizeParams: %zu\n", offsetof(Qnn_TensorV2_t, quantizeParams));
    printf("Offset of rank: %zu\n", offsetof(Qnn_TensorV2_t, rank));
    printf("Offset of dimensions: %zu\n", offsetof(Qnn_TensorV2_t, dimensions));
    printf("Offset of memType: %zu\n", offsetof(Qnn_TensorV2_t, memType));
    printf("Offset of clientBuf: %zu\n", offsetof(Qnn_TensorV2_t, clientBuf));
    printf("Offset of memHandle: %zu\n", offsetof(Qnn_TensorV2_t, memHandle));
    printf("Offset of isDynamicDimensions: %zu\n", offsetof(Qnn_TensorV2_t, isDynamicDimensions));
    printf("Offset of sparseParams: %zu\n", offsetof(Qnn_TensorV2_t, sparseParams));
    printf("Offset of isProduced: %zu\n", offsetof(Qnn_TensorV2_t, isProduced));

    printf("Size of Qnn_TensorV2_t: %zu\n", sizeof(Qnn_TensorV2_t));


  auto returnStatus = StatusCode::SUCCESS;
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    QNN_DEBUG("Starting execution for graphIdx: %d", graphIdx);
    if (graphIdx >= m_inputFileLists.size()) {
      QNN_ERROR("No Inputs available for: %d", graphIdx);
      returnStatus = StatusCode::FAILURE;
      break;
    }
    Qnn_Tensor_t* inputs  = nullptr;
    Qnn_Tensor_t* outputs = nullptr;
    if (iotensor::StatusCode::SUCCESS !=
        m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx])) {
      QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
      returnStatus = StatusCode::FAILURE;
      break;
    }
    auto inputFileList = m_inputFileLists[graphIdx];
    auto graphInfo     = (*m_graphsInfo)[graphIdx];
    if (!inputFileList.empty()) {
      size_t totalCount           = inputFileList[0].size();
      size_t inputFileIndexOffset = 0;
      while (inputFileIndexOffset < totalCount) {
        iotensor::StatusCode iotReturnStatus;
        size_t numInputFilesPopulated;
        size_t batchSize;
        std::tie(iotReturnStatus, numInputFilesPopulated, batchSize) =
            m_ioTensor.populateInputTensors(graphIdx,
                                            inputFileList,
                                            inputFileIndexOffset,
                                            false,
                                            m_inputNameToIndex[graphIdx],
                                            inputs,
                                            graphInfo,
                                            m_inputDataType);
        if (iotensor::StatusCode::SUCCESS != iotReturnStatus) {
          returnStatus = StatusCode::FAILURE;
        }
        if (StatusCode::SUCCESS == returnStatus) {
          QNN_DEBUG("Successfully populated input tensors for graphIdx: %d", graphIdx);
          Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;

          
          // Record the start time
          auto startTime = std::chrono::steady_clock::now();

          // Get the duration in seconds
          auto duration = std::chrono::seconds(seconds);

          while (true) {
              auto currentTime = std::chrono::steady_clock::now();
              auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);

              // Print the elapsed time
              std::cout << "Elapsed time: " << elapsedTime.count() << " seconds\n";

              // Check if the duration has been exceeded
              if (elapsedTime >= duration) {
                  break;
              }
              executeStatus =
                  m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                      inputs,
                      graphInfo.numInputTensors,
                      outputs,
                      graphInfo.numOutputTensors,
                      m_profileBackendHandle,
                      nullptr);
              if (QNN_GRAPH_NO_ERROR != executeStatus) {
                  returnStatus = StatusCode::FAILURE;
                  break;
              }
          }
          auto endTime = std::chrono::steady_clock::now();
          std::cout << "End time: " << std::chrono::duration_cast<std::chrono::seconds>(endTime.time_since_epoch()).count() << " seconds since epoch\n";

          if (StatusCode::SUCCESS == returnStatus) {
            QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
#ifndef __hexagon__
            if (iotensor::StatusCode::SUCCESS !=
                m_ioTensor.writeOutputTensors(graphIdx,
                                              inputFileIndexOffset,
                                              graphInfo.graphName,
                                              outputs,
                                              graphInfo.numOutputTensors,
                                              m_outputDataType,
                                              m_graphsCount,
                                              m_outputPath,
                                              numInputFilesPopulated,
                                              batchSize)) {
              returnStatus = StatusCode::FAILURE;
            }
#endif
          }
          inputFileIndexOffset += numInputFilesPopulated;
        }
        if (StatusCode::SUCCESS != returnStatus) {
          QNN_ERROR("Execution of Graph: %d failed!", graphIdx);
          break;
        }
      }
    }
    m_ioTensor.tearDownInputAndOutputTensors(
        inputs, outputs, graphInfo.numInputTensors, graphInfo.numOutputTensors);
    inputs  = nullptr;
    outputs = nullptr;
    if (StatusCode::SUCCESS != returnStatus) {
      break;
    }
  }

  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  m_graphsInfo = nullptr;
  return returnStatus;
}
