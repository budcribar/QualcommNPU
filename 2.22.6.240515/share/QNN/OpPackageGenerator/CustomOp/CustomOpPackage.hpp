//==============================================================================
//
// Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "CustomOpRegister.hpp"

namespace qnn {

namespace custom {

//============================================================================
// Backend Defined Behavior
//=============================================================================
/**
 * @brief This is an abstract class which that manages op creation and registration. It is a
 * singleton class that holds all op registration functions. Create and free kernels
 * must be implemented with BE specific behavior
 */
class CustomOpPackage {
 public:
  const char* m_packageName;
  /**
   * @brief This adds an op registration structure to an internal map based on its name
   * @param name
   * @param opRegistration
   */
  inline void addOpRegistration(const char* typeName, CustomOpRegistration_t* opRegistration) {
    m_registered_ops[typeName] = opRegistration;
  };

  /**
   * @brief Returns a singleton package instance
   */
  static std::shared_ptr<CustomOpPackage> getInstance() noexcept;

  /**
   * Initializes the package and sets the mutex
   * @param isInitialized
   */

  static void setIsInitialized(bool isInitialized);

  /**
   * Returns True if the package object is initialized, False otherwise.
   */

  static bool getIsInitialized();

  /**
   * @brief Returns an op registration pointer based on the operation type
   * @param typeName The type of the operation
   * @return
   */
  const CustomOpRegistration_t* getOpRegistration(const std::string& typeName) {
    if (m_registered_ops.find(typeName) == m_registered_ops.end()) {
      return nullptr;
    }
    return m_registered_ops[typeName];
  }

  /**
   * @brief Create Op implementation with executable content for a given node.
   * @file  See QnnOpPackage.h for details
   * @return Error code:
   *         - QNN_SUCCESS: Op implementation is created successfully
   *         - QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE: Failed to create op implementation
   *           due to invalid graph infrastructure content.
   *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: one or more invalid arguments (e.g. NULL)
   *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: API not supported
   *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
   */
  virtual Qnn_ErrorHandle_t createOpImpl(QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                         QnnOpPackage_Node_t node,
                                         QnnOpPackage_OpImpl_t* opImplPtr);

  /**
   * @brief Free the resources associated with Op implementation previously allocated by
   *        QnnOpPackage_CreateOpImplFn_t.
   * @file See QnnOpPackage.h for details
   * @return Error code:
   *         - QNN_SUCCESS if Op implementation resources are successfully freed.
   *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: _opImpl_ argument was NULL.
   *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: API not supported.
   *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
   */
  virtual Qnn_ErrorHandle_t freeOpImpl(QnnOpPackage_OpImpl_t opImpl);

  /**
   * @brief Retrieve a QnnOpPackage_Info_t struct from an Op package library describing all
   *        operations and optimizations provided by the library.
   * @file See QnnOpPackage.h for details
   * @return Error code:
   *         - QNN_SUCCESS: Info is fetched successfully.
   *         - QNN_OP_PACKAGE_ERROR_INVALID_INFO: 'info' argument was NULL or invalid.
   *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
   */
  virtual Qnn_ErrorHandle_t getPackageInfo(const QnnOpPackage_Info_t** info);

  /**
   * Retrieves a custom operation based on the handle
   * @param[in] handle A handle to an operation that will be used for retrieval
   * @param[out] op A shared ptr to a custom operation
   * @return QNN_SUCCESS if the operation exists
   */
  const CustomOpResolver* getOpResolver() const { return m_opResolver.get(); }

  /**
   * Reset Op package instance
   */
  static void destroyInstance();

  void freeResolver();

  ~CustomOpPackage() {
    freeResolver();
    destroyInstance();
  }

 protected:
  CustomOpPackage() = default;

  CustomOpPackage(const CustomOpPackage& other) = delete;

  CustomOpPackage& operator=(const CustomOpPackage& other) = delete;

  static std::mutex s_mtx;
  static bool s_isInitialized;
  static std::shared_ptr<CustomOpPackage> s_opPackageInstance;

  std::vector<const char*> m_operationNames;
  std::unique_ptr<CustomOpResolver> m_opResolver;
  std::map<std::string, CustomOpRegistration_t*> m_registered_ops;
  std::list<std::shared_ptr<QnnCpuOpPackage_OpImpl_t>> m_OpImplList;

  QnnOpPackage_Info_t m_packageInfo;
  Qnn_ApiVersion_t m_sdkApiVersion;
};

namespace macros {

// defaults to initializing the singleton instance
#define INIT_BE_OP_PACKAGE(name)                           \
  auto package           = CustomOpPackage::getInstance(); \
  package->m_packageName = #name;

#define REGISTER_PACKAGE_OP(opName)                        \
  using namespace qnn::custom;                             \
  extern OpRegistrationReceiver opName##_receiver;         \
  if (opName##_receiver.isOpReceiver(#opName)) {           \
    auto regFunction = opName##_receiver.pop();            \
    package->addOpRegistration(#opName, (*regFunction)()); \
  } else {                                                 \
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;   \
  }                                                        \
  // backend specific setup goes here

#define INIT_BE_PACKAGE_OPTIMIZATIONS()

}  // namespace macros
}  // namespace custom
}  // namespace qnn
