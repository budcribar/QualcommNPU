//==============================================================================
//
// Copyright (c) 2020, 2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string.h>

#include <map>
#include <memory>

#include "CustomOpUtils.hpp"

using opHandle = std::size_t;

namespace qnn {

namespace custom {

/**
 * This struct holds pointers to functions that a user defines and associates with a package object
 * or CustomOp. Only the execute, validate and free functions must be defined, all other functions
 * can be set to null at the users discretion.
 */
typedef struct _CustomOpRegistration_t {
  Qnn_ErrorHandle_t (*execute)(utils::CustomOp* operation);         // single mode
  Qnn_ErrorHandle_t (*finalize)(const utils::CustomOp* operation);  // single mode
  Qnn_ErrorHandle_t (*free)(utils::CustomOp& op);

  QnnOpPackage_ValidateOpConfigFn_t validateOpConfig;

  Qnn_ErrorHandle_t (*initialize)(const QnnOpPackage_Node_t opNode,
                                  QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                  utils::CustomOp* operation);
} CustomOpRegistration_t;

// This describes a function which returns a custom op registration defined below.
typedef CustomOpRegistration_t* (*CustomOpRegistrationFunction_t)();

/**
 * @brief Stores a pointer to function which should return a static opRegistration object when
 * called. The registration function stored here is temporary and will be transferred to an op
 * package object.
 */
struct OpRegistrationReceiver {
 public:
  OpRegistrationReceiver(const char* opTypeName, CustomOpRegistrationFunction_t regFunction) {
    opType = opTypeName;
    func.reset(new CustomOpRegistrationFunction_t(regFunction));
  }

  /**
   * @brief Checks if the operation type is associated with this receiver
   * @param opType The name of the operation
   * @return True if this receiver holds the registration function, false otherwise
   */
  bool isOpReceiver(const char* opTypeName) const { return strcmp(opType, opTypeName) == 0; }

  /**
   *
   * @return The function stored in this class
   */
  CustomOpRegistrationFunction_t* pop() { return func.release(); }

  ~OpRegistrationReceiver() {
    if (func) func.release();
  }

 private:
  const char* opType;
  std::unique_ptr<CustomOpRegistrationFunction_t> func;
};

/**
 * @brief This classes manages custom operations which are created from node objects. Each operation
 * is assigned a unique index and cached in a map.
 */
class CustomOpResolver {
 public:
  CustomOpResolver()                              = default;
  CustomOpResolver(const CustomOpResolver& other) = delete;

  /**
   * Registers a custom operation in a map
   * @param op A shared pointer to a populated custom operation
   * @return A unique handle based on the operation name
   */
  opHandle registerCustomOp(std::shared_ptr<utils::CustomOp>&& op) {
    opHandle handle     = (opHandle)(op.get());
    m_customOps[handle] = op;
    return handle;
  }

  /**
   * Retrieves a custom operation based on the handle
   * @param[in] handle A handle to an operation that will be used for retrieval
   * @param[out] op A shared ptr to a custom operation
   * @return QNN_SUCCESS if the operation exists
   */
  Qnn_ErrorHandle_t getCustomOp(opHandle handle, std::shared_ptr<utils::CustomOp>& op) const {
    if (m_customOps.find(handle) == m_customOps.end()) {
      return QNN_OP_PACKAGE_ERROR_GENERAL;
    }

    op = m_customOps.find(handle)->second;

    return QNN_SUCCESS;
  }

  /**
   * Removes the custom operation associated with the handle
   * @param handle A size t handle that should correspond to an operation in the map
   * @return QNN_SUCCESS if the operation exists
   *         QNN_OP_PACKAGE_ERROR_GENERAL if the handle is not found
   */
  Qnn_ErrorHandle_t removeCustomOp(opHandle handle) {
    if (m_customOps.find(handle) == m_customOps.end()) {
      return QNN_OP_PACKAGE_ERROR_GENERAL;
    }
    m_customOps.at(handle).reset();
    m_customOps.erase(handle);
    return QNN_SUCCESS;
  }

  virtual ~CustomOpResolver() = default;

 private:
  std::map<opHandle, std::shared_ptr<utils::CustomOp>> m_customOps;
};

namespace macros {
// Receives all registration functions that have been defined in client code
#define REGISTER_OP(opName, func)                          \
  OpRegistrationReceiver opName##_receiver(#opName, func); \
  // backend specific setup goes here
}  // namespace macros

}  // namespace custom
}  // namespace qnn
