//==============================================================================
//
// Copyright (c) 2019-2020, 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <vector>
#include <functional>

#include "SnpeUdo/UdoReg.h"
#include "SnpeUdo/UdoBase.h"
#include "UdoMacros.hpp"

namespace UdoUtil {

/**
* @brief A helper class to create the UDO version struct
* This class automatically fills the API version from the UDO Base header,
* and allows a developer to specify a library version.
*
*/
class UdoVersion
{
public:
  SnpeUdo_LibVersion_t m_Version;

  /**
   * \brief Base constructor. initializes the API version based on headers.
   * Library version is set to 0.0.0.0
   */
  UdoVersion();

  /**
   * \brief A function to set the library version, in case the default constructor was used
   * API version is not touched
   * Library version is set based on values passed
   */
  void setUdoVersion(uint32_t libMajor, uint32_t libMinor, uint32_t libPatch);

  /**
   * \brief A function to return the Library version struct
   * Note that it returns a pointer to the m_version class member.
   */
  SnpeUdo_ErrorType_t
  getLibraryVersion(SnpeUdo_LibVersion_t** version)
  {
    *version =  &m_Version;
    return SNPE_UDO_NO_ERROR;
  }
};

/**
 * @brief A helper class to create the Library information struct in the
 * registration object.
 * This class provides easy creation of library info using C++ constructor.
 * Once created, the class provides a copy method to serialize the info.
 */
class UdoLibraryInfo
{
public:
  std::string m_LibraryName;

  UdoLibraryInfo(std::string&& libraryName,
         SnpeUdo_CoreType_t udoCoreType)
    : m_LibraryName(libraryName), m_CoreType(udoCoreType) {}

  SnpeUdo_ErrorType_t copyLibraryInfo(SnpeUdo_LibraryInfo_t* libraryInfoPtr);

  SnpeUdo_CoreType_t getCoreType() { return m_CoreType; }

private:
  SnpeUdo_CoreType_t m_CoreType;

};

/**
* @brief A helper class to create Operation information struct
* This class provides the following :
* - Creation basic Operation info by using C++ constructor
* - Provide "add" methods for params and Core-specific info
* - Provide a serialization function ("copy") which allocates a C struct
*   and copies all information to it
*/
class UdoOperationInfo
{
public:
  std::string m_OperationType;

  UdoOperationInfo(std::string&& operationType,
           SnpeUdo_Bitmask_t supportedCores,
           int numOfInputs,
           int numOfOutputs);

private:
  int m_NumOfInputs;
  int m_NumOfOutputs;
  SnpeUdo_Bitmask_t m_SupportedCoreTypes;
};

class UdoRegLibrary
{
public:
  std::string m_PackageName; // Pointer to hold the package name string
  SnpeUdo_LibraryInfo_t* m_LibraryInfo; // pointer to hold array of library info

  UdoRegLibrary(const std::string &packageName, SnpeUdo_Bitmask_t supportedCoreTypes);

  /**
   * \brief This function creates a new implementation library info
   * object, and adds it to internal vector
   */
  void
  addImplLib(std::string&& libName, SnpeUdo_CoreType_t udoCoreType);

  /**
   * \brief This function creates a new Operation info
   * object, and adds it to internal vector.
   * The function returns the info object so that the user can add parameters and
   * per-core information
   */
  std::shared_ptr<UdoOperationInfo>
  addOperation(std::string&& operationType,
               SnpeUdo_Bitmask_t supportedCores,
               int numOfInputs,
               int numOfOutputs);

  /**
   * \brief This function creates the registration struct and populate it
   * with the libraries information, operations information etc.
   * The function should be called after all libraries and operations were
   * added to the registration object using "addImplLib", "addOperation" etc.
   */
  SnpeUdo_ErrorType_t
  createRegInfoStruct();

  /**
   * \brief
   *  Populates the registration library info pointer passed with the private regInfo struct
   *  for a UdoRegLibrary instance. This function should be called after CreateRegInfoStruct.
   */
  SnpeUdo_ErrorType_t
  getLibraryRegInfo(SnpeUdo_RegInfo_t** regInfo) {
    *regInfo = m_RegInfo.get();
    return SNPE_UDO_NO_ERROR;
  }

private:
  SnpeUdo_ErrorType_t createImplLibInfo();
  SnpeUdo_ErrorType_t createOperationString();
  SnpeUdo_Bitmask_t m_SupportedCoreTypes;
  std::string m_OperationsString;
  std::shared_ptr<SnpeUdo_RegInfo_t> m_RegInfo; // struct pointer to hold reglibinfo
  std::vector<std::unique_ptr<UdoLibraryInfo>> m_UdoImplLibs;
  std::vector<std::shared_ptr<UdoOperationInfo>> m_UdoOperations;
};

}
