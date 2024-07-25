//==============================================================================
//
// Copyright (c) 2019-2021, 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "UdoUtil.hpp"

using namespace UdoUtil;

#define UDO_CHECK_POINTER(ptr) {if (ptr == nullptr) return false;}

// Utility function to initialize all fields of SnpeUdo_RegInfo_t structure
bool
initUdoRegInfoStruct(std::shared_ptr<SnpeUdo_RegInfo_t> &regInfoStruct,
                     const std::string &packageName,
                     SnpeUdo_Bitmask_t supportedCoreTypes,
                     uint32_t numOfImplementationLibValue,
                     SnpeUdo_LibraryInfo_t* implementationLibValue,
                     const std::string &operationsString,
                     uint32_t numOfOperationsValue) {
    UDO_CHECK_POINTER(regInfoStruct);

    regInfoStruct->packageName = const_cast<char*>(packageName.c_str());
    // TODO: Check if supportedCoreTypes is needed or not
    regInfoStruct->supportedCoreTypes = supportedCoreTypes;
    regInfoStruct->numOfImplementationLib = numOfImplementationLibValue;
    regInfoStruct->implementationLib = implementationLibValue;
    regInfoStruct->operationsString = const_cast<char*>(operationsString.c_str());
    regInfoStruct->numOfOperations = numOfOperationsValue;

    return true;
}

bool
setVersionStruct(SnpeUdo_Version_t* versionStruct,
                 uint32_t majorValue,
                 uint32_t minorValue,
                 uint32_t patchValue) {
    UDO_CHECK_POINTER(versionStruct);

    versionStruct->major = majorValue;
    versionStruct->minor = minorValue;
    versionStruct->teeny = patchValue;
    return true;
}


UdoVersion::UdoVersion() {
    // Get API version from UdoBase header file
    setVersionStruct(&m_Version.apiVersion,
                     API_VERSION_MAJOR, API_VERSION_MINOR, API_VERSION_TEENY);
    // Set Library version to zero.
    setVersionStruct(&m_Version.libVersion, 0, 0, 0);
}

void
UdoVersion::setUdoVersion(uint32_t libMajor, uint32_t libMinor, uint32_t libPatch) {
    setVersionStruct(&m_Version.apiVersion,
                     API_VERSION_MAJOR, API_VERSION_MINOR, API_VERSION_TEENY);
    setVersionStruct(&m_Version.libVersion, libMajor, libMinor, libPatch);
}

SnpeUdo_ErrorType_t
UdoLibraryInfo::copyLibraryInfo(SnpeUdo_LibraryInfo_t* libraryInfoPtr) {
    UDO_VALIDATE_MSG(libraryInfoPtr == nullptr,
                 SNPE_UDO_INVALID_ARGUMENT,
                 "The provided library info struct is null")

    libraryInfoPtr->libraryName = const_cast<char*>(m_LibraryName.c_str());
    libraryInfoPtr->udoCoreType = m_CoreType;
    return SNPE_UDO_NO_ERROR;
}


UdoOperationInfo::UdoOperationInfo(std::string &&operationType,
                                   SnpeUdo_Bitmask_t supportedCores,
                                   int numOfInputs,
                                   int numOfOutputs) {
    m_OperationType = operationType;
    m_SupportedCoreTypes = supportedCores;
    m_NumOfInputs = numOfInputs;
    m_NumOfOutputs = numOfOutputs;
}


UdoRegLibrary::UdoRegLibrary(const std::string &packageName,
                             SnpeUdo_Bitmask_t supportedCoreTypes)
        : m_LibraryInfo(nullptr), m_SupportedCoreTypes(supportedCoreTypes) {
    m_RegInfo.reset(new SnpeUdo_RegInfo_t);
    m_PackageName = packageName;
    initUdoRegInfoStruct(m_RegInfo,
                         m_PackageName,
                         SNPE_UDO_CORETYPE_UNDEFINED,
                         0,
                         nullptr,
                         m_OperationsString,
                         0);
}

void
UdoRegLibrary::addImplLib(std::string &&libName, SnpeUdo_CoreType_t udoCoreType) {
    m_UdoImplLibs.emplace_back(new UdoLibraryInfo(std::move(libName), udoCoreType));
}

std::shared_ptr<UdoOperationInfo>
UdoRegLibrary::addOperation(std::string &&operationType,
                            SnpeUdo_Bitmask_t supportedCores,
                            int numOfInputs,
                            int numOfOutputs) {
    m_UdoOperations.emplace_back(new UdoOperationInfo(std::move(operationType),
                                                      supportedCores,
                                                      numOfInputs,
                                                      numOfOutputs));
    return m_UdoOperations.back();
}

SnpeUdo_ErrorType_t
UdoRegLibrary::createImplLibInfo() {
    UDO_VALIDATE_MSG(m_UdoImplLibs.empty(),
                 SNPE_UDO_UNKNOWN_ERROR,
                 "No implementation libraries present in package: " << m_PackageName)

    // m_libraryInfo marks the start of the array, need local pointer to travel...
    m_LibraryInfo = new SnpeUdo_LibraryInfo_t[m_UdoImplLibs.size()];
    SnpeUdo_LibraryInfo_t* localLiInfoPtr = m_LibraryInfo;

    for (auto &&implLibPtr: m_UdoImplLibs)
    {
        UDO_VALIDATE_RETURN_STATUS(implLibPtr->copyLibraryInfo(localLiInfoPtr))
        localLiInfoPtr++; // advance the pointer to next struct entry
    }

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
UdoRegLibrary::createOperationString() {
    UDO_VALIDATE_MSG(m_UdoOperations.empty(),
                 SNPE_UDO_UNKNOWN_ERROR,
                 "No operations were found in package: " << m_PackageName)

    // allocate memory for array of m_OperationInfo structures
    std::string combinedOperationsString;

    for (auto &operationVecPtr: m_UdoOperations)
    {
        combinedOperationsString.append(operationVecPtr->m_OperationType);
        combinedOperationsString.append(" ");
    }
    m_OperationsString = combinedOperationsString;

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
UdoRegLibrary::createRegInfoStruct() {
    UDO_VALIDATE_RETURN_STATUS(createImplLibInfo());
    UDO_VALIDATE_RETURN_STATUS(createOperationString());
    UDO_VALIDATE_MSG(!initUdoRegInfoStruct(m_RegInfo,                 // Registration info Struct
                                         m_PackageName,             // Package name
                                         m_SupportedCoreTypes,      // Supported Cores
                                         m_UdoImplLibs.size(),      // Number of Implementation libraries
                                         m_LibraryInfo,             // Pointer to array of Impl Libraries Info
                                         m_OperationsString,        // Support Operations string
                                         m_UdoOperations.size()),   // Number of operation info structs
                                         SNPE_UDO_UNKNOWN_ERROR, "Unknown failure while create the regInfo for package"<< m_PackageName)

    return SNPE_UDO_NO_ERROR;
}
