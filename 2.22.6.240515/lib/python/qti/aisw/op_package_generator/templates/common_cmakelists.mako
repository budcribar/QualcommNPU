# ==============================================================================
#
#  Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

cmake_minimum_required(VERSION 3.14)
get_filename_component(PACKAGE_NAME ${'${CMAKE_CURRENT_SOURCE_DIR}'} NAME)
project(${'${PACKAGE_NAME}'})

# Export QNN_API of UDO library
add_compile_definitions("QNN_API=__declspec(dllexport)")

# UDO library depends on MSVC library, add /MT flag so users don't need to copy MSVC library manually
set(CMAKE_CXX_FLAGS_DEBUG "${'${CMAKE_CXX_FLAGS_DEBUG}'} /MTd /guard:cf /ZH:SHA_256")
set(CMAKE_CXX_FLAGS_RELEASE "${'${CMAKE_CXX_FLAGS_RELEASE}'} /MT /O2 /Ob3 /guard:cf /ZH:SHA_256")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${'${CMAKE_SHARED_LINKER_FLAGS_DEBUG}'} /guard:cf /DYNAMICBASE \"ucrtd.lib\" /NODEFAULTLIB:\"libucrtd.lib\"")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${'${CMAKE_SHARED_LINKER_FLAGS_RELEASE}'} /guard:cf /DYNAMICBASE \"ucrt.lib\" /NODEFAULTLIB:\"libucrt.lib\"")

#============================ Define Common Variables ===============================================================
# Include paths
# QNN_SDK_ROOT should be set and points to the SDK path, it will be used.
if(DEFINED ENV{QNN_SDK_ROOT})
    set(UTIL_SRC_DIR ${'${CMAKE_CURRENT_SOURCE_DIR}'}/src/utils)
    # define directories
    set(CUSTOM_OP_DIR $ENV{QNN_SDK_ROOT}/share/QNN/OpPackageGenerator/CustomOp)

    # setup include paths
    set(PACKAGE_C_INCLUDES $ENV{QNN_SDK_ROOT}/include/QNN
                           $ENV{QNN_SDK_ROOT}/include/QNN/${package_info.backend.upper()}
                           ./include/
                           ${'${UTIL_SRC_DIR}'}
                           ${'${UTIL_SRC_DIR}'}/${package_info.backend.upper()}
                           ${'${CUSTOM_OP_DIR}'})
else()
    message(FATAL_ERROR "QNN_SDK_ROOT: Please set QNN_SDK_ROOT")
endif()

#============================ Build Target Library ===============================================================
set( SOURCES
    src/${'${PACKAGE_NAME}'}Interface.cpp
% for op in package_info.operators:
    src/ops/${op.type_name}.cpp
%endfor
## consider to add backend-specific src files to package_info.
% if package_info.backend.upper() == 'CPU':
    src/utils/CPU/CPUBackendUtils.cpp
    src/CpuCustomOpPackage.cpp
%endif
)
include_directories(${'${PACKAGE_C_INCLUDES}'})
add_library( ${'${PACKAGE_NAME}'} SHARED ${'${SOURCES}'} )