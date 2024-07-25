#==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

cmake_minimum_required (VERSION 3.14)
project (snpe-samplec)
set (APP "snpe-samplec")

# CMake option to control whether to build with VCRuntime libraries
option(BUILD_WITH_VCRUNTIME "Build the snpe-samplec with static vcruntime libraries." OFF)
message("Build snpe-samplec with vcruntime: ${BUILD_WITH_VCRUNTIME}")

set( APP_SOURCES
    "main.cpp"
        "LoadInputTensor.hpp"
        "LoadInputTensor.cpp"
        "Util.hpp"
        "Util.cpp"
        "SetBuilderOptions.hpp"
        "SetBuilderOptions.cpp"
        "CheckRuntime.hpp"
        "CheckRuntime.cpp"
        "LoadContainer.hpp"
        "LoadContainer.cpp"
        "PreprocessInput.hpp"
        "PreprocessInput.cpp"
        "SaveOutputTensor.hpp"
        "SaveOutputTensor.cpp"
        "GetOpt.cpp"
        "GetOpt.hpp"
)

set (SNPE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/SNPE)
set (SNPE_LIB_PREFIX ../../../../lib)
set (_dtuple_POSTFIX windows-msvc)

if(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    message("Linking with x64 SNPE")
    get_filename_component(SNPE_DLL_PATH "${SNPE_LIB_PREFIX}/x86_64-${_dtuple_POSTFIX}/SNPE.dll" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    get_filename_component(SNPE_IMPLIB_PATH "${SNPE_LIB_PREFIX}/x86_64-${_dtuple_POSTFIX}/SNPE.lib" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    message("Linking with ARM64 SNPE")
    get_filename_component(SNPE_DLL_PATH "${SNPE_LIB_PREFIX}/aarch64-${_dtuple_POSTFIX}/SNPE.dll" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    get_filename_component(SNPE_IMPLIB_PATH "${SNPE_LIB_PREFIX}/aarch64-${_dtuple_POSTFIX}/SNPE.lib" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(FATAL "Not Supported Platform")
endif()

add_library( SNPE SHARED IMPORTED )
set_target_properties(SNPE PROPERTIES
    IMPORTED_LOCATION ${SNPE_DLL_PATH}
    IMPORTED_IMPLIB ${SNPE_IMPLIB_PATH}
    INTERFACE_INCLUDE_DIRECTORIES ${SNPE_INCLUDE_DIR}
)

add_executable(${APP} ${APP_SOURCES})
target_compile_definitions(${APP} PUBLIC -D_CRT_SECURE_NO_WARNINGS)
if(${BUILD_WITH_VCRUNTIME})
    target_compile_options(${APP} PUBLIC /MT)
endif()
target_link_libraries (${APP} SNPE)
add_custom_command(TARGET ${APP} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${SNPE_DLL_PATH}
    $<TARGET_FILE_DIR:${APP}>)
