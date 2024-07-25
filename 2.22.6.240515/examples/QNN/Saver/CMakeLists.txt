#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

cmake_minimum_required(VERSION 3.14)
project(saver_windows)

# Include paths
# QNN_SDK_ROOT should be set and points to the SDK path, it will be used.
if(DEFINED ENV{QNN_SDK_ROOT})
    # define directories
    set(PACKAGE_INCLUDES $ENV{QNN_SDK_ROOT}/include/QNN)
else()
    message(FATAL_ERROR "QNN_SDK_ROOT: Please set QNN_SDK_ROOT")
endif()

set(APP_SOURCES "saver_output.c")

if(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    add_executable(cpu ${APP_SOURCES})
    set_target_properties(cpu PROPERTIES OUTPUT_NAME "saver_output_QnnCpu")
    target_link_libraries(cpu PRIVATE $ENV{QNN_SDK_ROOT}/lib/x86_64-windows-msvc/QnnCpu.lib)
elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    add_executable(cpu ${APP_SOURCES})
    set_target_properties(cpu PROPERTIES OUTPUT_NAME "saver_output_QnnCpu")
    target_link_libraries(cpu PRIVATE $ENV{QNN_SDK_ROOT}/lib/aarch64-windows-msvc/QnnCpu.lib)

    add_executable(dsp ${APP_SOURCES})
    set_target_properties(dsp PROPERTIES OUTPUT_NAME "saver_output_QnnDsp")
    target_link_libraries(dsp PRIVATE $ENV{QNN_SDK_ROOT}/lib/aarch64-windows-msvc/QnnDsp.lib)

    add_executable(htp ${APP_SOURCES})
    set_target_properties(htp PROPERTIES OUTPUT_NAME "saver_output_QnnHtp")
    target_link_libraries(htp PRIVATE $ENV{QNN_SDK_ROOT}/lib/aarch64-windows-msvc/QnnHtp.lib)
else()
    message(FATAL_ERROR "Please set platform in [x64, ARM64]")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /O2 /Ob3")

include_directories(${PACKAGE_INCLUDES})