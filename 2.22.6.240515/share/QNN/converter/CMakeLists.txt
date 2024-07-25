#==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

cmake_minimum_required(VERSION 3.14)
project(model-lib-windows)
set(APP "model-lib-windows")

if(NOT DEFINED ENV{QNN_SDK_ROOT})
    message(FATAL_ERROR "QNN_SDK_ROOT not found in enviromental variables.")
endif()

# Below configuration is to deal with the DEPENDENTS issue with VCRUNTIME140.dll
set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/DYNAMICBASE \"ucrtd.lib\" /NODEFAULTLIB:\"libucrtd.lib\"")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "/DYNAMICBASE \"ucrtd.lib\" /NODEFAULTLIB:\"libucrtd.lib\"")
set(CMAKE_CXX_FLAGS_RELEASE "/MT")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/DYNAMICBASE \"ucrt.lib\" /NODEFAULTLIB:\"libucrt.lib\"")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/DYNAMICBASE \"ucrt.lib\" /NODEFAULTLIB:\"libucrt.lib\"")

set(SDK_ROOT $ENV{QNN_SDK_ROOT})
set(APP_SOURCES
    "QnnModel.cpp"
    "QnnModelPal.cpp"
    "QnnWrapperUtils.cpp"
    "QnnNetworkModel.cpp")

add_definitions("-DQNN_API=__declspec(dllexport)")

# ARM64 is for windows platform only
if(${CMAKE_GENERATOR_PLATFORM} STREQUAL "ARM64")
    file(GLOB_RECURSE OBJ_SOURCES "obj/windows-aarch64/*.o")
else()
    file(GLOB_RECURSE OBJ_SOURCES "obj/windows-x86_64/*.o")
endif()
list(APPEND APP_SOURCES ${OBJ_SOURCES})
add_library(${APP} SHARED ${APP_SOURCES})
target_include_directories(${APP} PUBLIC ${SDK_ROOT}/include/QNN)