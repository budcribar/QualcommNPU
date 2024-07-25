#==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# set minimum cmake version
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# project name and language
project(qnn_delegate_skip_node_ids LANGUAGES CXX)

if (NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR  "QNN_SDK_ROOT is not defined. Please use cmake -DQNN_SDK=/path/to/unzipped/qnn_sdk")
endif()

if (NOT DEFINED TENSORFLOW_SOURCE_DIR)
  message(FATAL_ERROR  "TensorFlow path is not defined. Please use cmake -DTENSORFLOW_SOURCE_DIR=/path/to/tensorflow")
endif()

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-missing-braces -g -O0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-missing-braces -Os")
endif()

if(${ANDROID})
  find_library(android_log log)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

add_library(qnn_delegate SHARED IMPORTED)

set_property(TARGET qnn_delegate PROPERTY IMPORTED_LOCATION ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnTFLiteDelegate.so)
add_executable(main ${PROJECT_SOURCE_DIR}/main.cc)

target_include_directories(main PRIVATE
  ${QNN_SDK_ROOT}/include/QNN/TFLiteDelegate)

target_link_libraries(main PRIVATE qnn_delegate tensorflow-lite ${android_log})