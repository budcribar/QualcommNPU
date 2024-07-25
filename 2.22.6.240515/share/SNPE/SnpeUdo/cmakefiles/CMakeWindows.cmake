#==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

set( CMAKE_SYSTEM_NAME                   "Windows" CACHE INTERNAL "" FORCE )
set( CMAKE_SYSTEM_VERSION                    10.0  CACHE INTERNAL "" FORCE )

if( "${CMAKE_GENERATOR}" STREQUAL "Ninja" )
set(CMAKE_SYSTEM_VERSION $ENV{WindowsSdkVersion})
string(REGEX REPLACE "[/\\]" "/" VSInstallDir "$ENV{VSInstallDir}")
string(REGEX REPLACE "[/\\]" "/" WindowsSdkDir "$ENV{WindowsSdkDir}")
string(REGEX REPLACE "[/\\]" "/" WindowsSdkVersion "$ENV{WindowsSdkVersion}")

set(VC_TOOLCHAIN_ROOT ${VSInstallDir}/VC/Tools/MSVC/$ENV{VCToolsVersion})
set(VC_TOOLCHAIN_BIN_ROOT ${VC_TOOLCHAIN_ROOT}/bin/HostX64/arm64)
set(VC_TOOLCHAIN_LIB_ROOT ${VC_TOOLCHAIN_ROOT}/lib/arm64)

set(CLANG_TOOLCHAIN_ROOT ${VSInstallDir}/VC/Tools/Llvm/x64/bin)

set(WINDOWS_SDK_BIN_ROOT ${WindowsSdkDir}/bin/${WindowsSdkVersion}/x64) # always use x64
set(WINDOWS_SDK_LIB_ROOT ${WindowsSdkDir}/Lib/${WindowsSdkVersion})

set(CMAKE_C_COMPILER "${VC_TOOLCHAIN_BIN_ROOT}/cl.exe")
set(CMAKE_CXX_COMPILER "${VC_TOOLCHAIN_BIN_ROOT}/cl.exe")
set(CMAKE_LINKER "${VC_TOOLCHAIN_BIN_ROOT}/link.exe")
set(CMAKE_RC_COMPILER "${WINDOWS_SDK_BIN_ROOT}/rc.exe")

link_directories(
  ${VC_TOOLCHAIN_LIB_ROOT}
  ${WINDOWS_SDK_LIB_ROOT}/ucrt/arm64
  ${WINDOWS_SDK_LIB_ROOT}/um/arm64
)
endif()

set( CMAKE_CXX_FLAGS_DEBUG "/MTd /O2 /Ob2 /DNDEBUG" )
set( CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG" )
set( CMAKE_EXE_LINKER_FLAGS "/DYNAMICBASE \"ucrt.lib\" /NODEFAULTLIB:\"libucrt.lib\"" )
set( CMAKE_SHARED_LINKER_FLAGS "/DYNAMICBASE \"ucrt.lib\" /NODEFAULTLIB:\"libucrt.lib\"" )

message( STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}." )


# macro to install shared library
# platform info including RUNTIME, DESTINATION
macro( install_shared_lib_to_platform_dir target )
   install(
      TARGETS ${target}
      RUNTIME
      DESTINATION "${CMAKE_SOURCE_DIR}/libs/${CMAKE_VS_PLATFORM_NAME}_windows" )
endmacro()
