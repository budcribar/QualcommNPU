<%doc>
# ==============================================================================
#
#  Copyright (c) 2022-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""This template is used to generate CMakeList.txt for Reg, ImplCpu libraries.

Args from python:
   package_name (str): package name.
   lib_name_suffix (str): suffix for the target library name, e.g. Reg, ImplCpu.
   util_src_files (list of str): A list of the util source file name.
   src_files (list of str): A list of the the source file names.
   runtimes_of_lib_def (list of str): A list of the runtime name to add defintion.
      This is for Reg lib only, and empty list for Impl lib.
"""</%doc>
<% package_name = package_info.name %>
#================================================================================
# Auto Generated Code for ${package_name}
#================================================================================

<% runtimes_of_lib_def = [] %>
set( LIB_NAME Udo${package_name}${lib_name_suffix} )
set( UTIL_SOURCES "../utils/UdoUtil.cpp" )
set( SRC_DIR "${'${CMAKE_CURRENT_LIST_DIR}'}/src")
set( SRC_DIR_OPS "${'${CMAKE_CURRENT_LIST_DIR}'}/src/ops")
set( SRC_DIR_UTILS "${'${CMAKE_CURRENT_LIST_DIR}'}/src/utils/CPU")

% if lib_name_suffix != 'Reg':
set( SOURCES
    "src/${package_info.name}Interface.cpp"
    % for op in package_info.operators:
    "src/ops/${op.type_name}.cpp"
    %endfor
    ## consider to add backend-specific src files to package_info.
    % for runtime in package_info.supported_runtimes:
    % if runtime.upper() == 'CPU':
    "src/utils/CPU/CpuBackendUtils.cpp"
    %endif
    %endfor
   )
% else:
set( SOURCES "./${package_info.name}RegLib.cpp")
% for runtime in package_info.supported_runtimes:
    % if runtime.upper() != 'DSP':
        <% runtimes_of_lib_def.append(runtime.upper()) %>
    %endif
%endfor
%endif

set( LIB_INCLUDES ${'${ROOT_INCLUDES}'} )
set( DLL_EXPT
   "__declspec( dllexport )"
   )

% if lib_name_suffix != "Reg":
if( "$ENV{QNN_SDK_ROOT}" STREQUAL ""  )
   message(FATAL_ERROR "Error undefined QNN_SDK_ROOT: Please set environment variable QNN_SDK_ROOT"
                       " to qnn sdk installation")
endif()

set( CUSTOM_OP_DIR $ENV{QNN_SDK_ROOT}/share/QNN/OpPackageGenerator/CustomOp )
list(APPEND LIB_INCLUDES
     $ENV{QNN_SDK_ROOT}/include/QNN
     ${'${SRC_DIR}'}/utils
     ${'${SRC_DIR_UTILS}'}
     ${'${CUSTOM_OP_DIR}'} )

% if lib_name_suffix == "ImplCpu":
file(COPY ${'${CUSTOM_OP_DIR}'}/CPU/CpuCustomOpPackage.cpp DESTINATION ${'${SRC_DIR}'})
list(APPEND SOURCES ${'${SRC_DIR}'}/CpuCustomOpPackage.cpp)
list(APPEND LIB_INCLUDES $ENV{QNN_SDK_ROOT}/include/QNN/CPU)
%endif
%endif

add_library( ${'${LIB_NAME}'} SHARED ${'${UTIL_SOURCES}'} ${'${SOURCES}'} )
target_include_directories(
   ${'${LIB_NAME}'}
   PRIVATE
   ${'${LIB_INCLUDES}'} )

%for runtime in runtimes_of_lib_def:
<%
   flag_name = 'UDO_LIB_NAME_' + runtime.upper()
   lib_name = '${CMAKE_SHARED_LIBRARY_PREFIX}' + 'Udo' + package_name + 'Impl' + runtime.title() + '${CMAKE_SHARED_LIBRARY_SUFFIX}'
%>
if( NOT DEFINED ${flag_name} )
   set( ${flag_name} ${lib_name} )
endif()
target_compile_definitions(
   ${'${LIB_NAME}'}
   PRIVATE
   ${flag_name}="${'${' + flag_name + '}'}" )
%endfor

% if lib_name_suffix != "Reg":
target_compile_definitions(${'${LIB_NAME}'} PUBLIC QNN_API=${'${DLL_EXPT}'})
% endif

if( "${'${CMAKE_GENERATOR}'}" STREQUAL "Ninja"  )
target_compile_options(${'${LIB_NAME}'} PRIVATE /arm64EC /D_AMD64_ /D_ARM64EC_)

set( LIB_NAME_ARM64 "${'${LIB_NAME}'}_ARM64")
add_library( ${'${LIB_NAME_ARM64}'} OBJECT ${'${UTIL_SOURCES}'} ${'${SOURCES}'} )
target_include_directories(
   ${'${LIB_NAME_ARM64}'}
   PRIVATE
   ${'${LIB_INCLUDES}'} )

% if lib_name_suffix != "Reg":
target_compile_definitions(${'${LIB_NAME_ARM64}'} PUBLIC QNN_API=${'${DLL_EXPT}'})
% endif

% if lib_name_suffix == "Reg":
%for each_runtime in runtimes_of_lib_def:
<%
   flag_name_normal = 'UDO_LIB_NAME_' + each_runtime.upper()
   flag_name_arm64 = 'UDO_LIB_NAME_' + each_runtime.upper() + '_ARM64'
   lib_name_normal = '${CMAKE_SHARED_LIBRARY_PREFIX}' + 'Udo' + package_name + 'Impl' + each_runtime.title() + '${CMAKE_SHARED_LIBRARY_SUFFIX}'
%>
if( NOT DEFINED ${flag_name_arm64} )
   set( ${flag_name_arm64} ${lib_name_normal} )
endif()
target_compile_definitions(
   ${'${LIB_NAME_ARM64}'}
   PRIVATE
   ${flag_name_normal}="${'${' + flag_name_arm64 + '}'}" )
%endfor
% endif

target_link_options( ${'${LIB_NAME_ARM64}'} PRIVATE /machine:ARM64)
target_link_libraries(${'${LIB_NAME}'} PRIVATE ${'${LIB_NAME_ARM64}'})
target_link_options(${'${LIB_NAME}'} PRIVATE /machine:ARM64X /FORCE:MULTIPLE)
install(
   TARGETS ${'${LIB_NAME}'}
   RUNTIME
   DESTINATION "${'${CMAKE_SOURCE_DIR}'}/libs/arm64x_windows" )
else()
if( COMMAND install_shared_lib_to_platform_dir )
   install_shared_lib_to_platform_dir( ${'${LIB_NAME}'} )
endif()
endif()
