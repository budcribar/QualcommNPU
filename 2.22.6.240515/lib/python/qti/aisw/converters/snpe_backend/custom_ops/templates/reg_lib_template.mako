<%doc> define all relevant variables</%doc>
<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2021, 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%
 package_name = package_info.name
 runtimes = package_info.supported_runtimes
 coretypes = package_info.core_types
 op_catalog = package_info.op_catalog_info
 operators = package_info.operators
 calculation_types = package_info.calculation_types
%>

<%def name="_to_bitmask(qnn_types)" filter="trim" buffered="True">
%if isinstance(qnn_types, list):
<% formatted_string = ' | '.join([qnn_type for qnn_type in qnn_types]) %>
${formatted_string}
%else:
${qnn_types}
%endif
</%def>

//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
#include <iostream>
#include "utils/UdoUtil.hpp"

%for runtime in runtimes:
#ifndef UDO_LIB_NAME_${str(runtime).upper()}
#define UDO_LIB_NAME_${str(runtime).upper()} "libUdo${package_name}Impl${str(runtime).title()}.so"
#endif
%endfor

extern "C"
{

std::unique_ptr<UdoUtil::UdoVersion> regLibraryVersion;
std::unique_ptr<UdoUtil::UdoRegLibrary> regLibraryInfo;

SnpeUdo_ErrorType_t
SnpeUdo_initRegLibrary(void)
{
    regLibraryInfo.reset(new UdoUtil::UdoRegLibrary("${package_name}",
                                                   ${_to_bitmask(coretypes)}));

    regLibraryVersion.reset(new UdoUtil::UdoVersion);

    regLibraryVersion->setUdoVersion(1, 0, 0);

<%doc> Add library names : one for each supported coretype </%doc>
    /*
    ** User should fill in implementation library path here as needed.
    ** Note: The Implementation library path set here is relative, meaning each library to be used
    ** must be discoverable by the linker.
    */
%for idx, runtime in enumerate(runtimes):
    regLibraryInfo->addImplLib(UDO_LIB_NAME_${str(runtime).upper()}, ${_to_bitmask(coretypes[idx])}); //adding implementation libraries
%endfor

    %for operator in operators:
    //==============================================================================
    // Auto Generated Code for ${operator.type_name}
    //==============================================================================
    auto ${operator.type_name}Info = regLibraryInfo->addOperation("${operator.type_name}", ${_to_bitmask(operator.core_types)}, ${len(operator.input)}, ${len(operator.output)});

    %endfor
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->createRegInfoStruct())

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_getVersion(SnpeUdo_LibVersion_t** version) {

    UDO_VALIDATE_RETURN_STATUS(regLibraryVersion->getLibraryVersion(version))

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_getRegInfo(SnpeUdo_RegInfo_t** registrationInfo) {

    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->getLibraryRegInfo(registrationInfo))

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_terminateRegLibrary(void) {
    regLibraryInfo.reset();
    regLibraryVersion.reset();

    return SNPE_UDO_NO_ERROR;
}

}; //extern C
