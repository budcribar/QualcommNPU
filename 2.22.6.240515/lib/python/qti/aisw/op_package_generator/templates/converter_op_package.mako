//==============================================================================
// Auto Generated Code for ExamplePackage
//==============================================================================
#include <iostream>
#include <string>

#include "QnnTypes.h"
#include "QnnOpPackage.h"

#ifndef EXPORT_API
#if defined _MSC_VER
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif
#endif

extern "C" {
%for operator in package_info.operators:
EXPORT_API Qnn_ErrorHandle_t ${operator.type_name}ShapeInference(Qnn_OpConfig_t *opConfig) {

/**
* Add code here
**/

return QNN_SUCCESS;
}
%endfor

%for operator in package_info.operators:
Qnn_ErrorHandle_t (*${operator.type_name}OutputInfoInferencePtr)(Qnn_OpConfig_t *)  = &${operator.type_name}ShapeInference;
%endfor

%for operator in package_info.operators:
EXPORT_API Qnn_ErrorHandle_t ${operator.type_name}DataTypeInference(Qnn_OpConfig_t *opConfig) {

/**
* Add code here
**/

return QNN_SUCCESS;
}
%endfor

%for operator in package_info.operators:
Qnn_ErrorHandle_t (*${operator.type_name}DataTypeInferencePtr)(Qnn_OpConfig_t *)  = &${operator.type_name}DataTypeInference;
%endfor

} // extern "C"