#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstddef>
#include <sys/stat.h>
#include "QnnTypes.h"

#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(dir) _mkdir(dir)
#else
#include <sys/types.h>
#define CREATE_DIR(dir) mkdir(dir, 0777)
#endif
#include <QnnWrapperUtils.hpp>
#include <QnnInterface.h>
using namespace qnn_wrapper_api;

class StructOffsetGenerator {
public:
	static void generateAndWriteOffsets(const std::string& outputDir) {
		createDirectory(outputDir);

		writeOffsets<Qnn_Tensor_t>(outputDir, "Qnn_Tensor_t");
		writeOffsets<Qnn_TensorSetV1_t>(outputDir, "Qnn_TensorSetV1_t");
		writeOffsets<Qnn_TensorSet_t>(outputDir, "Qnn_TensorSet_t");
		writeOffsets<Qnn_OpConfigV1_t>(outputDir, "Qnn_OpConfigV1_t");
		writeOffsets<Qnn_OpConfig_t>(outputDir, "Qnn_OpConfig_t");

		writeOffsets<GraphConfigInfo_t>(outputDir, "GraphConfigInfo_t");
		writeOffsets<GraphInfo_t>(outputDir, "GraphInfo_t");
		writeOffsets<Qnn_Version_t>(outputDir, "Qnn_Version_t");
		writeOffsets<Qnn_Tensor_t>(outputDir, "Qnn_Tensor_t");
		writeOffsets<Qnn_BwAxisScaleOffset_t>(outputDir, "Qnn_BwAxisScaleOffset_t");
		writeOffsets<Qnn_ScaleOffset_t>(outputDir, "Qnn_ScaleOffset_t");
		writeOffsets<Qnn_QuantizeParams_t>(outputDir, "Qnn_QuantizeParams_t");
		writeOffsets<Qnn_ClientBuffer_t>(outputDir, "Qnn_ClientBuffer_t");
		writeOffsets<Qnn_SparseParams_t>(outputDir, "Qnn_SparseParams_t");
		writeOffsets<Qnn_SparseLayoutHybridCoo_t>(outputDir, "Qnn_SparseLayoutHybridCoo_t");
		writeOffsets<Qnn_TensorV1_t>(outputDir, "Qnn_TensorV1_t");
		writeOffsets<Qnn_TensorV2_t>(outputDir, "Qnn_TensorV2_t");
		writeOffsets<QnnInterface_t>(outputDir, "QnnInterface_t");
		writeOffsets<Qnn_ApiVersion_t>(outputDir, "Qnn_ApiVersion_t");
		std::cout << "Offset files have been written to: " << outputDir << std::endl;
	}

private:
	static void createDirectory(const std::string& dir) {
		CREATE_DIR(dir.c_str());
	}

	template<typename T>
	static void writeOffsets(const std::string& outputDir, const std::string& structName) {
		std::stringstream ss;
		ss << "{\n";
		ss << "    \"StructName\": \"" << structName << "\",\n";
		ss << "    \"Offsets\": {\n";

		addOffsets<T>(ss);

		// Remove the last comma and newline
		ss.seekp(-2, std::ios_base::end);
		ss << "\n    },\n";

		ss << "    \"TotalSize\": " << sizeof(T) << "\n";
		ss << "}\n";

		std::string filename = outputDir + "/" + structName + "_offsets.json";
		std::ofstream outFile(filename);
		if (outFile.is_open()) {
			outFile << ss.str();
			outFile.close();
			std::cout << "Written: " << filename << std::endl;
		}
		else {
			std::cerr << "Unable to open file: " << filename << std::endl;
		}
	}

	static void addOffset(std::stringstream& ss, const std::string& name, size_t offset) {
		ss << "        \"" << name << "\": " << offset << ",\n";
	}

	// Base template (does nothing)
	template<typename T>
	static void addOffsets(std::stringstream&) {}

	// Template specializations for each struct type
	template<>
	static void addOffsets<Qnn_TensorV1_t>(std::stringstream& ss) {
		addOffset(ss, "id", offsetof(Qnn_TensorV1_t, id));
		addOffset(ss, "name", offsetof(Qnn_TensorV1_t, name));
		addOffset(ss, "type", offsetof(Qnn_TensorV1_t, type));
		addOffset(ss, "dataFormat", offsetof(Qnn_TensorV1_t, dataFormat));
		addOffset(ss, "dataType", offsetof(Qnn_TensorV1_t, dataType));
		addOffset(ss, "quantizeParams", offsetof(Qnn_TensorV1_t, quantizeParams));
		addOffset(ss, "rank", offsetof(Qnn_TensorV1_t, rank));
		addOffset(ss, "dimensions", offsetof(Qnn_TensorV1_t, dimensions));
		addOffset(ss, "memType", offsetof(Qnn_TensorV1_t, memType));
		addOffset(ss, "clientBuf", offsetof(Qnn_TensorV1_t, clientBuf));
		addOffset(ss, "memHandle", offsetof(Qnn_TensorV1_t, memHandle));
	}

	template<>
	static void addOffsets<Qnn_TensorV2_t>(std::stringstream& ss) {
		addOffset(ss, "id", offsetof(Qnn_TensorV2_t, id));
		addOffset(ss, "name", offsetof(Qnn_TensorV2_t, name));
		addOffset(ss, "type", offsetof(Qnn_TensorV2_t, type));
		addOffset(ss, "dataFormat", offsetof(Qnn_TensorV2_t, dataFormat));
		addOffset(ss, "dataType", offsetof(Qnn_TensorV2_t, dataType));
		addOffset(ss, "quantizeParams", offsetof(Qnn_TensorV2_t, quantizeParams));
		addOffset(ss, "rank", offsetof(Qnn_TensorV2_t, rank));
		addOffset(ss, "dimensions", offsetof(Qnn_TensorV2_t, dimensions));
		addOffset(ss, "memType", offsetof(Qnn_TensorV2_t, memType));
		addOffset(ss, "clientBuf", offsetof(Qnn_TensorV2_t, clientBuf));
		addOffset(ss, "memHandle", offsetof(Qnn_TensorV2_t, memHandle));
		addOffset(ss, "isDynamicDimensions", offsetof(Qnn_TensorV2_t, isDynamicDimensions));
		addOffset(ss, "sparseParams", offsetof(Qnn_TensorV2_t, sparseParams));
		addOffset(ss, "isProduced", offsetof(Qnn_TensorV2_t, isProduced));
	}

	template<>
	static void addOffsets<Qnn_Tensor_t>(std::stringstream& ss) {
		addOffset(ss, "version", offsetof(Qnn_Tensor_t, version));
		addOffset(ss, "v1", offsetof(Qnn_Tensor_t, v1));
		addOffset(ss, "v2", offsetof(Qnn_Tensor_t, v2));
	}

	template<>
	static void addOffsets<Qnn_TensorSetV1_t>(std::stringstream& ss) {
		addOffset(ss, "numInputs", offsetof(Qnn_TensorSetV1_t, numInputs));
		addOffset(ss, "inputs", offsetof(Qnn_TensorSetV1_t, inputs));
		addOffset(ss, "numOutputs", offsetof(Qnn_TensorSetV1_t, numOutputs));
		addOffset(ss, "outputs", offsetof(Qnn_TensorSetV1_t, outputs));
	}

	template<>
	static void addOffsets<Qnn_TensorSet_t>(std::stringstream& ss) {
		addOffset(ss, "version", offsetof(Qnn_TensorSet_t, version));
		addOffset(ss, "v1", offsetof(Qnn_TensorSet_t, v1));
	}

	template<>
	static void addOffsets<Qnn_OpConfigV1_t>(std::stringstream& ss) {
		addOffset(ss, "name", offsetof(Qnn_OpConfigV1_t, name));
		addOffset(ss, "packageName", offsetof(Qnn_OpConfigV1_t, packageName));
		addOffset(ss, "typeName", offsetof(Qnn_OpConfigV1_t, typeName));
		addOffset(ss, "numOfParams", offsetof(Qnn_OpConfigV1_t, numOfParams));
		addOffset(ss, "params", offsetof(Qnn_OpConfigV1_t, params));
		addOffset(ss, "numOfInputs", offsetof(Qnn_OpConfigV1_t, numOfInputs));
		addOffset(ss, "inputTensors", offsetof(Qnn_OpConfigV1_t, inputTensors));
		addOffset(ss, "numOfOutputs", offsetof(Qnn_OpConfigV1_t, numOfOutputs));
		addOffset(ss, "outputTensors", offsetof(Qnn_OpConfigV1_t, outputTensors));
	}

	template<>
	static void addOffsets<Qnn_OpConfig_t>(std::stringstream& ss) {
		addOffset(ss, "version", offsetof(Qnn_OpConfig_t, version));
		addOffset(ss, "v1", offsetof(Qnn_OpConfig_t, v1));
	}
	template<>
	static void addOffsets<GraphConfigInfo_t>(std::stringstream& ss) {
		addOffset(ss, "graphName", offsetof(GraphConfigInfo_t, graphName));
		addOffset(ss, "graphConfigs", offsetof(GraphConfigInfo_t, graphConfigs));
	}

	template<>
	static void addOffsets<GraphInfo_t>(std::stringstream& ss) {
		addOffset(ss, "graph", offsetof(GraphInfo_t, graph));
		addOffset(ss, "graphName", offsetof(GraphInfo_t, graphName));
		addOffset(ss, "inputTensors", offsetof(GraphInfo_t, inputTensors));
		addOffset(ss, "numInputTensors", offsetof(GraphInfo_t, numInputTensors));
		addOffset(ss, "outputTensors", offsetof(GraphInfo_t, outputTensors));
		addOffset(ss, "numOutputTensors", offsetof(GraphInfo_t, numOutputTensors));
	}

	template<>
	static void addOffsets<Qnn_Version_t>(std::stringstream& ss) {
		addOffset(ss, "major", offsetof(Qnn_Version_t, major));
		addOffset(ss, "minor", offsetof(Qnn_Version_t, minor));
		addOffset(ss, "patch", offsetof(Qnn_Version_t, patch));
	}

	template<>
	static void addOffsets<Qnn_ApiVersion_t>(std::stringstream& ss) {
		addOffset(ss, "coreApiVersion", offsetof(Qnn_ApiVersion_t, coreApiVersion));
		addOffset(ss, "backendApiVersion", offsetof(Qnn_ApiVersion_t, backendApiVersion));

	}

	template<>
	static void addOffsets<Qnn_BwAxisScaleOffset_t>(std::stringstream& ss) {
		addOffset(ss, "bitwidth", offsetof(Qnn_BwAxisScaleOffset_t, bitwidth));
		addOffset(ss, "axis", offsetof(Qnn_BwAxisScaleOffset_t, axis));
		addOffset(ss, "numElements", offsetof(Qnn_BwAxisScaleOffset_t, numElements));
		addOffset(ss, "scales", offsetof(Qnn_BwAxisScaleOffset_t, scales));
		addOffset(ss, "offsets", offsetof(Qnn_BwAxisScaleOffset_t, offsets));
	}

	template<>
	static void addOffsets<Qnn_ScaleOffset_t>(std::stringstream& ss) {
		addOffset(ss, "scale", offsetof(Qnn_ScaleOffset_t, scale));
		addOffset(ss, "offset", offsetof(Qnn_ScaleOffset_t, offset));
	}

	template<>
	static void addOffsets<Qnn_QuantizeParams_t>(std::stringstream& ss) {
		addOffset(ss, "encodingDefinition", offsetof(Qnn_QuantizeParams_t, encodingDefinition));
		addOffset(ss, "quantizationEncoding", offsetof(Qnn_QuantizeParams_t, quantizationEncoding));
		addOffset(ss, "scaleOffsetEncoding", offsetof(Qnn_QuantizeParams_t, scaleOffsetEncoding));
		addOffset(ss, "axisScaleOffsetEncoding", offsetof(Qnn_QuantizeParams_t, axisScaleOffsetEncoding));
		addOffset(ss, "bwScaleOffsetEncoding", offsetof(Qnn_QuantizeParams_t, bwScaleOffsetEncoding));
		addOffset(ss, "bwAxisScaleOffsetEncoding", offsetof(Qnn_QuantizeParams_t, bwAxisScaleOffsetEncoding));
	}

	template<>
	static void addOffsets<Qnn_ClientBuffer_t>(std::stringstream& ss) {
		addOffset(ss, "data", offsetof(Qnn_ClientBuffer_t, data));
		addOffset(ss, "dataSize", offsetof(Qnn_ClientBuffer_t, dataSize));
	}

	template<>
	static void addOffsets<Qnn_SparseParams_t>(std::stringstream& ss) {
		addOffset(ss, "type", offsetof(Qnn_SparseParams_t, type));
		addOffset(ss, "hybridCoo", offsetof(Qnn_SparseParams_t, hybridCoo));
	}

	template<>
	static void addOffsets<Qnn_SparseLayoutHybridCoo_t>(std::stringstream& ss) {
		addOffset(ss, "numSpecifiedElements", offsetof(Qnn_SparseLayoutHybridCoo_t, numSpecifiedElements));
		addOffset(ss, "numSparseDimensions", offsetof(Qnn_SparseLayoutHybridCoo_t, numSparseDimensions));
	}

	template<>
	static void addOffsets<QnnInterface_t>(std::stringstream& ss) {
		addOffset(ss, "backendId", offsetof(QnnInterface_t, backendId));
		addOffset(ss, "providerName", offsetof(QnnInterface_t, providerName));
		addOffset(ss, "apiVersion", offsetof(QnnInterface_t, apiVersion));

		// Calculate the base offset of the union within QnnInterface_t
		uint16_t baseOffset = 40;

		// Offsets for function pointers in the union
		addOffset(ss, "propertyHasCapability", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, propertyHasCapability));
		addOffset(ss, "backendCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendCreate));
		addOffset(ss, "backendSetConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendSetConfig));
		addOffset(ss, "backendGetApiVersion", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendGetApiVersion));
		addOffset(ss, "backendGetBuildId", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendGetBuildId));
		addOffset(ss, "backendRegisterOpPackage", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendRegisterOpPackage));
		addOffset(ss, "backendGetSupportedOperations", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendGetSupportedOperations));
		addOffset(ss, "backendValidateOpConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendValidateOpConfig));
		addOffset(ss, "backendFree", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, backendFree));

		addOffset(ss, "contextCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextCreate));
		addOffset(ss, "contextSetConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextSetConfig));
		addOffset(ss, "contextGetBinarySize", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextGetBinarySize));
		addOffset(ss, "contextGetBinary", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextGetBinary));
		addOffset(ss, "contextCreateFromBinary", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextCreateFromBinary));
		addOffset(ss, "contextFree", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextFree));

		addOffset(ss, "graphCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphCreate));
		addOffset(ss, "graphCreateSubgraph", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphCreateSubgraph));
		addOffset(ss, "graphSetConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphSetConfig));
		addOffset(ss, "graphAddNode", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphAddNode));
		addOffset(ss, "graphFinalize", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphFinalize));
		addOffset(ss, "graphRetrieve", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphRetrieve));
		addOffset(ss, "graphExecute", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphExecute));
		addOffset(ss, "graphExecuteAsync", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphExecuteAsync));

		addOffset(ss, "tensorCreateContextTensor", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, tensorCreateContextTensor));
		addOffset(ss, "tensorCreateGraphTensor", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, tensorCreateGraphTensor));

		addOffset(ss, "logCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, logCreate));
		addOffset(ss, "logSetLogLevel", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, logSetLogLevel));
		addOffset(ss, "logFree", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, logFree));

		addOffset(ss, "profileCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileCreate));
		addOffset(ss, "profileSetConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileSetConfig));
		addOffset(ss, "profileGetEvents", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileGetEvents));
		addOffset(ss, "profileGetSubEvents", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileGetSubEvents));
		addOffset(ss, "profileGetEventData", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileGetEventData));
		addOffset(ss, "profileGetExtendedEventData", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileGetExtendedEventData));
		addOffset(ss, "profileFree", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, profileFree));

		addOffset(ss, "memRegister", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, memRegister));
		addOffset(ss, "memDeRegister", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, memDeRegister));

		addOffset(ss, "deviceGetPlatformInfo", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceGetPlatformInfo));
		addOffset(ss, "deviceFreePlatformInfo", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceFreePlatformInfo));
		addOffset(ss, "deviceGetInfrastructure", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceGetInfrastructure));
		addOffset(ss, "deviceCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceCreate));
		addOffset(ss, "deviceSetConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceSetConfig));
		addOffset(ss, "deviceGetInfo", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceGetInfo));
		addOffset(ss, "deviceFree", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, deviceFree));

		addOffset(ss, "signalCreate", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, signalCreate));
		addOffset(ss, "signalSetConfig", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, signalSetConfig));
		addOffset(ss, "signalTrigger", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, signalTrigger));
		addOffset(ss, "signalFree", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, signalFree));

		addOffset(ss, "errorGetMessage", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, errorGetMessage));
		addOffset(ss, "errorGetVerboseMessage", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, errorGetVerboseMessage));
		addOffset(ss, "errorFreeVerboseMessage", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, errorFreeVerboseMessage));

		addOffset(ss, "graphPrepareExecutionEnvironment", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphPrepareExecutionEnvironment));
		addOffset(ss, "graphReleaseExecutionEnvironment", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphReleaseExecutionEnvironment));
		addOffset(ss, "graphGetProperty", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, graphGetProperty));

		addOffset(ss, "contextValidateBinary", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextValidateBinary));
		addOffset(ss, "contextCreateFromBinaryWithSignal", baseOffset + offsetof(QNN_INTERFACE_VER_TYPE, contextCreateFromBinaryWithSignal));
	}

};

//int main() {
//    StructOffsetGenerator::generateAndWriteOffsets("./struct_offsets");
//    return 0;
//}