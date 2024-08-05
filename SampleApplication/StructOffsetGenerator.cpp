#include "StructOffsetGenerator.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstddef>
#include "QnnTypes.h"

class StructOffsetGenerator {
public:
    static void generateAndPrintOffsets() {
        printOffsets<Qnn_TensorV1_t>("Qnn_TensorV1_t");
        printOffsets<Qnn_TensorV2_t>("Qnn_TensorV2_t");
        printOffsets<Qnn_Tensor_t>("Qnn_Tensor_t");
        printOffsets<Qnn_TensorSetV1_t>("Qnn_TensorSetV1_t");
        printOffsets<Qnn_TensorSet_t>("Qnn_TensorSet_t");
        printOffsets<Qnn_OpConfigV1_t>("Qnn_OpConfigV1_t");
        printOffsets<Qnn_OpConfig_t>("Qnn_OpConfig_t");
    }

private:
    template<typename T>
    static void printOffsets(const std::string& structName) {
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

        std::cout << ss.str() << std::endl;
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
};

// Usage example
//int main() {
//    StructOffsetGenerator::generateAndPrintOffsets();
//    return 0;
//}
