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

class StructOffsetGenerator {
public:
    static void generateAndWriteOffsets(const std::string& outputDir) {
        createDirectory(outputDir);
        writeOffsets<Qnn_TensorV1_t>(outputDir, "Qnn_TensorV1_t");
        writeOffsets<Qnn_TensorV2_t>(outputDir, "Qnn_TensorV2_t");
        writeOffsets<Qnn_Tensor_t>(outputDir, "Qnn_Tensor_t");
        writeOffsets<Qnn_TensorSetV1_t>(outputDir, "Qnn_TensorSetV1_t");
        writeOffsets<Qnn_TensorSet_t>(outputDir, "Qnn_TensorSet_t");
        writeOffsets<Qnn_OpConfigV1_t>(outputDir, "Qnn_OpConfigV1_t");
        writeOffsets<Qnn_OpConfig_t>(outputDir, "Qnn_OpConfig_t");
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
};

//int main() {
//    StructOffsetGenerator::generateAndWriteOffsets("./struct_offsets");
//    return 0;
//}