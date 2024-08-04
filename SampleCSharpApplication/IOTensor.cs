using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static SampleCSharpApplication.QnnDelegates;

namespace SampleCSharpApplication
{
    public class IOTensor
    {
        public enum StatusCode
        {
            SUCCESS,
            FAILURE
        }

        public enum InputDataType
        {
            FLOAT,
            NATIVE,
            INVALID
        }

        public enum OutputDataType
        {
            FLOAT_ONLY,
            NATIVE_ONLY,
            FLOAT_AND_NATIVE,
            INVALID
        }

        public struct PopulateInputTensorsRetType
        {
            public StatusCode Status;
            public int NumFilesPopulated;
            public int BatchSize;
        }

        // Helper method to read data from files to a buffer.
        public PopulateInputTensorsRetType ReadDataAndAllocateBuffer(
            List<string> filePaths,
            int filePathsIndexOffset,
            bool loopBackToStart,
            List<int> dims,
            Qnn_DataType_t dataType,
            out byte[] bufferToCopy)
        {
            // Implementation needed
            bufferToCopy = null;
            return new PopulateInputTensorsRetType();
        }

        // Helper method to copy a float buffer, quantize it, and copy
        // it to a tensor (Qnn_Tensor_t) buffer.
        public StatusCode CopyFromFloatToNative(float[] floatBuffer, Qnn_Tensor_t tensor)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        // Helper method to populate an input tensor in the graph during execution.
        public PopulateInputTensorsRetType PopulateInputTensor(
            List<string> filePaths,
            int filePathsIndexOffset,
            bool loopBackToStart,
            Qnn_Tensor_t input,
            InputDataType inputDataType)
        {
            // Implementation needed
            return new PopulateInputTensorsRetType();
        }

        // Helper method to populate all input tensors during execution.
        public PopulateInputTensorsRetType PopulateInputTensors(
            uint graphIdx,
            List<List<string>> filePathsVector,
            int filePathsIndexOffset,
            bool loopBackToStart,
            Dictionary<string, uint> inputNameToIndex,
            Qnn_Tensor_t[] inputs,
            GraphInfo_t graphInfo,
            InputDataType inputDataType)
        {
            // Implementation needed
            return new PopulateInputTensorsRetType();
        }

        // Setup details for Qnn_Tensor_t for execution
        public StatusCode SetupTensors(
            out Qnn_Tensor_t[] tensors,
            uint tensorCount,
            Qnn_Tensor_t[] tensorWrappers)
        {
            // Implementation needed
            tensors = null;
            return StatusCode.SUCCESS;
        }

        // Setup details for all input and output tensors for graph execution.
        public StatusCode SetupInputAndOutputTensors(
            out Qnn_Tensor_t[] inputs,
            out Qnn_Tensor_t[] outputs,
            GraphInfo_t graphInfo)
        {
            // Implementation needed
            inputs = null;
            outputs = null;
            return StatusCode.SUCCESS;
        }

        // Clean up all tensors related data after execution.
        public StatusCode TearDownTensors(Qnn_Tensor_t[] tensors, uint tensorCount)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        // Clean up all input and output tensors after execution.
        public StatusCode TearDownInputAndOutputTensors(
            Qnn_Tensor_t[] inputs,
            Qnn_Tensor_t[] outputs,
            int numInputTensors,
            int numOutputTensors)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        // Helper method to allocate a buffer.
        public StatusCode AllocateBuffer(out byte[] buffer, List<int> dims, Qnn_DataType_t dataType)
        {
            // Implementation needed
            buffer = null;
            return StatusCode.SUCCESS;
        }

        // Helper method to allocate a buffer.
        public StatusCode AllocateBuffer<T>(out T[] buffer, int elementCount)
        {
            // Implementation needed
            buffer = null;
            return StatusCode.SUCCESS;
        }

        // Convert data to float or de-quantization.
        public StatusCode ConvertToFloat(out float[] output, Qnn_Tensor_t tensor)
        {
            // Implementation needed
            output = null;
            return StatusCode.SUCCESS;
        }

        // Helper method to convert Output tensors to float and write them out to files.
        public StatusCode ConvertAndWriteOutputTensorInFloat(
            Qnn_Tensor_t output,
            List<string> outputPaths,
            string fileName,
            int outputBatchSize)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        // Helper method to write out output. There is no de-quantization here.
        public StatusCode WriteOutputTensor(
            Qnn_Tensor_t output,
            List<string> outputPaths,
            string fileName,
            int outputBatchSize)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        // Write out all output tensors to files.
        public StatusCode WriteOutputTensors(
            uint graphIdx,
            int startIdx,
            string graphName,
            Qnn_Tensor_t[] outputs,
            uint numOutputs,
            OutputDataType outputDatatype,
            uint graphsCount,
            string outputPath,
            int numInputFilesPopulated,
            int outputBatchSize)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        // Helper method to allocate a buffer and copy data to it.
        public StatusCode AllocateAndCopyBuffer(out byte[] buffer, Qnn_Tensor_t tensor)
        {
            // Implementation needed
            buffer = null;
            return StatusCode.SUCCESS;
        }

        public StatusCode FillDims(List<int> dims, uint[] inDimensions, uint rank)
        {
            // Implementation needed
            return StatusCode.SUCCESS;
        }

        public static OutputDataType ParseOutputDataType(string dataTypeString)
        {
            var lowercaseDataType = dataTypeString.ToLower();
            switch (lowercaseDataType)
            {
                case "float_only":
                    return OutputDataType.FLOAT_ONLY;
                case "native_only":
                    return OutputDataType.NATIVE_ONLY;
                case "float_and_native":
                    return OutputDataType.FLOAT_AND_NATIVE;
                default:
                    return OutputDataType.INVALID;
            }
        }

        public static InputDataType ParseInputDataType(string dataTypeString)
        {
            var lowercaseDataType = dataTypeString.ToLower();
            switch (lowercaseDataType)
            {
                case "float":
                    return InputDataType.FLOAT;
                case "native":
                    return InputDataType.NATIVE;
                default:
                    return InputDataType.INVALID;
            }
        }
    }
}
