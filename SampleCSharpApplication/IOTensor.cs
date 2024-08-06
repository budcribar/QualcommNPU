using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static SampleCSharpApplication.QnnDelegates;

namespace SampleCSharpApplication
{
    public class IOTensor
    {
        public enum OutputDataType
        {
            /// <summary>
            /// Float data type only.
            /// </summary>
            FLOAT_ONLY,

            /// <summary>
            /// Native data type only.
            /// </summary>
            NATIVE_ONLY,

            /// <summary>
            /// Both float and native data types.
            /// </summary>
            FLOAT_AND_NATIVE,

            /// <summary>
            /// Invalid or unspecified data type.
            /// </summary>
            INVALID
        }

        /// <summary>
        /// Represents the type of input data.
        /// </summary>
        public enum InputDataType
        {
            /// <summary>
            /// Float data type.
            /// </summary>
            FLOAT,

            /// <summary>
            /// Native data type.
            /// </summary>
            NATIVE,

            /// <summary>
            /// Invalid or unspecified data type.
            /// </summary>
            INVALID
        }
        public enum StatusCode
        {
            SUCCESS,
            FAILURE
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
        private void FreeTensorResources(ref Qnn_Tensor_t tensor)
        {
            if (tensor.v2.Dimensions != null)
            {
                //tensor.v2.Dimensions = null;
            }

            if (tensor.v2.memType == Qnn_TensorMemType_t.QNN_TENSORMEMTYPE_RAW && tensor.v2.clientBuf.data != IntPtr.Zero)
            {
                //var clientBuf = Marshal.PtrToStructure<Qnn_ClientBuffer_t>(tensor.v2.memoryUnion);
                //if (clientBuf.data != IntPtr.Zero)
                //{
                //    Marshal.FreeHGlobal(clientBuf.data);
                //}
                //Marshal.FreeHGlobal(tensor.v2.memoryUnion);
                //tensor.v2.memoryUnion = IntPtr.Zero;
            }

            if (tensor.v2.isDynamicDimensions != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(tensor.v2.isDynamicDimensions);
                tensor.v2.isDynamicDimensions = IntPtr.Zero;
            }

            // Add any other necessary cleanup here
        }

        private static long CalculateElementCount(List<long> dims)
        {
            long count = 1;
            foreach (var dim in dims)
            {
                count *= dim;
            }
            return count;
        }

        private static StatusCode AllocateBuffer<T>(ref IntPtr buffer, ref int length,  long elementCount) where T : unmanaged
        {
            try
            {
                length = (int)(elementCount * Marshal.SizeOf<T>());
                buffer = Marshal.AllocHGlobal(length);
                return StatusCode.SUCCESS;
            }
            catch (OutOfMemoryException)
            {
                buffer = IntPtr.Zero;
                Console.WriteLine("Failed to allocate memory");
                return StatusCode.FAILURE;
            }
        }

        public static StatusCode AllocateBuffer(ref IntPtr buffer, ref int length, List<long> dims, Qnn_DataType_t dataType)
        {
            long elementCount = CalculateElementCount(dims);
            StatusCode returnStatus = StatusCode.SUCCESS;

            switch (dataType)
            {
                case Qnn_DataType_t.QNN_DATATYPE_FLOAT_32:
                    Console.WriteLine("Allocating float buffer");
                    returnStatus = AllocateBuffer<float>(ref buffer,ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UINT_8:
                case Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8:
                    Console.WriteLine("Allocating uint8_t buffer");
                    returnStatus = AllocateBuffer<byte>(ref buffer, ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UINT_16:
                case Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16:
                    Console.WriteLine("Allocating uint16_t buffer");
                    returnStatus = AllocateBuffer<ushort>(ref buffer, ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UINT_32:
                    Console.WriteLine("Allocating uint32_t buffer");
                    returnStatus = AllocateBuffer<uint>(ref buffer, ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_INT_8:
                    Console.WriteLine("Allocating int8_t buffer");
                    returnStatus = AllocateBuffer<sbyte>(ref buffer, ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_INT_16:
                    Console.WriteLine("Allocating int16_t buffer");
                    returnStatus = AllocateBuffer<short>(ref buffer, ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_INT_32:
                    Console.WriteLine("Allocating int32_t buffer");
                    returnStatus = AllocateBuffer<int>(ref buffer, ref length, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_BOOL_8:
                    Console.WriteLine("Allocating bool buffer");
                    returnStatus = AllocateBuffer<byte>(ref buffer, ref length, elementCount);
                    break;
                default:
                    Console.WriteLine("Datatype not supported yet!");
                    returnStatus = StatusCode.FAILURE;
                    break;
            }

            return returnStatus;
        }
        private bool DeepCopyQnnTensorInfo(ref Qnn_Tensor_t dest, Qnn_Tensor_t src)
        {
            try
            {
                // Copy version
                dest.version = src.version;

                // Deep copy Qnn_TensorV2_t
                dest.v2 = new Qnn_TensorV2_t();

                // Copy simple fields
                dest.v2.id = src.v2.id;
                dest.v2.type = src.v2.type;
                dest.v2.dataFormat = src.v2.dataFormat;
                dest.v2.dataType = src.v2.dataType;
                dest.v2.Rank = src.v2.Rank;
                dest.v2.memType = src.v2.memType;
                dest.v2.isProduced = src.v2.isProduced;

                dest.v2.Name = src.v2.Name;
                
                // Deep copy dimensions
                if (src.v2.Dimensions != null && src.v2.Rank > 0)
                {
                    dest.v2.Dimensions = src.v2.Dimensions;
                }

                // Deep copy quantization parameters
                dest.v2.quantizeParams = new Qnn_QuantizeParams_t
                {
                    encodingDefinition = src.v2.quantizeParams.encodingDefinition,
                    quantizationEncoding = src.v2.quantizeParams.quantizationEncoding
                    // You may need to add more logic here depending on the union structure in Qnn_QuantizeParams_t
                };

                // Handle memoryUnion based on memType
                if (src.v2.memType == Qnn_TensorMemType_t.QNN_TENSORMEMTYPE_RAW)
                {


                    //var srcClientBuf = Marshal.PtrToStructure<Qnn_ClientBuffer_t>(src.v2.clientBuf.data);
                    //var destClientBuf = new Qnn_ClientBuffer_t();
                    //if (srcClientBuf.data != IntPtr.Zero && srcClientBuf.dataSize > 0)
                    //{
                    //    destClientBuf.data = Marshal.AllocHGlobal((int)srcClientBuf.dataSize);
                    //    Marshal.Copy(srcClientBuf.data, new byte[srcClientBuf.dataSize], 0, (int)srcClientBuf.dataSize);
                    //    destClientBuf.dataSize = srcClientBuf.dataSize;
                    //}
                    //dest.v2.memoryUnion = Marshal.AllocHGlobal(Marshal.SizeOf<Qnn_ClientBuffer_t>());
                    //Marshal.StructureToPtr(destClientBuf, dest.v2.memoryUnion, false);
                }
                else if (src.v2.memType == Qnn_TensorMemType_t.QNN_TENSORMEMTYPE_MEMHANDLE)
                {
                    //dest.v2.memoryUnion = src.v2.memoryUnion; // Just copy the handle
                }

                // Deep copy dynamic dimensions flag
                if (src.v2.isDynamicDimensions != IntPtr.Zero && src.v2.Rank > 0)
                {
                    int size = (int)src.v2.Rank * sizeof(byte);
                    dest.v2.isDynamicDimensions = Marshal.AllocHGlobal(size);
                    Marshal.Copy(src.v2.isDynamicDimensions, new byte[size], 0, size);
                }

                // Deep copy sparse parameters
                dest.v2.sparseParams = new Qnn_SparseParams_t
                {
                    type = src.v2.sparseParams.type
                    // You may need to add more logic here depending on the structure of Qnn_SparseParams_t
                };

                return true;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error in DeepCopyQnnTensorInfo: {ex.Message}");
                // Clean up any allocated memory in case of failure
                FreeTensorResources(ref dest);
                return false;
            }
        }

        // Setup details for Qnn_Tensor_t for execution
        unsafe private StatusCode SetupTensors(out Qnn_Tensor_t[] tensors, uint tensorCount, Qnn_Tensor_t* tensorWrappers)
        {
            tensors = null;
            if (tensorWrappers == null)
            {
                Console.Error.WriteLine("tensorWrappers is null");
                return StatusCode.FAILURE;
            }
            if (tensorCount == 0)
            {
                Console.WriteLine("tensor count is 0. Nothing to setup.");
                return StatusCode.SUCCESS;
            }

            try
            {
                tensors = new Qnn_Tensor_t[tensorCount];

                for (int tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++)
                {
                    // Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
                    Qnn_Tensor_t wrapperTensor = *tensorWrappers;
                    var dims = new List<long>();
                    FillDims(dims, wrapperTensor.v2.Dimensions, wrapperTensor.v2.Rank);

                    tensors[tensorIdx] = new Qnn_Tensor_t(); // Assuming Qnn_Tensor_t has a default constructor
                    if (!DeepCopyQnnTensorInfo(ref tensors[tensorIdx], wrapperTensor))
                    {
                        throw new Exception("Failed to deep copy QnnTensorInfo");
                    }

                    tensors[tensorIdx].v2.memType = Qnn_TensorMemType_t.QNN_TENSORMEMTYPE_RAW;

                    var clientBuffer = new Qnn_ClientBuffer_t();

                    IntPtr buffer = IntPtr.Zero;
                    int length = 0;
                    if (AllocateBuffer(ref buffer,ref length, dims, tensors[tensorIdx].v2.dataType) != StatusCode.SUCCESS)
                    {
                        throw new Exception("Failed to allocate buffer");
                    }

                    clientBuffer.data = Marshal.AllocHGlobal(length);
                    clientBuffer.dataSize = (uint)length;

                    try
                    {
                        clientBuffer.data = Marshal.AllocHGlobal(length);
                        unsafe
                        {
                            Buffer.MemoryCopy(buffer.ToPointer(), clientBuffer.data.ToPointer(), length, length);
                        }
                    }
                    finally
                    {
                        // Free the temporary buffer
                        if (buffer != IntPtr.Zero)
                        {
                            Marshal.FreeHGlobal(buffer);
                        }
                    }


                    tensors[tensorIdx].v2.clientBuf = clientBuffer;
                }

                return StatusCode.SUCCESS;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failure in SetupTensors: {ex.Message}");
                TearDownTensors(tensors, tensorCount);
                tensors = null;
                return StatusCode.FAILURE;
            }
        }

        // Setup details for all input and output tensors for graph execution.
        unsafe public StatusCode SetupInputAndOutputTensors(
            out Qnn_Tensor_t[] inputs,
            out Qnn_Tensor_t[] outputs,
            GraphInfo_t graphInfo)
        {
            inputs = null;
            outputs = null;
            StatusCode returnStatus = StatusCode.SUCCESS;

            try
            {
                // Setup input tensors
                if (SetupTensors(out inputs, graphInfo.numInputTensors, graphInfo.inputTensors) != StatusCode.SUCCESS)
                {
                    throw new Exception("Failure in setting up input tensors");
                }

                // Setup output tensors
                if (SetupTensors(out outputs, graphInfo.numOutputTensors, graphInfo.outputTensors) != StatusCode.SUCCESS)
                {
                    throw new Exception("Failure in setting up output tensors");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failure in setupInputAndOutputTensors: {ex.Message}");
                returnStatus = StatusCode.FAILURE;

                // Clean up resources if an error occurred
                if (inputs != null)
                {
                    TearDownTensors(inputs, graphInfo.numInputTensors);
                    inputs = null;
                }
                if (outputs != null)
                {
                    TearDownTensors(outputs, graphInfo.numOutputTensors);
                    outputs = null;
                }
            }

            return returnStatus;
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

        public StatusCode FillDims(List<long> dims, uint[] inDimensions, uint rank)
        {
            if (inDimensions == null)
            {
                Console.Error.WriteLine("Input dimensions is null");
                return StatusCode.FAILURE;
            }

            dims.Clear(); // Ensure the list is empty before filling

            try
            {
                for (int r = 0; r < rank; r++)
                {
                    dims.Add(inDimensions[r]);
                }
                return StatusCode.SUCCESS;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error in FillDims: {ex.Message}");
                return StatusCode.FAILURE;
            }
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
