using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static SampleCSharpApplication.QnnDelegates;

namespace SampleCSharpApplication
{
    public class IOTensor
    {
        [DllImport("kernel32.dll", EntryPoint = "RtlZeroMemory", SetLastError = false)]
        private static extern void ZeroMemory(IntPtr dest, IntPtr size);

        public static void ZeroMemory(IntPtr buffer, int startOffset, long length)
        {
            // Ensure we're not trying to write beyond the end of the buffer
            if (startOffset < 0 || length < 0)
            {
                throw new ArgumentOutOfRangeException("Offset and length must be non-negative.");
            }

            // Use the Win32 ZeroMemory function via P/Invoke
            ZeroMemory(buffer + startOffset, (IntPtr)length);
        }

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
            public long BatchSize;

            public PopulateInputTensorsRetType(StatusCode status, int numFilesPopulated, long numBatchSize)
            {
                Status = status;
                NumFilesPopulated = numFilesPopulated;
                BatchSize = numBatchSize;
            }
        }

        public struct ReadBatchDataRetType
        {
            public DataStatusCode Status;
            public int NumInputsCopied;
            public long NumBatchSize;

            public ReadBatchDataRetType(DataStatusCode status, int numInputsCopied, long numBatchSize)
            {
                Status = status;
                NumInputsCopied = numInputsCopied;
                NumBatchSize = numBatchSize;
            }
        }

        public struct CalculateLengthResult
        {
            public DataStatusCode Status;
            public long Length;

            public CalculateLengthResult(DataStatusCode status, long length)
            {
                Status = status;
                Length = length;
            }
        }

        public enum DataStatusCode
        {
            SUCCESS,
            DATA_READ_FAIL,
            DATA_WRITE_FAIL,
            FILE_OPEN_FAIL,
            DIRECTORY_CREATE_FAIL,
            INVALID_DIMENSIONS,
            INVALID_DATA_TYPE,
            DATA_SIZE_MISMATCH,
            INVALID_BUFFER
        }

        private static (DataStatusCode, int) GetDataTypeSizeInBytes(Qnn_DataType_t dataType)
        {
            switch (dataType)
            {
                case Qnn_DataType_t.QNN_DATATYPE_FLOAT_32:
                    return (DataStatusCode.SUCCESS, sizeof(float));
                case Qnn_DataType_t.QNN_DATATYPE_UINT_8:
                case Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8:
                    return (DataStatusCode.SUCCESS, sizeof(byte));
                case Qnn_DataType_t.QNN_DATATYPE_UINT_16:
                case Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16:
                    return (DataStatusCode.SUCCESS, sizeof(short));
                case Qnn_DataType_t.QNN_DATATYPE_UINT_32:
                case Qnn_DataType_t.QNN_DATATYPE_INT_32:
                    return (DataStatusCode.SUCCESS, sizeof(int));
                // Add more cases as needed
                default:
                    Console.WriteLine($"Unsupported data type: {dataType}");
                    return (DataStatusCode.INVALID_DATA_TYPE, 0);
            }
        }
        public static CalculateLengthResult CalculateLength(List<long> dims, Qnn_DataType_t dataType)
        {
            if (dims.Count == 0)
            {
                Console.WriteLine("dims.Count is zero");
                return new CalculateLengthResult(DataStatusCode.INVALID_DIMENSIONS, 0);
            }

            var (returnStatus, elementSize) = GetDataTypeSizeInBytes(dataType);
            if (returnStatus != DataStatusCode.SUCCESS)
            {
                return new CalculateLengthResult(returnStatus, 0);
            }

            long length = elementSize * CalculateElementCount(dims);
            return new CalculateLengthResult(DataStatusCode.SUCCESS, length);
        }


        public static ReadBatchDataRetType ReadBatchData(
        List<string> filePaths,
        int filePathsIndexOffset,
        bool loopBackToStart,
        List<long> dims,
        Qnn_DataType_t dataType,
        IntPtr buffer)
        {
            uint tensorLength = 0;
            if (buffer == IntPtr.Zero)
            {
                Console.WriteLine("buffer is null");
                return new ReadBatchDataRetType(DataStatusCode.INVALID_BUFFER, 0, 0);
            }

            CalculateLengthResult result = CalculateLength(dims, dataType);
            if (result.Status != DataStatusCode.SUCCESS)
            {
                return new ReadBatchDataRetType(result.Status, 0, 0);
            }

            int numInputsCopied = 0;
            long numBatchSize = 0;
            int totalLength = 0;
            int fileIndex = filePathsIndexOffset;

            while (true)
            {
                if (fileIndex >= filePaths.Count)
                {
                    if (loopBackToStart)
                    {
                        fileIndex = fileIndex % filePaths.Count;
                    }
                    else
                    {
                        numBatchSize += (tensorLength - totalLength) / (totalLength / numBatchSize);
                        // pad the vector with zeros
                        ZeroMemory(buffer, totalLength, tensorLength - totalLength);
                        break;
                    }
                }

                try
                {
                    using (FileStream fileStream = new FileStream(filePaths[fileIndex], FileMode.Open, FileAccess.Read))
                    {
                        int fileSize = (int)fileStream.Length;

                        if ((tensorLength % fileSize) != 0 || fileSize > tensorLength || fileSize == 0)
                        {
                            Console.WriteLine($"Given input file {filePaths[fileIndex]} with file size in bytes {fileSize}. " +
                                              $"If the model expects a batch size of one, the file size should match the tensor extent: {tensorLength} bytes. " +
                                              $"If the model expects a batch size > 1, the file size should evenly divide the tensor extent: {tensorLength} bytes.");
                            return new ReadBatchDataRetType(DataStatusCode.DATA_SIZE_MISMATCH, numInputsCopied, numBatchSize);
                        }

                        byte[] tempBuffer = new byte[fileSize];
                        if (fileStream.Read(tempBuffer, 0, fileSize) != fileSize)
                        {
                            Console.WriteLine($"Failed to read the contents of: {filePaths[fileIndex]}");
                            return new ReadBatchDataRetType(DataStatusCode.DATA_READ_FAIL, numInputsCopied, numBatchSize);
                        }

                        Marshal.Copy(tempBuffer, 0, buffer + (numInputsCopied * fileSize), fileSize);

                        totalLength += fileSize;
                        numInputsCopied += 1;
                        numBatchSize += 1;
                        fileIndex += 1;

                        if (totalLength >= tensorLength)
                        {
                            break;
                        }
                    }
                }
                catch (IOException)
                {
                    Console.WriteLine($"Failed to open input file: {filePaths[fileIndex]}");
                    return new ReadBatchDataRetType(DataStatusCode.FILE_OPEN_FAIL, numInputsCopied, numBatchSize);
                }
            }

            return new ReadBatchDataRetType(DataStatusCode.SUCCESS, numInputsCopied, numBatchSize);
        }


        public static PopulateInputTensorsRetType ReadDataAndAllocateBuffer(
            List<string> filePaths,
            int filePathsIndexOffset,
            bool loopBackToStart,
            List<long> dims,
            Qnn_DataType_t dataType,
            out IntPtr bufferToCopy)
        {
            StatusCode returnStatus = StatusCode.SUCCESS;
            bufferToCopy = IntPtr.Zero;

            returnStatus = AllocateBuffer(out bufferToCopy, dims, dataType);
            int numFilesPopulated = 0;
            long batchSize = 0;
            //(status, filesPopulated, size)
            ReadBatchDataRetType rbd = ReadBatchData(
                filePaths,
                filePathsIndexOffset,
                loopBackToStart,
                dims,
                dataType,
                bufferToCopy);

            if (rbd.Status != DataStatusCode.SUCCESS)
            {
                Console.Error.WriteLine("Failure in DataUtil.ReadBatchData");
                returnStatus = StatusCode.FAILURE;
            }

            if (returnStatus != StatusCode.SUCCESS)
            {
                if (bufferToCopy != IntPtr.Zero)
                {
                    UnmanagedMemoryTracker.FreeMemory(bufferToCopy);
                    bufferToCopy = IntPtr.Zero;
                }
            }

            numFilesPopulated = rbd.NumInputsCopied;
            batchSize = rbd.NumBatchSize;

            return new PopulateInputTensorsRetType
            {
                Status = returnStatus,
                NumFilesPopulated = numFilesPopulated,
                BatchSize = batchSize
            };
        }


        // TODO
        private static StatusCode AllocateBuffer(out IntPtr buffer, List<long> dims, Qnn_DataType_t dataType)
        {
            // Implement buffer allocation logic here
            buffer = IntPtr.Zero;
            return StatusCode.SUCCESS;
        }

        public static void GenericMarshalCopy<T>(T[] sourceArray, IntPtr destination, int length) where T : struct
        {
            int sizeOfT = Marshal.SizeOf<T>();
            int totalSize = length * sizeOfT;

            byte[] byteArray = new byte[totalSize];
            Buffer.BlockCopy(sourceArray, 0, byteArray, 0, totalSize);
            Marshal.Copy(byteArray, 0, destination, totalSize);
        }

        public static DataStatusCode FloatToTfN<T>(IntPtr outPtr, float[] inArray, int offset, float scale, int numElements) where T : struct, IConvertible
        {
            if (outPtr == IntPtr.Zero || inArray == null)
            {
                Console.Error.WriteLine("Received a null argument");
                return DataStatusCode.INVALID_BUFFER;
            }
            if (!typeof(T).IsValueType || !IsUnsignedInteger<T>())
            {
                throw new ArgumentException("FloatToTfN supports unsigned integers only!");
            }
            int dataTypeSizeInBytes = Marshal.SizeOf<T>();
            int bitWidth = dataTypeSizeInBytes * 8; // 8 bits per byte
            double trueBitWidthMax = Math.Pow(2, bitWidth) - 1;
            double encodingMin = offset * scale;
            double encodingMax = (trueBitWidthMax + offset) * scale;
            double encodingRange = encodingMax - encodingMin;
            T[] outArray = new T[numElements];
            for (int i = 0; i < numElements; ++i)
            {
                int quantizedValue = (int)Math.Round(trueBitWidthMax * (inArray[i] - encodingMin) / encodingRange);
                quantizedValue = Math.Max(0, Math.Min(quantizedValue, (int)trueBitWidthMax));
                outArray[i] = (T)Convert.ChangeType(quantizedValue, typeof(T));
            }

            GenericMarshalCopy(outArray, outPtr, numElements);
            return DataStatusCode.SUCCESS;
        }
        public static DataStatusCode CastFromFloat<T>(IntPtr outPtr, float[] inArray, int numElements) where T : struct
        {
            if (outPtr == IntPtr.Zero || inArray == null)
            {
                Console.Error.WriteLine("Received a null argument");
                return DataStatusCode.INVALID_BUFFER;
            }

            T[] outArray = new T[numElements];

            for (int i = 0; i < numElements; i++)
            {
                outArray[i] = (T)Convert.ChangeType(inArray[i], typeof(T));
            }
            GenericMarshalCopy(outArray, outPtr, numElements);

            return DataStatusCode.SUCCESS;
        }

        private static bool IsUnsignedInteger<T>() where T : struct, IConvertible
        {
            return typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) || typeof(T) == typeof(uint) || typeof(T) == typeof(ulong);
        }

        // Helper method to copy a float buffer, quantize it, and copy
        // it to a tensor (Qnn_Tensor_t) buffer.
        public static StatusCode CopyFromFloatToNative(IntPtr floatBuffer, Qnn_Tensor_t tensor)
        {
            if (floatBuffer == IntPtr.Zero)
            {
                Console.Error.WriteLine("CopyFromFloatToNative(): received a null argument");
                return StatusCode.FAILURE;
            }
            StatusCode returnStatus = StatusCode.SUCCESS;
            int elementCount = CalculateElementCount(tensor.v2.Dimensions);

            // Create a float array from the IntPtr
            float[] floatArray = new float[elementCount];
            Marshal.Copy(floatBuffer, floatArray, 0, elementCount);

            switch (tensor.v2.dataType)
            {
                case Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8:
                    FloatToTfN<byte>(tensor.v2.clientBuf.data, floatArray, tensor.v2.quantizeParams.scaleOffsetEncoding.offset, tensor.v2.quantizeParams.scaleOffsetEncoding.scale, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16:
                    FloatToTfN<ushort>(tensor.v2.clientBuf.data, floatArray, tensor.v2.quantizeParams.scaleOffsetEncoding.offset, tensor.v2.quantizeParams.scaleOffsetEncoding.scale, elementCount);
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UINT_8:
                    if (CastFromFloat<byte>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<byte>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UINT_16:
                    if (CastFromFloat<ushort>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<ushort>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_UINT_32:
                    if (CastFromFloat<uint>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<uint>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_INT_8:
                    if (CastFromFloat<sbyte>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<sbyte>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_INT_16:
                    if (CastFromFloat<short>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<short>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_INT_32:
                    if (CastFromFloat<int>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<int>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                case Qnn_DataType_t.QNN_DATATYPE_BOOL_8:
                    if (CastFromFloat<byte>(tensor.v2.clientBuf.data, floatArray, elementCount) != DataStatusCode.SUCCESS)
                    {
                        Console.Error.WriteLine("Failure in CastFromFloat<bool>");
                        returnStatus = StatusCode.FAILURE;
                    }
                    break;
                default:
                    Console.Error.WriteLine("Datatype not supported yet!");
                    returnStatus = StatusCode.FAILURE;
                    break;
            }
            return returnStatus;
        }
        private static int CalculateElementCount(uint[] dimensions)
        {
            int count = 1;
            foreach (int dim in dimensions)
            {
                count *= dim;
            }
            return count;
        }

        // Helper method to populate an input tensor in the graph during execution.

        public static PopulateInputTensorsRetType PopulateInputTensor(
            List<string> filePaths,
            int filePathsIndexOffset,
            bool loopBackToStart,
            Qnn_Tensor_t input,
            InputDataType inputDataType)
        {
            // TODO
            //if (input == null)
            //{
            //    Console.WriteLine("input is null");
            //    return new PopulateInputTensorsRetType(StatusCode.FAILURE, 0, 0);
            //}

            //StatusCode returnStatus = StatusCode.SUCCESS;
            //int numFilesPopulated = 0;
            //int batchSize = 0;
            ReadBatchDataRetType rbd = new ReadBatchDataRetType();

            StatusCode status = StatusCode.SUCCESS;
            List<long> dims = new List<long>();
            FillDims(dims, input.v2.Dimensions, input.v2.Rank);

            if (inputDataType == InputDataType.FLOAT && input.v2.dataType != Qnn_DataType_t.QNN_DATATYPE_FLOAT_32)
            {
                IntPtr fileToBuffer = IntPtr.Zero;
                try
                {
                    PopulateInputTensorsRetType rda = ReadDataAndAllocateBuffer(filePaths, filePathsIndexOffset, loopBackToStart, dims, Qnn_DataType_t.QNN_DATATYPE_FLOAT_32, out fileToBuffer);

                    if (rda.Status == StatusCode.SUCCESS)
                    {
                        Console.WriteLine("readDataFromFileToBuffer successful");
                        rda.Status = CopyFromFloatToNative(fileToBuffer, input);
                    }
                }
                finally
                {
                    if (fileToBuffer != IntPtr.Zero)
                    {
                        UnmanagedMemoryTracker.FreeMemory(fileToBuffer);
                       
                    }
                }
            }
            else
            {
                rbd = ReadBatchData(
                    filePaths,
                    filePathsIndexOffset,
                    loopBackToStart,
                    dims,
                    input.v2.dataType,
                    input.v2.clientBuf.data);

                if (rbd.Status != DataStatusCode.SUCCESS)
                {
                    Console.WriteLine("Failure in DataUtil.ReadBatchData");
                    status = StatusCode.FAILURE;
                }
            }

            return new PopulateInputTensorsRetType(status, rbd.NumInputsCopied, rbd.NumBatchSize);// numFilesPopulated, batchSize);
        }

        // Helper method to populate all input tensors during execution.
        public static PopulateInputTensorsRetType PopulateInputTensors(uint graphIdx, List<List<string>> filePathsVector, int filePathsIndexOffset,
               bool loopBackToStart, Dictionary<string, uint> inputNameToIndex, Qnn_Tensor_t[] inputs, GraphInfo_t graphInfo, InputDataType inputDataType)
        {
            Console.WriteLine($"populateInputTensors() graphIndx {graphIdx}");
            if (inputs == null)
            {
                Console.WriteLine("inputs is null");
                return new PopulateInputTensorsRetType(StatusCode.FAILURE, 0, 0);
            }

            uint inputCount = graphInfo.numInputTensors;
            if (filePathsVector.Count != inputCount)
            {
                Console.WriteLine($"Incorrect amount of Input files for graphIdx: {graphIdx}. " +
                                  $"Expected: {inputCount}, received: {filePathsVector.Count}");
                return new PopulateInputTensorsRetType(StatusCode.FAILURE, 0, 0);
            }

            int numFilesPopulated = 0;
            long numBatchSize = 0;

            for (int inputIdx = 0; inputIdx < inputCount; inputIdx++)
            {
                int inputNameIdx = inputIdx;
                Console.WriteLine($"index = {inputIdx} input column index = {inputNameIdx}");

                // TODO
                //string inputNodeName = graphInfo.inputTensors.inputTensors[inputIdx]
                string inputNodeName = "TODO";
                //if (!string.IsNullOrEmpty(inputNodeName) && inputNameToIndex.ContainsKey(inputNodeName))
                //{
                //    inputNameIdx = (int)inputNameToIndex[inputNodeName];
                //}

                var pit = // (returnStatus, currentInputNumFilesPopulated, currentInputNumBatchSize) =
                    PopulateInputTensor(filePathsVector[inputNameIdx],
                                        filePathsIndexOffset,
                                        loopBackToStart,
                                        inputs[inputIdx],
                                        inputDataType);

                if (pit.Status != StatusCode.SUCCESS)
                {
                    // TODO
                    //Console.WriteLine($"populateInputTensorFromFiles failed for input {inputNodeName} with index {inputIdx}");
                    return new PopulateInputTensorsRetType(StatusCode.FAILURE, pit.NumFilesPopulated, pit.BatchSize);
                }

                if (inputIdx == 0)
                {
                    numFilesPopulated = pit.NumFilesPopulated;
                    numBatchSize = pit.BatchSize;
                }
                else if (numFilesPopulated != pit.NumFilesPopulated || numBatchSize != pit.BatchSize)
                {
                    Console.WriteLine($"Current input tensor with name: {inputNodeName} with index {inputIdx} " +
                                      $"files populated = {pit.NumFilesPopulated}, batch size = {pit.BatchSize} " +
                                      $"does not match with expected files populated = {numFilesPopulated}, batch size = {numBatchSize}");
                    return new PopulateInputTensorsRetType(StatusCode.FAILURE, numFilesPopulated, numBatchSize);
                }
            }

            return new PopulateInputTensorsRetType(StatusCode.SUCCESS, numFilesPopulated, numBatchSize);
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
                UnmanagedMemoryTracker.FreeMemory(tensor.v2.isDynamicDimensions);
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

        private static StatusCode AllocateBuffer<T>(ref IntPtr buffer, ref int length, long elementCount) where T : unmanaged
        {
            try
            {
                length = (int)(elementCount * Marshal.SizeOf<T>());
                buffer = UnmanagedMemoryTracker.AllocateMemory(length);
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
                    returnStatus = AllocateBuffer<float>(ref buffer, ref length, elementCount);
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
                    // TODO
                    // You may need to add more logic here depending on the union structure in Qnn_QuantizeParams_t
                };

                // Handle memoryUnion based on memType
                if (src.v2.memType == Qnn_TensorMemType_t.QNN_TENSORMEMTYPE_RAW)
                {
                    // TODO

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
                    dest.v2.isDynamicDimensions = UnmanagedMemoryTracker.AllocateMemory(size); 
                    Marshal.Copy(src.v2.isDynamicDimensions, new byte[size], 0, size);
                }

                // Deep copy sparse parameters
                dest.v2.sparseParams = new Qnn_SparseParams_t
                {
                    type = src.v2.sparseParams.type
                    // TODO
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
            tensors = Array.Empty<Qnn_Tensor_t>();
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
                    if (AllocateBuffer(ref buffer, ref length, dims, tensors[tensorIdx].v2.dataType) != StatusCode.SUCCESS)
                    {
                        throw new Exception("Failed to allocate buffer");
                    }

                    clientBuffer.data = buffer;
                    clientBuffer.dataSize = (uint)length;

                    //try
                    //{
                    //    //clientBuffer.data = UnmanagedMemoryTracker.AllocateMemory(length); 
                    //    unsafe
                    //    {
                    //        Buffer.MemoryCopy(buffer.ToPointer(), clientBuffer.data.ToPointer(), length, length);
                    //    }
                    //}
                    //finally
                    //{
                    //    // Free the temporary buffer
                    //    if (buffer != IntPtr.Zero)
                    //    {
                    //        UnmanagedMemoryTracker.FreeMemory(buffer);
                    //    }
                    //}


                    tensors[tensorIdx].v2.clientBuf = clientBuffer;
                }

                return StatusCode.SUCCESS;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failure in SetupTensors: {ex.Message}");
                TearDownTensors(tensors, tensorCount);
                tensors = Array.Empty<Qnn_Tensor_t>();
                return StatusCode.FAILURE;
            }
        }

        // Setup details for all input and output tensors for graph execution.
        unsafe public StatusCode SetupInputAndOutputTensors(
            out Qnn_Tensor_t[] inputs,
            out Qnn_Tensor_t[] outputs,
            GraphInfo_t graphInfo)
        {
            inputs = Array.Empty<Qnn_Tensor_t>();
            outputs = Array.Empty<Qnn_Tensor_t>();
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
                    inputs = Array.Empty<Qnn_Tensor_t>();
                }
                if (outputs != null)
                {
                    TearDownTensors(outputs, graphInfo.numOutputTensors);
                    outputs = Array.Empty<Qnn_Tensor_t>();
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
        //public StatusCode AllocateBuffer<T>(out T[] buffer, int elementCount)
        //{
        //    // Implementation needed
        //    buffer = null;
        //    return StatusCode.SUCCESS;
        //}

        // Convert data to float or de-quantization.
        //public StatusCode ConvertToFloat(out float[] output, Qnn_Tensor_t tensor)
        //{
        //    // Implementation needed
        //    output = null;
        //    return StatusCode.SUCCESS;
        //}

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
        //public StatusCode AllocateAndCopyBuffer(out byte[] buffer, Qnn_Tensor_t tensor)
        //{
        //    // Implementation needed
        //    buffer = null;
        //    return StatusCode.SUCCESS;
        //}

        public static StatusCode FillDims(List<long> dims, uint[] inDimensions, uint rank)
        {
            if (inDimensions == null || inDimensions.Count() == 0)
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
