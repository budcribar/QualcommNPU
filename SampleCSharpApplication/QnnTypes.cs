using System.Runtime.InteropServices;

namespace SampleCSharpApplication
{
    // TODO QnnDevice_Error_t

    [StructLayout(LayoutKind.Sequential)]
    public struct GraphConfigInfo_t
    {
        private readonly IntPtr graphName;  // char* in C++
        public IntPtr GraphConfigs;  // const QnnGraph_Config_t** in C++

        public string GraphName
        {
            get
            {
                if (graphName == IntPtr.Zero)
                    return string.Empty;
                return Marshal.PtrToStringAnsi(graphName) ?? string.Empty;
            }
        }
    }

    public struct GraphInfo_t
    {
        public Qnn_GraphHandle_t graph;
        public IntPtr graphName;             // char* in C++ is IntPtr in C#
        unsafe public Qnn_Tensor_t* inputTensors;
        public uint numInputTensors;
        unsafe public Qnn_Tensor_t* outputTensors;
        public uint numOutputTensors;

        public string GraphNameString
        {
            get
            {
                if (graphName == IntPtr.Zero)
                    return string.Empty;
                return Marshal.PtrToStringAnsi(graphName) ?? string.Empty;
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Qnn_Version_t
    {
        public uint Major;
        public uint Minor;
        public uint Patch;
    }

    public static class QnnConstants
    {
        public const uint QNN_MIN_ERROR_COMMON = 1000;  // Assuming this value, please adjust if different
        public const uint QNN_MAX_ERROR_COMMON = 1999;  // Assuming this value, please adjust if different
        public const uint QNN_MIN_ERROR_PROFILE = 12000;  // Assuming this value, please adjust if different
        public const uint QNN_MAX_ERROR_PROFILE = 12999;  // Assuming this value, please adjust if different
        public const uint QNN_SUCCESS = 0;
        public const uint QNN_MIN_ERROR_BACKEND = 4000;  // Assuming this value, please adjust if different
        public const uint QNN_MAX_ERROR_BACKEND = 4999;  // Assuming this value, please adjust if different
        public const uint QNN_MIN_ERROR_CONTEXT = 5000;
        public const uint QNN_MAX_ERROR_CONTEXT = 5999;
    }

    public enum QnnBackend_Error_t : uint
    {
        QNN_BACKEND_MIN_ERROR = QnnConstants.QNN_MIN_ERROR_BACKEND,

        /// <summary>
        /// Qnn Backend success
        /// </summary>
        QNN_BACKEND_NO_ERROR = QnnConstants.QNN_SUCCESS,

        /// <summary>
        /// General error relating to memory allocation in Backend API
        /// </summary>
        QNN_BACKEND_ERROR_MEM_ALLOC = QnnCommon_Error_t.QNN_COMMON_ERROR_MEM_ALLOC,

        /// <summary>
        /// Backend attempted to be created on an unsupported platform
        /// </summary>
        QNN_BACKEND_ERROR_UNSUPPORTED_PLATFORM = QnnCommon_Error_t.QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED,

        /// <summary>
        /// Backend failed to initialize
        /// </summary>
        QNN_BACKEND_ERROR_CANNOT_INITIALIZE = QnnConstants.QNN_MIN_ERROR_BACKEND + 0,

        /// <summary>
        /// Failed to free allocated resources during termination
        /// </summary>
        QNN_BACKEND_ERROR_TERMINATE_FAILED = QnnConstants.QNN_MIN_ERROR_BACKEND + 2,

        /// <summary>
        /// Backend does not support requested functionality
        /// </summary>
        QNN_BACKEND_ERROR_NOT_SUPPORTED = QnnConstants.QNN_MIN_ERROR_BACKEND + 3,

        /// <summary>
        /// Invalid function argument
        /// </summary>
        QNN_BACKEND_ERROR_INVALID_ARGUMENT = QnnConstants.QNN_MIN_ERROR_BACKEND + 4,

        /// <summary>
        /// Could not find specified op package
        /// </summary>
        QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND = QnnConstants.QNN_MIN_ERROR_BACKEND + 5,

        /// <summary>
        /// Could not load interface provider from op package library
        /// </summary>
        QNN_BACKEND_ERROR_OP_PACKAGE_IF_PROVIDER_NOT_FOUND = QnnConstants.QNN_MIN_ERROR_BACKEND + 6,

        /// <summary>
        /// Failed to register op package
        /// </summary>
        QNN_BACKEND_ERROR_OP_PACKAGE_REGISTRATION_FAILED = QnnConstants.QNN_MIN_ERROR_BACKEND + 7,

        /// <summary>
        /// Backend does not support the op config's interface version
        /// </summary>
        QNN_BACKEND_ERROR_OP_PACKAGE_UNSUPPORTED_VERSION = QnnConstants.QNN_MIN_ERROR_BACKEND + 8,

        /// <summary>
        /// An Op with the same package name and op name was already registered
        /// </summary>
        QNN_BACKEND_ERROR_OP_PACKAGE_DUPLICATE = QnnConstants.QNN_MIN_ERROR_BACKEND + 9,

        /// <summary>
        /// Inconsistent backend configuration
        /// </summary>
        QNN_BACKEND_ERROR_INCONSISTENT_CONFIG = QnnConstants.QNN_MIN_ERROR_BACKEND + 10,

        /// <summary>
        /// Invalid backend handle
        /// </summary>
        QNN_BACKEND_ERROR_INVALID_HANDLE = QnnConstants.QNN_MIN_ERROR_BACKEND + 11,

        /// <summary>
        /// Invalid config
        /// </summary>
        QNN_BACKEND_ERROR_INVALID_CONFIG = QnnConstants.QNN_MIN_ERROR_BACKEND + 12,

        QNN_BACKEND_MAX_ERROR = QnnConstants.QNN_MAX_ERROR_BACKEND,

        /// <summary>
        /// Unused, present to ensure 32 bits.
        /// </summary>
        QNN_BACKEND_ERROR_UNDEFINED = 0x7FFFFFFF
    }

    [Flags]
    public enum QnnCommon_Error_t : uint
    {
        QNN_COMMON_MIN_ERROR = QnnConstants.QNN_MIN_ERROR_COMMON,

        /// <summary>
        /// API or feature is not supported by implementation.
        /// </summary>
        QNN_COMMON_ERROR_NOT_SUPPORTED = QnnConstants.QNN_MIN_ERROR_COMMON + 0,

        /// <summary>
        /// Memory allocation related error.
        /// </summary>
        QNN_COMMON_ERROR_MEM_ALLOC = QnnConstants.QNN_MIN_ERROR_COMMON + 2,

        /// <summary>
        /// System level error, such as related to platform / OS services
        /// </summary>
        QNN_COMMON_ERROR_SYSTEM = QnnConstants.QNN_MIN_ERROR_COMMON + 3,

        /// <summary>
        /// Invalid function argument
        /// </summary>
        QNN_COMMON_ERROR_INVALID_ARGUMENT = QnnConstants.QNN_MIN_ERROR_COMMON + 4,

        /// <summary>
        /// Illegal operation or sequence of operations
        /// </summary>
        QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED = QnnConstants.QNN_MIN_ERROR_COMMON + 5,

        /// <summary>
        /// Attempt to use QNN API on an unsupported platform
        /// </summary>
        QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED = QnnConstants.QNN_MIN_ERROR_COMMON + 6,

        /// <summary>
        /// Communication errors with platform / OS service
        /// </summary>
        QNN_COMMON_ERROR_SYSTEM_COMMUNICATION = QnnConstants.QNN_MIN_ERROR_COMMON + 7,

        /// <summary>
        /// Loaded libraries are of incompatible versions
        /// </summary>
        QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES = QnnConstants.QNN_MIN_ERROR_COMMON + 8,

        /// <summary>
        /// Attempt to reload library already loaded in this process
        /// </summary>
        QNN_COMMON_ERROR_LOADING_BINARIES = QnnConstants.QNN_MIN_ERROR_COMMON + 9,

        /// <summary>
        /// Resource allocation related error.
        /// </summary>
        QNN_COMMON_ERROR_RESOURCE_UNAVAILABLE = QnnConstants.QNN_MIN_ERROR_COMMON + 10,

        /// <summary>
        /// General error, which has not been identified as any other error type.
        /// </summary>
        QNN_COMMON_ERROR_GENERAL = QnnConstants.QNN_MIN_ERROR_COMMON + 100,

        QNN_COMMON_MAX_ERROR = QnnConstants.QNN_MAX_ERROR_COMMON,

        /// <summary>
        /// Unused, present to ensure 32 bits.
        /// </summary>
        QNN_COMMON_ERROR_UNDEFINED = 0x7FFFFFFF
    }
    public enum QnnContextError : uint
    {
        QNN_CONTEXT_MIN_ERROR = QnnConstants.QNN_MIN_ERROR_CONTEXT,

        ////////////////////////////////////////////

        /// Qnn context success
        QNN_CONTEXT_NO_ERROR = QnnConstants.QNN_SUCCESS,
        /// There is optional API component that is not supported yet. See QnnProperty.
        QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE = QnnCommon_Error_t.QNN_COMMON_ERROR_NOT_SUPPORTED,
        /// Context-specific memory allocation/deallocation failure
        QNN_CONTEXT_ERROR_MEM_ALLOC = QnnCommon_Error_t.QNN_COMMON_ERROR_MEM_ALLOC,
        /// An argument to QNN context API is deemed invalid by a backend
        QNN_CONTEXT_ERROR_INVALID_ARGUMENT = QnnConstants.QNN_MIN_ERROR_CONTEXT + 0,
        /// A QNN context has not yet been created in the backend
        QNN_CONTEXT_ERROR_CTX_DOES_NOT_EXIST = QnnConstants.QNN_MIN_ERROR_CONTEXT + 1,
        /// Invalid/NULL QNN context handle
        QNN_CONTEXT_ERROR_INVALID_HANDLE = QnnConstants.QNN_MIN_ERROR_CONTEXT + 2,
        /// Attempting an operation when graphs in a context haven't been finalized
        QNN_CONTEXT_ERROR_NOT_FINALIZED = QnnConstants.QNN_MIN_ERROR_CONTEXT + 3,
        /// Attempt to access context binary with an incompatible version
        QNN_CONTEXT_ERROR_BINARY_VERSION = QnnConstants.QNN_MIN_ERROR_CONTEXT + 4,
        /// Failure to create context from binary
        QNN_CONTEXT_ERROR_CREATE_FROM_BINARY = QnnConstants.QNN_MIN_ERROR_CONTEXT + 5,
        /// Failure to get size of a QNN serialized context
        QNN_CONTEXT_ERROR_GET_BINARY_SIZE_FAILED = QnnConstants.QNN_MIN_ERROR_CONTEXT + 6,
        /// Failure to generate a QNN serialized context
        QNN_CONTEXT_ERROR_GET_BINARY_FAILED = QnnConstants.QNN_MIN_ERROR_CONTEXT + 7,
        /// Invalid context binary configuration
        QNN_CONTEXT_ERROR_BINARY_CONFIGURATION = QnnConstants.QNN_MIN_ERROR_CONTEXT + 8,
        /// Failure to set profile
        QNN_CONTEXT_ERROR_SET_PROFILE = QnnConstants.QNN_MIN_ERROR_CONTEXT + 9,
        /// Invalid config
        QNN_CONTEXT_ERROR_INVALID_CONFIG = QnnConstants.QNN_MIN_ERROR_CONTEXT + 10,
        /// Attempt to create a context from suboptimal binary
        QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL = QnnConstants.QNN_MIN_ERROR_CONTEXT + 11,
        /// Call aborted early due to a QnnSignal_trigger call issued
        /// to the observed signal object.
        QNN_CONTEXT_ERROR_ABORTED = QnnConstants.QNN_MIN_ERROR_CONTEXT + 12,
        /// Call aborted early due to a QnnSignal timeout
        QNN_CONTEXT_ERROR_TIMED_OUT = QnnConstants.QNN_MIN_ERROR_CONTEXT + 13,

        ////////////////////////////////////////////

        QNN_CONTEXT_MAX_ERROR = QnnConstants.QNN_MAX_ERROR_CONTEXT,
        // Unused, present to ensure 32 bits.
        QNN_CONTEXT_ERROR_UNDEFINED = 0x7FFFFFFF
    }
    public static class QnnDelegates
    {
        [Flags]
        public enum QnnProfile_Error_t : uint
        {
            QNN_PROFILE_MIN_ERROR = QnnConstants.QNN_MIN_ERROR_PROFILE,

            /// <summary>
            /// Qnn Profile success
            /// </summary>
            QNN_PROFILE_NO_ERROR = QnnConstants.QNN_SUCCESS,

            /// <summary>
            /// Backend does not support requested functionality
            /// </summary>
            QNN_PROFILE_ERROR_UNSUPPORTED = QnnCommon_Error_t.QNN_COMMON_ERROR_NOT_SUPPORTED,

            /// <summary>
            /// Invalid function argument
            /// </summary>
            QNN_PROFILE_ERROR_INVALID_ARGUMENT = QnnCommon_Error_t.QNN_COMMON_ERROR_INVALID_ARGUMENT,

            /// <summary>
            /// General error relating to memory allocation in Profile API
            /// </summary>
            QNN_PROFILE_ERROR_MEM_ALLOC = QnnCommon_Error_t.QNN_COMMON_ERROR_MEM_ALLOC,

            /// <summary>
            /// Invalid/NULL QNN profile handle
            /// </summary>
            QNN_PROFILE_ERROR_INVALID_HANDLE = QnnConstants.QNN_MIN_ERROR_PROFILE + 0,

            /// <summary>
            /// Attempt to free or reconfigure a profile handle that is in-use
            /// </summary>
            QNN_PROFILE_ERROR_HANDLE_IN_USE = QnnConstants.QNN_MIN_ERROR_PROFILE + 1,

            /// <summary>
            /// Event is incompatible with API
            /// </summary>
            QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT = QnnConstants.QNN_MIN_ERROR_PROFILE + 2,

            QNN_PROFILE_MAX_ERROR = QnnConstants.QNN_MAX_ERROR_PROFILE,

            /// <summary>
            /// Unused, present to ensure 32 bits.
            /// </summary>
            QNN_PROFILE_ERROR_UNDEFINED = 0x7FFFFFFF
        }
    }
    public enum ModelError_t
    {
        MODEL_NO_ERROR = 0,
        MODEL_TENSOR_ERROR = 1,
        MODEL_PARAMS_ERROR = 2,
        MODEL_NODES_ERROR = 3,
        MODEL_GRAPH_ERROR = 4,
        MODEL_CONTEXT_ERROR = 5,
        MODEL_GENERATION_ERROR = 6,
        MODEL_SETUP_ERROR = 7,
        MODEL_INVALID_ARGUMENT_ERROR = 8,
        MODEL_FILE_ERROR = 9,
        MODEL_MEMORY_ALLOCATE_ERROR = 10,
        MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
    }

    public enum QnnLog_Level_t
    {
        QNN_LOG_LEVEL_ERROR,
        QNN_LOG_LEVEL_WARN,
        QNN_LOG_LEVEL_INFO,
        QNN_LOG_LEVEL_DEBUG
    }


    [Flags]
    public enum Qnn_TensorVersion_t : uint
    {
        /// <summary>
        /// Enum to choose usage of Qnn_TensorV1_t in Qnn_Tensor_t
        /// </summary>
        QNN_TENSOR_VERSION_1 = 1,

        /// <summary>
        /// Enum to choose usage of Qnn_TensorV2_t in Qnn_Tensor_t
        /// </summary>
        QNN_TENSOR_VERSION_2 = 2,

        /// <summary>
        /// Unused, present to ensure 32 bits.
        /// </summary>
        QNN_TENSOR_VERSION_UNDEFINED = 0x7FFFFFFF
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct Qnn_Tensor_t
    {
        /// <summary>
        /// Version of the QNN tensor
        /// </summary>
        [FieldOffset(0)]
        public Qnn_TensorVersion_t version;

        [FieldOffset(8)]  // Offset includes padding
        public Qnn_TensorV2_t v2;

        [FieldOffset(8)]
        public Qnn_TensorV1_t v1;

        public void Dispose()
        {
            if (version == Qnn_TensorVersion_t.QNN_TENSOR_VERSION_1)
            {
            }
            else
            {
                v2.FreeDimensions();
                v2.FreeName();
                v2.FreeClientBuffer();
            }
        }
    }

    public enum Qnn_TensorType_t : uint
    {
        QNN_TENSOR_TYPE_APP_WRITE = 0,
        QNN_TENSOR_TYPE_APP_READ = 1,
        QNN_TENSOR_TYPE_APP_READWRITE = 2,
        QNN_TENSOR_TYPE_NATIVE = 3,
        QNN_TENSOR_TYPE_STATIC = 4,
        QNN_TENSOR_TYPE_NULL = 5,
        QNN_TENSOR_TYPE_UNDEFINED = 0x7FFFFFFF
    }

    public enum Qnn_DataType_t : uint
    {
        QNN_DATATYPE_INT_8 = 0x0008,
        QNN_DATATYPE_INT_16 = 0x0016,
        QNN_DATATYPE_INT_32 = 0x0032,
        QNN_DATATYPE_INT_64 = 0x0064,
        QNN_DATATYPE_UINT_8 = 0x0108,
        QNN_DATATYPE_UINT_16 = 0x0116,
        QNN_DATATYPE_UINT_32 = 0x0132,
        QNN_DATATYPE_UINT_64 = 0x0164,
        QNN_DATATYPE_FLOAT_16 = 0x0216,
        QNN_DATATYPE_FLOAT_32 = 0x0232,
        QNN_DATATYPE_FLOAT_64 = 0x0264,
        QNN_DATATYPE_SFIXED_POINT_4 = 0x0304,
        QNN_DATATYPE_SFIXED_POINT_8 = 0x0308,
        QNN_DATATYPE_SFIXED_POINT_16 = 0x0316,
        QNN_DATATYPE_SFIXED_POINT_32 = 0x0332,
        QNN_DATATYPE_UFIXED_POINT_4 = 0x0404,
        QNN_DATATYPE_UFIXED_POINT_8 = 0x0408,
        QNN_DATATYPE_UFIXED_POINT_16 = 0x0416,
        QNN_DATATYPE_UFIXED_POINT_32 = 0x0432,
        QNN_DATATYPE_BOOL_8 = 0x0508,
        QNN_DATATYPE_STRING = 0x0608,
        QNN_DATATYPE_UNDEFINED = 0x7FFFFFFF
    }
    public enum Qnn_Definition_t : uint
    {
        /// <summary>
        /// Indicates backend implementation to update or decide
        /// </summary>
        QNN_DEFINITION_IMPL_GENERATED = 0,

        /// <summary>
        /// Indicates that provided definition needs to be used
        /// </summary>
        QNN_DEFINITION_DEFINED = 1,

        /// <summary>
        /// Unused, present to ensure 32 bits.
        /// </summary>
        QNN_DEFINITION_UNDEFINED = 0x7FFFFFFF
    }
    public enum Qnn_QuantizationEncoding_t : uint
    {
        /// <summary>
        /// Indicates Qnn_ScaleOffset_t encoding type
        /// </summary>
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET = 0,

        /// <summary>
        /// Indicates Qnn_AxisScaleOffset_t encoding type
        /// </summary>
        QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET = 1,

        /// <summary>
        /// Indicates Qnn_BwScaleOffset_t encoding type
        /// </summary>
        QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET = 2,

        /// <summary>
        /// Indicates Qnn_BwAxisScaleOffset_t encoding type
        /// </summary>
        QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET = 3,

        /// <summary>
        /// Unused, present to ensure 32 bits.
        /// </summary>
        QNN_QUANTIZATION_ENCODING_UNDEFINED = 0x7FFFFFFF
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Qnn_BwAxisScaleOffset_t
    {
        /// <summary>
        /// Bitwidth must be <= number of bits specified by data type of tensor
        /// </summary>
        public uint Bitwidth;

        public int Axis;

        /// <summary>
        /// NumElements applies to both scales and offsets and they are supposed to be a one-to-one match
        /// </summary>
        public uint NumElements;

        /// <summary>
        /// Scales must be strictly positive
        /// </summary>
        public IntPtr Scales;

        /// <summary>
        /// Offsets must match scales in their dimension except when it can be IntPtr.Zero to indicate that the
        /// value is symmetrically quantized and hence, offset = 0
        /// </summary>
        public IntPtr Offsets;

        // Helper methods to work with the unmanaged pointers
        public float[] GetScales()
        {
            if (Scales == IntPtr.Zero) return Array.Empty<float>();
            float[] result = new float[NumElements];
            Marshal.Copy(Scales, result, 0, (int)NumElements);
            return result;
        }

        public void SetScales(float[] scales)
        {
            if (scales == null)
            {
                Scales = IntPtr.Zero;
                return;
            }
            Scales = UnmanagedMemoryTracker.AllocateMemory(scales.Length * sizeof(float));
            Marshal.Copy(scales, 0, Scales, scales.Length);
            NumElements = (uint)scales.Length;
        }

        public int[] GetOffsets()
        {
            if (Offsets == IntPtr.Zero) return Array.Empty<int>();
            int[] result = new int[NumElements];
            Marshal.Copy(Offsets, result, 0, (int)NumElements);
            return result;
        }

        public void SetOffsets(int[] offsets)
        {
            if (offsets == null)
            {
                Offsets = IntPtr.Zero;
                return;
            }
            Offsets = UnmanagedMemoryTracker.AllocateMemory(offsets.Length * sizeof(int));
            Marshal.Copy(offsets, 0, Offsets, offsets.Length);
            NumElements = (uint)offsets.Length;
        }

        // Don't forget to free the allocated memory when you're done with the struct
        public void Dispose()
        {
            if (Scales != IntPtr.Zero)
                UnmanagedMemoryTracker.FreeMemory(Scales);
            if (Offsets != IntPtr.Zero)
                UnmanagedMemoryTracker.FreeMemory(Offsets);
        }
    }

    public struct Qnn_ScaleOffset_t
    {
        /// <summary>
        /// Scale must be strictly positive
        /// </summary>
        public float scale;

        public int offset;
    }

    public struct Qnn_BwScaleOffset_t
    {
        /// <summary>
        /// bitwidth must be <= number of bits specified by data type of tensor
        /// </summary>
        public uint bitwidth;

        /// <summary>
        /// scale must be strictly positive
        /// </summary>
        public float scale;

        public int offset;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct Qnn_QuantizeParams_t
    {
        [FieldOffset(0)]
        public Qnn_Definition_t encodingDefinition;

        /// <summary>
        /// Quantization encoding type identifying quantization encoding structure to use
        /// </summary>
        [FieldOffset(sizeof(Qnn_Definition_t))]
        public Qnn_QuantizationEncoding_t quantizationEncoding;

        [FieldOffset(sizeof(Qnn_Definition_t) + sizeof(Qnn_QuantizationEncoding_t))]
        public Qnn_ScaleOffset_t scaleOffsetEncoding;

        [FieldOffset(sizeof(Qnn_Definition_t) + sizeof(Qnn_QuantizationEncoding_t))]
        public Qnn_AxisScaleOffset_t axisScaleOffsetEncoding;

        [FieldOffset(sizeof(Qnn_Definition_t) + sizeof(Qnn_QuantizationEncoding_t))]
        public Qnn_BwScaleOffset_t bwScaleOffsetEncoding;

        [FieldOffset(sizeof(Qnn_Definition_t) + sizeof(Qnn_QuantizationEncoding_t))]
        public Qnn_BwAxisScaleOffset_t bwAxisScaleOffsetEncoding;
    }
    public struct Qnn_AxisScaleOffset_t
    {
        public int axis;
        public uint numScaleOffsets;
        public IntPtr scaleOffset; // Use IntPtr for the pointer to Qnn_ScaleOffset_t
    }



    [StructLayout(LayoutKind.Sequential)]
    public struct Qnn_ClientBuffer_t
    {
        public IntPtr data;
        public uint dataSize;
    }

    public enum Qnn_TensorMemType_t : uint
    {
        QNN_TENSORMEMTYPE_RAW = 0,
        QNN_TENSORMEMTYPE_MEMHANDLE = 1,
        QNN_TENSORMEMTYPE_UNDEFINED = 0x7FFFFFFF
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Qnn_SparseParams_t
    {
        /// <summary>
        /// Specifies the sparse tensor layout
        /// </summary>
        public Qnn_SparseLayoutType_t type;

        /// <summary>
        /// Union of different sparse layout types
        /// </summary>
        public Qnn_SparseLayoutHybridCoo_t hybridCoo;
    }


    // You'll need to define these types:
    public enum Qnn_SparseLayoutType_t : uint
    {
        // Define the enum values here
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Qnn_SparseLayoutHybridCoo_t
    {
        /// <summary>
        /// Number of specified elements of a sparse tensor. Treated as the maximum when creating a tensor.
        /// </summary>
        public uint NumSpecifiedElements;

        /// <summary>
        /// Size of the index for a hybrid COO sparse tensor. The size of the index can range from 1 to
        /// the rank of the tensor. This feature allows for partially sparse tensors.
        /// </summary>
        public uint NumSparseDimensions;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct Qnn_TensorV1_t
    {
        [FieldOffset(0)]
        public uint id;

        [FieldOffset(8)]
        private IntPtr name;

        [FieldOffset(16)]
        public Qnn_TensorType_t type;

        [FieldOffset(20)]
        public Qnn_TensorDataFormat_t dataFormat;

        [FieldOffset(24)]
        public Qnn_DataType_t dataType;

        [FieldOffset(32)]
        public Qnn_QuantizeParams_t quantizeParams;

        [FieldOffset(72)]
        public uint Rank; // Renamed to lowercase 'rank' for consistency

        [FieldOffset(80)]
        private IntPtr dimensions;

        [FieldOffset(88)]
        public Qnn_TensorMemType_t memType;

        [FieldOffset(96)]
        public Qnn_ClientBuffer_t clientBuf;

        [FieldOffset(96)]
        public Qnn_MemHandle_t memHandle;
        //public IntPtr isDynamicDimensions;
        //public Qnn_SparseParams_t sparseParams;
        //public byte isProduced;

        public string Name
        {
            get
            {
                if (name == IntPtr.Zero)
                    return string.Empty;
                return Marshal.PtrToStringAnsi(name) ?? string.Empty;
            }
            set
            {
                // Free the existing name if it exists
                if (name != IntPtr.Zero)
                {
                    UnmanagedMemoryTracker.FreeMemory(name);
                    name = IntPtr.Zero;
                }

                // If the new value is not null or empty, allocate memory and copy the string
                if (!string.IsNullOrEmpty(value))
                {
                    name = Marshal.StringToHGlobalAnsi(value);
                }
            }
        }
        public void FreeName()
        {
            if (name != IntPtr.Zero)
            {
                UnmanagedMemoryTracker.FreeMemory(name);
                name = IntPtr.Zero;
            }
        }
        public uint[] Dimensions
        {
            get
            {
                if (dimensions == IntPtr.Zero || Rank == 0)
                    return Array.Empty<uint>();
                uint[] dims = new uint[Rank];
                for (int i = 0; i < Rank; i++)
                {
                    dims[i] = (uint)Marshal.ReadInt32(dimensions, i * sizeof(uint));
                }
                return dims;
            }
            set
            {
                if (value == null)
                {
                    dimensions = IntPtr.Zero;
                    Rank = 0;
                    return;
                }

                Rank = (uint)value.Length;
                if (dimensions != IntPtr.Zero)
                {
                    UnmanagedMemoryTracker.FreeMemory(dimensions);
                }
                dimensions = UnmanagedMemoryTracker.AllocateMemory(value.Length * sizeof(uint));
                for (int i = 0; i < value.Length; i++)
                {
                    Marshal.WriteInt32(dimensions, i * sizeof(uint), (int)value[i]);
                }
            }
        }

        public void FreeDimensions()
        {
            if (dimensions != IntPtr.Zero)
            {
                UnmanagedMemoryTracker.FreeMemory(dimensions);
                dimensions = IntPtr.Zero;
                Rank = 0;
            }
        }
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct Qnn_TensorV2_t
    {
        public Qnn_TensorV2_t() { }

        [FieldOffset(0)]
        public uint id;

        [FieldOffset(8)]
        private IntPtr name;

        [FieldOffset(16)]
        public Qnn_TensorType_t type;

        [FieldOffset(20)]
        public Qnn_TensorDataFormat_t dataFormat;

        [FieldOffset(24)]
        public Qnn_DataType_t dataType;

        [FieldOffset(32)]
        public Qnn_QuantizeParams_t quantizeParams;

        [FieldOffset(72)]
        public uint Rank; // Renamed to lowercase 'rank' for consistency

        [FieldOffset(80)]
        private IntPtr dimensions;

        [FieldOffset(88)]
        public Qnn_TensorMemType_t memType;

        [FieldOffset(96)]
        public Qnn_ClientBuffer_t clientBuf;

        [FieldOffset(96)] // Same offset as clientBuf for union behavior
        public Qnn_MemHandle_t memHandle;

        [FieldOffset(112)]
        public IntPtr isDynamicDimensions;

        [FieldOffset(120)]
        public Qnn_SparseParams_t sparseParams;

        [FieldOffset(132)]
        public byte isProduced;


        public string Name
        {
            get
            {
                if (name == IntPtr.Zero)
                    return string.Empty;
                return Marshal.PtrToStringAnsi(name) ?? string.Empty;
            }
            set
            {
                // Free the existing name if it exists
                if (name != IntPtr.Zero)
                {
                    UnmanagedMemoryTracker.FreeMemory(name);
                    name = IntPtr.Zero;
                }

                // If the new value is not null or empty, allocate memory and copy the string
                if (!string.IsNullOrEmpty(value))
                {
                    name = Marshal.StringToHGlobalAnsi(value);
                }
            }
        }
        public void FreeName()
        {
            if (name != IntPtr.Zero)
            {
                UnmanagedMemoryTracker.FreeMemory(name);
                name = IntPtr.Zero;
            }
        }
        public uint[] Dimensions
        {
            get
            {
                if (dimensions == IntPtr.Zero || Rank == 0)
                    return Array.Empty<uint>();
                uint[] dims = new uint[Rank];
                for (int i = 0; i < Rank; i++)
                {
                    dims[i] = (uint)Marshal.ReadInt32(dimensions, i * sizeof(uint));
                }
                return dims;
            }
            set
            {
                if (value == null)
                {
                    dimensions = IntPtr.Zero;
                    Rank = 0;
                    return;
                }

                Rank = (uint)value.Length;
                if (dimensions != IntPtr.Zero)
                {
                    UnmanagedMemoryTracker.FreeMemory(dimensions);
                }
                dimensions = UnmanagedMemoryTracker.AllocateMemory(value.Length * sizeof(uint));
                for (int i = 0; i < value.Length; i++)
                {
                    Marshal.WriteInt32(dimensions, i * sizeof(uint), (int)value[i]);
                }
            }
        }

        public void FreeClientBuffer()
        {
            if (clientBuf.data != IntPtr.Zero)
            {
                UnmanagedMemoryTracker.FreeMemory(clientBuf.data);
                clientBuf.data = IntPtr.Zero;
            }
        }

        public void FreeDimensions()
        {
            if (dimensions != IntPtr.Zero)
            {
                UnmanagedMemoryTracker.FreeMemory(dimensions);
                dimensions = IntPtr.Zero;
                Rank = 0;
            }
        }
    }
}


