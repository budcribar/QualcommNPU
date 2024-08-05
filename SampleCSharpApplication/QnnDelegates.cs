using System;
using System.Runtime.InteropServices;
using static SampleCSharpApplication.QnnDelegates;

namespace SampleCSharpApplication
{
    public static class QnnDelegates
    {
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
        [StructLayout(LayoutKind.Sequential)]
        public struct GraphConfigInfo_t
        {
            public IntPtr GraphName;  // char* in C++
            public IntPtr GraphConfigs;  // const QnnGraph_Config_t** in C++
        }

        public enum QnnLog_Level_t
        {
            QNN_LOG_LEVEL_ERROR,
            QNN_LOG_LEVEL_WARN,
            QNN_LOG_LEVEL_INFO,
            QNN_LOG_LEVEL_DEBUG
        }

        public struct GraphInfo_t
        {
            public Qnn_GraphHandle_t graph;                
            public IntPtr graphName;             // char* in C++ is IntPtr in C#
            unsafe public Qnn_Tensor_t* inputTensors;          
            public uint numInputTensors;
            unsafe public Qnn_Tensor_t* outputTensors;         
            public uint numOutputTensors;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct CoreApiVersion
        {
            public uint Major;
            public uint Minor;
            public uint Patch;
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

        [StructLayout(LayoutKind.Sequential)]
        public struct Qnn_Tensor_t
        {
            public Qnn_TensorVersion_t version;
            public Qnn_TensorV2_t v2;
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
        public struct Qnn_QuantizeParams_t
        {
            public Qnn_Definition_t encodingDefinition;
            public Qnn_QuantizationEncoding_t quantizationEncoding;
            public IntPtr encodingUnion; // This will need to be handled carefully in managed code
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
            public SparseLayoutUnion layoutUnion;
        }

        [StructLayout(LayoutKind.Explicit)]
        public struct SparseLayoutUnion
        {
            /// <summary>
            /// Hybrid coordinate list layout. Used when type is QNN_SPARSE_LAYOUT_HYBRID_COO.
            /// </summary>
            [FieldOffset(0)]
            public Qnn_SparseLayoutHybridCoo_t hybridCoo;
        }

        // You'll need to define these types:
        public enum Qnn_SparseLayoutType_t : uint
        {
            // Define the enum values here
        }

        public struct Qnn_SparseLayoutHybridCoo_t
        {
            // Define the struct fields here
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Qnn_TensorV2_t
        {
            public uint id;
            public IntPtr name;
            public Qnn_TensorType_t type;
            public Qnn_TensorDataFormat_t dataFormat;
            public Qnn_DataType_t dataType;
            public Qnn_QuantizeParams_t quantizeParams;
            public uint rank;
            public IntPtr[] dimensions;
            public Qnn_TensorMemType_t memType;

            public Qnn_ClientBuffer_t clientBuf;
            public Qnn_MemHandle_t memHandle;
            public IntPtr memoryUnion; // This will need to be handled carefully in managed code
            public IntPtr isDynamicDimensions;
            public Qnn_SparseParams_t sparseParams;
            public byte isProduced;

            public string NameString
            {
                get
                {
                    if (name == IntPtr.Zero)
                        return null;
                    return Marshal.PtrToStringAnsi(name);
                }
            }
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void QnnLog_Callback_t(int level, IntPtr msg);

        // Type definitions
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnLog_CreateFn_t(IntPtr logCallback, int logLevel, ref Qnn_LogHandle_t logHandle);

        //[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        //public delegate void QnnLog_CallbackFn_t(int level, IntPtr msg);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        unsafe public delegate Qnn_ErrorHandle_t QnnBackend_CreateFn_t(Qnn_LogHandle_t logger, IntPtr* config, ref Qnn_BackendHandle_t backend);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnProperty_HasCapabilityFn_t(QnnProperty_Key_t key);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnDevice_CreateFn_t(Qnn_LogHandle_t logger,IntPtr config, ref Qnn_DeviceHandle_t device);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnContext_CreateFn_t(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,IntPtr config, ref Qnn_ContextHandle_t context);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        unsafe public delegate ModelError_t ComposeGraphsFnHandleType_t(
            Qnn_BackendHandle_t backendHandle,
            IntPtr qnnInterface,
            Qnn_ContextHandle_t context,
            [In] IntPtr[] graphConfigInfos,  // const qnn_wrapper_api::GraphConfigInfo_t**
            uint graphConfigInfosCount,
            out GraphInfo_t** graphInfos,  // qnn_wrapper_api::GraphInfo_t***
            out uint graphInfosCount,
            [MarshalAs(UnmanagedType.I1)] bool debug,
            QnnLog_Callback_t logCallback,
            QnnLog_Level_t logLevel);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnGraph_FinalizeFn_t(
            Qnn_GraphHandle_t graphHandle,
            Qnn_ProfileHandle_t profileHandle,
            Qnn_SignalHandle_t signalHandle);
    }
}

