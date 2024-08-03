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


        // GraphInfo_t structure (you'll need to define this based on your C++ definition)
        public struct GraphInfo_t
        {
            // Define the fields of GraphInfo_t here
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
        public delegate ModelError_t ComposeGraphsFnHandleType_t(
            Qnn_BackendHandle_t backendHandle,
            IntPtr qnnInterface,
            Qnn_ContextHandle_t context,
            [In] IntPtr[] graphConfigInfos,  // const qnn_wrapper_api::GraphConfigInfo_t**
            uint graphConfigInfosCount,
            out IntPtr graphInfos,  // qnn_wrapper_api::GraphInfo_t***
            out uint graphInfosCount,
            [MarshalAs(UnmanagedType.I1)] bool debug,
            QnnLog_Callback_t logCallback,
            QnnLog_Level_t logLevel);
    }
}

