using System;
using System.Runtime.InteropServices;

namespace SampleCSharpApplication
{
    public static class QnnDelegates
    {
        // Type definitions
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnLog_CreateFn_t(IntPtr logCallback, int logLevel, ref Qnn_LogHandle_t logHandle);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void QnnLog_CallbackFn_t(int level, IntPtr msg);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        unsafe public delegate Qnn_ErrorHandle_t QnnBackend_CreateFn_t(Qnn_LogHandle_t logger, IntPtr* config, ref Qnn_BackendHandle_t backend);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate Qnn_ErrorHandle_t QnnProperty_HasCapabilityFn_t(QnnProperty_Key_t key);

    }
}

