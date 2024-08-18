using System.Runtime.InteropServices;
using static SampleCSharpApplication.QnnDelegates;

namespace SampleCSharpApplication
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void QnnLog_Callback_t(int level, IntPtr msg);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate Qnn_ErrorHandle_t QnnLog_CreateFn_t(IntPtr logCallback, int logLevel, ref Qnn_LogHandle_t logHandle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate ulong QnnGraphExecuteDelegate(IntPtr graphHandle,[In] Qnn_Tensor_t[] inputs, uint numInputs, [In, Out] Qnn_Tensor_t[] outputs, uint numOutputs, IntPtr profileHandle, IntPtr signalHandle );

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    unsafe public delegate Qnn_ErrorHandle_t QnnBackend_CreateFn_t(Qnn_LogHandle_t logger, IntPtr* config, ref Qnn_BackendHandle_t backend);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate Qnn_ErrorHandle_t QnnProperty_HasCapabilityFn_t(QnnProperty_Key_t key);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate Qnn_ErrorHandle_t QnnDevice_CreateFn_t(Qnn_LogHandle_t logger, IntPtr config, ref Qnn_DeviceHandle_t device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate Qnn_ErrorHandle_t QnnContext_CreateFn_t(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device, IntPtr config, ref Qnn_ContextHandle_t context);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate Qnn_ErrorHandle_t QnnDevice_FreeFn_t(Qnn_DeviceHandle_t device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    unsafe public delegate ModelError_t ComposeGraphsFnHandleType_t(Qnn_BackendHandle_t backendHandle,IntPtr qnnInterface,Qnn_ContextHandle_t context, [In] IntPtr[]? graphConfigInfos,uint graphConfigInfosCount, out GraphInfo_t** graphInfos, out uint graphInfosCount, [MarshalAs(UnmanagedType.I1)] bool debug,QnnLog_Callback_t? logCallback, QnnLog_Level_t logLevel);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate Qnn_ErrorHandle_t QnnGraph_FinalizeFn_t(Qnn_GraphHandle_t graphHandle,Qnn_ProfileHandle_t profileHandle,Qnn_SignalHandle_t signalHandle);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate QnnProfile_Error_t QnnProfile_FreeFn_t(Qnn_ProfileHandle_t profile);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate QnnBackend_Error_t QnnBackend_FreeFn_t(Qnn_BackendHandle_t backend);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate QnnProfile_Error_t QnnLog_FreeFn_t(Qnn_LogHandle_t logger);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate QnnContextError QnnContext_FreeFn_t(Qnn_ContextHandle_t context,Qnn_ProfileHandle_t profile);
}


