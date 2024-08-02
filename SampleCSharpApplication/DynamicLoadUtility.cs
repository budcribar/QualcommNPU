using System.Runtime.InteropServices;

namespace SampleCSharpApplication
{
    public static class DynamicLoadUtil
    {
        // Windows API imports
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FreeLibrary(IntPtr hModule);

        [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll")]
        private static extern uint GetLastError();

        private const int QNN_API_VERSION_MAJOR = 2;
        private const int QNN_API_VERSION_MINOR = 15;

        // Delegate types
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate Qnn_ErrorHandle_t QnnInterfaceGetProvidersFn_t(
            out IntPtr* providerList,
            ref uint numProviders);

        private static void SetLastErrorMessage(string message)
        {
            Console.Error.WriteLine($"{message}. Error code: {GetLastError()}");
        }

        public unsafe static StatusCode GetQnnFunctionPointers(
            string backendPath,
            string modelPath,
            ref QnnFunctionPointers qnnFunctionPointers,
            out IntPtr backendHandle,
            bool loadModelLib,
            out IntPtr modelHandle)
        {
            backendHandle = IntPtr.Zero;
            modelHandle = IntPtr.Zero;

            // Load backend library
            IntPtr libBackendHandle = LoadLibrary(backendPath);
            if (libBackendHandle == IntPtr.Zero)
            {
                SetLastErrorMessage($"Unable to load backend: {backendPath}");
                return StatusCode.FAIL_LOAD_BACKEND;
            }
            backendHandle = libBackendHandle;

            // Get QNN Interface
            IntPtr getInterfaceProvidersPtr = GetProcAddress(libBackendHandle, "QnnInterface_getProviders");
            if (getInterfaceProvidersPtr == IntPtr.Zero)
            {
                SetLastErrorMessage("Failed to get QnnInterface_getProviders function");
                FreeLibrary(libBackendHandle);
                return StatusCode.FAIL_SYM_FUNCTION;
            }

            var getProviders = Marshal.GetDelegateForFunctionPointer<QnnInterfaceGetProvidersFn_t>(getInterfaceProvidersPtr);

            IntPtr* providerListPtr;
            uint numProviders = 0;
            Qnn_ErrorHandle_t result = getProviders(out providerListPtr, ref numProviders);

            if (result != 0 || numProviders == 0 || providerListPtr == null)
            {
                Console.Error.WriteLine("Failed to get interface providers.");
                FreeLibrary(libBackendHandle);
                return StatusCode.FAIL_GET_INTERFACE_PROVIDERS;
            }

            bool foundValidInterface = false;

            for (int i = 0; i < numProviders; i++)
            {
                IntPtr providerPtr = providerListPtr[i];
                QnnInterface_t provider = Marshal.PtrToStructure<QnnInterface_t>(providerPtr);

                if (QNN_API_VERSION_MAJOR == provider.ApiVersion.CoreApiVersion.Major &&
                    QNN_API_VERSION_MINOR <= provider.ApiVersion.CoreApiVersion.Minor)
                {
                    foundValidInterface = true;//0x000001defa1347a8 is the start of the interface table
                    qnnFunctionPointers.QnnInterface = Marshal.PtrToStructure<QnnInterface_t>(providerPtr);
                    //qnnFunctionPointers.QnnInterface = (IntPtr**)Marshal.ReadIntPtr(providerPtr, 40);


                    break;
                }
            }

            if (!foundValidInterface)
            {
                Console.Error.WriteLine("Unable to find a valid interface.");
                FreeLibrary(libBackendHandle);
                return StatusCode.FAIL_GET_INTERFACE_PROVIDERS;
            }

            if (loadModelLib)
            {
                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"Error: Could not load model: {modelPath}");
                    FreeLibrary(libBackendHandle);
                    return StatusCode.FAIL_LOAD_MODEL;
                }

                Console.WriteLine("Loading model shared library ([model].dll)");
                IntPtr libModelHandle = LoadLibrary(modelPath);
                if (libModelHandle == IntPtr.Zero)
                {
                    SetLastErrorMessage($"Unable to load model: {modelPath}");
                    FreeLibrary(libBackendHandle);
                    return StatusCode.FAIL_LOAD_MODEL;
                }
                modelHandle = libModelHandle;

                string modelPrepareFunc = "QnnModel_composeGraphs";
                qnnFunctionPointers.ComposeGraphsFnHandle = GetProcAddress(libModelHandle, modelPrepareFunc);
                if (qnnFunctionPointers.ComposeGraphsFnHandle == IntPtr.Zero)
                {
                    SetLastErrorMessage($"Failed to get {modelPrepareFunc} function");
                    FreeLibrary(libBackendHandle);
                    FreeLibrary(libModelHandle);
                    return StatusCode.FAIL_SYM_FUNCTION;
                }

                string modelFreeFunc = "QnnModel_freeGraphsInfo";
                qnnFunctionPointers.FreeGraphInfoFnHandle = GetProcAddress(libModelHandle, modelFreeFunc);
                if (qnnFunctionPointers.FreeGraphInfoFnHandle == IntPtr.Zero)
                {
                    SetLastErrorMessage($"Failed to get {modelFreeFunc} function");
                    FreeLibrary(libBackendHandle);
                    FreeLibrary(libModelHandle);
                    return StatusCode.FAIL_SYM_FUNCTION;
                }
            }
            else
            {
                Console.WriteLine("Model wasn't loaded from a shared library.");
            }

            return StatusCode.SUCCESS;
        }
    }
}