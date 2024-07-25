using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
    
    public static class DynamicLoadUtil
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Ansi)]
        private static extern IntPtr LoadLibraryExA(string lpFileName, IntPtr hFile, uint dwFlags);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FreeLibrary(IntPtr hModule);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();

        [DllImport("psapi.dll", SetLastError = true)]
        private static extern bool EnumProcessModules(IntPtr hProcess, IntPtr[] lphModule, uint cb, out uint lpcbNeeded);

        // Constants
        private const uint LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;
        private const int DL_NOW = 2;
        private const int DL_LOCAL = 0;
        private const int DL_GLOBAL = 1;

        // Set to store locally loaded modules
        private static HashSet<IntPtr> modHandles = new HashSet<IntPtr>();

        // Delegate types
        private delegate int QnnInterfaceGetProvidersFn(out IntPtr providerList, out uint numProviders);

        private static IntPtr DlOpen(string filename, int flags)
        {
            if (string.IsNullOrEmpty(filename))
            {
                SetLastErrorMessage("filename is null or empty");
                return IntPtr.Zero;
            }

            if ((flags & DL_NOW) == 0)
            {
                SetLastErrorMessage("flags must include DL_NOW");
                return IntPtr.Zero;
            }

            IntPtr curProc = GetCurrentProcess();
            uint asIs;
            if (!EnumProcessModules(curProc, null, 0, out asIs))
            {
                SetLastErrorMessage("enumerate modules failed before loading module");
                return IntPtr.Zero;
            }

            IntPtr mod = LoadLibraryExA(filename, IntPtr.Zero, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (mod == IntPtr.Zero)
            {
                SetLastErrorMessage("load library failed");
                return IntPtr.Zero;
            }

            uint toBe;
            if (!EnumProcessModules(curProc, null, 0, out toBe))
            {
                SetLastErrorMessage("enumerate modules failed after loading module");
                FreeLibrary(mod);
                return IntPtr.Zero;
            }

            bool loadedBefore = asIs == toBe;

            if (!loadedBefore && (flags & DL_LOCAL) != 0)
            {
                modHandles.Add(mod);
            }

            if ((flags & DL_GLOBAL) != 0)
            {
                modHandles.Remove(mod);
            }

            return mod;
        }

        private static void SetLastErrorMessage(string message)
        {
            // You might want to implement a more sophisticated error handling mechanism
            Console.Error.WriteLine(message);
        }
        public struct ApiVersion
        {
            public CoreApiVersion CoreApiVersion;
            // Add other necessary fields
        }

        public struct CoreApiVersion
        {
            public int Major;
            public int Minor;
        }

        public struct QnnInterface
        {
            public ApiVersion ApiVersion;
            public IntPtr QNN_INTERFACE_VER_NAME;
            // Add other necessary fields
        }

      
        public static StatusCode GetQnnFunctionPointers(
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
            IntPtr libBackendHandle = DlOpen(backendPath, DL_NOW | DL_LOCAL);
            if (libBackendHandle == IntPtr.Zero)
            {
                Console.Error.WriteLine($"Unable to load backend. Error: {Marshal.GetLastWin32Error()}");
                return StatusCode.FAIL_LOAD_BACKEND;
            }
            backendHandle = libBackendHandle;

            // Get QNN Interface
            IntPtr getInterfaceProvidersPtr = GetProcAddress(libBackendHandle, "QnnInterface_getProviders");
            if (getInterfaceProvidersPtr == IntPtr.Zero)
            {
                return StatusCode.FAIL_SYM_FUNCTION;
            }

            var getInterfaceProviders = Marshal.GetDelegateForFunctionPointer<QnnInterfaceGetProvidersFn>(getInterfaceProvidersPtr);

            IntPtr interfaceProviders;
            uint numProviders;
            if (getInterfaceProviders(out interfaceProviders, out numProviders) != 0)
            {
                Console.Error.WriteLine("Failed to get interface providers.");
                return StatusCode.FAIL_GET_INTERFACE_PROVIDERS;
            }

            if (interfaceProviders == IntPtr.Zero)
            {
                Console.Error.WriteLine("Failed to get interface providers: null interface providers received.");
                return StatusCode.FAIL_GET_INTERFACE_PROVIDERS;
            }

            if (numProviders == 0)
            {
                Console.Error.WriteLine("Failed to get interface providers: 0 interface providers.");
                return StatusCode.FAIL_GET_INTERFACE_PROVIDERS;
            }

            bool foundValidInterface = false;
            for (int pIdx = 0; pIdx < numProviders; pIdx++)
            {
                // You'll need to define the QnnInterface structure and marshal it properly
                // This is a simplified version
                var interfaceProvider = Marshal.PtrToStructure<QnnInterface>(interfaceProviders + pIdx * Marshal.SizeOf<QnnInterface>());

                // TODO
                //if (QNN_API_VERSION_MAJOR == interfaceProvider.ApiVersion.CoreApiVersion.Major &&
                //    QNN_API_VERSION_MINOR <= interfaceProvider.ApiVersion.CoreApiVersion.Minor)
                //{
                //    foundValidInterface = true;
                //    qnnFunctionPointers.QnnInterface = interfaceProvider.QNN_INTERFACE_VER_NAME;
                //    break;
                //}
            }

            if (!foundValidInterface)
            {
                Console.Error.WriteLine("Unable to find a valid interface.");
                backendHandle = IntPtr.Zero;
                return StatusCode.FAIL_GET_INTERFACE_PROVIDERS;
            }

            if (loadModelLib)
            {
                Console.WriteLine("Loading model shared library ([model].dll)");
                IntPtr libModelHandle = DlOpen(modelPath, DL_NOW | DL_LOCAL);
                if (libModelHandle == IntPtr.Zero)
                {
                    Console.Error.WriteLine($"Unable to load model. Error: {Marshal.GetLastWin32Error()}");
                    return StatusCode.FAIL_LOAD_MODEL;
                }
                modelHandle = libModelHandle;

                string modelPrepareFunc = "QnnModel_composeGraphs";
                qnnFunctionPointers.ComposeGraphsFnHandle = GetProcAddress(libModelHandle, modelPrepareFunc);
                if (qnnFunctionPointers.ComposeGraphsFnHandle == IntPtr.Zero)
                {
                    return StatusCode.FAIL_SYM_FUNCTION;
                }

                string modelFreeFunc = "QnnModel_freeGraphsInfo";
                qnnFunctionPointers.FreeGraphInfoFnHandle = GetProcAddress(libModelHandle, modelFreeFunc);
                if (qnnFunctionPointers.FreeGraphInfoFnHandle == IntPtr.Zero)
                {
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


