using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static SampleCSharpApplication.DynamicLoadUtil;

namespace SampleCSharpApplication
{
    // Constants
   

    [StructLayout(LayoutKind.Sequential)]
    public struct CoreApiVersion
    {
        public uint Major;
        public uint Minor;
        public uint Patch;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct QnnInterface_t
    {
        public uint BackendId;
        public uint pad;
        public IntPtr ProviderName;  // const char* in C++ becomes IntPtr in C# 
        public ApiVersion ApiVersion;
        unsafe public IntPtr QNN_INTERFACE_VER_NAME;  // We'll treat the union as an IntPtr
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct ApiVersion
    {
        public CoreApiVersion CoreApiVersion;
        public CoreApiVersion BackendApiVersion;
    }

}
