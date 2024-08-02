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
        //unsafe public IntPtr QNN_INTERFACE_VER_NAME;  // We'll treat the union as an IntPtr
        public IntPtr PropertyHasCapability;
        public IntPtr BackendCreate;
        public IntPtr BackendSetConfig;
        public IntPtr BackendGetApiVersion;
        public IntPtr BackendGetBuildId;
        public IntPtr BackendRegisterOpPackage;
        public IntPtr BackendGetSupportedOperations;
        public IntPtr BackendValidateOpConfig;
        public IntPtr BackendFree;
        public IntPtr ContextCreate;
        public IntPtr ContextSetConfig;
        public IntPtr ContextGetBinarySize;
        public IntPtr ContextGetBinary;
        public IntPtr ContextCreateFromBinary;
        public IntPtr ContextFree;
        public IntPtr GraphCreate;
        public IntPtr GraphCreateSubgraph;
        public IntPtr GraphSetConfig;
        public IntPtr GraphAddNode;
        public IntPtr GraphFinalize;
        public IntPtr GraphRetrieve;
        public IntPtr GraphExecute;
        public IntPtr GraphExecuteAsync;
        public IntPtr TensorCreateContextTensor;
        public IntPtr TensorCreateGraphTensor;
        public IntPtr LogCreate;
        public IntPtr LogSetLogLevel;
        public IntPtr LogFree;
        public IntPtr ProfileCreate;
        public IntPtr ProfileSetConfig;
        public IntPtr ProfileGetEvents;
        public IntPtr ProfileGetSubEvents;
        public IntPtr ProfileGetEventData;
        public IntPtr ProfileGetExtendedEventData;
        public IntPtr ProfileFree;
        public IntPtr MemRegister;
        public IntPtr MemDeRegister;
        public IntPtr DeviceGetPlatformInfo;
        public IntPtr DeviceFreePlatformInfo;
        public IntPtr DeviceGetInfrastructure;
        public IntPtr DeviceCreate;
        public IntPtr DeviceSetConfig;
        public IntPtr DeviceGetInfo;
        public IntPtr DeviceFree;
        public IntPtr SignalCreate;
        public IntPtr SignalSetConfig;
        public IntPtr SignalTrigger;
        public IntPtr SignalFree;
        public IntPtr ErrorGetMessage;
        public IntPtr ErrorGetVerboseMessage;
        public IntPtr ErrorFreeVerboseMessage;
        public IntPtr GraphPrepareExecutionEnvironment;
        public IntPtr GraphReleaseExecutionEnvironment;
        public IntPtr GraphGetProperty;
        public IntPtr ContextValidateBinary;
        public IntPtr ContextCreateFromBinaryWithSignal;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct ApiVersion
    {
        public CoreApiVersion CoreApiVersion;
        public CoreApiVersion BackendApiVersion;
    }

}
