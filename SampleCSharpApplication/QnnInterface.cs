﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Runtime.InteropServices;
namespace SampleCSharpApplication
{
    [StructLayout(LayoutKind.Sequential)]
    public struct QnnInterface2
    {
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

    [StructLayout(LayoutKind.Explicit, Size = 448)]
    public struct QnnInterface
    {
        [FieldOffset(0)]
        public IntPtr PropertyHasCapability;

        [FieldOffset(8)]
        public IntPtr BackendCreate;

        [FieldOffset(16)]
        public IntPtr BackendSetConfig;

        [FieldOffset(24)]
        public IntPtr BackendGetApiVersion;

        [FieldOffset(32)]
        public IntPtr BackendGetBuildId;

        [FieldOffset(40)]
        public IntPtr BackendRegisterOpPackage;

        [FieldOffset(48)]
        public IntPtr BackendGetSupportedOperations;

        [FieldOffset(56)]
        public IntPtr BackendValidateOpConfig;

        [FieldOffset(64)]
        public IntPtr BackendFree;

        [FieldOffset(72)]
        public IntPtr ContextCreate;

        [FieldOffset(80)]
        public IntPtr ContextSetConfig;

        [FieldOffset(88)]
        public IntPtr ContextGetBinarySize;

        [FieldOffset(96)]
        public IntPtr ContextGetBinary;

        [FieldOffset(104)]
        public IntPtr ContextCreateFromBinary;

        [FieldOffset(112)]
        public IntPtr ContextFree;

        [FieldOffset(120)]
        public IntPtr GraphCreate;

        [FieldOffset(128)]
        public IntPtr GraphCreateSubgraph;

        [FieldOffset(136)]
        public IntPtr GraphSetConfig;

        [FieldOffset(144)]
        public IntPtr GraphAddNode;

        [FieldOffset(152)]
        public IntPtr GraphFinalize;

        [FieldOffset(160)]
        public IntPtr GraphRetrieve;

        [FieldOffset(168)]
        public IntPtr GraphExecute;

        [FieldOffset(176)]
        public IntPtr GraphExecuteAsync;

        [FieldOffset(184)]
        public IntPtr TensorCreateContextTensor;

        [FieldOffset(192)]
        public IntPtr TensorCreateGraphTensor;

        [FieldOffset(200)]
        public IntPtr LogCreate;

        [FieldOffset(208)]
        public IntPtr LogSetLogLevel;

        [FieldOffset(216)]
        public IntPtr LogFree;

        [FieldOffset(224)]
        public IntPtr ProfileCreate;

        [FieldOffset(232)]
        public IntPtr ProfileSetConfig;

        [FieldOffset(240)]
        public IntPtr ProfileGetEvents;

        [FieldOffset(248)]
        public IntPtr ProfileGetSubEvents;

        [FieldOffset(256)]
        public IntPtr ProfileGetEventData;

        [FieldOffset(264)]
        public IntPtr ProfileGetExtendedEventData;

        [FieldOffset(272)]
        public IntPtr ProfileFree;

        [FieldOffset(280)]
        public IntPtr MemRegister;

        [FieldOffset(288)]
        public IntPtr MemDeRegister;

        [FieldOffset(296)]
        public IntPtr DeviceGetPlatformInfo;

        [FieldOffset(304)]
        public IntPtr DeviceFreePlatformInfo;

        [FieldOffset(312)]
        public IntPtr DeviceGetInfrastructure;

        [FieldOffset(320)]
        public IntPtr DeviceCreate;

        [FieldOffset(328)]
        public IntPtr DeviceSetConfig;

        [FieldOffset(336)]
        public IntPtr DeviceGetInfo;

        [FieldOffset(344)]
        public IntPtr DeviceFree;

        [FieldOffset(352)]
        public IntPtr SignalCreate;

        [FieldOffset(360)]
        public IntPtr SignalSetConfig;

        [FieldOffset(368)]
        public IntPtr SignalTrigger;

        [FieldOffset(376)]
        public IntPtr SignalFree;

        [FieldOffset(384)]
        public IntPtr ErrorGetMessage;

        [FieldOffset(392)]
        public IntPtr ErrorGetVerboseMessage;

        [FieldOffset(400)]
        public IntPtr ErrorFreeVerboseMessage;

        [FieldOffset(408)]
        public IntPtr GraphPrepareExecutionEnvironment;

        [FieldOffset(416)]
        public IntPtr GraphReleaseExecutionEnvironment;

        [FieldOffset(424)]
        public IntPtr GraphGetProperty;

        [FieldOffset(432)]
        public IntPtr ContextValidateBinary;

        [FieldOffset(440)]
        public IntPtr ContextCreateFromBinaryWithSignal;
    }
}