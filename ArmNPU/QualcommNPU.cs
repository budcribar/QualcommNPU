using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.Runtime.InteropServices;

namespace ArmNPU
{
    
    public static class SnpeWrapper
    {
        private const string SnpeDll = "SNPE";

        [DllImport(SnpeDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SNPEFactory_IsRuntimeAvailable(int runtime);

        [DllImport(SnpeDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SNPEFactory_CreateFromOnnx(string modelPath);

        [DllImport(SnpeDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SNPEFactory_GetBuildId();

        [DllImport(SnpeDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Delete_SNPE(IntPtr snpeHandle);

        // Add more function declarations as needed for input/output tensor creation and model execution
    }
    public static class QualcommNPU
    {
        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnInterface_getProviders(out IntPtr providerList, out uint numProviders);


    }
    public static class QualcommNPUng
    {
        // Importing methods from Qualcomm AI Engine Direct libraries
        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnBackend_Create(out IntPtr backend);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnBackend_Initialize(IntPtr backend);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnContext_Create(IntPtr backend, out IntPtr context);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnModel_Load(IntPtr context, string modelPath, out IntPtr model);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnModel_Run(IntPtr model);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnContext_Destroy(IntPtr context);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnBackend_Destroy(IntPtr backend);
    }
    public static class QualcommNPUbad
    {
        // Importing methods from Qualcomm AI Engine Direct libraries
        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr QnnBackend_Create();

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnBackend_Initialize(IntPtr backend);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnContext_Create(IntPtr backend, out IntPtr context);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnModel_Load(IntPtr context, string modelPath, out IntPtr model);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnModel_Run(IntPtr model);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnContext_Destroy(IntPtr context);

        [DllImport("QnnHtp.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int QnnBackend_Destroy(IntPtr backend);
    }
}
