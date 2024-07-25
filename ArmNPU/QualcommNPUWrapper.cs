using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArmNPU
{
    using System;
    using System.Runtime.InteropServices;
   
    public static class QualcommNPUWrapper
    {
        private const string DllName = "QualcommNPUSDK.dll";

        [DllImport(DllName)]
        private static extern int Initialize();

        [DllImport(DllName)]
        private static extern int LoadModel(string path);

        [DllImport(DllName)]
        private static extern int RunInference(float[] input, int inputLength, float[] output, int outputLength);

        public static void InitializeNPU()
        {
            int result = Initialize();
            if (result != 0)
                throw new Exception($"Failed to initialize NPU. Error code: {result}");
        }

        public static void LoadNPUModel(string modelPath)
        {
            int result = LoadModel(modelPath);
            if (result != 0)
                throw new Exception($"Failed to load model. Error code: {result}");
        }

        public static float[] RunNPUInference(float[] input)
        {
            float[] output = new float[1024];
            int result = RunInference(input, input.Length, output, output.Length);
            if (result != 0)
                throw new Exception($"Inference failed. Error code: {result}");
            return output;
        }
    }
}
