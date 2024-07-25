using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
    public class QnnInitializer
    {
        private IntPtr sg_backendHandle;
        private IntPtr sg_modelHandle;

        public void Initialize(string backEndPath, string modelPath, bool loadFromCachedBinary)
        {
            QnnFunctionPointers qnnFunctionPointers = new QnnFunctionPointers();

            // Load backend and model .so and validate all the required function symbols are resolved
            StatusCode statusCode = DynamicLoadUtil.GetQnnFunctionPointers(
                backEndPath,
                modelPath,
                ref qnnFunctionPointers,
                out sg_backendHandle,
                !loadFromCachedBinary,
                out sg_modelHandle);

            if (statusCode != StatusCode.SUCCESS)
            {
                if (statusCode == StatusCode.FAIL_LOAD_BACKEND)
                {
                    ExitWithMessage($"Error initializing QNN Function Pointers: could not load backend: {backEndPath}", 1);
                }
                else if (statusCode == StatusCode.FAIL_LOAD_MODEL)
                {
                    ExitWithMessage($"Error initializing QNN Function Pointers: could not load model: {modelPath}", 1);
                }
                else
                {
                    ExitWithMessage("Error initializing QNN Function Pointers", 1);
                }
            }
        }

        private void ExitWithMessage(string message, int exitCode)
        {
            Console.WriteLine(message);
            Environment.Exit(exitCode);
        }
    }
}
