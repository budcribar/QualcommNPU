using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
   
    public class QnnInitializer
    {
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FreeLibrary(IntPtr hModule);

        private IntPtr sg_backendHandle;
        private IntPtr sg_modelHandle;

        public QnnFunctionPointers Initialize(string backEndPath, string modelPath, bool loadFromCachedBinary) 
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

            return qnnFunctionPointers;
        }

        private void ExitWithMessage(string message, int exitCode)
        {
            Console.WriteLine(message);
            throw new Exception(message);
        }

        public StatusCode InitializeBackend(QnnInterface_t qnnInterface)
        {
            Console.WriteLine("Entering InitializeBackend method");
            Console.WriteLine($"QnnInterface pointer: {qnnInterface}");

            if (qnnInterface.BackendCreate == IntPtr.Zero)
            {
                Console.Error.WriteLine("QnnInterface pointer is null");
                return StatusCode.FAILURE;
            }

            IntPtr backendCreatePtr = qnnInterface.BackendCreate;
            if (backendCreatePtr == IntPtr.Zero)
            {
                Console.Error.WriteLine("Failed to get function pointer for QnnBackend_Create");
                return StatusCode.FAILURE;
            }

            QnnBackend_CreateFn_t backendCreate = Marshal.GetDelegateForFunctionPointer<QnnBackend_CreateFn_t>(backendCreatePtr);

            try
            {
                //Qnn_ErrorHandle_t qnnStatus = backendCreate(m_logHandle, IntPtr.Zero, ref m_backendHandle);

                //if (qnnStatus != QNN_SUCCESS)
                //{
                //    Console.Error.WriteLine($"Could not initialize backend due to error = {qnnStatus}");
                //    return StatusCode.FAILURE;
                //}

                //Console.WriteLine($"Initialize Backend Returned Status = {qnnStatus}");
                //m_isBackendInitialized = true;
                return StatusCode.SUCCESS;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Exception occurred while calling backendCreate: {ex.Message}");
                return StatusCode.FAILURE;
            }
        }


        public void Unload()
        {
            if (sg_backendHandle != IntPtr.Zero)
            {
                FreeLibrary(sg_backendHandle);
                sg_backendHandle = IntPtr.Zero;
            }

            if (sg_modelHandle != IntPtr.Zero)
            {
                FreeLibrary(sg_modelHandle);
                sg_modelHandle = IntPtr.Zero;
            }
        }
    }
}
