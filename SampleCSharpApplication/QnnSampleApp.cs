using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Qnn_ErrorHandle_t = System.Int32;
using System.Security.Cryptography;
using Qnn_LogHandle_t = System.IntPtr;
using Qnn_BackendHandle_t = System.IntPtr;




namespace SampleCSharpApplication
{
   

    

   

    [StructLayout(LayoutKind.Sequential)]
    public struct QnnApiVersion
    {
        public uint Major;
        public uint Minor;
        public uint Patch;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Qnn_ApiVersion_t
    {
        public uint Major;
        public uint Minor;
        public uint Patch;
    }

   

    public unsafe class QnnSampleApp
    {
        [DllImport("../../../QnnHtpArm64.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern Qnn_ErrorHandle_t QnnLog_Create(  IntPtr logCallback,  int logLevel,  ref Qnn_LogHandle_t logHandle);

        private QnnLog_CallbackFn_t m_logCallback; // Class member to keep the delegate alive
        private QnnFunctionPointers m_qnnFunctionPointers = null;
        private Qnn_LogHandle_t m_logHandle;
        private Qnn_BackendHandle_t m_backendHandle;
        private bool m_isBackendInitialized;
        private IntPtr* m_backendConfig;

        private string model;
        private string backend;
        private string inputList;
        private int duration;

        // Constants
        private const int QNN_API_VERSION_MAJOR = 2;
        private const int QNN_API_VERSION_MINOR = 15;
        private const int QNN_SUCCESS = 0;

        // Logging related delegates and enums
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void LogCallback(int level, string message);

        private enum LogLevel
        {
            ERROR = 1,
            WARN = 2,
            INFO = 3,
            VERBOSE = 4,
            DEBUG = 5
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate Qnn_ErrorHandle_t QnnLog_CreateFn_t(
        IntPtr logCallback,
        int logLevel,
        ref Qnn_LogHandle_t logHandle);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void QnnLog_CallbackFn_t(int level, IntPtr msg);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate Qnn_ErrorHandle_t QnnBackend_CreateFn_t(
            Qnn_LogHandle_t logger,
            IntPtr* config,
            ref Qnn_BackendHandle_t backend);

        public QnnSampleApp(string model, string backend, string inputList, int duration)
        {
            this.model = model;
            this.backend = backend;
            this.inputList = inputList;
            this.duration = duration;
        }

        public int Run()
        {
            Console.WriteLine($"Model: {model}");
            Console.WriteLine($"Backend: {backend}");

            if (!File.Exists(model))
            {
                Console.WriteLine($"Error: Could not load model: {model}");
                return 1;
            }

            if (!File.Exists(backend))
            {
                Console.WriteLine($"Error: Could not load backend: {backend}");
                return 1;
            }

            if (!File.Exists(inputList))
            {
                Console.WriteLine($"Error: Could not find input list: {inputList}");
                return 1;
            }

            Console.WriteLine("Initializing...");
            var initializer = new QnnInitializer();
            m_qnnFunctionPointers = initializer.Initialize(backend, model, false);

            InitializeLogging();

            Console.WriteLine("Initializing backend...");
            var status = InitializeBackend();
            if (status == StatusCode.FAILURE)
            {
                Console.WriteLine("Backend Initialization failure");
                return 1;
            }

            Console.WriteLine("Creating device...");
            if (!CreateDevice())
            {
                Console.WriteLine("Device Creation failure");
                return 1;
            }

            Console.WriteLine("Initializing profiling...");
            if (!InitializeProfiling())
            {
                Console.WriteLine("Profiling Initialization failure");
                return 1;
            }

            Console.WriteLine("Registering op packages...");
            if (!RegisterOpPackages())
            {
                Console.WriteLine("Register Op Packages failure");
                return 1;
            }

            Console.WriteLine("Creating context...");
            if (!CreateContext())
            {
                Console.WriteLine("Context Creation failure");
                return 1;
            }

            Console.WriteLine("Composing graphs...");
            if (!ComposeGraphs())
            {
                Console.WriteLine("Graph Prepare failure");
                return 1;
            }

            Console.WriteLine("Finalizing graphs...");
            if (!FinalizeGraphs())
            {
                Console.WriteLine("Graph Finalize failure");
                return 1;
            }

            Console.WriteLine("Executing graphs...");
            if (!ExecuteGraphs())
            {
                Console.WriteLine("Graph Execution failure");
                return 1;
            }

            Console.WriteLine("Freeing context...");
            if (!FreeContext())
            {
                Console.WriteLine("Context Free failure");
                return 1;
            }

            Console.WriteLine("Freeing device...");
            if (!FreeDevice())
            {
                Console.WriteLine("Device Free failure");
                return 1;
            }

            return 0;
        }

        private void InitializeLogging()
        {
            if (IsLogInitialized())
            {
                m_logCallback = LogMessage;
                var logLevel = GetLogLevel();
                Console.WriteLine($"Initializing logging in the backend. Callback: [{m_logCallback.Method.Name}], Log Level: [{logLevel}]");

                try
                {
                   
                    IntPtr logCallbackPtr = Marshal.GetFunctionPointerForDelegate(m_logCallback);

                    IntPtr logCreatePtr = GetFunctionPointerForDelegate("QnnLog_Create");
                    if (logCreatePtr == IntPtr.Zero)
                    {
                        Console.Error.WriteLine("Failed to get function pointer for QnnLog_Create");
                        return;
                    }

                    QnnLog_CreateFn_t backendCreate = Marshal.GetDelegateForFunctionPointer<QnnLog_CreateFn_t>(logCreatePtr);

                    Qnn_ErrorHandle_t result = backendCreate(new IntPtr(0), (int)logLevel, ref m_logHandle);
                   // Qnn_ErrorHandle_t result = backendCreate(logCallbackPtr, (int)logLevel, ref m_logHandle);
                    //Qnn_ErrorHandle_t result = QnnLog_Create(logCallbackPtr, (int)logLevel, ref m_logHandle);

                    if (result != QNN_SUCCESS)
                    {
                        Console.WriteLine($"Unable to initialize logging in the backend. Error code: {result}");
                    }
                    else
                    {
                        Console.WriteLine("Logging initialized successfully");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Exception occurred while initializing logging: {ex.Message}");
                }
            }
            else
            {
                Console.WriteLine("Logging not available in the backend.");
            }
        }

        //private void InitializeLogging()
        //{
        //    if (IsLogInitialized())
        //    {
        //        m_logCallback = LogMessage; // Store the delegate as a class member
        //        var logLevel = GetLogLevel();
        //        Console.WriteLine($"Initializing logging in the backend. Callback: [{m_logCallback.Method.Name}], Log Level: [{logLevel}]");

        //        try
        //        {
        //            IntPtr logCreatePtr = GetFunctionPointerForDelegate("QnnLog_Create");
        //            if (logCreatePtr == IntPtr.Zero)
        //            {
        //                Console.WriteLine("Unable to get function pointer for QnnLog_Create");
        //                return;
        //            }

        //            var logCreateFn = Marshal.GetDelegateForFunctionPointer<QnnLog_CreateFn_t>(logCreatePtr);
        //            IntPtr logCallbackPtr = Marshal.GetFunctionPointerForDelegate(m_logCallback);
        //            Qnn_ErrorHandle_t result = logCreateFn(logCallbackPtr, (int)logLevel, ref m_logHandle);

        //            if (result != QNN_SUCCESS)
        //            {
        //                Console.WriteLine($"Unable to initialize logging in the backend. Error code: {result}");
        //            }
        //            else
        //            {
        //                Console.WriteLine("Logging initialized successfully");
        //            }
        //        }
        //        catch (Exception ex)
        //        {
        //            Console.WriteLine($"Exception occurred while initializing logging: {ex.Message}");
        //        }
        //    }
        //    else
        //    {
        //        Console.WriteLine("Logging not available in the backend.");
        //    }
        //}

        private bool IsLogInitialized()
        {
            // Implement your logic to check if logging is initialized
            return true; // Placeholder
        }

       

        private LogLevel GetLogLevel()
        {
            // Implement your logic to get the current log level
            return LogLevel.INFO; // Placeholder
        }

        private void LogMessage(int level, IntPtr msgPtr)
        {
            string message = Marshal.PtrToStringAnsi(msgPtr);
            Console.WriteLine($"[{(LogLevel)level}] {message}");
        }

        public IntPtr GetOffsetQnnInterface(int offset)
        {
            if (m_qnnFunctionPointers.QnnInterface == IntPtr.Zero)
            {
                throw new Exception("QnnInterface pointer is null");
            }

            byte* ptr = (byte*)m_qnnFunctionPointers.QnnInterface.ToPointer();
            ptr += offset;
            return new IntPtr(ptr);
        }

        private IntPtr GetFunctionPointerForDelegate(string functionName)
        {
            if (m_qnnFunctionPointers.QnnInterface == IntPtr.Zero)
            {
                Console.WriteLine("QnnInterface pointer is null");
                return IntPtr.Zero;
            }

            //IntPtr* interfacePtr = (IntPtr*)m_qnnFunctionPointers.QnnInterface;

            //QnnInterface* interfacePtr = (QnnInterface*)m_qnnFunctionPointers.QnnInterface;
            //QnnInterface interfacex = m_qnnFunctionPointers.QnnInterface 

            switch (functionName)
            {
                case "QnnProperty_HasCapability":
                    return GetOffsetQnnInterface(0);
                case "QnnBackend_Create":
                    return GetOffsetQnnInterface(1*8);
                case "QnnBackend_SetConfig":
                    return GetOffsetQnnInterface(2 * 8);
                 
                case "QnnBackend_GetApiVersion":
                    return GetOffsetQnnInterface(3 * 8);
                   
                case "QnnBackend_GetBuildId":
                    return GetOffsetQnnInterface(4 * 8);
                   
                // Add more cases as needed, incrementing the index for each function pointer
                case "QnnLog_Create":
                    return GetOffsetQnnInterface(400);
                    //return interfacePtr[25]; // Adjust this index based on the actual position in the struct
                                           // ... other cases ...
                default:
                    Console.WriteLine($"Unknown function name: {functionName}");
                    return IntPtr.Zero;
            }
        }
        public unsafe StatusCode InitializeBackend()
        {
            Console.WriteLine("Entering InitializeBackend method");
            Console.WriteLine($"QnnInterface pointer: {m_qnnFunctionPointers.QnnInterface}");

            if (m_qnnFunctionPointers.QnnInterface == IntPtr.Zero)
            {
                Console.Error.WriteLine("QnnInterface pointer is null");
                return StatusCode.FAILURE;
            }

            IntPtr backendCreatePtr = GetFunctionPointerForDelegate("QnnBackend_Create");
            if (backendCreatePtr == IntPtr.Zero)
            {
                Console.Error.WriteLine("Failed to get function pointer for QnnBackend_Create");
                return StatusCode.FAILURE;
            }

            QnnBackend_CreateFn_t backendCreate = Marshal.GetDelegateForFunctionPointer<QnnBackend_CreateFn_t>(backendCreatePtr);

            try
            {
                Qnn_ErrorHandle_t qnnStatus = backendCreate(m_logHandle, m_backendConfig, ref m_backendHandle);

                if (qnnStatus != QNN_SUCCESS)
                {
                    Console.Error.WriteLine($"Could not initialize backend due to error = {qnnStatus}");
                    return StatusCode.FAILURE;
                }

                Console.WriteLine($"Initialize Backend Returned Status = {qnnStatus}");
                m_isBackendInitialized = true;
                return StatusCode.SUCCESS;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Exception occurred while calling backendCreate: {ex.Message}");
                return StatusCode.FAILURE;
            }
        }

        private bool CreateDevice()
        {
            // Implementation for device creation
            return true;
        }

        private bool InitializeProfiling()
        {
            // Implementation for profiling initialization
            return true;
        }

        private bool RegisterOpPackages()
        {
            // Implementation for registering op packages
            return true;
        }

        private bool CreateContext()
        {
            // Implementation for context creation
            return true;
        }

        private bool ComposeGraphs()
        {
            // Implementation for composing graphs
            return true;
        }

        private bool FinalizeGraphs()
        {
            // Implementation for finalizing graphs
            return true;
        }

        private bool ExecuteGraphs()
        {
            Console.WriteLine($"Executing for {duration} seconds...");
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            while (stopwatch.Elapsed.TotalSeconds < duration)
            {
                // Simulating graph execution
                Console.WriteLine($"Elapsed time: {stopwatch.Elapsed.TotalSeconds} seconds");
                System.Threading.Thread.Sleep(1000); // Sleep for 1 second
            }

            stopwatch.Stop();
            Console.WriteLine($"End time: {DateTimeOffset.Now.ToUnixTimeSeconds()} seconds since epoch");

            return true;
        }

        private bool FreeContext()
        {
            // Implementation for freeing context
            return true;
        }

        private bool FreeDevice()
        {
            // Implementation for freeing device
            return true;
        }
    }
}