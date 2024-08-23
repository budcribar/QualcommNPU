using System.Runtime.InteropServices;
using System.Diagnostics;
using static SampleCSharpApplication.QnnDelegates;
using static SampleCSharpApplication.DynamicLoadUtil;
using System;
using System.Text.RegularExpressions;
using System.Reflection;
using System.Xml.Linq;

namespace SampleCSharpApplication
{
  
    public unsafe class QnnSampleApp : IDisposable
    {
        private QnnLog_Callback_t? m_logCallback = null;
        private QnnFunctionPointers? m_qnnFunctionPointers = null;
        private Qnn_LogHandle_t m_logHandle;
        private Qnn_BackendHandle_t m_backendHandle;
        private Qnn_DeviceHandle_t m_deviceHandle = IntPtr.Zero;
        private Qnn_ContextHandle_t m_context = IntPtr.Zero;
        private GraphInfoManager? m_graphInfoManager;
        private bool m_isBackendInitialized = false;
        private readonly IntPtr m_backendConfig;
        private readonly IntPtr[]? m_graphConfigsInfo;
        private readonly IOTensor m_iOTensor = new();
        private ReadInputListsResult readInputListsResult = new();
        private Qnn_ProfileHandle_t m_profileBackendHandle = IntPtr.Zero;
        private bool m_isContextCreated = false;
        private readonly string model;
        private readonly string backend;
        private readonly string inputList;
        private readonly int duration;

        // Constants
        private const int QNN_API_VERSION_MAJOR = 2;
        private const int QNN_API_VERSION_MINOR = 15;
      

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

        public struct ReadInputListsResult
        {
            public List<List<List<string>>> FilePathsLists;         
            public List<Dictionary<string, uint>> InputNameToIndexMaps;   
            public bool ReadSuccess;             // Indicates success or failure of the read operation
        }

        private static Dictionary<string, uint> ExtractInputNameIndices(string inputLine, string separator)
        {
            var inputFilePaths = inputLine.Split(' ').ToList();
            var inputNameToIndex = new Dictionary<string, uint>();
            int inputCount = 0; // Change idx to int

            for (int idx = 0; idx < inputFilePaths.Count; idx++)
            {
                int position = inputFilePaths[idx].IndexOf(separator, StringComparison.OrdinalIgnoreCase);
                if (position != -1)
                {
                    var unsanitizedTensorName = inputFilePaths[idx][..position];
                    var sanitizedTensorName = SanitizeTensorName(unsanitizedTensorName);

                    if (sanitizedTensorName != unsanitizedTensorName)
                    {
                        inputNameToIndex[unsanitizedTensorName] = (uint)idx; // Add explicit cast
                    }

                    inputNameToIndex[sanitizedTensorName] = (uint)idx; // Add explicit cast
                    inputCount++;
                }
            }
            if (inputCount != inputFilePaths.Count)
            {
                //Handle Error. 
            }
            return inputNameToIndex;
        }

        private static (List<List<string>>, Dictionary<string, uint>, bool) ReadInputList(string inputFileListPath)
        {
            var lines = new Queue<string>();
            var filePathsList = new List<List<string>>();
            var inputNameToIndex = new Dictionary<string, uint>();

            try
            {
                using StreamReader fileListStream = new(inputFileListPath);
               
                string? fileLine;
                while ((fileLine = fileListStream.ReadLine()) != null)
                {
                    if (!string.IsNullOrEmpty(fileLine))
                    {
                        lines.Enqueue(fileLine);
                    }
                }
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading input file: {ex.Message}");
                return (filePathsList, inputNameToIndex, false); // Indicate failure
            }

            // Remove lines starting with '#' or '%'
            while (lines.Any() && (lines.Peek().StartsWith('#') || lines.Peek().StartsWith('%')))
            {
                lines.Dequeue();
            }

            const string separator = ":=";
            if (lines.Any())
            {
                inputNameToIndex = ExtractInputNameIndices(lines.Peek(), separator);
            }

            while (lines.Any())
            {
                var paths = lines.Dequeue().Split(' ').ToList();
                filePathsList.Add(ParseInputFilePaths(paths, separator));
            }

            return (filePathsList, inputNameToIndex, true); // Indicate success
        }

        private static List<string> ParseInputFilePaths(List<string> inputFilePaths, string separator)
        {
            var paths = new List<string>();

            foreach (var inputInfo in inputFilePaths)
            {
                int position = inputInfo.IndexOf(separator);
                if (position != -1)  // -1 is the C# equivalent of std::string::npos
                {
                    string path = inputInfo[(position + separator.Length)..];
                    paths.Add(path);
                }
                else
                {
                    paths.Add(inputInfo);
                }
            }

            return paths;
        }

        private static string SanitizeTensorName(string name)
        {
            string sanitizedName = Regex.Replace(name, "\\W+", "_");
            if (!char.IsLetter(sanitizedName[0]) && sanitizedName[0] != '_')
            {
                sanitizedName = "_" + sanitizedName;
            }
            return sanitizedName;
        }

        public static ReadInputListsResult ReadInputLists(string[] inputFileListPaths)
        {
            var filePathsLists = new List<List<List<string>>>();
            var inputNameToIndexMaps = new List<Dictionary<string, uint>>();

            foreach (var path in inputFileListPaths)
            {
                var (filePathList, inputNameToIndex, readSuccess) = ReadInputList(path);

                if (!readSuccess)
                {
                    filePathsLists.Clear(); // Clear any data read so far
                    return new ReadInputListsResult
                    {
                        FilePathsLists = new(),
                        InputNameToIndexMaps = new(),
                        ReadSuccess = false
                    };
                }

                filePathsLists.Add(filePathList);
                inputNameToIndexMaps.Add(inputNameToIndex);
            }

            return new ReadInputListsResult
            {
                FilePathsLists = filePathsLists,
                InputNameToIndexMaps = inputNameToIndexMaps,
                ReadSuccess = true
            };
        }
    

    public QnnSampleApp(string model, string backend, string inputList, int duration)
        {
            this.model = model;
            this.backend = backend;
            this.inputList = inputList;
            this.duration = duration;
        }

        public StatusCode Run()
        {
            Console.WriteLine($"Model: {model}");
            Console.WriteLine($"Backend: {backend}");

            if (!File.Exists(model))
            {
                Console.WriteLine($"Error: Could not load model: {model}");
                return StatusCode.FAILURE;
            }

            if (!File.Exists(backend))
            {
                Console.WriteLine($"Error: Could not load backend: {backend}");
                return StatusCode.FAILURE;
            }

            if (!File.Exists(inputList))
            {
                Console.WriteLine($"Error: Could not find input list: {inputList}");
                return StatusCode.FAILURE;
            }
            readInputListsResult = ReadInputLists(inputList.Split(','));
            if (!readInputListsResult.ReadSuccess)
            {
                Console.WriteLine($"Error: Could not read inputList");
                return StatusCode.FAILURE;
            }

            Console.WriteLine("Initializing...");
            var initializer = new QnnInitializer();
            m_qnnFunctionPointers = initializer.Initialize(backend, model, false);
           
            InitializeLogging();

            Console.WriteLine("Initializing backend...");
            var status = initializer.InitializeBackend(m_qnnFunctionPointers.QnnInterface);
            if (status == StatusCode.FAILURE)
            {
                Console.WriteLine("Backend Initialization failure");
                return StatusCode.FAILURE;
            }

            if (!initializer.IsDevicePropertySupported(m_qnnFunctionPointers.QnnInterface))
            {
                return StatusCode.FAILURE;
            }

            Console.WriteLine("Creating device...");
            initializer.CreateDevice(m_qnnFunctionPointers.QnnInterface);
            

            //Console.WriteLine("Initializing profiling...");
            //if (!InitializeProfiling())
            //{
            //    Console.WriteLine("Profiling Initialization failure");
            //    return 1;
            //}

            //Console.WriteLine("Registering op packages...");
            //if (!RegisterOpPackages())
            //{
            //    Console.WriteLine("Register Op Packages failure");
            //    return 1;
            //}

            Console.WriteLine("Creating context...");
            initializer.CreateContext();

            Console.WriteLine("Composing graphs...");
            initializer.ComposeGraphs();
           
            Console.WriteLine("Finalizing graphs...");
            initializer.FinalizeGraphs();
            

            Console.WriteLine("Executing graphs...");

            initializer.InitializeTensors(0, out Qnn_Tensor_t[] inputs, out Qnn_Tensor_t[] outputs, out GraphInfo_t graphInfo);

            Stopwatch stopwatch = new();
            stopwatch.Start();

            while (stopwatch.Elapsed.TotalSeconds < duration)
            {
                initializer.ExecuteTensors(graphInfo, inputs, outputs, m_qnnFunctionPointers);
            }


            //if (ExecuteGraphs(readInputListsResult.FilePathsLists) != StatusCode.SUCCESS)
            //{
            //    Console.WriteLine("Graph Execution failure");
            //    return 1;
            //}

            //Console.WriteLine("Freeing context...");
            //if (FreeContext() != StatusCode.SUCCESS)
            //{
            //    Console.WriteLine("Context Free failure");
            //    return StatusCode.FAILURE;
            //}

            Console.WriteLine("Freeing device...");
            //if (FreeDevice() != StatusCode.SUCCESS)
            //{
            //    Console.WriteLine("Device Free failure");
            //    return StatusCode.FAILURE;
            //}

            return StatusCode.SUCCESS;
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
                    Qnn_LogHandle_t logHandle = IntPtr.Zero;
                    IntPtr logCallbackPtr = Marshal.GetFunctionPointerForDelegate(m_logCallback);

                    IntPtr logCreatePtr = m_qnnFunctionPointers?.QnnInterface.LogCreate ?? IntPtr.Zero;
                   
                    if (logCreatePtr == IntPtr.Zero)
                    {
                        Console.Error.WriteLine("Failed to get function pointer for QnnLog_Create");
                        return;
                    }

                    QnnLog_CreateFn_t logCreate = Marshal.GetDelegateForFunctionPointer<QnnLog_CreateFn_t>(logCreatePtr);

                    Qnn_ErrorHandle_t result = logCreate(new IntPtr(0), (int)logLevel, ref m_logHandle);
                   
                    if (result != QnnConstants.QNN_SUCCESS)
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

      
        private static bool IsLogInitialized()
        {
            // Implement your logic to check if logging is initialized
            return true; // Placeholder
        }

       

        private static LogLevel GetLogLevel()
        {
            // Implement your logic to get the current log level
            return LogLevel.INFO; // Placeholder
        }

        private void LogMessage(int level, IntPtr msgPtr)
        {
            string message = Marshal.PtrToStringAnsi(msgPtr) ?? string.Empty;
            Console.WriteLine($"[{(LogLevel)level}] {message}");
        }
     
       
       

        //private bool InitializeProfiling()
        //{
        //    // Implementation for profiling initialization
        //    return true;
        //}

        //private bool RegisterOpPackages()
        //{
        //    // Implementation for registering op packages
        //    return true;
        //}

       

       
        
      

       

       

        //private StatusCode ExecuteGraphs(List<List<List<string>>> filePathsLists)
        //{
        //    var returnStatus = StatusCode.SUCCESS; // TODO

        //    for (uint graphIdx = 0; graphIdx < m_graphInfoManager?.Count; graphIdx++)
        //    {
        //        if (graphIdx >= filePathsLists.Count)
        //        {
        //            Console.WriteLine($"No Inputs available for: {graphIdx}");
        //            return StatusCode.FAILURE;

        //        }

        //        if (initializer.InitializeTensors(graphIdx, out Qnn_Tensor_t[] inputs, out Qnn_Tensor_t[] outputs, out GraphInfo_t graphInfo) == StatusCode.FAILURE)
        //            return StatusCode.FAILURE;

        //        //populateInputTensors
        //        var inputFileList = filePathsLists[(int)graphIdx];

        //        if (inputFileList.Any())
        //        {
        //            int totalCount = inputFileList[0].Count;
        //            int inputFileIndexOffset = 0;
        //            while (inputFileIndexOffset < totalCount)
        //            {
        //                var result = IOTensor.PopulateInputTensors(graphIdx, inputFileList, inputFileIndexOffset, false, readInputListsResult.InputNameToIndexMaps[(int)graphIdx], inputs, graphInfo, IOTensor.InputDataType.FLOAT);

        //                if (result.Status != IOTensor.StatusCode.SUCCESS)
        //                {
        //                    returnStatus = StatusCode.FAILURE;
        //                }

        //                if (returnStatus == StatusCode.SUCCESS)
        //                {
        //                    Console.WriteLine("Successfully populated input tensors for graphIdx: {graphIdx}");
        //                    QnnGraph_Error_t executeStatus = ExecuteTensors(graphInfo, inputs, outputs, m_qnnFunctionPointers, duration);

        //                    if (executeStatus != QnnGraph_Error_t.QNN_GRAPH_NO_ERROR)
        //                        returnStatus = StatusCode.FAILURE;
        //                }
        //                inputFileIndexOffset++;
                      
        //                UnmanagedMemoryTracker.PrintMemoryUsage();
        //                for (int i = 0; i < inputs.Length; i++)
        //                {
        //                    inputs[i].Dispose();
        //                }
        //                for (int i = 0; i < outputs.Length; i++)
        //                {
        //                    outputs[i].Dispose();
        //                }
        //                UnmanagedMemoryTracker.PrintMemoryUsage();
        //                return StatusCode.SUCCESS;
        //            }

        //        }
        //    }

        //    return StatusCode.SUCCESS;
        //}

       
        private bool disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                // Free Profiling object if it was created
                if (m_profileBackendHandle != IntPtr.Zero)
                {
                    Console.WriteLine("Freeing backend profile object.");
                    if (m_qnnFunctionPointers?.QnnInterface.ProfileFree != IntPtr.Zero)
                    {
                        var profileFreeFn = Marshal.GetDelegateForFunctionPointer<QnnProfile_FreeFn_t>(m_qnnFunctionPointers?.QnnInterface.ProfileFree ?? IntPtr.Zero);
                        
                        if (profileFreeFn(m_profileBackendHandle) != QnnProfile_Error_t.QNN_PROFILE_NO_ERROR)
                        {
                            Console.WriteLine("Could not free backend profile handle.");
                        }
                    }
                }

                // Free context if not already done
                //if (m_isContextCreated)
                //{
                //    FreeContext();             
                //}
                //m_isContextCreated = false;

                // Terminate backend
                if (m_isBackendInitialized && m_qnnFunctionPointers?.QnnInterface.BackendFree != IntPtr.Zero)
                {
                    Console.WriteLine("Freeing backend");
                    var backendFreeFn = Marshal.GetDelegateForFunctionPointer<QnnBackend_FreeFn_t>(m_qnnFunctionPointers?.QnnInterface.BackendFree ?? IntPtr.Zero);
                    if (backendFreeFn(m_backendHandle) != QnnBackend_Error_t.QNN_BACKEND_NO_ERROR)
                    {
                        Console.WriteLine("Could not free backend");
                    }
                }
                m_isBackendInitialized = false;

                // Terminate logging in the backend
                if (m_qnnFunctionPointers?.QnnInterface.LogFree != IntPtr.Zero && m_logHandle != IntPtr.Zero)
                {
                    var logFreeFn = Marshal.GetDelegateForFunctionPointer<QnnLog_FreeFn_t>(m_qnnFunctionPointers?.QnnInterface.LogFree ?? IntPtr.Zero);
                    if (logFreeFn(m_logHandle) != QnnConstants.QNN_SUCCESS)
                    {
                        Console.WriteLine("Unable to terminate logging in the backend.");
                    }
                }
            }

            disposed = true;
            UnmanagedMemoryTracker.PrintMemoryUsage();
        }
        ~QnnSampleApp()
        {
            Dispose(false);
        }


    }
}