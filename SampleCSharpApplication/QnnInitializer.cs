using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        private Qnn_BackendHandle_t m_backendHandle;
        private bool m_isBackendInitialized = false;
        private IntPtr m_deviceHandle;
        private bool m_deviceCreated = false;
        private QnnFunctionPointers m_qnnFunctionPointers = new QnnFunctionPointers();
        private bool qnnFunctionPointersLoaded = false;
        private bool m_isContextCreated = false;
        private Qnn_ContextHandle_t m_context = IntPtr.Zero;
        private GraphInfoManager? m_graphInfoManager;
        private readonly IntPtr[]? m_graphConfigsInfo;
        private readonly IOTensor m_iOTensor = new();

        private void ExitWithMessage(string message, int exitCode)
        {
            Console.WriteLine(message);
            throw new Exception(message);
        }

        public QnnFunctionPointers Initialize(string backEndPath, string modelPath, bool loadFromCachedBinary) 
        {
            // Load backend and model .so and validate all the required function symbols are resolved
            StatusCode statusCode = DynamicLoadUtil.GetQnnFunctionPointers(
                backEndPath,
                modelPath,
                ref m_qnnFunctionPointers,
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
            qnnFunctionPointersLoaded = true;
            return m_qnnFunctionPointers;
        }

        public bool IsDevicePropertySupported(QnnInterface_t qnnInterface)
        {
            if (qnnInterface.PropertyHasCapability == IntPtr.Zero)
            {
                throw new Exception("PropertyHasCapability pointer is null");
            }
            IntPtr propertyHasCapabilityPtr = qnnInterface.PropertyHasCapability;

            QnnProperty_HasCapabilityFn_t propertyHasCapability = Marshal.GetDelegateForFunctionPointer<QnnProperty_HasCapabilityFn_t>(propertyHasCapabilityPtr);

            Qnn_ErrorHandle_t qnnStatus = propertyHasCapability(1501);

            if (qnnStatus != QnnConstants.QNN_SUCCESS)
            {
                Console.Error.WriteLine($"Device property is not supported {qnnStatus}");
                return false;
            }

            return true;
        }
      
        public void CreateDevice(QnnInterface_t qnnInterface)
        {
            if (qnnInterface.DeviceCreate == IntPtr.Zero)
            {
                throw new Exception("DeviceCreate pointer is null");
            }
            IntPtr deviceCreatePtr = qnnInterface.DeviceCreate;

            QnnDevice_CreateFn_t deviceCreate = Marshal.GetDelegateForFunctionPointer<QnnDevice_CreateFn_t>(deviceCreatePtr);
           
            Qnn_ErrorHandle_t qnnStatus = deviceCreate(IntPtr.Zero, IntPtr.Zero, ref m_deviceHandle);

            if (qnnStatus != QnnConstants.QNN_SUCCESS)
            {
                throw new Exception($"DeviceCreate failed {qnnStatus}");
            }
            m_deviceCreated = true;
        }

        public void CreateContext()
        {
            if (m_qnnFunctionPointers?.QnnInterface.ContextCreate == IntPtr.Zero)
            {
                throw new Exception ("ContextCreate pointer is null");
            }
            IntPtr contextCreatePtr = m_qnnFunctionPointers?.QnnInterface.ContextCreate ?? IntPtr.Zero;

            QnnContext_CreateFn_t contextCreate = Marshal.GetDelegateForFunctionPointer<QnnContext_CreateFn_t>(contextCreatePtr);
            Qnn_ErrorHandle_t qnnStatus = contextCreate(m_backendHandle, m_deviceHandle, IntPtr.Zero, ref m_context);

            if (qnnStatus != QnnConstants.QNN_SUCCESS)
            {
                throw new Exception($"contextCreate failed {qnnStatus}");
            }
            m_isContextCreated = true;
        }

        public unsafe GraphInfoManager ComposeGraphs()
        {
            if (m_qnnFunctionPointers?.ComposeGraphsFnHandle == IntPtr.Zero)
            {
                throw new Exception("ComposeGraphsFnHandle pointer is null");
            }
            IntPtr composeGraphsPtr = m_qnnFunctionPointers?.ComposeGraphsFnHandle ?? IntPtr.Zero;

            ComposeGraphsFnHandleType_t composeGraphs = Marshal.GetDelegateForFunctionPointer<ComposeGraphsFnHandleType_t>(composeGraphsPtr);

            uint graphConfigInfosCount = 0;

            QnnLog_Level_t log_level = QnnLog_Level_t.QNN_LOG_LEVEL_ERROR;

            if (m_qnnFunctionPointers == null)
                throw new Exception("Function Pointers is null");

            ModelError_t qnnStatus;
            fixed (QnnInterface_t* qnnInterfacePtr = &m_qnnFunctionPointers.QnnInterface)
            {
                // Calculate the address of the PropertyHasCapability field
                IntPtr propertyHasCapabilityPtrAddress = (IntPtr)(&qnnInterfacePtr->PropertyHasCapability);
                qnnStatus = composeGraphs(m_backendHandle, propertyHasCapabilityPtrAddress, m_context, m_graphConfigsInfo, graphConfigInfosCount, out IntPtr graphsInfos, out uint graphsCount, false, null, log_level);
                if (qnnStatus != QnnConstants.QNN_SUCCESS)
                {
                    throw new Exception($"composeGraphs failed {qnnStatus}");
                }

                m_graphInfoManager = new GraphInfoManager(graphsInfos, graphsCount);
                return m_graphInfoManager;
            }         
        }

        public void FinalizeGraphs()
        {
            for (uint graphIdx = 0; graphIdx < m_graphInfoManager?.Count; graphIdx++)
            {
                if (m_qnnFunctionPointers?.QnnInterface.GraphFinalize == IntPtr.Zero)
                    throw new Exception("GraphFinalize pointer is null");

                IntPtr graphFinalizePtr = m_qnnFunctionPointers?.QnnInterface.GraphFinalize ?? IntPtr.Zero;

                QnnGraph_FinalizeFn_t graphFinalize = Marshal.GetDelegateForFunctionPointer<QnnGraph_FinalizeFn_t>(graphFinalizePtr);

                GraphInfo_t graphInfo = m_graphInfoManager[graphIdx];

                Qnn_ErrorHandle_t qnnStatus = graphFinalize(graphInfo.graph, IntPtr.Zero, IntPtr.Zero);

                if (qnnStatus != QnnConstants.QNN_SUCCESS)
                    throw new Exception($"GraphFinalize failed {qnnStatus}");
            }
        }

        public void InitializeTensors(uint graphIdx, out Qnn_Tensor_t[] inputs, out Qnn_Tensor_t[] outputs, out GraphInfo_t graphInfo)
        {
            inputs = Array.Empty<Qnn_Tensor_t>();
            outputs = Array.Empty<Qnn_Tensor_t>();
            graphInfo = m_graphInfoManager?[graphIdx] ?? new();
            if (m_iOTensor.SetupInputAndOutputTensors(out inputs, out outputs, graphInfo) != IOTensor.StatusCode.SUCCESS)
            {
                throw new Exception("Error in setting up Input and output Tensors for graphIdx: {graphIdx}");
            }
         
        }
        public void ExecuteTensors(GraphInfo_t graphInfo, Qnn_Tensor_t[] inputs, Qnn_Tensor_t[] outputs, QnnFunctionPointers qnnFunctionPointers)
        {
            QnnGraph_Error_t returnStatus = QnnGraph_Error_t.QNN_GRAPH_NO_ERROR;

            IntPtr graphExecutePtr = qnnFunctionPointers.QnnInterface.GraphExecute;
            if (graphExecutePtr == IntPtr.Zero)
                throw new Exception("Failed to get function pointer for GraphExecute");

            QnnGraphExecuteDelegate executeGraph = Marshal.GetDelegateForFunctionPointer<QnnGraphExecuteDelegate>(graphExecutePtr);

            returnStatus = executeGraph(graphInfo.graph, inputs, graphInfo.numInputTensors, outputs, graphInfo.numOutputTensors, IntPtr.Zero, IntPtr.Zero);

            if (QnnGraph_Error_t.QNN_GRAPH_NO_ERROR != returnStatus)
            {
                throw new Exception($"GraphExecute returned {returnStatus}");
            }
        }

    
        public StatusCode InitializeBackend(QnnInterface_t qnnInterface)
        {
            if (qnnInterface.BackendCreate == IntPtr.Zero)
            {
                throw new Exception("QnnInterface pointer is null");
            }

            IntPtr backendCreatePtr = qnnInterface.BackendCreate;
            if (backendCreatePtr == IntPtr.Zero)
            {
                throw new ("Failed to get function pointer for QnnBackend_Create");
            }

            QnnBackend_CreateFn_t backendCreate = Marshal.GetDelegateForFunctionPointer<QnnBackend_CreateFn_t>(backendCreatePtr);

            try
            {
                Qnn_ErrorHandle_t qnnStatus = backendCreate(IntPtr.Zero, IntPtr.Zero, ref m_backendHandle);

                if (qnnStatus != QnnConstants.QNN_SUCCESS)
                {
                    throw new Exception($"Could not initialize backend due to error = {qnnStatus}");
                }

                Console.WriteLine($"Initialize Backend Returned Status = {qnnStatus}");
                m_isBackendInitialized = true;
                return StatusCode.SUCCESS;
            }
            catch (Exception ex)
            {
                throw new Exception($"Exception occurred while calling backendCreate: {ex.Message}");
            }
        }

        private StatusCode FreeDevice()
        {
            if (m_qnnFunctionPointers?.QnnInterface.DeviceFree == IntPtr.Zero)
            {
                Console.Error.WriteLine("DeviceFree pointer is null");
                return StatusCode.FAILURE;
            }
            IntPtr deviceFreePtr = m_qnnFunctionPointers?.QnnInterface.DeviceFree ?? IntPtr.Zero;

            QnnDevice_FreeFn_t deviceFree = Marshal.GetDelegateForFunctionPointer<QnnDevice_FreeFn_t>(deviceFreePtr);

            Qnn_ErrorHandle_t qnnStatus = deviceFree(m_deviceHandle);

            if (qnnStatus != QnnConstants.QNN_SUCCESS && qnnStatus != (ulong)QnnCommon_Error_t.QNN_COMMON_ERROR_NOT_SUPPORTED) // TODO QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE
            {
                Console.Error.WriteLine("Failed to free device");
                return StatusCode.FAILURE;
            }
            
            return StatusCode.SUCCESS;
        }

        private StatusCode FreeBackend()
        {
            if (m_qnnFunctionPointers?.QnnInterface.BackendFree == IntPtr.Zero)
            {
                Console.Error.WriteLine("BackendFree pointer is null");
                return StatusCode.FAILURE;
            }
            var backendFreeFn = Marshal.GetDelegateForFunctionPointer<QnnBackend_FreeFn_t>(m_qnnFunctionPointers?.QnnInterface.BackendFree ?? IntPtr.Zero);

            var result = backendFreeFn(m_backendHandle);
            if (result != QnnBackend_Error_t.QNN_BACKEND_NO_ERROR)
            {
                Console.WriteLine("Could not free backend");
                return StatusCode.FAILURE;
            }
            return StatusCode.SUCCESS;
        }

        private StatusCode FreeContext()
        {
            if (!m_isContextCreated) return StatusCode.SUCCESS;

            Console.WriteLine("Freeing context");
            if (m_qnnFunctionPointers?.QnnInterface.ContextFree != IntPtr.Zero)
            {
                var contextFreeFn = Marshal.GetDelegateForFunctionPointer<QnnContext_FreeFn_t>(m_qnnFunctionPointers?.QnnInterface.ContextFree ?? IntPtr.Zero);
                if (contextFreeFn(m_context, IntPtr.Zero) != QnnContextError.QNN_CONTEXT_NO_ERROR)
                {
                    Console.WriteLine("Could not free context");
                    return StatusCode.FAILURE;
                }
            }
            m_isContextCreated = false;
            // Implementation for freeing context
            return StatusCode.SUCCESS;
        }


        public void Unload()
        {
            if (m_isBackendInitialized && m_backendHandle != IntPtr.Zero)
            {
                FreeBackend();
                m_isBackendInitialized = false;
            }
            if (m_deviceCreated && m_deviceHandle != IntPtr.Zero)
            {
                FreeDevice();
                m_deviceCreated = false;
            }
               

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
