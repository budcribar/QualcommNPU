﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
    internal class NPUInference : INPUInference, IDisposable
    {
        private QnnFunctionPointers? m_qnnFunctionPointers = null;
        private QnnInitializer? initializer;
        private bool disposedValue;
        private bool initialized;
        private Qnn_Tensor_t[] inputs = Array.Empty<Qnn_Tensor_t>();
        private Qnn_Tensor_t[] outputs = Array.Empty<Qnn_Tensor_t>();
        private GraphInfoManager m_graphInfoManager = new(IntPtr.Zero, 0);
        public string Name { get; } = "NPU";

        public void Execute()
        {
            if (m_qnnFunctionPointers == null)
                throw new Exception("Function Pointers is null");

            if (!initialized || inputs.Length == 0 || outputs.Length == 0 || m_graphInfoManager.Count == 0)
                throw new Exception("NPU has not been initialized");

            initializer?.ExecuteTensors(m_graphInfoManager[0], inputs, outputs, m_qnnFunctionPointers);
        }

        public void SetupInference(string modelPath, string backendPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new Exception($"Error: Could not load model: {modelPath}");
            }
            if (!File.Exists(backendPath))
            {
                throw new Exception($"Error: Could not load backend: {backendPath}");
            }

            initializer = new QnnInitializer();
            m_qnnFunctionPointers = initializer.Initialize(backendPath, modelPath, false);

            initializer.InitializeBackend(m_qnnFunctionPointers.QnnInterface);

            if (!initializer.IsDevicePropertySupported(m_qnnFunctionPointers.QnnInterface))
            {
                throw new Exception("Device not supported");
            }

            initializer.CreateDevice(m_qnnFunctionPointers.QnnInterface);
            initializer.CreateContext();
            m_graphInfoManager = initializer.ComposeGraphs();
            initializer.FinalizeGraphs();
            initializer.InitializeTensors(0, out inputs, out outputs, out GraphInfo_t graphInfo);
            initialized = true;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    foreach (var input in inputs)
                        input.Dispose(); 
                    inputs = Array.Empty<Qnn_Tensor_t>();

                    foreach (var output in outputs) output.Dispose();
                    outputs = Array.Empty<Qnn_Tensor_t>();
                }

                initializer?.Unload();
                disposedValue = true;
            }
        }

        ~NPUInference()
        {
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
