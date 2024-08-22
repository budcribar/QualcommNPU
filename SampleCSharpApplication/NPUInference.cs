using System;
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

        public string Name { get; } = "NPU";

      

        public void Execute()
        {
            
        }

        public void SetupInference(string modelPath, string backendPath = "")
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


        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                initializer?.Unload();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~NPUInference()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
