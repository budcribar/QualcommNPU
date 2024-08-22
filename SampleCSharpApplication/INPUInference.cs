

namespace SampleCSharpApplication
{
    internal interface INPUInference : IDisposable
    {
        string Name { get; }
        void SetupInference(string modelPath, string backendPath="");

        void Execute();
    }
}
