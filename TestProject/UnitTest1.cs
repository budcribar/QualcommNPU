
using NUnit.Framework;
using Newtonsoft.Json.Linq;
using System.IO;

namespace TestProject // Replace with your actual namespace
{
    [TestFixture]
    public class StructureOffsetTests
    {
        private const string relative = @"..\..\..\..\SampleCSharpApplication\bin\Debug\net7.0\";

        private const string CppOffsetDir = relative + "CPPStructOffsets"; // Path to C++ offset files
        private const string CSharpOffsetDir = relative + "StructOffsets"; // Path to C# offset files

        private static readonly string[] StructureNames =
        {
            "GraphConfigInfo_t",
            "GraphInfo_t",
            "CoreApiVersion",
            "Qnn_Tensor_t",
            "QnnBwAxisScaleOffset",
            "Qnn_ScaleOffset_t",
            "Qnn_QuantizeParams_t",
            "Qnn_ClientBuffer_t",
            "Qnn_SparseParams_t",
            "Qnn_SparseLayoutHybridCoo_t",
            "Qnn_TensorV1_t",
            "Qnn_TensorV2_t",
            "QnnInterface",
            "QnnInterface_t",
            "ApiVersion",
            "PopulateInputTensorsRetType",
            "ReadBatchDataRetType",
            "CalculateLengthResult"
        };

        [Test, TestCaseSource(nameof(StructureNames))]
        public void CompareStructureOffsets(string structureName)
        {
            // C:\Users\Andaz-8CBE\source\repos\QualcommNPU\TestProject\SampleCSharpApplication\bin\Debug\net7.0\StructOffsets\ApiVersion_offsets.json
            // Load JSON offset files
            string cppOffsetFile = Path.Combine(CppOffsetDir, $"{structureName}_offsets.json");
            string csharpOffsetFile = Path.Combine(CSharpOffsetDir, $"{structureName}_offsets.json");

            JObject? csharpOffsets = null;
            JObject? cppOffsets = null;

            try
            { 
                csharpOffsets = JObject.Parse(File.ReadAllText(csharpOffsetFile));
                cppOffsets = JObject.Parse(File.ReadAllText(cppOffsetFile));


            }
            catch (Exception ex) { }
            Assert.IsNotNull(csharpOffsets, $"Could not find C# {structureName}");
            Assert.IsNotNull(cppOffsets,$"Could not find C++ {structureName}");

            var memberNames = cppOffsets.Properties().Select(p => p.Name).ToList();

            // Iterate through the member names and compare offsets
            foreach (var memberName in memberNames)
            {
                // Check if the member exists in both C++ and C# offset files
                Assert.IsTrue(cppOffsets.ContainsKey(memberName),
                    $"Member '{memberName}' not found in C++ offsets for structure '{structureName}'.");

                Assert.IsTrue(csharpOffsets.ContainsKey(memberName),
                    $"Member '{memberName}' not found in C# offsets for structure '{structureName}'.");

                // Get the offset values
                int cppOffset = cppOffsets[memberName].Value<int>();
                int csharpOffset = csharpOffsets[memberName].Value<int>();

                // Compare the offset values
                Assert.AreEqual(cppOffset, csharpOffset,
                    $"Offset mismatch for member '{memberName}' in structure '{structureName}' (C++: {cppOffset}, C#: {csharpOffset}).");
            }
        }


        //    // Compare offsets
        //    foreach (var cppProperty in cppOffsets.Properties())
        //    {
        //        string propertyName = cppProperty.Name;
        //        int cppOffset = cppProperty.Value.Value<int>();

        //        // Check if the property exists in the C# offsets
        //        Assert.IsTrue(csharpOffsets.ContainsKey(propertyName),
        //            $"Property '{propertyName}' not found in C# offsets for structure '{structureName}'.");

        //        int csharpOffset = csharpOffsets[propertyName].Value<int>();

        //        // Compare the offset values
        //        Assert.AreEqual(cppOffset, csharpOffset,
        //            $"Offset mismatch for property '{propertyName}' in structure '{structureName}' (C++: {cppOffset}, C#: {csharpOffset}).");
        //    }
        //}
    }
}