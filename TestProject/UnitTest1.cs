
using NUnit.Framework;
using Newtonsoft.Json.Linq;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;

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
            "Qnn_Version_t",
            "Qnn_Tensor_t",
            "Qnn_BwAxisScaleOffset_t",
            "Qnn_ScaleOffset_t",
            "Qnn_QuantizeParams_t",
            "Qnn_ClientBuffer_t",
            "Qnn_SparseParams_t",
            "Qnn_SparseLayoutHybridCoo_t",
            "Qnn_TensorV1_t",
            "Qnn_TensorV2_t",
            "QnnInterface_t",
            "Qnn_ApiVersion_t",
        };

        [Test, TestCaseSource(nameof(StructureNames))]
        public void CompareStructureOffsets(string structureName)
        {
            // C:\Users\Andaz-8CBE\source\repos\QualcommNPU\TestProject\SampleCSharpApplication\bin\Debug\net7.0\StructOffsets\ApiVersion_offsets.json
            // Load JSON offset files
            string cppOffsetFile = Path.Combine(CppOffsetDir, $"{structureName}_offsets.json");
            string csharpOffsetFile = Path.Combine(CSharpOffsetDir, $"{structureName}.json");

            JObject? csharpJson = null;
            JObject? cppJson = null;

            var p = Path.Combine(Directory.GetCurrentDirectory(), Path.GetDirectoryName(cppOffsetFile));
            var q = Path.GetFullPath(p);
            ;
            var f = File.ReadAllText(cppOffsetFile);

            try
            {
                csharpJson = JObject.Parse(File.ReadAllText(csharpOffsetFile));
                cppJson = JObject.Parse(File.ReadAllText(cppOffsetFile));


            }
            catch (Exception ex) { }
            Assert.IsNotNull(csharpJson, $"Could not find C# {structureName}");
            Assert.IsNotNull(cppJson, $"Could not find C++ {structureName}");

            // Extract data from JSON objects
            string cppStructName = cppJson["StructName"].Value<string>();
            JObject cppOffsets = cppJson["Offsets"].Value<JObject>();
            int cppTotalSize = cppJson["TotalSize"].Value<int>();

            string csharpStructName = csharpJson["StructName"].Value<string>();
            JObject csharpOffsets = csharpJson["Offsets"].Value<JObject>();
            int csharpTotalSize = csharpJson["TotalSize"].Value<int>();

            // Assert structure names match
            Assert.AreEqual(cppStructName, csharpStructName,
                $"Structure name mismatch for '{structureName}' (C++: {cppStructName}, C#: {csharpStructName}).");

            // Assert total sizes match
            Assert.AreEqual(cppTotalSize, csharpTotalSize,
                $"Total size mismatch for structure '{structureName}' (C++: {cppTotalSize}, C#: {csharpTotalSize}).");

            // Compare offsets
            foreach (var cppProperty in cppOffsets.Properties())
            {
                string memberName = cppProperty.Name;
                // Convert the first character to lowercase
                memberName = char.ToLowerInvariant(memberName[0]) + memberName.Substring(1);
                int cppOffset = cppProperty.Value.Value<int>();

                // Case-insensitive check using LINQ
                Assert.IsTrue(csharpOffsets.Properties().Any(p => p.Name.Equals(memberName, StringComparison.OrdinalIgnoreCase)),
                    $"Member '{memberName}' not found in C# offsets for structure '{structureName}'.");

                if (!csharpOffsets.ContainsKey(memberName))
                    memberName = char.ToUpperInvariant(memberName[0]) + memberName.Substring(1);
                
                int csharpOffset = csharpOffsets[memberName].Value<int>();

                // Compare the offset values
                Assert.AreEqual(cppOffset, csharpOffset,
                    $"Offset mismatch for member '{memberName}' in structure '{structureName}' (C++: {cppOffset}, C#: {csharpOffset}).");
            }
        }


    } 
}

