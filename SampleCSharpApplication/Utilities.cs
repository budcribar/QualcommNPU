using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Newtonsoft.Json.Linq;

namespace SampleCSharpApplication
{
    public static class StructOffsetGenerator
    {
        public static bool CompareOffsetDirectories(string cppDir, string csharpDir)
        {
            var cppFiles = Directory.GetFiles(cppDir, "*.json");
            var csharpFiles = Directory.GetFiles(csharpDir, "*.json");

            if (cppFiles.Length != csharpFiles.Length)
            {
                Console.WriteLine($"Number of files mismatch: C++ ({cppFiles.Length}) vs C# ({csharpFiles.Length})");
                //return false;
            }

            bool allMatch = true;

            foreach (var cppFile in cppFiles)
            {
                var fileName = Path.GetFileName(cppFile);
                var csharpFile = Path.Combine(csharpDir, fileName);

                if (!File.Exists(csharpFile))
                {
                    Console.WriteLine($"File {fileName} not found in C# directory");
                    allMatch = false;
                    continue;
                }

                var cppJson = JObject.Parse(File.ReadAllText(cppFile));
                var csharpJson = JObject.Parse(File.ReadAllText(csharpFile));

                var differences = cppJson.Properties()
                    .Select(p => new
                    {
                        Property = p.Name,
                        CppValue = p.Value.ToString(),
                        CSharpValue = csharpJson[p.Name]?.ToString()
                    })
                    .Where(d => d.CppValue != d.CSharpValue)
                    .ToList();

                if (differences.Any())
                {
                    Console.WriteLine($"Differences found in {fileName}:");
                    foreach (var diff in differences)
                    {
                        Console.WriteLine($"  {diff.Property}: C++ = {diff.CppValue}, C# = {diff.CSharpValue}");
                    }
                    allMatch = false;
                }
            }

            return allMatch;
        }
        public static void GenerateStructOffsetsJson(string outputDirectory)
        {
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            var structTypes = Assembly.GetExecutingAssembly().GetTypes()
                .Where(t => t.Namespace == "SampleCSharpApplication" &&
                             t.IsValueType &&
                             !t.IsEnum)
                .ToList();

            foreach (var structType in structTypes)
            {
                var offsets = new SortedDictionary<string, int>();
                var fields = structType.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

                foreach (var field in fields)
                {
                    try
                    {
                        int offset = (int)Marshal.OffsetOf(structType, field.Name).ToInt64();
                        offsets.Add(field.Name, offset);
                    }
                    catch { } // Consider adding more specific error handling
                }

                if (offsets.Count == 0) continue;
                // Create the output object
                var outputObject = new
                {
                    StructName = structType.Name,
                    Offsets = offsets,
                    TotalSize = Marshal.SizeOf(structType)
                };

                string json = JsonSerializer.Serialize(outputObject, new JsonSerializerOptions { WriteIndented = true });
                string fileName = $"{structType.Name}.json";
                File.WriteAllText(Path.Combine(outputDirectory, fileName), json);

                Console.WriteLine($"Generated offset file for {structType.Name}");
            }
        }
    }
}