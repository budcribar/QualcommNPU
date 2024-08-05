using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace SampleCSharpApplication
{
    public static class StructOffsetGenerator
    {
        public static void GenerateStructOffsetsJson(string outputDirectory)
        {
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            var structTypes = typeof(QnnDelegates).GetNestedTypes()
                .Where(t => t.IsValueType && !t.IsEnum)
                .ToList();

            foreach (var structType in structTypes)
            {
                var offsets = new System.Collections.Generic.Dictionary<string, long>();
                var fields = structType.GetFields();

                foreach (var field in fields)
                {
                    long offset = Marshal.OffsetOf(structType, field.Name).ToInt64();
                    offsets.Add(field.Name, offset);
                }

                // Add total size of the struct
                offsets.Add("TotalSize", Marshal.SizeOf(structType));

                string json = JsonSerializer.Serialize(offsets, new JsonSerializerOptions { WriteIndented = true });
                string fileName = $"{structType.Name}_offsets.json";
                File.WriteAllText(Path.Combine(outputDirectory, fileName), json);

                Console.WriteLine($"Generated offset file for {structType.Name}");
            }
        }
    }
}