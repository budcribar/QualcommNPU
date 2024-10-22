﻿using System;
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
                    Offsets = offsets.OrderBy(x => x.Value)
                                        .ToDictionary(x => x.Key, x => x.Value),
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