using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Reflection;
using System.Management;

namespace SampleCSharpApplication
{
    public class Program
    {
        public static int Main(string[] args)
        {
            try
            {
                ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_Processor");

                //foreach (ManagementObject obj in searcher.Get())
                //{
                //    Console.WriteLine("Processor Information:");
                //    Console.WriteLine("Name: " + obj["Name"]);
                //    Console.WriteLine("Description: " + obj["Description"]);
                //    Console.WriteLine("Manufacturer: " + obj["Manufacturer"]);
                //}

                // Attempt to find more specific information about the NPU
                searcher = new ManagementObjectSearcher("SELECT * FROM Win32_PnPEntity WHERE  Name LIKE '%Hexagon%NPU%'");

                foreach (ManagementObject obj in searcher.Get())
                {
                    Console.WriteLine("\nPossible NPU Information:");
                    Console.WriteLine("Name: " + obj["Name"]);
                    Console.WriteLine("Description: " + obj["Description"]);
                    Console.WriteLine("DeviceID: " + obj["DeviceID"]);
                }
                // ACPI\QCOM0D0A\2&DABA3FF&1
            }
            catch (Exception e)
            {
                Console.WriteLine("An error occurred: " + e.Message);
            }
            string model = string.Empty;
            string backend = string.Empty;
           
            int duration = 3; // Default duration
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model":
                        model = args[++i];
                        break;
                    case "--backend":
                        backend = args[++i];
                        break;
                    case "--duration":
                        if (int.TryParse(args[++i], out int parsedDuration))
                        {
                            duration = parsedDuration;
                        }
                        break;
                }
            }

            for (int i = 0; i<100; i++)
            {
                try
                {
                    using INPUInference inference = new NPUInference();

                    inference.SetupInference(model, backend);

                    Stopwatch stopwatch = new();
                    stopwatch.Start();

                    while (stopwatch.Elapsed.TotalSeconds < duration)
                    {
                        inference.Execute();
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                    Console.ReadLine();
                }
                UnmanagedMemoryTracker.PrintMemoryUsage();
            }
           
            Console.WriteLine("Finished");
            UnmanagedMemoryTracker.PrintMemoryUsage();
            Console.ReadLine();
            return 0;
        }


        public static int Main2(string[] args)
        {

            StructOffsetGenerator.GenerateStructOffsetsJson("StructOffsets");
            Console.WriteLine("Struct offset JSON files have been generated.");

            string model = string.Empty;
            string backend = string.Empty;
            string inputList = string.Empty;
            int duration = 3; // Default duration

           // QnnInterface q1 = new QnnInterface();

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model":
                        model = args[++i];
                        break;
                    case "--backend":
                        backend = args[++i];
                        break;
                    case "--input_list":
                        inputList = args[++i];
                        break;
                    case "--duration":
                        if (int.TryParse(args[++i], out int parsedDuration))
                        {
                            duration = parsedDuration;
                        }
                        break;
                }
            }

            if (string.IsNullOrEmpty(model) || string.IsNullOrEmpty(backend) || string.IsNullOrEmpty(inputList))
            {
                Console.WriteLine("Missing required arguments. Usage: program --model <FILE> --backend <FILE> --input_list <FILE> [--duration <SECONDS>]");
                return 1;
            }
            // Console.ReadLine();
            UnmanagedMemoryTracker.PrintMemoryUsage();
            for (int i = 0; i<10; i++)
            {
                using QnnSampleApp app = new QnnSampleApp(model, backend, inputList, duration);
                int res;
                try
                {
                    res = (int)app.Run();
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.ToString());
                }
                UnmanagedMemoryTracker.PrintMemoryUsage();
            }
           
            //Console.ReadLine();

            //Environment.Exit(res);
            return 0;
        }
    }
}
