using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Reflection;

namespace SampleCSharpApplication
{
    public class Program
    {
        public static int Main(string[] args)
        {
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

            try
            {
                using INPUInference inference = new NPUInference();

                inference.SetupInference("../../../Inception_v3_quantizedArm64.dll", "../../../QnnHtp.dll");

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
            }
           
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
