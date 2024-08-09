﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;

namespace SampleCSharpApplication
{
    public class Program
    {
        public unsafe static int Main(string[] args)
        {

            StructOffsetGenerator.GenerateStructOffsetsJson("StructOffsets");
            StructOffsetGenerator.CompareOffsetDirectories("CPPStructOffsets", "StructOffsets");
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
            Console.ReadLine();
            for (int i = 0; i<1000; i++)
            {
                using QnnSampleApp app = new QnnSampleApp(model, backend, inputList, duration);
                int res = app.Run();
            }
           
            Console.ReadLine();

            //Environment.Exit(res);
            return 0;
        }
    }
}
