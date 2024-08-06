using System;
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

            string model = null;
            string backend = null;
            string inputList = null;
            int duration = 3; // Default duration

            QnnInterface q1 = new QnnInterface();
            QnnInterface2 q2 = new QnnInterface2();

            var x = sizeof(QnnInterface);
            var xy = sizeof(QnnInterface2);


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

            QnnSampleApp app = new QnnSampleApp(model, backend, inputList, duration);
            return app.Run();
        }
    }
}
