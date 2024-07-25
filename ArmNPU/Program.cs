// See https://aka.ms/new-console-template for more information
using ArmNPU;
using System.Runtime.InteropServices;
using System;
using static ArmNPU.QualcommNPU;

Console.WriteLine("Hello, World!");
try
{
    IntPtr providerListPtr;
    uint numProviders;

    // Call the QnnInterface_getProviders function
    int result = QnnInterface_getProviders(out providerListPtr, out numProviders);

    // Check if the function call was successful
    if (result == 0)
    {
        Console.WriteLine("Number of providers: " + numProviders);

        // Convert the provider list pointer to an array of pointers
        IntPtr[] providerPtrs = new IntPtr[numProviders];
        Marshal.Copy(providerListPtr, providerPtrs, 0, (int)numProviders);

        // Iterate through the provider pointers
        for (int i = 0; i < numProviders; i++)
        {
            IntPtr providerPtr = providerPtrs[i];
            string provider = Marshal.PtrToStringAnsi(providerPtr);
            Console.WriteLine($"Provider {i}: {provider}");
        }
    }
    else
    {
        Console.WriteLine("Failed to get providers. Error code: " + result);
    }
}
catch (Exception ex)
{
    Console.WriteLine("An error occurred: " + ex.Message);
}