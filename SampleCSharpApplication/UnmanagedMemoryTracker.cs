using System;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

namespace SampleCSharpApplication
{
   

    public static class UnmanagedMemoryTracker
    {
        private static ConcurrentDictionary<IntPtr, long> allocations = new ConcurrentDictionary<IntPtr, long>();
        private static long totalAllocated = 0;

        public static IntPtr AllocateMemory(int size)
        {
            IntPtr ptr = Marshal.AllocHGlobal(size);
            allocations[ptr] = size;
            Interlocked.Add(ref totalAllocated, size);
            return ptr;
        }

        public static void FreeMemory(IntPtr ptr)
        {
            if (allocations.TryRemove(ptr, out long size))
            {
                Marshal.FreeHGlobal(ptr);
                Interlocked.Add(ref totalAllocated, -size);
            }
        }

        public static long GetTotalAllocatedMemory()
        {
            return totalAllocated;
        }

        public static void PrintMemoryUsage()
        {
            foreach (var ptr in allocations.Values)
            {
                Console.WriteLine(ptr.ToString());
            }
            Console.WriteLine($"Total unmanaged memory allocated: {totalAllocated} bytes");
            Console.WriteLine($"Number of active allocations: {allocations.Count}");
        }
    }
}
