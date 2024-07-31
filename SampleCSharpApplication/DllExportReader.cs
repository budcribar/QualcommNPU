using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class DllExportReader
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);

    private IntPtr m_libHandle;

    public DllExportReader(string dllPath)
    {
        m_libHandle = LoadLibrary(dllPath);
        if (m_libHandle == IntPtr.Zero)
        {
            throw new Exception($"Failed to load library: {dllPath}");
        }
    }

    public List<string> ListExports()
    {
        List<string> exports = new List<string>();

        try
        {
            // Get the DOS header
            IMAGE_DOS_HEADER dosHeader = Marshal.PtrToStructure<IMAGE_DOS_HEADER>(m_libHandle);

            // Get the NT headers
            IntPtr ntHeadersPtr = IntPtr.Add(m_libHandle, dosHeader.e_lfanew);
            IMAGE_NT_HEADERS64 ntHeaders = Marshal.PtrToStructure<IMAGE_NT_HEADERS64>(ntHeadersPtr);

            // Get the export directory
            uint exportDirRva = ntHeaders.OptionalHeader.DataDirectory[0].VirtualAddress;
            if (exportDirRva == 0)
            {
                Console.WriteLine("No export directory found");
                return exports;
            }

            int exportDirRvaInt = ConvertUIntToInt(exportDirRva, "Export directory RVA");

            IntPtr exportDirPtr = IntPtr.Add(m_libHandle, exportDirRvaInt);
            IMAGE_EXPORT_DIRECTORY exportDir = Marshal.PtrToStructure<IMAGE_EXPORT_DIRECTORY>(exportDirPtr);

            // Get the arrays of export information
            int[] addressOfFunctions = new int[exportDir.NumberOfFunctions];
            int[] addressOfNames = new int[exportDir.NumberOfNames];
            short[] addressOfNameOrdinals = new short[exportDir.NumberOfNames];

            int addressOfFunctionsInt = ConvertUIntToInt(exportDir.AddressOfFunctions, "AddressOfFunctions");
            int addressOfNamesInt = ConvertUIntToInt(exportDir.AddressOfNames, "AddressOfNames");
            int addressOfNameOrdinalsInt = ConvertUIntToInt(exportDir.AddressOfNameOrdinals, "AddressOfNameOrdinals");

            Marshal.Copy(IntPtr.Add(m_libHandle, addressOfFunctionsInt), addressOfFunctions, 0, (int)exportDir.NumberOfFunctions);
            Marshal.Copy(IntPtr.Add(m_libHandle, addressOfNamesInt), addressOfNames, 0, (int)exportDir.NumberOfNames);
            Marshal.Copy(IntPtr.Add(m_libHandle, addressOfNameOrdinalsInt), addressOfNameOrdinals, 0, (int)exportDir.NumberOfNames);

            // Get the exported function names
            for (int i = 0; i < exportDir.NumberOfNames; i++)
            {
                int nameRva = ConvertUIntToInt((uint)addressOfNames[i], $"Name RVA for function {i}");
                string functionName = Marshal.PtrToStringAnsi(IntPtr.Add(m_libHandle, nameRva));
                exports.Add(functionName);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error reading exports: {ex.Message}");
        }

        return exports;
    }

    private int ConvertUIntToInt(uint value, string valueName)
    {
        int result;
        unchecked
        {
            result = (int)value;
        }
        if (value > int.MaxValue)
        {
            Console.WriteLine($"Warning: {valueName} is larger than int.MaxValue. Some data might not be readable.");
        }
        return result;
    }
    public void Dispose()
    {
        if (m_libHandle != IntPtr.Zero)
        {
            FreeLibrary(m_libHandle);
            m_libHandle = IntPtr.Zero;
        }
    }

    // Structure definitions
    [StructLayout(LayoutKind.Sequential)]
    private struct IMAGE_DOS_HEADER
    {
        public ushort e_magic;
        public ushort e_cblp;
        public ushort e_cp;
        public ushort e_crlc;
        public ushort e_cparhdr;
        public ushort e_minalloc;
        public ushort e_maxalloc;
        public ushort e_ss;
        public ushort e_sp;
        public ushort e_csum;
        public ushort e_ip;
        public ushort e_cs;
        public ushort e_lfarlc;
        public ushort e_ovno;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public ushort[] e_res1;
        public ushort e_oemid;
        public ushort e_oeminfo;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 10)]
        public ushort[] e_res2;
        public int e_lfanew;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IMAGE_NT_HEADERS64
    {
        public uint Signature;
        public IMAGE_FILE_HEADER FileHeader;
        public IMAGE_OPTIONAL_HEADER64 OptionalHeader;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IMAGE_FILE_HEADER
    {
        public ushort Machine;
        public ushort NumberOfSections;
        public uint TimeDateStamp;
        public uint PointerToSymbolTable;
        public uint NumberOfSymbols;
        public ushort SizeOfOptionalHeader;
        public ushort Characteristics;
    }

    

    [StructLayout(LayoutKind.Sequential)]
    private struct IMAGE_DATA_DIRECTORY
    {
        public uint VirtualAddress;
        public uint Size;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IMAGE_EXPORT_DIRECTORY
    {
        public uint Characteristics;
        public uint TimeDateStamp;
        public ushort MajorVersion;
        public ushort MinorVersion;
        public uint Name;
        public uint Base;
        public uint NumberOfFunctions;
        public uint NumberOfNames;
        public uint AddressOfFunctions;
        public uint AddressOfNames;
        public uint AddressOfNameOrdinals;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IMAGE_OPTIONAL_HEADER64
    {
        public ushort Magic;
        public byte MajorLinkerVersion;
        public byte MinorLinkerVersion;
        public uint SizeOfCode;
        public uint SizeOfInitializedData;
        public uint SizeOfUninitializedData;
        public uint AddressOfEntryPoint;
        public uint BaseOfCode;
        public ulong ImageBase;
        public uint SectionAlignment;
        public uint FileAlignment;
        public ushort MajorOperatingSystemVersion;
        public ushort MinorOperatingSystemVersion;
        public ushort MajorImageVersion;
        public ushort MinorImageVersion;
        public ushort MajorSubsystemVersion;
        public ushort MinorSubsystemVersion;
        public uint Win32VersionValue;
        public uint SizeOfImage;
        public uint SizeOfHeaders;
        public uint CheckSum;
        public ushort Subsystem;
        public ushort DllCharacteristics;
        public ulong SizeOfStackReserve;
        public ulong SizeOfStackCommit;
        public ulong SizeOfHeapReserve;
        public ulong SizeOfHeapCommit;
        public uint LoaderFlags;
        public uint NumberOfRvaAndSizes;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public IMAGE_DATA_DIRECTORY[] DataDirectory;
    }
}