using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace SampleCSharpApplication
{
    
    public class GraphInfoManager : IDisposable
    {
        private IntPtr GraphInfos { get; set; }
        public uint Count { get; private set; }

        public GraphInfoManager(IntPtr graphInfos, uint count) {
            GraphInfos = graphInfos;
            Count = count;    
        }

        public unsafe GraphInfo_t this[uint index]
        {
            get
            {
                if (index < 0 || index >= Count)
                    throw new IndexOutOfRangeException();

                IntPtr ptr = Marshal.ReadIntPtr(GraphInfos, (int)(index * IntPtr.Size));
                return Marshal.PtrToStructure<GraphInfo_t>(ptr);
            }
        }

        public unsafe void Dispose()
        {
            if (GraphInfos != IntPtr.Zero)
            {
                //FreeGraphInfos(m_graphInfos);
                GraphInfos = IntPtr.Zero;
            }
        }
    }
}
