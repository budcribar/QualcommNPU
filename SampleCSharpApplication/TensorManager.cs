using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
    
    public class TensorManager : IDisposable
    {
        private IntPtr Tensors { get; set; }
        public uint Count { get; private set; }

        public TensorManager(IntPtr tensors, uint count)
        {
            Tensors = tensors;
            Count = count;
        }

        public Qnn_Tensor_t this[uint index]
        {
            get
            {
                if (index < 0 || index >= Count)
                    throw new IndexOutOfRangeException();

                IntPtr ptr = new IntPtr(Tensors+ (int)(index * IntPtr.Size));
                return Marshal.PtrToStructure<Qnn_Tensor_t>(ptr);
            }
        }

        public void Dispose()
        {
            if (Tensors != IntPtr.Zero)
            {
                //FreeGraphInfos(m_graphInfos);
                Tensors = IntPtr.Zero;
            }
        }
    }
}
