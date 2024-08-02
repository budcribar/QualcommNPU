using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
    public class QnnFunctionPointers
    {
    
        public IntPtr ComposeGraphsFnHandle;
        public IntPtr FreeGraphInfoFnHandle;
        unsafe public QnnInterface_t QnnInterface;
    }
   

}
