using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SampleCSharpApplication
{
  
    public enum StatusCode
    {
        SUCCESS,
        FAILURE,
        FAIL_LOAD_BACKEND,
        FAIL_LOAD_MODEL,
        FAIL_SYM_FUNCTION,
        FAIL_GET_INTERFACE_PROVIDERS,
        FAIL_LOAD_SYSTEM_LIB
    }
}
