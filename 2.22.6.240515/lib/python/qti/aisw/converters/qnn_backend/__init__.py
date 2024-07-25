# ==============================================================================
#
#  Copyright (c) 2022-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import sys
import platform

if platform.system() == "Linux":
    if platform.machine() == "x86_64":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-x86_64'))
    else:
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))
elif platform.system() == "Windows":
    if "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "ARMv8" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), cpu_isa))
else:
    raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))

try:
    from qti.aisw.converters.common import ir_graph
    from qti.aisw.converters.common import qnn_ir
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        import libPyQnnDefinitions36 as qnn_definitions
    elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
        import libPyQnnDefinitions38 as qnn_definitions
    else:
        import libPyQnnDefinitions as qnn_definitions
except ImportError as e:
    try:
        from qti.aisw.converters.common import ir_graph
        from qti.aisw.converters.common import qnn_ir
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            from . import libPyQnnDefinitions36 as qnn_definitions
        elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
            from . import libPyQnnDefinitions38 as qnn_definitions
        else:
            from . import libPyQnnDefinitions as qnn_definitions
    except ImportError:
        try:
            import libPyQnnDefines as qnn_definitions
        except ImportError:
            raise ImportError(e)
