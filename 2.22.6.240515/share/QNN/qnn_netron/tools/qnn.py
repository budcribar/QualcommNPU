# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys, os
import tarfile
from enum import Enum

import numpy as np

assert len(sys.argv) == 4, "Incorrect number of parameter passed for loading model bin. Excepted filename, name to extract and dataType. Got {}".format(sys.argv[1:])
line = sys.argv[1]
name = sys.argv[2]
dataType = sys.argv[3]
assert os.path.isfile(line.strip()), "Cannot open {} file".format(line)

class DataType(Enum):
    INT8 = 8
    INT16 = 22
    INT32 = 50
    INT64 = 100
    UINT8 = 264
    UINT16 = 278
    UINT32 = 306
    UINT64 = 356
    FLOAT16 = 534
    FLOAT32 = 562
    SFIXEDPOINT8 = 776
    SFIXEDPOINT16 = 790
    SFIXEDPOINT32 = 818
    UFIXEDPOINT8 = 1032
    UFIXEDPOINT16 = 1046
    UFIXEDPOINT32 = 1074

tar=tarfile.open(line.strip())
for member in tar.getmembers():
    if name == member.name:
        f = tar.extractfile(member)
        tarContent=''
        if "FLOAT32" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.float32)
            print(tarContent)
        elif "FLOAT16" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.float16)
            print(tarContent)
        elif "INT8" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.int8)
            print(tarContent)
        elif "INT16" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.int16)
            print(tarContent)
        elif "INT32" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.int32)
            print(tarContent)
        elif "INT64" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.int64)
            print(tarContent)
        elif "UINT8" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint8)
            print(tarContent)
        elif "SFIXEDPOINT8" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint8)
            print(tarContent)
        elif "UFIXEDPOINT8" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint8)
            print(tarContent)
        elif "UINT16" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint16)
            print(tarContent)
        elif "SFIXEDPOINT16" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint16)
            print(tarContent)
        elif "UFIXEDPOINT16" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint16)
            print(tarContent)
        elif "UINT32" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint32)
            print(tarContent)
        elif "SFIXEDPOINT32" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint32)
            print(tarContent)
        elif "UFIXEDPOINT32" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint32)
            print(tarContent)
        elif "UINT64" == dataType:
            tarContent = np.frombuffer(f.read(), dtype=np.uint64)
            print(tarContent)
        else:
            print("unknown data type: {}".format(dataType))

sys.exit()
tar.close()

