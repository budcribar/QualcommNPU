#!/usr/bin/python3
#==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

import numpy as np
import os
import sys

def usage(msg='unknown error'):
    print('%s' % msg)
    exit(1)

if len(sys.argv) != 2:
    usage('Invalid argument.')

raw_output_file = sys.argv[1]

if not os.path.isfile(raw_output_file) or not os.access(raw_output_file, os.R_OK):
    usage('Raw output file not accessible.')

# load floats from file
float_array = np.fromfile(raw_output_file, dtype=np.float32)

if len(float_array) != 10:
    usage('Cannot read 10 floats from raw output file.')

max_prob = float_array[0]
max_idx = 0

# print out index and value pair, saving index with highest value
for i in range(len(float_array)):
    prob = float_array[i]
    if prob >= max_prob:
        max_prob = prob
        max_idx = i
    print(' %d : %f' % (i, float_array[i]))

print('Classification Result: Class %d.' % max_idx)
