#!/usr/bin/python3
#==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
#  MIT License
#
#  Copyright (c) 2018 Mohsin Baig
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#==============================================================================

import librosa
import numpy as np
import os
import sys

def usage(msg='unknown error'):
    print('%s' % msg)
    exit(1)

def extract_mfcc(file_path, utterance_length):
    # load raw .wav data with librosa
    raw_w, sampling_rate = librosa.load(file_path, mono=True)
    # get mfcc features
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    if mfcc_features.shape[1] > utterance_length:
        mfcc_features = mfcc_features[:, 0:utterance_length]
    else:
        mfcc_features = np.pad(mfcc_features,
                               ((0, 0),
                               (0, utterance_length - mfcc_features.shape[1])),
                               mode='constant',
                               constant_values=0)
    return mfcc_features

if len(sys.argv) != 2:
    usage('Invalid argument.')

input_file = sys.argv[1]

if not os.path.isfile(input_file) or not os.access(input_file, os.R_OK):
    usage('Raw input file not accessible.')

# load input data and get mfcc features
mfcc_features = extract_mfcc(input_file, utterance_length=35)
mfcc_features = mfcc_features.reshape((1,mfcc_features.shape[0],mfcc_features.shape[1]))
# write into file
fid = open('input.raw', 'wb')
mfcc_features.tofile(fid)
fid.close()

print('Processed input data: %s.' % input_file)
