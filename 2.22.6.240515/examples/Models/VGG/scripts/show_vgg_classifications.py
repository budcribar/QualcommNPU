#!/usr/bin/python3
#==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

import argparse
import heapq
import numpy as np
import os
from scipy.special import softmax

def main():
    parser = argparse.ArgumentParser(description='Display vgg classification results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_list',
                        help='File containing input list used to generate the name of output_dir.', required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Output directory containing Result_X/prob.raw files matching input_list.', required=True)
    parser.add_argument('-l', '--labels_file',
                        help='Path to synset.txt', required=True)
    args = parser.parse_args()

    input_list = os.path.abspath(args.input_list)
    output_dir = os.path.abspath(args.output_dir)
    labels_file = os.path.abspath(args.labels_file)

    if not os.path.isfile(input_list):
        raise RuntimeError('input_list %s does not exist' % input_list)
    if not os.path.isdir(output_dir):
        raise RuntimeError('output_dir %s does not exist' % output_dir)
    if not os.path.isfile(labels_file):
        raise RuntimeError('labels_file %s does not exist' % labels_file)
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if len(labels) != 1000:
        raise RuntimeError('Invalid labels_file: need 1000 categories')
    with open(input_list, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]

    if len(input_files) <= 0:
        print('No files listed in input_files')
    else:
        print('Classification results')
        max_filename_len = max([len(file) for file in input_files])

        for idx, val in enumerate(input_files):
            cur_results_dir = 'Result_' + str(idx)
            cur_results_file = os.path.join(output_dir, cur_results_dir, 'vgg0_dense2_fwd.raw')
            if not os.path.isfile(cur_results_file):
                raise RuntimeError('missing results file: ' + cur_results_file)

            float_array = np.fromfile(cur_results_file, dtype=np.float32)
            if len(float_array) != 1000:
                raise RuntimeError(str(len(float_array)) + ' outputs in ' + cur_results_file)

            float_array = softmax(float_array)
            a = np.argsort(float_array)[::-1]
            for i in a[0:5]:
                print('probability=%f ; class=%s ' %(float_array[i],labels[i]))


if __name__ == '__main__':
    main()
