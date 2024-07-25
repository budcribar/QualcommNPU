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
#  Copyright (c) 2017 Rowel Atienza
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

import argparse
import collections
import numpy as np
import os

def create_training_data(filename):
    with open(filename) as f:
        data = f.readlines()
    data = [word.strip() for word in data]
    data = [word for i in range(len(data)) for word in data[i].split()]
    data = np.array(data)
    return data

def create_embedding(words):
    count = collections.Counter(words).most_common()
    embedding = dict()
    for word, _ in count:
        embedding[word] = len(embedding)
    reverse_embedding = dict(zip(embedding.values(), embedding.keys()))
    return embedding, reverse_embedding

def create_command(use_cpu, use_gpu, use_dsp, use_aip):
    cmd = []
    if use_cpu:
        print('Use device cpu.')
        cmd = ['adb', 'shell', '"cd', '/data/local/tmp/word_rnn', '&&', 'sh', 'word_rnn_adb.sh', 'cpu"']
    elif use_gpu:
        print('Use device gpu.')
        cmd = ['adb', 'shell', '"cd', '/data/local/tmp/word_rnn', '&&', 'sh', 'word_rnn_adb.sh', 'gpu"']
    elif use_dsp:
        print('Use device dsp.')
        cmd = ['adb', 'shell', '"cd', '/data/local/tmp/word_rnn', '&&', 'sh', 'word_rnn_adb.sh', 'dsp"']
    elif use_aip:
        print('Use device aip.')
        cmd = ['adb', 'shell', '"cd', '/data/local/tmp/word_rnn', '&&', 'sh', 'word_rnn_adb.sh', 'aip"']
    else:
        print('Use host cpu.')
        cmd = ['snpe-net-run', '--container', 'word_rnn.dlc', '--input_list', 'input_list.txt']
    return cmd

def Inference(training_filename, output_length, use_cpu, use_gpu, use_dsp, use_aip):
    # prepare training data
    training_data = create_training_data(training_filename)
    print('Load training file %s.' % (training_filename))
    embedding, reverse_embedding = create_embedding(training_data)
    print('Embedding created.')
    # number of units in rnn cell
    hidden = vocab_size = len(embedding.keys())
    # define user input length
    input_length = 4
    # create command
    cmd = create_command(use_cpu, use_gpu, use_dsp, use_aip)
    # launch the inference loop
    while True:
        print('Display word embedding keys:')
        print(embedding.keys())
        prompt = 'Please input %s words: ' % input_length
        sentence = input(prompt)
        words = sentence.strip().split(' ')
        if len(words) != input_length:
            print('Please input exactly %s words.' % (input_length))
            continue
        try:
            input_keys = [embedding[str(words[i])] for i in range(len(words))]
        except:
            print('Word not in embedding')
            continue
        for i in range(output_length):
            keys = np.reshape(np.array(input_keys, dtype=np.float32), [-1, input_length, 1])
            input_raw_file = open('./input.raw', 'wb')
            keys.tofile(input_raw_file)
            input_raw_file.close()
            # Run SNPE
            if os.system(' '.join(['rm', '-rf', 'output'])):
                print('Encounter error with remove output folder.')
                return
            if (use_cpu or use_gpu or use_dsp or use_aip) and \
                    os.system(' '.join(['adb', 'push', 'input.raw', '/data/local/tmp/word_rnn'])):
                print('Encounter error with adb push.')
                return
            if os.system(' '.join(cmd)):
                print('Encounter error with SNPE.')
                return
            if (use_cpu or use_gpu or use_dsp or use_aip) and \
                    os.system(' '.join(['adb', 'pull', '/data/local/tmp/word_rnn/output', 'output' ])):
                print('Encounter error with adb pull.')
                return
            pred = np.fromfile('./output/Result_0/rnn/basic_lstm_cell/Mul_11:0_all_time_steps.raw',
                               dtype=np.float32)[(vocab_size * (input_length - 1)) : ]
            pred_index = np.argmax(pred)
            sentence = '%s %s' % (sentence,reverse_embedding[pred_index])
            input_keys = input_keys[1:]
            input_keys.append(pred_index)
        print('Inference result: %s' % (sentence))

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Word-RNN model inference with SNPE.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--training_filename',
                        help='Training text filename.',
                        default='belling_the_cat.txt')
    parser.add_argument('-ol', '--output_length',
                        help='Word output length.',
                        default=32)
    parser.add_argument('--use_cpu', action='store_true',
                        help='Add this tag if using device cpu.',
                        default=False)
    parser.add_argument('--use_gpu', action='store_true',
                        help='Add this tag if using device gpu.',
                        default=False)
    parser.add_argument('--use_dsp', action='store_true',
                        help='Add this tag if using device dsp.',
                        default=False)
    parser.add_argument('--use_aip', action='store_true',
                        help='Add this tag if using device aip.',
                        default=False)
    args = parser.parse_args()
    Inference(args.training_filename, int(args.output_length),
              bool(args.use_cpu), bool(args.use_gpu), bool(args.use_dsp), bool(args.use_aip))

if __name__ == '__main__':
    main()
