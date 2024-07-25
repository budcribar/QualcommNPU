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
import random
import tensorflow as tf

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

def create_rnn(X, hidden, input_length):
    # input processing
    X = tf.reshape(X, [-1, input_length])
    X = tf.split(X,input_length,1)
    # tensorflow rnn module
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden)
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, X, dtype=tf.float32)
    return outputs[-1]

def word_rnn(log_dir, training_filename, learning_rate, training_iter):
    # create training log
    writer = tf.summary.FileWriter(log_dir)
    print('Training will be logged in %s.' % (log_dir))
    # prepare training data
    training_data = create_training_data(training_filename)
    print('Load training file %s.' % (training_filename))
    embedding, reverse_embedding = create_embedding(training_data)
    print('Embedding created.')
    # number of units in rnn cell
    hidden = vocab_size = len(embedding.keys())
    # define user input length
    input_length = 4
    # tensorflow graph input
    X = tf.placeholder('float', [None, input_length, 1])
    Y = tf.placeholder('float', [None, vocab_size])
    # create rnn model
    pred = create_rnn(X, hidden, input_length)
    # loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    # model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # initialize all variables
    init = tf.global_variables_initializer()
    # launch TF training session
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0,input_length+1)
        end_offset = input_length + 1
        writer.add_graph(session.graph)
        while step < training_iter:
            # generate a minibatch and add some randomness on selection process.
            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, input_length+1)
            input_keys = [[embedding[str(training_data[i])]] for i in range(offset, offset+input_length)]
            input_keys = np.reshape(np.array(input_keys), [-1, input_length, 1])
            output_onehot = np.zeros([vocab_size], dtype=float)
            output_onehot[embedding[str(training_data[offset+input_length])]] = 1.0
            output_onehot = np.reshape(output_onehot,[1,-1])
            session.run([optimizer, accuracy, cost, pred], feed_dict={X: input_keys, Y: output_onehot})
            if (step+1) % 1000 == 0:
                print('Iter= %s' % (step+1))
            step += 1
            offset += (input_length+1)
        print('Optimization done.')
        constant_graph = tf.graph_util.convert_variables_to_constants(session, session.graph_def, ['rnn/basic_lstm_cell/Mul_11'])
        with tf.gfile.FastGFile('word_rnn.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        print('Save frozen graph in word_rnn.pb.')

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Create and Train Word-RNN model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--log_dir',
                        help='Log directory.',
                        default='word_rnn_log')
    parser.add_argument('-t', '--training_filename',
                        help='Training text filename.',
                        default='belling_the_cat.txt')
    parser.add_argument('-lr', '--learning_rate',
                        help='Learning rate.',
                        default=0.001)
    parser.add_argument('-iter', '--training_iter',
                        help='Training iteration.',
                        default=5000)
    args = parser.parse_args()
    word_rnn(args.log_dir, args.training_filename, float(args.learning_rate), int(args.training_iter))

if __name__ == '__main__':
    main()
