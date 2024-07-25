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

import argparse
import librosa
import numpy as np
import os
import random
import shutil
import tensorflow as tf
import tflearn

def split_train_test(src_path, train_path, test_path):
    os.makedirs(train_path, exist_ok =True)
    os.makedirs(test_path, exist_ok =True)
    for filename in os.listdir(src_path):
        first_split = filename.rsplit('_', 1)[1]
        second_split = first_split.rsplit('.', 1)[0]
        if int(second_split) <= 4:
            shutil.copyfile(src_path + '/' + filename, test_path + '/' + filename)
        else:
            shutil.copyfile(src_path + '/' + filename, train_path + '/' + filename)

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

def get_mfcc_batch(file_path, batch_size, utterance_length):
    files = os.listdir(file_path)
    feature_batch = []
    label_batch = []
    while True:
        # shuffle files
        random.shuffle(files)
        for file_name in files:
            # make sure raw files are in .wav format
            if not file_name.endswith('.wav'):
                continue
            # get mfcc features from file_path
            mfcc_features = extract_mfcc(file_path + file_name, utterance_length)
            # one-hot encoded label from 0-9
            label = np.eye(10)[int(file_name[0])]
            # label batch
            label_batch.append(label)
            # feature batch
            feature_batch.append(mfcc_features)
            if len(feature_batch) >= batch_size:
                # yield feature and label batches
                yield feature_batch, label_batch
                # reset batches
                feature_batch = []
                labels_batch = []

def create_model(learning_rate, training_epochs, training_batch):
    # create neural network with four fully connected layers and adam optimizer
    sp_network = tflearn.input_data([None, 20, 35])
    sp_network = tflearn.fully_connected(sp_network, 256, activation='relu')
    sp_network = tflearn.fully_connected(sp_network, 128, activation='relu')
    sp_network = tflearn.fully_connected(sp_network, 64, activation='relu')
    sp_network = tflearn.fully_connected(sp_network, 10, activation='softmax')
    sp_network = tflearn.regression(sp_network,
                                    optimizer='adam',
                                    learning_rate=learning_rate,
                                    loss='categorical_crossentropy')
    sp_model = tflearn.DNN(sp_network, tensorboard_verbose=0)
    return sp_model

def serialize_model():
    # use TensorFlow session to freeze tflearn checkpoint
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('model/spoken_digit.tflearn.meta')
        saver.restore(session, tf.train.latest_checkpoint('model/'))
        # the last layer name is FullyConnected_3/Softmax
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session,
            session.graph_def,
            ['FullyConnected_3/Softmax'])
        # serialize to protobuf file
        with open ('spoken_digit.pb', 'wb') as f:
            f.write(frozen_graph.SerializeToString())

def spoken_digit(learning_rate, training_epochs, training_batch):
    # split training and testing data
    split_train_test('free-spoken-digit-dataset/recordings/',
                     'train/',
                     'test/')
    print('Successfully split free-spoken-digit-dataset training/testing data.')
    # get training data
    train_batch = get_mfcc_batch('train/', training_batch*4, utterance_length=35)
    print('Training data created.')
    # create model
    sp_model = create_model(learning_rate, training_epochs, training_batch)
    # train model
    X_train, y_train = next(train_batch)
    X_test, y_test = next(train_batch)
    sp_model.fit(X_train,
                 y_train,
                 n_epoch=training_epochs,
                 validation_set=(X_test, y_test),
                 batch_size=training_batch)
    print('Optimization done.')
    # delete training ops
    with sp_model.net.graph.as_default():
        del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
    # save model in tflearn format
    sp_model.save('model/spoken_digit.tflearn')
    # save model in protobuf format
    serialize_model()
    print('Save frozen graph in spoken_digit.pb.')

def main():
    # argument parser
    parser = argparse.ArgumentParser(description='Create and Train Spoken Digit Neural Network model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate',
                        help='Learning rate.',
                        default=0.001)
    parser.add_argument('-epochs', '--training_epochs',
                        help='Training epochs.',
                        default=20)
    parser.add_argument('-batch', '--training_batch',
                        help='Training batch size.',
                        default=128)
    args = parser.parse_args()
    spoken_digit(float(args.learning_rate), int(args.training_epochs), int(args.training_batch))

if __name__ == '__main__':
    main()
