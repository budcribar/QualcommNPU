#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) 2015-2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import tensorflow as tf
import numpy as np
import os
import tempfile

from mock import Mock
from mock import PropertyMock
from tensorflow.python.framework import graph_util
from qti.aisw.converters.tensorflow.common import LayerDescriptor
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.tf_to_ir import (
    GraphMatcher,
    GraphHelper,
    TFGraphBuilder
)
from qti.aisw.converters.tensorflow import tf_compat_v1
from functools import wraps


def within_session(create_graph_func):

    @wraps(create_graph_func)
    def within_session_d(test_func):
        @wraps(test_func)
        def wrapper(self):
            session = tf_compat_v1.Session(graph=tf.Graph())
            with session.graph.as_default():
                with session.as_default():
                    self.ops = create_graph_func(session)
                    session.run(tf.global_variables_initializer())
                    self.graph_helper = GraphHelper(session, model=None, ops=self.ops)
                    test_func(self)
        return wrapper
    return within_session_d


class TestUtils(object):
    @classmethod
    def create_test_graph(cls, input_node_name):
        with tf_compat_v1.Session(graph=tf.Graph()) as session:
            with session.graph.as_default():
                with session.as_default():
                    network = tflearn.input_data((3, 3, 3), name=input_node_name)
                    network = tflearn.fully_connected(network, 1)
                    graph_path = cls.store_session_graph(session)
        return graph_path, network.op.name

    @classmethod
    def store_session_graph(cls, session):
        tempdir = tempfile.mkdtemp()
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        saver.save(session, os.path.join(tempdir, 'model'))
        graph_path = os.path.join(tempdir, 'model')
        return graph_path + '.meta'

    @classmethod
    def freeze(cls, meta_graph_path, output_node_name):
        graph = tf.Graph()
        with tf_compat_v1.Session(graph=graph) as session:
            with session.graph.as_default():
                saver = tf.train.import_meta_graph(meta_graph_path)
                checkpoint = meta_graph_path.split('.meta')[0]
                saver.restore(session, checkpoint)

                graph_def = session.graph.as_graph_def(add_shapes=True)
                frozen_def = graph_util.convert_variables_to_constants(session, graph_def, [output_node_name])
                frozen_path = tempfile.TemporaryFile()
                with open(frozen_path.name, "wb") as f:
                    f.write(frozen_def.SerializeToString())
                return frozen_path.name

    @classmethod
    def mock_op(cls, name='someop', op_type='none'):
        mock = Mock(spec=tf.Operation, type=op_type, inputs=[])
        type(mock).name = PropertyMock(return_value=name)
        return mock

    @classmethod
    def mock_tensor(cls, name='sometensor'):
        mock = Mock(spec=tf.Tensor)
        mock.op = cls.mock_op()
        type(mock).name = PropertyMock(return_value=name)
        return mock

    @classmethod
    def scoped_ops(cls, ops, scope_name):
        scoped_ops = []
        scope_segments = scope_name.split('/')
        for op in ops:
            op_name = op.name.split('/')
            if op_name == scope_segments or op_name[:-1] == scope_segments:
                scoped_ops.append(op)
        return scoped_ops

    @classmethod
    def non_const_or_variable_ops(cls, ops):
        return [op for op in ops if op.type not in ['Const', 'Variable', 'VariableV2']]

    @classmethod
    def create_graph(cls, test_lambda):
        session = tf_compat_v1.Session(graph=tf.Graph())
        with session.graph.as_default():
            with session.as_default():
                return test_lambda(session)

    @classmethod
    def test_within_session(cls, test_lambda):
        session = tf_compat_v1.Session(graph=tf.Graph())
        with session.graph.as_default():
            with session.as_default():
                test_lambda(session)

    @classmethod
    def create_graph_matcher(cls, ops, graph_helper):
        graph_builder = TFGraphBuilder(ops)
        graph_builder.link_nodes()
        return GraphMatcher(graph_builder.nodes, graph_helper)

    @classmethod
    def get_tensor_shape(cls, tensor):
        return [int(dim) if dim is not None else 1 for dim in tensor.get_shape().as_list()]

    @classmethod
    def assert_tensor(cls, test, tensor, shape):
        test.assertIsInstance(tensor, np.ndarray)
        test.assertEqual(np.shape(tensor), shape)
