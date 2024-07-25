#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) 2015-2020, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import unittest
import tensorflow as tf
import logging
import os
import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
from qti.aisw.converters.tensorflow.common import LayerResolver, LayerDescriptor
from qti.aisw.converters.tensorflow.layers import FullyConnectedLayerResolver
from qti.aisw.converters.tensorflow.testutils import TestUtils
from qti.aisw.converters.tensorflow.loader import ModelLoader
from qti.aisw.converters.tensorflow.tf_to_ir import ConverterContext
from qti.aisw.converters.tensorflow.tf_to_ir import ConverterError
from qti.aisw.converters.tensorflow.tf_to_ir import TopologyResolver
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.tf_to_ir import TFConverterFrontend
from nose.plugins.attrib import attr


@attr(profile='ci')
class ConverterContextTest(unittest.TestCase):

    def setUp(self):
        super(ConverterContextTest, self).setUp()
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        session = tf.Session(graph=tf.Graph())
        with session.as_default():
            loader = ModelLoader(logging.getLogger())
            self.model = loader.load(graph_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], session)

        self.topology_resolver = TopologyResolver()
        self.context = ConverterContext(converter_model=self.model,
                                        dnn_model=modeltools.Model(),
                                        graph_helper=GraphHelper(session, self.model, session.graph.get_operations()),
                                        topology_resolver=self.topology_resolver,
                                        logger=logging.getLogger())

    def test_get_tensor_shape(self):
        l1 = self.context.graph.get_operation_by_name('input/X')

        self.assertItemsEqual((1, 3, 3, 3), self.context.graph_helper.get_op_output_shape(l1))

    def test_get_previous_layers_output_ops_returns_empty_for_op_not_in_layer(self):
        l1 = self.context.graph.get_operation_by_name('input/X')
        l2 = self.context.graph.get_operation_by_name('FullyConnected/MatMul')

        with self.assertRaises(ConverterError):
            self.context._get_input_layer_output_op_for(l1)

        with self.assertRaises(ConverterError):
            self.context._get_input_layer_output_op_for(l2)

    def test_get_previous_layers_output_ops(self):
        l1 = self.context.graph.get_operation_by_name('input/X')
        l2 = self.context.graph.get_operation_by_name('FullyConnected/Reshape')
        descriptor1 = LayerDescriptor('l1', 'type', [l1])
        descriptor2 = LayerDescriptor('l2', 'type', [o for o in self.context.graph.get_operations() if o != l1])
        self.topology_resolver.resolve_topology([descriptor1, descriptor2])

        self.assertEqual(l1, self.context._get_input_layer_output_op_for(l2))

    def test_get_previous_layers_output_ops_for_input_layer_is_empty(self):
        l1 = self.context.graph.get_operation_by_name('input/X')
        descriptor1 = LayerDescriptor('l1', 'type', [l1])
        descriptor2 = LayerDescriptor('l2', 'type', [o for o in self.context.graph.get_operations() if o != l1])
        self.topology_resolver.resolve_topology([descriptor1, descriptor2])

        with self.assertRaises(ConverterError):
            self.context._get_input_layer_output_op_for(l1)

    def test_get_op_input_layer_output_op_raises_when_output_not_found(self):
        l2 = self.context.graph.get_operation_by_name('FullyConnected/MatMul')
        with self.assertRaises(ConverterError):
            self.context._get_input_layer_output_op_for(l2)

    def test_get_op_input_layers_output_tensor_when_output_not_found(self):
        l2 = self.context.graph.get_operation_by_name('FullyConnected/MatMul')
        descriptor = FullyConnectedLayerResolver.Descriptor('name', [l2], l2, l2, None, None)
        self.topology_resolver.resolve_topology([descriptor])
        self.assertEqual([], self.context._get_input_layers_output_tensors_for(l2))


@attr(profile='ci')
class DlcConverterTest(unittest.TestCase):

    @classmethod
    def create_test_graph_with_unconsumed_ops(cls, input_node_name):
        with tf.Session(graph=tf.Graph()) as session:
            with session.graph.as_default():
                with session.as_default():
                    network = tf.placeholder(tf.float32, (1, 3, 3, 3), name=input_node_name)
                    network = tf.layers.conv2d_transpose(network,
                                                         filters=1,
                                                         kernel_size=[2, 2],
                                                         strides=[1, 1],
                                                         kernel_initializer=tf.ones_initializer(),
                                                         padding='VALID')
                    graph_path = TestUtils.store_session_graph(session)
        return graph_path, network.op.name

    def setUp(self):
        super(DlcConverterTest, self).setUp()
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        session = tf.Session(graph=tf.Graph())
        with session.as_default():
            loader = ModelLoader(logging.getLogger())
            self.model = loader.load(graph_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], session)

    def tearDown(self):
        if os.path.exists('/tmp/model.dlc'):
            os.remove('/tmp/model.dlc')

    def test_convert(self):
        converter = TFConverterFrontend(self.model, strict_node_resolution=True)
        with self.model.session.as_default():
            converter.convert('/tmp/model.dlc', '', '')
        self.assertTrue(os.path.exists('/tmp/model.dlc'))

    def test_convert_with_quantization_off(self):
        converter = TFConverterFrontend(self.model, strict_node_resolution=True)
        with self.model.session.as_default():
            converter.convert('/tmp/model.dlc', '', '')
        self.assertTrue(os.path.exists('/tmp/model.dlc'))

    def test_convert_fails_given_unsupported_nodes_on_strict_mode(self):
        self.assert_strict_mode(True)

    def test_convert_given_unsupported_nodes_on_non_strict_mode(self):
        self.assert_strict_mode(False)

    def assert_strict_mode(self, strict_node_resolution):
        graph_path, output_node_name = self.create_test_graph_with_unconsumed_ops('input')
        session = tf.Session(graph=tf.Graph())
        with session.as_default():
            loader = ModelLoader(logging.getLogger())
            self.model = loader.load(graph_path, ['input'], ['1,3,3,3'], None, [output_node_name], session)

        converter = TFConverterFrontend(self.model, strict_node_resolution)
        with self.model.session.as_default():
            if strict_node_resolution:
                with self.assertRaises(ConverterError):
                    converter.convert('/tmp/model.dlc', '', '')
            else:
                converter.convert('/tmp/model.dlc', '', '')
        self.assertEqual(not strict_node_resolution, os.path.exists('/tmp/model.dlc'))
