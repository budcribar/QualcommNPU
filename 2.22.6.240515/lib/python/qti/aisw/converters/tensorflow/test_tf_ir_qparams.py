#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) 2019-2022, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import unittest
import numpy as np

import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
from qti.aisw.converters.tensorflow.layers.convolution import ConvolutionLayerResolver
from qti.aisw.converters.tensorflow.layers.batchnorm import BatchNormWithEltwiseLayerResolver
from qti.aisw.converters.tensorflow.layers.fake_quant import FakeQuantLayerResolver
from qti.aisw.converters.tensorflow.layers.convolution import ConvolutionLayerBuilder
from qti.aisw.converters.tensorflow.layers.fake_quant import FakeQuantLayerBuilder
from qti.aisw.converters.tensorflow.tf_to_ir import (
    ConverterContext,
    LayerDescriptor
)
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.common.converter_ir import op_graph, axis_tracker, op_policies
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.testutils import TestUtils
from qti.aisw.converters.tensorflow.testutils import within_session
from mock import Mock
from nose.plugins.attrib import attr
from mock import patch, Mock

from tflearn import layers
import tensorflow as tf


@attr(profile='ci')
class BaseIRQuantParamTest(unittest.TestCase):
    _CONV_OP_TYPE = 'Conv2D'
    _BN_OP_TYPE = 'BatchNorm'
    _FQ_OP_TYPE = 'FakeQuantWithMinMaxVars'
    _CONV_OUTPUT_NAME = 'conv_output'
    _BN_OUTPUT_NAME = 'bn_output'
    _FQ_OUTPUT_NAME = 'fq_output'
    _BN_SCALE = None
    _BN_BETA = None
    _INPUT_NAME = 'input'
    _INPUT_SHAPE = [4, 4, 3]

    @staticmethod
    def get_ops_from_graph(session):
        graph_def = session.graph.as_graph_def(add_shapes=True)
        return [session.graph.get_operation_by_name(node.name) for node in graph_def.node]

    @classmethod
    def _get_input_node(cls):
        return layers.core.input_data(shape=cls._INPUT_SHAPE, name=cls._INPUT_NAME)

    @classmethod
    def _create_base_conv_graph(cls, input_node, weights_init='uniform_scaling', bias=True, bias_init='zeros'):
        return layers.conv_2d(input_node, nb_filter=3, filter_size=3, name=cls._CONV_OUTPUT_NAME,
                              weights_init=weights_init, bias=bias, bias_init=bias_init)

    @classmethod
    def _create_base_bn_graph(cls, input_node):
        input_shape = TestUtils.get_tensor_shape(input_node)
        channel_shape = input_shape[-1]
        cls._BN_SCALE = tf.Variable(tf.ones(channel_shape), name='scale')
        cls._BN_BETA = tf.Variable(tf.zeros(channel_shape), name='beta')
        pop_mean = tf.Variable(tf.random_normal([channel_shape]), trainable=False, name='mean')
        pop_var = tf.Variable(tf.ones([channel_shape]), trainable=False, name='variance')
        epsilon = 1e-3
        return tf.nn.batch_normalization(input_node, pop_mean, pop_var, cls._BN_BETA, cls._BN_SCALE,
                                         epsilon, name=cls._BN_OUTPUT_NAME)

    @classmethod
    def _create_base_conv_bn_graph(cls, input_node):
        conv_node = cls._create_base_conv_graph(input_node)
        return cls._create_base_bn_graph(conv_node)

    @classmethod
    def create_conv_graph(cls, session):
        cls._create_base_conv_graph(cls._get_input_node())
        return cls.get_ops_from_graph(session)

    @classmethod
    def create_conv_bn_no_fq_graph(cls, session):
        cls._create_base_conv_bn_graph(cls._get_input_node())
        return cls.get_ops_from_graph(session)

    @classmethod
    def create_conv_param_fq_graph(cls, session):
        weights = tf.Variable(tf.random_normal([3, 3, 3, 3]), trainable=False, name='weights')  # shape based on filters
        fq_node = tf.fake_quant_with_min_max_vars(weights, min=-1, max=1, name=cls._FQ_OUTPUT_NAME)
        cls._create_base_conv_graph(cls._get_input_node(), weights_init=fq_node)
        return cls.get_ops_from_graph(session)

    @classmethod
    def create_conv_output_fq_graph(cls, session):
        conv_node = cls._create_base_conv_graph(cls._get_input_node(), bias=False)
        _ = tf.fake_quant_with_min_max_vars(conv_node, min=-1, max=1, name=cls._FQ_OUTPUT_NAME)
        return cls.get_ops_from_graph(session)

    @classmethod
    def create_conv_bn_outputs_fq_graph(cls, session):
        conv_node = cls._create_base_conv_graph(cls._get_input_node(), bias=False)
        fq_node = tf.fake_quant_with_min_max_vars(conv_node, min=-1, max=1, name=cls._FQ_OUTPUT_NAME)
        bn_node = cls._create_base_bn_graph(fq_node)
        _ = tf.fake_quant_with_min_max_vars(bn_node, min=-1, max=1, name=cls._FQ_OUTPUT_NAME)
        return cls.get_ops_from_graph(session)

    def setUp(self):
        super(BaseIRQuantParamTest, self).setUp()
        self.ops = []
        self.graph_helper = None
        self.mock_context = Mock(spec=ConverterContext)
        self.conv_resolver = ConvolutionLayerResolver()
        self.bn_resolver = BatchNormWithEltwiseLayerResolver()
        self.fq_resolver = FakeQuantLayerResolver()
        self.ir_graph_mock = Mock(spec=op_graph.IROpGraph)
        self.ir_graph = op_graph.IROpGraph(op_policies.ConversionNamePolicy, None, [], [], axis_tracker.AxisOrders.TF)


@attr(profile='ci')
class TFQuantParamTests(BaseIRQuantParamTest):

    def setup_mock(self, graph_helper, input_resolver, current_resolver, output_resolver):
        graph_matcher = TestUtils.create_graph_matcher(self.ops, graph_helper)
        descriptors = current_resolver.resolve_layer(graph_matcher, graph_helper)
        self.mock_context.get_input_layer_output_shape_for.return_value = self._INPUT_SHAPE
        self.mock_context.graph_helper.get_op_output_shape.return_value = self._INPUT_SHAPE

        if input_resolver:
            input_descriptors = input_resolver.resolve_layer(graph_matcher, graph_helper)
        else:
            input_descriptors = [Mock(spec=LayerDescriptor)]
            input_descriptors[0].get_output_names_for.return_value = ['in']

        if output_resolver:
            output_descriptors = output_resolver.resolve_layer(graph_matcher, graph_helper)
        else:
            output_descriptors = [Mock(spec=LayerDescriptor)]
            output_descriptors[0].get_output_names_for.return_value = ['out']

        return input_descriptors, descriptors, output_descriptors

    @within_session(BaseIRQuantParamTest.create_conv_graph)
    @patch('qti.aisw.converters.tensorflow.converter.GraphHelper')
    def test_no_quant_params(self, GraphHelperMock):
        input_descriptors, descriptors, output_descriptors = self.setup_mock(GraphHelperMock, None, self.conv_resolver, None)
        builder = ConvolutionLayerBuilder()
        builder.transform_layer(self.ir_graph_mock, self.mock_context, descriptors[0], input_descriptors, output_descriptors)
        self.assertFalse(self.ir_graph_mock.add_quantization_params.called, "Adding Qparams to IR called when no "
                                                                            "quantization param in graph.")

    @within_session(BaseIRQuantParamTest.create_conv_bn_no_fq_graph)
    @patch('qti.aisw.converters.tensorflow.converter.GraphHelper')
    def test_bn_params(self, GraphHelperMock):
        input_descriptors, descriptors,  output_descriptors = self.setup_mock(GraphHelperMock, None, self.conv_resolver, self.bn_resolver)
        builder = ConvolutionLayerBuilder()

        # test functions called for updating quantization parameters
        builder.transform_layer(self.ir_graph_mock, self.mock_context, descriptors[0], input_descriptors, output_descriptors)
        self.ir_graph_mock.add_quantization_params.assert_called_once()
        self.ir_graph_mock.merge_quantization_params.assert_called_once()

        # test actual insertion
        builder.transform_layer(self.ir_graph, self.mock_context, descriptors[0], input_descriptors, output_descriptors)
        conv_name = descriptors[0].layer_name
        self.assertIn(conv_name, self.ir_graph.quantization_params)
        conv_quant_params = self.ir_graph.get_layer_quantization_param(conv_name)
        self.assertIn("bn_params", conv_quant_params)
        self.assertIn("gamma", conv_quant_params['bn_params'])
        self.assertIn("beta", conv_quant_params['bn_params'])

    @within_session(BaseIRQuantParamTest.create_conv_param_fq_graph)
    @patch('qti.aisw.converters.tensorflow.converter.GraphHelper')
    def test_params_encodings(self, GraphHelperMock):
        graph_helper_mock = GraphHelperMock()
        graph_helper_mock.check_op_const_origin.side_effect = GraphHelper.check_op_const_origin
        input_descriptors, descriptors,  output_descriptors = self.setup_mock(graph_helper_mock, None, self.fq_resolver, self.conv_resolver)
        fq_builder = FakeQuantLayerBuilder()

        # test param encodings identified properly
        self.assertFalse(descriptors[0].is_act_quant, "weight encoding identified as output/activation")
        # test functions called for updating quantization parameters
        fq_builder.transform_layer(self.ir_graph_mock, self.mock_context, descriptors[0], input_descriptors, output_descriptors)
        self.ir_graph_mock.add_quantization_params.assert_called_once()
        op_name, kwargs = self.ir_graph_mock.add_quantization_params.call_args
        self.assertEquals(output_descriptors[0].layer_name, op_name[0])
        self.assertEquals("weights", kwargs['param_encodings']['name'])

    @within_session(BaseIRQuantParamTest.create_conv_output_fq_graph)
    @patch('qti.aisw.converters.tensorflow.converter.GraphHelper')
    def test_outputs_encodings(self, GraphHelperMock):
        graph_helper_mock = GraphHelperMock()
        graph_helper_mock.check_op_const_origin.side_effect = GraphHelper.check_op_const_origin
        input_descriptors, descriptors,  output_descriptors = self.setup_mock(graph_helper_mock, self.conv_resolver, self.fq_resolver, None)
        fq_builder = FakeQuantLayerBuilder()

        # test output encodings identified properly
        self.assertTrue(descriptors[0].is_act_quant, "output/activation encoding identified as param/weight.")
        # test functions called for updating quantization parameters
        fq_builder.transform_layer(self.ir_graph_mock, self.mock_context, descriptors[0], input_descriptors, output_descriptors)
        self.ir_graph_mock.add_quantization_params.assert_called_once()
        op_name, kwargs = self.ir_graph_mock.add_quantization_params.call_args
        self.assertEquals(input_descriptors[0].layer_name, op_name[0])
        self.assertEquals(descriptors[0].input_tensor_name, kwargs['output_encodings']['name'])

    @within_session(BaseIRQuantParamTest.create_conv_bn_outputs_fq_graph)
    @patch('qti.aisw.converters.tensorflow.converter.GraphHelper')
    def test_bn_folding_output_encodings(self, GraphHelperMock):
        graph_helper_mock = GraphHelperMock()
        graph_helper_mock.check_op_const_origin.side_effect = GraphHelper.check_op_const_origin
        conv_descriptors, bn_descriptors, fq_descriptors = self.setup_mock(graph_helper_mock, self.conv_resolver, self.bn_resolver, self.fq_resolver)
        fq_builder = FakeQuantLayerBuilder()
        conv_builder = ConvolutionLayerBuilder()

        # test functions called for updating quantization parameters
        fq_builder.transform_layer(self.ir_graph_mock, self.mock_context, fq_descriptors[0], conv_descriptors, bn_descriptors)
        fq_builder.transform_layer(self.ir_graph_mock, self.mock_context, fq_descriptors[1], bn_descriptors, [])
        conv_builder.transform_layer(self.ir_graph_mock, self.mock_context, conv_descriptors[0], [], bn_descriptors)
        self.assertEquals(3, self.ir_graph_mock.add_quantization_params.call_count)
        self.ir_graph_mock.merge_quantization_params.assert_called_once()

        # test actual merging
        conv_descriptors, bn_descriptors, fq_descriptors = self.setup_mock(graph_helper_mock, self.conv_resolver, self.bn_resolver, self.fq_resolver)
        fq_builder.transform_layer(self.ir_graph, self.mock_context, fq_descriptors[0], conv_descriptors, bn_descriptors)
        fq_builder.transform_layer(self.ir_graph, self.mock_context, fq_descriptors[1], bn_descriptors, [])
        conv_builder.transform_layer(self.ir_graph, self.mock_context, conv_descriptors[0], [], bn_descriptors)  # fold
        bn_name = bn_descriptors[0].layer_name
        conv_name = conv_descriptors[0].layer_name
        conv_quant_params = self.ir_graph.get_layer_quantization_param(conv_name)
        self.assertNotIn(bn_name, self.ir_graph.quantization_params)  # folded op should be removed
        self.assertEquals(conv_descriptors[0].output_names[0], bn_descriptors[0].output_names[0]) # conv output_encoding name should be replaced with bn due to merge
        self.assertEqual(conv_descriptors[0].output_names[0], conv_quant_params['output_encodings'][0]['name'])
