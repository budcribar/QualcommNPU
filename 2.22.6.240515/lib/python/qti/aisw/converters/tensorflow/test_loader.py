#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) 2015-2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
import tempfile
import unittest
from nose.plugins.attrib import attr

import tensorflow as tf

from qti.aisw.converters.tensorflow.loader import Model
from qti.aisw.converters.tensorflow.loader import ModelLoader
from qti.aisw.converters.tensorflow.testutils import TestUtils
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow import tf_compat_v1
from qti.aisw.converters.common.utils.converter_utils import *


import numpy as np
from tensorflow.python.saved_model import saved_model, save_options
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking
from tensorflow.python.framework import test_util
from tensorflow.python.training.tracking import tracking
from tensorflow.python.eager import def_function
from tensorflow.python.ops import variables
from tensorflow.python.lib.io import file_io

@attr(profile='ci')
# The unit test for TF2
class ModelLoaderTest(unittest.TestCase):

    def setUp(self):
        super(ModelLoaderTest, self).setUp()
        self.session = tf_compat_v1.Session(graph=tf.Graph())
        self._tempdir = ""
        setup_logging(0)

    def tearDown(self):
        super(ModelLoaderTest, self).tearDown()

    def get_temp_dir(self):
        """Returns a unique temporary directory for the test to use.

        If you call this method multiple times during in a test, it will return the
        same folder. However, across different runs the directories will be
        different. This will ensure that across different runs tests will not be
        able to pollute each others environment.
        If you need multiple unique directories within a single test, you should
        use tempfile.mkdtemp as follows:
        tempfile.mkdtemp(dir=self.get_temp_dir()):

        Returns:
        string, the path to the unique temporary directory created for this test.
        """
        if not self._tempdir:
            self._tempdir = tempfile.NamedTemporaryFile().name
        print("tempdir=", self._tempdir)
        return self._tempdir

    def _getMultiFunctionModel(self):
        class BasicModel(tracking.AutoTrackable):
            def __init__(self):
                self.y = None
                self.z = None

            @def_function.function
            def add(self, x):
                if self.y is None:
                    self.y = variables.Variable(2.)
                return x + self.y

            @def_function.function
            def sub(self, x):
                if self.z is None:
                    self.z = variables.Variable(3.)
                return x - self.z

        return BasicModel()

    def _getSimpleVariableModel(self):
        root = tracking.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        return root

    def _createV1SavedModel(self, shape):
        """Create a simple SavedModel."""
        saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                in_tensor_1 = tf.compat.v1.placeholder(
                    shape=shape, dtype=tf.float32, name='inputB')
                in_tensor_2 = tf.compat.v1.placeholder(
                    shape=shape, dtype=tf.float32, name='inputA')
                variable_node = tf.Variable(1.0, name='variable_node')
                out_tensor = in_tensor_1 + in_tensor_2 * variable_node
                inputs = {'x': in_tensor_1, 'y': in_tensor_2}
                outputs = {'z': out_tensor}
                sess.run(tf.compat.v1.variables_initializer([variable_node]))
                tf_compat_v1.saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir

    @test_util.run_v2_only
    def testV1SimpleModel(self):
        """Test a SavedModel."""
        with tf.Graph().as_default():
            saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

        # Convert model and ensure model is not None.
        ModelLoader(self.session, saved_model_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testTF1HubFormattedModel(self):
        """Test a TF1 hub formatted model."""
        print("====")
        print("testTF1HubFormattedModel")
        print("====")
        saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

        # TF1 hub model is based on V1 saved model and they omit the saved model
        # schema version setting.
        saved_model_proto = parse_saved_model(saved_model_dir)
        saved_model_proto.saved_model_schema_version = 0

        saved_model_pb_file_path = os.path.join(saved_model_dir, 'saved_model.pb')
        with file_io.FileIO(saved_model_pb_file_path, 'wb') as writer:
            writer.write(saved_model_proto.SerializeToString())

        # Convert model and ensure model is not None.
        ModelLoader(self.session, saved_model_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testConstModel(self):
        """Test a basic model with functions to make sure functions are inlined."""
        print("====")
        print("testConstModel")
        print("====")
        input_data = tf.constant(1., shape=[1])
        root = tracking.AutoTrackable()
        root.f = tf.function(lambda x: 2. * x)
        to_save = root.f.get_concrete_function(input_data)

        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)

        # Convert model and ensure model is not None.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testVariableModel(self):
        """Test a basic model with Variables with saving/loading the SavedModel."""
        print("====")
        print("testVariableModel")
        print("====")
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1., shape=[1])
        to_save = root.f.get_concrete_function(input_data)

        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)

        # Convert model and ensure model is not None.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testSignatures(self):
        """Test values for `signature_keys` argument."""
        print("====")
        print("testSignatures")
        print("====")
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1., shape=[1])
        to_save = root.f.get_concrete_function(input_data)

        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)

        # Convert model with invalid `signature_keys`.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", 'INVALID')

        # Convert model with empty `signature_keys`.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testMultipleFunctionModel(self):
        """Convert multiple functions in a multi-functional model."""
        print("====")
        print("testMultipleFunctionModel")
        print("====")
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1., shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        sub_func = root.sub.get_concrete_function(input_data)

        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'add': add_func, 'sub': sub_func})

        # Try converting multiple functions.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testNoConcreteFunctionModel(self):

        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir)

        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testKerasSequentialModel(self):
        """Test a simple sequential tf.Keras model."""
        print("====")
        print("testKerasSequentialModel")
        print("====")
        input_data = tf.constant(1., shape=[1, 1])

        x = np.array([[1.], [2.]])
        y = np.array([[2.], [4.]])

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x, y, epochs=1)

        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(model, save_dir)

        # Convert model and ensure model is not None.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")

    @test_util.run_v2_only
    def testGraphDebugInfo(self):
        """Test a SavedModel has debug info captured."""
        print("====")
        print("testGraphDebugInfo")
        print("====")
        input_data = tf.constant(1., shape=[1])
        root = tracking.AutoTrackable()
        root.f = tf.function(lambda x: 2. * x)
        to_save = root.f.get_concrete_function(input_data)
        options = save_options.SaveOptions(save_debug_info=True)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save, options)

        # Convert model and ensure model is not None.
        ModelLoader(self.session, save_dir, [], [], [], [], "serve", "serving_default")


# The unit test for TF1.
class ModelLoaderV1Test(unittest.TestCase):

    def setUp(self):
        super(ModelLoaderTest, self).setUp()
        self.session = tf_compat_v1.Session(graph=tf.Graph())
        self._tempdir = ""
        setup_logging(0)

    def tearDown(self):
        super(ModelLoaderTest, self).tearDown()

    def test_raises_when_graph_does_not_exist(self):
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, '/tmp/nowhere', ['node'], ['3,3,3'], None, ['node'], self.session)

    def test_raises_when_invalid_graph_fle(self):
        invalid_graph = tempfile.NamedTemporaryFile()
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, invalid_graph.name, ['node'], ['3,3,3'], None, ['node'], self.session)

    def test_raises_when_invalid_graph_meta_fle(self):
        invalid_graph = tempfile.NamedTemporaryFile()
        meta_graph = os.path.join(os.path.dirname(invalid_graph.name), os.path.basename(invalid_graph.name) + '.meta')
        with open(meta_graph, mode='a'):
            with self.assertRaises(ConverterError) as error:
                ModelLoader(self.session, meta_graph, ['node'], ['3,3,3'], None, ['node'], self.session)

    def test_raises_when_output_node_is_invalid(self):
        graph_path, _ = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            with self.session.as_default():
                ModelLoader(self.session, graph_path, ['node'], ['3,3,3'], None, ['node'], self.session)

    def test_raises_when_input_node_is_invalid(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, graph_path, ['node'], ['3,3,3'], None, [output_node_name], self.session)

    def test_raises_when_input_dimensions_are_empty(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, graph_path, ['input/X'], [''], None, [output_node_name], self.session)

    def test_raises_when_graph_inputs_and_dimensions_dont_match(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, graph_path, ['input/X'], [], None, [output_node_name], self.session)

    def test_raises_when_input_dimensions_are_invalid(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, graph_path, ['input/X'], ['a,b'], None, [output_node_name], self.session)

    def test_raises_when_input_dimensions_missmatch(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, graph_path, ['input/X'], ['3,3'], None, [output_node_name], self.session)

    def test_load_model_from_meta(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        model = ModelLoader(self.session, graph_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], self.session)
        self.assertIsInstance(model, Model)

    def test_load_model_from_frozen(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        frozen_path = TestUtils.freeze(graph_path, output_node_name)
        model = ModelLoader(self.session, frozen_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], self.session)
        self.assertIsInstance(model, Model)

    def test_raises_when_graph_inputs_and_types_dont_match(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with self.assertRaises(ConverterError) as error:
            ModelLoader(self.session, graph_path, ['input/X'], ['3,3,3'], [], [output_node_name], self.session)

if __name__ == '__main__':
    unittest.main()
