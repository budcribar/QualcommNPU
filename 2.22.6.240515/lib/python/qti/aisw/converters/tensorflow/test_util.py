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

from mock import patch
from mock import ANY

from qti.aisw.converters.tensorflow.util import VisitableGraph
from qti.aisw.converters.tensorflow.util import GraphVisitor
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import scoped_op_name
from qti.aisw.converters.tensorflow.util import OperationNotFoundError
from qti.aisw.converters.tensorflow.loader import ModelLoader
from qti.aisw.converters.tensorflow.testutils import TestUtils
from nose.plugins.attrib import attr


class DummyVisitor(GraphVisitor):
    def visit_operation(self, node_def):
        pass

    def visit_scope(self, scope, nodes_defs):
        pass


@attr(profile='ci')
class VisitableGraphTest(unittest.TestCase):

    @staticmethod
    def test_visit_empty_graph_does_not_call_visitor():
        visitable_graph = VisitableGraph([])
        visitor = DummyVisitor()
        with patch.object(visitor, 'visit_operation') as mock1:
            with patch.object(visitor, 'visit_scope') as mock2:
                visitable_graph.accept(visitor)
        mock1.assert_not_called()
        mock2.assert_not_called()

    @staticmethod
    def test_visit_graph_calls_visitor():
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with tf.Graph().as_default():
            with tf.Session() as session:
                loader = ModelLoader(logging.getLogger())
                model = loader.load(graph_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], session)
                operations = model.session.graph.get_operations()
                visitable_graph = VisitableGraph(operations)
                visitor = DummyVisitor()
                with patch.object(visitor, 'visit_operation') as mock1:
                    with patch.object(visitor, 'visit_scope') as mock2:
                        visitable_graph.accept(visitor)

                for op in operations:
                    mock1.assert_any_call(op)

                mock2.assert_any_call(ANY, ANY)

    def test_visit_graph_puts_node_on_single_scope(self):
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        with tf.Graph().as_default():
            with tf.Session() as session:
                loader = ModelLoader(logging.getLogger())
                model = loader.load(graph_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], session)
                operations = model.session.graph.get_operations()
                visitable_graph = VisitableGraph(operations)
                visitor = DummyVisitor()
                with patch.object(visitor, 'visit_scope') as mock:
                    visitable_graph.accept(visitor)

                for op in operations:
                    scopes = []
                    for args, kwargs in mock.call_args_list:
                        scope_name, scope_ops = args
                        if op in scope_ops:
                            scopes.append(scope_name)
                    self.assertEqual(1, len(scopes), msg='Operation not in any scope {}'.format(op.name))


@attr(profile='ci')
class GraphHelperTest(unittest.TestCase):

    def setUp(self):
        super(GraphHelperTest, self).setUp()
        graph_path, output_node_name = TestUtils.create_test_graph('input')
        session = tf.Session(graph=tf.Graph())
        with session.as_default():
            loader = ModelLoader(logging.getLogger())
            self.model = loader.load(graph_path, ['input/X'], ['1,3,3,3'], None, [output_node_name], session)

        self.graph = session.graph

    def test_indexed_tensor_name(self):
        self.assertEqual('input/X:0', GraphHelper.indexed_tensor_name('input/X:0'))
        self.assertEqual('input/X:0', GraphHelper.indexed_tensor_name('input/X'))

    def test_filter_ops_by_type(self):
        ops = GraphHelper.filter_ops_by_type(self.graph.get_operations(), 'MATMUL')
        self.assertIsNotNone(ops)
        self.assertEqual(len(ops), 1)

    def test_filter_single_op_by_type(self):
        ops = GraphHelper.filter_ops_by_type(self.graph.get_operations(), 'MATMUL')
        self.assertIsNotNone(ops)

    def test_filter_single_op_by_type_raise_when_none(self):
        with self.assertRaises(OperationNotFoundError) as error:
            _ = GraphHelper.filter_single_op_by_type(self.graph.get_operations(), 'CRAZYOP')

    def test_filter_single_op_by_type_raise_when_multiple(self):
        with self.assertRaises(OperationNotFoundError) as error:
            _ = GraphHelper.filter_single_op_by_type(self.graph.get_operations(), 'CONST')

    def test_scoped_op_name(self):
        self.assertEqual('a/b/b', scoped_op_name('a/b', TestUtils.mock_op('a/b')))
        self.assertEqual('a/b', scoped_op_name('a', TestUtils.mock_op('a/b')))

