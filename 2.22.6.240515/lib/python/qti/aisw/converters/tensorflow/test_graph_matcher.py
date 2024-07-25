#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) 2016-2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import unittest
from nose.plugins.attrib import attr

from qti.aisw.converters.tensorflow.graph_matcher import(
    ConverterSequenceNode as Node,
    GraphMatcher,
    GraphSequence
)


class DummyGraph:
    def get_tensor_by_name(self, *args):
        pass

    def indexed_tensor_name(self, *args):
        pass


class GraphHelper:
    def __init__(self):
        self._graph = DummyGraph()

    def get_none_identity_input(self, *args):
        pass

    def evaluate_tensors_output(self, *args):
        return []


@attr(profile='ci')
class GraphMatcherTest(unittest.TestCase):

    def setUp(self):
        super(GraphMatcherTest, self).setUp()

    def test_graph_match_single_node(self):
        graph_sequence = GraphSequence([Node('A', 'Add')])
        graph_sequence.set_inputs('A', [])
        graph_sequence.set_outputs(['A'])
        #define GraphB
        graph = [Node('1', 'Add')]
        graph_helper = GraphHelper()
        matcher = GraphMatcher(graph, graph_helper)
        potential_map_matches = matcher.match_sequence(graph_sequence)
        self.assertEqual(1, len(potential_map_matches))
        self.assertEqual(potential_map_matches[0]['A'], graph[0])

    def test_graph_match_miso(self):
        graph_sequence = GraphSequence([Node('A', 'Add'),
                                        Node('B', '?'),
                                        Node('C', 'RELU')])
        graph_sequence.set_inputs('C', ['A', 'B'])
        graph_sequence.set_outputs(['C'])
        #define GraphB
        graph = GraphSequence([Node('1', 'Sub'),
                               Node('2', 'Add'),
                               Node('3', 'Conv2D'),
                               Node('4', 'RELU'),
                               Node('5', 'Add'),
                               Node('6', 'RELU')])
        graph.set_inputs('2', ['1'])
        graph.set_inputs('3', ['1'])
        graph.set_inputs('4', ['2', '3'])
        graph.set_inputs('5', ['4'])
        graph.set_inputs('6', ['4'])

        graph = list(graph.values())
        graph_helper = GraphHelper()
        matcher = GraphMatcher(graph, graph_helper)
        potential_map_matches = matcher.match_sequence(graph_sequence)
        self.assertEqual(1, len(potential_map_matches))
        self.assertEqual(3, len(potential_map_matches[0]))
        self.assertEqual(potential_map_matches[0]['A'], graph[1])
        self.assertEqual(potential_map_matches[0]['B'], graph[2])
        self.assertEqual(potential_map_matches[0]['C'], graph[3])

    def test_graph_match_simo(self):
        graph_sequence = GraphSequence([Node('A', ['Add']),
                                        Node('B', ['Sub']),
                                        Node('C', ['AddN'])])
        graph_sequence.set_inputs('B', ['A'])
        graph_sequence.set_inputs('C', ['A'])
        graph_sequence.set_outputs(['B', 'C'])
        #define Graph
        graph = GraphSequence([Node('1', ['AddN']),
                               Node('2', ['Sub']),
                               Node('3', ['Add']),
                               Node('4', ['AddN']),
                               Node('5', ['Sub']),
                               Node('6', ['RELU'])])
        graph.set_inputs('2', ['1'])
        graph.set_inputs('3', ['1'])
        graph.set_inputs('4', ['3'])
        graph.set_inputs('5', ['3'])
        graph.set_inputs('6', ['2', '4'])
        graph = list(graph.values())
        graph_helper = GraphHelper()
        matcher = GraphMatcher(graph, graph_helper)
        potential_map_matches = matcher.match_sequence(graph_sequence)
        self.assertEqual(1, len(potential_map_matches))
        self.assertEqual(3, len(potential_map_matches[0]))
        self.assertEqual(potential_map_matches[0]['A'], graph[2])
        self.assertEqual(potential_map_matches[0]['B'], graph[4])
        self.assertEqual(potential_map_matches[0]['C'], graph[3])

    def test_graph_match_simo_multiple_matches(self):
        graph_sequence = GraphSequence([Node('A', ['Add']),
                                        Node('B', ['Sub']),
                                        Node('C', ['AddN'])])
        graph_sequence.set_inputs('B', ['A'])
        graph_sequence.set_inputs('C', ['A'])
        graph_sequence.set_outputs(['B', 'C'])
        #define GraphB
        graph = GraphSequence([Node('1', ['AddN']),
                               Node('2', ['Sub']),
                               Node('3', ['Add']),
                               Node('4', ['AddN']),
                               Node('5', ['Sub']),
                               Node('6', ['RELU']),
                               Node('7', ['Add']),
                               Node('8', ['AddN']),
                               Node('9', ['Sub'])])
        graph.set_inputs('2', ['1'])
        graph.set_inputs('3', ['1'])
        graph.set_inputs('4', ['3'])
        graph.set_inputs('5', ['3'])
        graph.set_inputs('6', ['2', '4'])
        graph.set_inputs('7', ['5'])
        graph.set_inputs('8', ['7'])
        graph.set_inputs('9', ['7'])

        graph = list(graph.values())
        graph_helper = GraphHelper()
        matcher = GraphMatcher(graph, graph_helper)
        potential_map_matches = matcher.match_sequence(graph_sequence)
        self.assertEqual(2, len(potential_map_matches))
        self.assertEqual(3, len(potential_map_matches[0]))
        self.assertEqual(potential_map_matches[0]['A'], graph[2])
        self.assertEqual(potential_map_matches[0]['B'], graph[4])
        self.assertEqual(potential_map_matches[0]['C'], graph[3])
        self.assertEqual(potential_map_matches[1]['A'], graph[6])
        self.assertEqual(potential_map_matches[1]['B'], graph[8])
        self.assertEqual(potential_map_matches[1]['C'], graph[7])

    def test_graph_match_multiple_levels(self):
        graph_sequence = GraphSequence([Node('A', ['Add']),
                                        Node('B', ['Sub']),
                                        Node('C', ['AddN']),
                                        Node('D', ['RELU']),
                                        Node('E', ['Conv']),
                                        Node('F', ['Mul']),
                                        Node('G', ['?']),
                                        Node('H', ['?'])])
        graph_sequence.set_inputs('B', ['A', 'G', 'H'])
        graph_sequence.set_inputs('C', ['A'])
        graph_sequence.set_inputs('D', ['B', 'C'])
        graph_sequence.set_inputs('E', ['D'])
        graph_sequence.set_inputs('F', ['D'])
        graph_sequence.set_outputs(['E', 'F'])

        #define GraphB
        graph = GraphSequence([Node('1', ['Conv']),
                               Node('2', ['AddN']),
                               Node('3', ['Add']),
                               Node('4', ['Sub']),
                               Node('5', ['AddN']),
                               Node('6', ['RELU']),
                               Node('7', ['Conv']),
                               Node('8', ['Mul']),
                               Node('9', ['Mul']),
                               Node('10', ['Relu']),
                               Node('11', ['Add'])])
        graph.set_inputs('4', ['1', '2', '3'])
        graph.set_inputs('5', ['3'])
        graph.set_inputs('6', ['4', '5'])
        graph.set_inputs('7', ['6'])
        graph.set_inputs('8', ['6'])
        graph.set_inputs('9', ['7'])
        graph.set_inputs('10', ['7', '8'])
        graph.set_inputs('11', ['8'])
        graph = list(graph.values())
        graph_helper = GraphHelper()
        matcher = GraphMatcher(graph, graph_helper)
        potential_map_matches = matcher.match_sequence(graph_sequence)
        self.assertEqual(1, len(potential_map_matches))
        self.assertEqual(8, len(potential_map_matches[0]))
        self.assertEqual(potential_map_matches[0]['A'], graph[2])
        self.assertEqual(potential_map_matches[0]['B'], graph[3])
        self.assertEqual(potential_map_matches[0]['C'], graph[4])
        self.assertEqual(potential_map_matches[0]['D'], graph[5])
        self.assertEqual(potential_map_matches[0]['E'], graph[6])
        self.assertEqual(potential_map_matches[0]['F'], graph[7])

    def test_graph_match_multiple_levels_with_unknown_type_in_the_middle(self):
        graph_sequence = GraphSequence([Node('A', ['Add']),
                                        Node('B', ['Sub']),
                                        Node('C', ['AddN']),
                                        Node('D', ['?']),
                                        Node('E', ['Conv']),
                                        Node('F', ['Mul']),
                                        Node('G', ['?']),
                                        Node('H', ['?'])])

        graph_sequence.set_inputs('B', ['A', 'G', 'H'])
        graph_sequence.set_inputs('C', ['A'])
        graph_sequence.set_inputs('D', ['B', 'C'])
        graph_sequence.set_inputs('E', ['D'])
        graph_sequence.set_inputs('F', ['D'])
        graph_sequence.set_outputs(['E', 'F'])
        #define GraphB
        graph = GraphSequence([Node('1', ['Conv']),
                               Node('2', ['AddN']),
                               Node('3', ['Add']),
                               Node('4', ['Sub']),
                               Node('5', ['AddN']),
                               Node('6', ['RELU']),
                               Node('7', ['Conv']),
                               Node('8', ['Mul']),
                               Node('9', ['Mul']),
                               Node('10', ['Relu']),
                               Node('11', ['Add'])])
        graph.set_inputs('4', ['1', '2', '3'])
        graph.set_inputs('5', ['3'])
        graph.set_inputs('6', ['4', '5'])
        graph.set_inputs('7', ['6'])
        graph.set_inputs('8', ['6'])
        graph.set_inputs('9', ['7'])
        graph.set_inputs('10', ['7', '8'])
        graph.set_inputs('11', ['8'])
        graph = list(graph.values())
        graph_helper = GraphHelper()
        matcher = GraphMatcher(graph, graph_helper)
        potential_map_matches = matcher.match_sequence(graph_sequence)
        self.assertEqual(1, len(potential_map_matches))
        self.assertEqual(8, len(potential_map_matches[0]))
        self.assertEqual(potential_map_matches[0]['A'], graph[2])
        self.assertEqual(potential_map_matches[0]['B'], graph[3])
        self.assertEqual(potential_map_matches[0]['C'], graph[4])
        self.assertEqual(potential_map_matches[0]['D'], graph[5])
        self.assertEqual(potential_map_matches[0]['E'], graph[6])
        self.assertEqual(potential_map_matches[0]['F'], graph[7])
