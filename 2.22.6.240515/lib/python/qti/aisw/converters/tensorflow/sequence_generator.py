# =============================================================================
#
#  Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import tensorflow as tf
tf_compat_v1 = tf.compat.v1
import argparse

from collections import OrderedDict


class SequenceNode:
    def __init__(self, name, type):
        self.name = name
        self.type = type


class SequenceGenerator:
    def __init__(self, graph_path):
        self.graph_def = self.__import_from_frozen_graph(graph_path)
        self.id_op_map = dict()

    @classmethod
    def __import_from_frozen_graph(cls, graph_path):
        graph_def = tf_compat_v1.GraphDef()
        with open(graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    def generate(self, input_names, output_names):
        session = tf_compat_v1.Session(graph=tf_compat_v1.Graph())
        with session.graph.as_default():
            tf_compat_v1.import_graph_def(self.graph_def, name="")

            for o in session.graph.get_operations():
                self.id_op_map[o.name] = o

            output_ops = []
            for n in output_names:
                output_ops.append(self.id_op_map[n])
            input_ops = []
            for n in input_names:
                input_ops.append(self.id_op_map[n])

            path_nodes = self._resolve_ops_in_path(input_ops, output_ops)
            for n in list(path_nodes.values()):
                tf_op = self.id_op_map[n.name]
                if len(tf_op.inputs) == 0:
                    path_nodes.pop(n.name)

            path_edges = dict()
            for n in list(path_nodes.values()):
                tf_op = self.id_op_map[n.name]
                input_ops = []
                for input_tensor in tf_op.inputs:
                    input_node_name = input_tensor.op.name
                    if input_node_name not in path_nodes:
                        stub_node_name = 'stub_{}'.format(len(path_nodes))
                        path_nodes[input_node_name] = SequenceNode(stub_node_name, '?')
                    input_ops.append(path_nodes[input_node_name].name)

                path_edges[n] = input_ops

            lines = ['self.cell_sequence = GraphSequence([']
            for n in list(path_nodes.values()):
                if n.type in ['?', 'Const', 'Identity'] and n.name not in output_names:
                    node_type = 'NonConsumableConverterSequenceNode'
                else:
                    node_type = 'ConverterSequenceNode'
                lines.append('    {}(\'{}\', [\'{}\']),'.format(node_type, n.name, n.type))
            lines.append('])')

            for n, input_names in path_edges.items():
                node_inputs = ','.join(['\'{}\''.format(name) for name in input_names])
                lines.append('self.cell_sequence.set_inputs(\'{}\', [{}])'.format(n.name, node_inputs))

            output_root_names = ','.join(['\'{}\''.format(name) for name in output_names])
            lines.append('self.cell_sequence.set_outputs([{}])'.format(output_root_names))

            with open('sequence.txt', 'w') as f:
                for l in lines:
                    f.write(l + '\n')

    def _resolve_ops_in_path(self, input_ops, output_ops):
        path_nodes = []
        queue = output_ops[:]
        visited = set()
        while len(queue) > 0:
            current_op = queue.pop(0)
            if current_op in visited:
                continue
            visited.add(current_op)
            path_nodes.append(current_op)
            if current_op in input_ops:
                continue
            for input_tensor in current_op.inputs:
                input_op = input_tensor.op
                queue.append(self.id_op_map[input_op.name])

        ops = OrderedDict()
        for op in reversed(path_nodes):
            ops[op.name] = SequenceNode(op.name, op.type)
        return ops


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generates graph sequences for the Tensorflow converter.")

    required = parser.add_argument_group('required arguments')
    required.add_argument('-g', type=str, required=True,
                          help='Path to TensorFlow graph def (.pb saved as binary) or graph meta (.meta) file.')
    required.add_argument('-o', action='append', required=True,
                          help='The names of the cell_sequence output roots')
    required.add_argument('-i', action='append', required=False, default=[],
                          help='The names of the cell_sequence inputs')

    args = parser.parse_args()
    generator = SequenceGenerator(args.g)
    generator.generate(args.i, args.o)
