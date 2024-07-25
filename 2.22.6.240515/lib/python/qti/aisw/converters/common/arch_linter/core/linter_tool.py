# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import glob
import os
import numpy as np
import pandas as pd
import itertools

from qti.aisw.converters.common.arch_linter.core.node_pool import NodePool
from qti.aisw.converters.common.arch_linter.core.node import Node
import qti.aisw.converters.common.arch_linter.core.constants as const

import qti.aisw.converters.common.arch_linter.core.utils as utils

from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common import ir_graph
OP_ADAPTER_CONV_OPS = [op_adapter.Conv2dOp.TRANSLATION_KEY,
                op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY,
                op_adapter.TransposeConv2dOp.TRANSLATION_KEY]

def parse_model(model, is_quantized):
    with open(model) as f:
        data = json.load(f)
    node_pool = NodePool()
    node_pool.set_converter_command(data["converter_command"])
    # activation bitwidth defaulted to 8
    if 'act_bw=16' in node_pool.get_converter_command() or not is_quantized:
        node_pool.set_elsize(2)
    tensors = data["graph"]["tensors"]
    nodes = data["graph"]["nodes"]
    io_tensors = {}
    for tensor_name in tensors:
        if tensors[tensor_name]["type"] == 0 or tensors[tensor_name]["type"] == 1:
            io_tensors[tensor_name] = tensors[tensor_name]
    node_pool.set_io_tensors(io_tensors)

    for node_name in nodes:
        node_config = nodes[node_name]
        node_config["node_io_tensors"] = {}
        for name in node_config["input_names"]:
            node_config["node_io_tensors"][name] = tensors[name]
        for name in node_config["output_names"]:
            node_config["node_io_tensors"][name] = tensors[name]
        node = Node(node_name, node_config)
        node_pool.add_node(node_name, node)
    return node_pool

def run_checks(node_pool, constraints_json, optimized_graph, backend):
    with open(constraints_json) as f:
        constraints = json.load(f)

    df = pd.DataFrame(columns=const.OUTPUT_CSV_HEADER, dtype=object)
    # graph check
    for constraint_id in constraints["graph"]:
        check = constraints["graph"][constraint_id]
        if eval(check["condition"]):
            tmp_df = dict.fromkeys(const.OUTPUT_CSV_HEADER, 'N/A')
            tmp_df[const.O_C_GRAPH_NODENAME] = "Graph"
            tmp_df[const.O_C_ISSUE] = eval(check["issue"])
            tmp_df[const.O_C_RECOMM] = eval(check["recomm"])
            df = pd.concat([df, pd.DataFrame([tmp_df])], ignore_index=True, sort=False)

    # single node check
    for node_name in node_pool.get_nodes():
        node = node_pool.get_node_by_name(node_name)
        for rule_id in constraints["single_node"]:
            check = constraints["single_node"][rule_id]
            if eval(check["condition"]):
                tmp_df = dict.fromkeys(const.OUTPUT_CSV_HEADER, 'N/A')
                tmp_df[const.O_C_GRAPH_NODENAME] = node.node_name
                tmp_df[const.O_C_INPUTS] = node.get_io_info(const.INPUT_NAMES)
                tmp_df[const.O_C_OUTPUTS] = node.get_io_info(const.OUTPUT_NAMES)
                tmp_df[const.O_C_ISSUE] = eval(check["issue"])
                tmp_df[const.O_C_RECOMM] = eval(check["recomm"])
                tmp_df[const.O_C_PARAMS] = node.node_config
                df = pd.concat([df, pd.DataFrame([tmp_df])], ignore_index=True, sort=False)

    # patterns check
    for rule_id in constraints["patterns"]:
        check = constraints["patterns"][rule_id]
        nodes = eval(check["condition"])
        for node in nodes:
            tmp_df = dict.fromkeys(const.OUTPUT_CSV_HEADER, 'N/A')
            tmp_df[const.O_C_GRAPH_NODENAME] = node.node_name
            tmp_df[const.O_C_INPUTS] = node.get_io_info(const.INPUT_NAMES)
            tmp_df[const.O_C_OUTPUTS] = node.get_io_info(const.OUTPUT_NAMES)
            tmp_df[const.O_C_ISSUE] = eval(check["issue"])
            if isinstance(nodes, dict) and nodes[node]:
                # return value for p-2 is a dict, nodes with value True needs to take recomm_alt
                tmp_df[const.O_C_RECOMM] = eval(check["recomm_alt"])
            else:
                # return value for p-1 is a list, and
                # nodes with value False from p-2's dict needs recomm
                tmp_df[const.O_C_RECOMM] = eval(check["recomm"])
            tmp_df[const.O_C_PARAMS] = node.node_config
            df = pd.concat([df, pd.DataFrame([tmp_df])], ignore_index=True, sort=False)

    return df

def is_start_conv_seq(opgraph_node, optimized_graph):
    parent = optimized_graph.get_buffer(opgraph_node.input_names[0]).producer
    if parent.op.type in OP_ADAPTER_CONV_OPS:
        return False
    if parent.op.type == ir_graph.IR_OP_NEURON:
        grand_parent = optimized_graph.get_buffer(parent.input_names[0]).producer
        if grand_parent.op.type in OP_ADAPTER_CONV_OPS:
            return False
    return True

def conv_seq_low_channel(node_pool, optimized_graph, backend, min_channel):
    """
    Identifies convolution [sequence] with low channel

    Walks through graph, if a convolution is found check if above minimum number of channels,
    if below minimum number of channels check for seqeuence or single node, add node to bad
    nodes list with flag for sequence or single node

    :param node_pool: set of nodes generated from json
    :param optimized_graph: intermediate representation of nodes
    :param backend:
    :param min_channel: minimum channel threshold
    :return: list of bad nodes
    """
    node_list = optimized_graph.list_nodes()
    bad_nodes = {}
    # Run through nodes...
    for opgraph_node in node_list:
        # If this node is convolution...
        if opgraph_node.op.type in OP_ADAPTER_CONV_OPS:
            node = node_pool.get_node_by_name(backend.sanitize_name(opgraph_node.op.name))
            # If this node has a low channel...
            if node.conv_channel_less_than(min_channel):
                # If sequence of convolutions...
                if not is_start_conv_seq(opgraph_node, optimized_graph):
                    bad_nodes[node] = True
                else:
                    bad_nodes[node] = False
    return bad_nodes

def conv_padding_in_middle(node_pool, optimized_graph, backend):
    node_list = optimized_graph.list_nodes()
    bad_nodes = []
    for opgraph_node in node_list:
        if opgraph_node.op.type in OP_ADAPTER_CONV_OPS:
            node = node_pool.get_node_by_name(backend.sanitize_name(opgraph_node.op.name))
            padding = node.get_conv_padding()
            if np.sum(padding) != 0 and not is_start_conv_seq(opgraph_node, optimized_graph):
                bad_nodes.append(node)
    return bad_nodes

def get_sn5_recomm(node):
    n_type = node.get_type()
    recomm = 'Try not to use {} if possible.'.format(n_type)
    if n_type == 'Transpose':
        recomm += ' If necessary, Reshape is better than Transpose.'
        perm = node.get_transpose_perm()
        if perm == [0,3,1,2] or perm == [0,2,3,1]:
            recomm += ' This Transpose is {}, which is probably alright.'.format(perm)
    return recomm

def get_html(df, html_out_path, model_json_path):
    html_out = const.HTML_STYLE
    html_out += '<h1>Architecture Checker Results</h1>'
    html_out += '<h2>Input Model: {}</h2>\n'.format(model_json_path)
    issues_count = df.shape[0]
    html_out += '<h2>Total Number of Potential Improvements: {}</h2>\n'.format(issues_count)

    for i in range(len(df[const.O_C_PARAMS])):
        cur_data = df[const.O_C_PARAMS][i]
        if cur_data != "N/A":
            formatted_data = json.dumps(cur_data, indent=4).replace("\n", "<br>" )
            new_data = '<pre>' + formatted_data + '</pre>'
            df.at[i, const.O_C_PARAMS] = new_data

    html_out += df.to_html(index=False, classes='table table-stripped', justify='center', escape=False)

    with open(html_out_path,'w') as f:
        f.write(html_out)
