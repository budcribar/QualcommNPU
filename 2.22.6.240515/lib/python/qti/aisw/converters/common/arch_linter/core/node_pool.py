# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from collections import OrderedDict
import qti.aisw.converters.common.arch_linter.core.constants as const

class NodePool():
    def __init__(self):
        self.nodes = OrderedDict()
        self.converter_command = ""
        self.io_tensors = {}
        self.elsize = 1

    def add_node(self, node_name, node):
        self.nodes[node_name] = node

    def set_io_tensors(self, tensors):
        self.io_tensors = tensors

    def set_converter_command(self, command):
        self.converter_command = command

    def get_converter_command(self):
        return self.converter_command

    def set_elsize(self, elsize):
        self.elsize = elsize

    def get_elsize(self):
        return self.elsize

    def get_nodes(self):
        return self.nodes

    def get_node_by_name(self, name):
        return self.nodes[name]
