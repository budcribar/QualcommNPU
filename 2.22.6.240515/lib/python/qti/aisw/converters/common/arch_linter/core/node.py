# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

import qti.aisw.converters.common.arch_linter.core.utils as utils
import qti.aisw.converters.common.arch_linter.core.constants as const

class Node():
    def __init__(self, node_name, node_config):
        self.node_name = node_name
        self.node_config = node_config

    def get_io_info(self, name_category):
        names = self.node_config[name_category]
        ret = ""
        for n in names:
            ret += n + ":" + str(self.node_config[const.NODE_IO_TENSORS][n][const.CURRENT_DIMS]) + ", "
        return ret[:-2]

    def get_type(self):
        return self.node_config["type"]

    def get_name(self):
        return self.node_name

    def get_conv_padding(self):
        if self.is_conv():
            pad = self.node_config["tensor_params"]["pad_amount"]
            for key in pad:
                return pad[key]["data"]
        else:
            raise Exception("Not a convolution")

    def is_unary_elwise_op(self):
        return self.node_config["type"] in const.UNARY_ELWISE_OP

    def is_binary_elwise_op(self):
        return self.node_config["type"] in const.BINARY_ELWISE_OP

    def is_ternary_elwise_op(self):
        return self.node_config["type"] in const.TERNARY_ELWISE_OP

    def is_conv(self):
        return self.node_config["type"] in const.CONV_OP

    def conv_channel_less_than(self, min_channel):
        input_name = self.node_config[const.INPUT_NAMES][0]
        back_filled_in_dim = utils.back_fill_shape(self.node_config[const.NODE_IO_TENSORS][input_name][const.CURRENT_DIMS])
        if back_filled_in_dim[const.INDEX_CHANNEL] < min_channel:
            return True
        output_name = self.node_config[const.OUTPUT_NAMES][0]
        back_filled_out_dim = utils.back_fill_shape(self.node_config[const.NODE_IO_TENSORS][output_name][const.CURRENT_DIMS])
        if back_filled_out_dim[const.INDEX_CHANNEL] < min_channel:
            return True
        return False

    def conv_channel_mul_of(self, channel):
        names = [self.node_config[const.INPUT_NAMES][0], self.node_config[const.OUTPUT_NAMES][0]]
        for name in names:
            back_filled_shape = utils.back_fill_shape(self.node_config[const.NODE_IO_TENSORS][name][const.CURRENT_DIMS])
            if back_filled_shape[const.INDEX_CHANNEL] % channel != 0:
                return False
        return True

    def get_transpose_perm(self):
        return list(self.node_config[const.TENSOR_PARAMS][const.PERM].values())[0][const.DATA]

    def get_eltwise_params(self):
        res = []
        if const.SCALAR_PARAMS in self.node_config:
            if const.ELTWISE_TYPE in self.node_config[const.SCALAR_PARAMS]:
                res =list(self.node_config[const.SCALAR_PARAMS][const.ELTWISE_TYPE].values())
        return res

    def is_divide_by_const(self):
        if self.get_type() == 'ElementWiseDivide' or 'ElementWiseDivide' in self.get_eltwise_params():
            div_by_name = self.node_config[const.INPUT_NAMES][1]
            # QNN_TENSOR_TYPE_STATIC==4,QNN_TENSOR_TYPE_APP_WRITE==0
            if self.node_config[const.NODE_IO_TENSORS][div_by_name]["type"] in [0,4]:
                return True
        return False