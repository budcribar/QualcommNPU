# ==============================================================================
#
#  Copyright (c) 2021, 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# import torch before qti.aisw.converters.xxxx modules
# TODO: Investigate the dependency causing the erroneous behavior
import torch

from qti.aisw.converters.relay.relay_to_ir import RelayConverterFrontend
from qti.aisw.converters.relay.importers.pytorch_importer import PyTorchImporter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrders
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings
from qti.aisw.converters.common.utils.converter_utils import log_debug1
import sys

def get_valid_type_name(type_name):
    """
    :param type_name: The type name of Pytorch source op
    :return type_name which is valid as a cpp identifier
    """
    # For Pytorch,
    #   The type name of Pytorch source op contains "::".
    #   Discard it along with the namespace because "::" has a different meaning in C++.
    return type_name.split('::')[-1]


def get_domain_and_valid_type_name(type_name):
    """
    :param type_name: The type name of Pytorch source op
    :return domain_name: The domain name
    :return type_name: A valid as a cpp identifier
    """
    names = type_name.split('::')
    return names[-2], names[-1]


class PyTorchConverterFrontend(RelayConverterFrontend):
    class ArgParser(RelayConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(PyTorchConverterFrontend.ArgParser, self).__init__(conflict_handler='resolve',
                                                                    parents=[PyTorchImporter.ArgParser()],
                                                                    **kwargs)
            self.add_optional_argument('--pytorch_custom_op_lib', "-pcl", type=str, default="",
                                        help="Absolute path to PyTorch library that contains the definition/declaration of\n"
                                             "the Custom Op. Must be separated by a comma for multiple custom op libraries.\n"
                                             "Example 1: --pytorch_custom_op_lib absolute_path_to/Example.so\n"
                                             "Example 2: -pcl absolute_path_to/Example1.so,absolute_path_to/Example2.so")

    def __init__(self, args, custom_op_factory=None, **kwargs):
        super(PyTorchConverterFrontend, self).__init__(args,
                                                       importer=PyTorchImporter(args, custom_op_factory=custom_op_factory),
                                                       axis_order=AxisOrders.PYTORCH,
                                                       **kwargs)
        if self.dump_io_config_template:
            self.dump_io_config_yaml_template()
            sys.exit(0)

        # set default input_layout if user doesn't specify it in command
        for input_name, input_shape in self.importer.shape_dict.items():
            if input_name not in self.graph.input_axis_formats:
                # handle time_series formats based on input enconding
                encodings = self.graph.get_input_encodings()
                input_in_encodings = [input_encoding[0] for input_encoding in encodings]
                if InputEncodings.OTHER in input_in_encodings and len(input_shape) in [3, 4]:
                    self.graph.input_axis_formats[input_name] = AxisTracker.AxisFormat.NONTRIVIAL
                else:
                    # Override time_series_format based on encoding
                    time_series_format = False
                    if InputEncodings.TIME_SERIES in input_in_encodings and len(input_shape) == 3:
                        time_series_format = True
                    self.graph.input_axis_formats[input_name] = self.graph.src_axis_order.get_default_input_axis_format(len(input_shape),
                                                                                                                        time_series_format=time_series_format)
            log_debug1("Set input axis-format for {} with {}".format(input_name, self.graph.input_axis_formats[input_name]))
