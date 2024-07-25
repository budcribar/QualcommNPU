# ==============================================================================
#
#  Copyright (c) 2020-2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

'''
This file contains things common to all blocks of the Converter Stack.
It will contain things common to the Frontend and Backend.
'''

import sys
from abc import ABC
from typing import Any, Text
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter
from qti.aisw.converters.common.utils.io_utils import check_validity
from qti.aisw.converters.common.utils.validation_utils import check_filename_encoding
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
import traceback


class ConverterBase(ABC):

    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(ConverterBase.ArgParser, self).__init__(
                formatter_class=CustomHelpFormatter, **kwargs)

            self.add_required_argument("--input_network", "-i", type=str,
                                       action=validation_utils.validate_pathname_arg(
                                           must_exist=True),
                                       help="Path to the source framework model.")

            self.add_optional_argument("--debug", type=int, nargs='?', default=-1,
                                       help="Run the converter in debug mode.")
            self.add_optional_argument('--keep_int64_inputs', action='store_true',
                                       help=argparse.SUPPRESS, default=False)

            # add command-line options for custom qnn converters
            custom_op_group = self.add_argument_group(
                title='Custom Op Package Options')

            custom_op_group.add_argument('--op_package_lib', '-opl', type=str, default="",
                                         help='Use this argument to pass an op package library for quantization. '
                                              'Must be in the form <op_package_lib_path:interfaceProviderName> and'
                                              ' be separated by a comma for multiple package libs')
            custom_op_group_me = custom_op_group.add_mutually_exclusive_group()

            custom_op_group_me.add_argument('-p', '--package_name', type=str,
                                            help='A global package name to be used for each node in the '
                                                 'Model.cpp file. Defaults to Qnn header defined package name')

            custom_op_group_me.add_argument("--op_package_config", "-opc", nargs='+',
                                            action=validation_utils.check_xml(),
                                            dest="custom_op_config_paths",
                                            help="Path to a Qnn Op Package XML configuration "
                                                 "file that contains user defined custom operations.")

            custom_op_group.add_argument("--converter_op_package_lib", "-cpl", type=str, default="",
                                         dest="converter_op_package_lib",
                                         help="Absolute path to converter op package library compiled by the OpPackage "
                                              "generator. Must be separated by a comma for multiple package libraries.\n"
                                              "Note: Order of converter op package libraries must follow the order of xmls.\n"
                                              "Ex1: --converter_op_package_lib absolute_path_to/libExample.so\n"
                                              "Ex2: -cpl absolute_path_to/libExample1.so,absolute_path_to/libExample2.so")

            self.add_mutually_exclusive_args(
                "op_package_config", "package_name")

    class ArgParserv2(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(ConverterBase.ArgParserv2, self).__init__(
                formatter_class=CustomHelpFormatter, **kwargs)

            self.add_required_argument("--input_network", "-i", type=str,
                                       action=validation_utils.validate_pathname_arg(
                                           must_exist=True),
                                       help="Path to the source framework model.")

            self.add_optional_argument("--debug", type=int, nargs='?', default=-1,
                                       help="Run the converter in debug mode.")
            self.add_optional_argument('--keep_int64_inputs', action='store_true',
                                       help=argparse.SUPPRESS, default=False)
            # add command-line options for custom qnn converters
            custom_op_group = self.add_argument_group(
                title='Custom Op Package Options')

            custom_op_group_me = custom_op_group.add_mutually_exclusive_group()

            custom_op_group_me.add_argument('--package_name', '-p', type=str,
                                            help='A global package name to be used for each node in the '
                                                 'Model.cpp file. Defaults to Qnn header defined package name')

            custom_op_group_me.add_argument("--op_package_config", "-opc", nargs='+',
                                            action=validation_utils.check_xml(),
                                            dest="custom_op_config_paths",
                                            help="Path to a Qnn Op Package XML configuration "
                                                 "file that contains user defined custom operations.")

            custom_op_group.add_argument("--converter_op_package_lib", "-cpl", type=str, default="",
                                         dest="converter_op_package_lib",
                                         help="Absolute path to converter op package library compiled by the OpPackage "
                                              "generator. Must be separated by a comma for multiple package libraries.\n"
                                              "Note: Order of converter op package libraries must follow the order of xmls.\n"
                                              "Ex1: --converter_op_package_lib absolute_path_to/libExample.so\n"
                                              "Ex2: -cpl absolute_path_to/libExample1.so,absolute_path_to/libExample2.so")

            self.add_mutually_exclusive_args("op_package_config", "package_name")

    def __init__(self, args, custom_op_factory: CustomOpFactory = None):
        """
        initialize base class for Converter Base.

        :param CustomOpFactory custom_op_factory: CustomOPFactory instance.
        :param args: argument pass to the converters.
        """
        self.input_model_path = args.input_network
        self.debug = args.debug
        self.custom_op_factory = custom_op_factory


        if hasattr(args, "converter_op_package_lib"):
            self.converter_op_package_lib = args.converter_op_package_lib
        else:
            self.converter_op_package_lib = None

        if hasattr(args, "package_name"):
            self.package_name = args.package_name
        else:
            self.package_name = None

        if hasattr(args, "custom_op_config_paths"):
            self.custom_op_config_paths = args.custom_op_config_paths
        else:
            self.custom_op_config_paths = None

        if hasattr(args, "op_package_lib"):
            self.op_package_lib = args.op_package_lib
        else:
            self.op_package_lib = None

        if self.debug is None:
            # If --debug provided without any argument, enable all the debug modes upto log_debug3
            self.debug = 3
        setup_logging(self.debug)
        self.keep_int64_inputs = args.keep_int64_inputs
