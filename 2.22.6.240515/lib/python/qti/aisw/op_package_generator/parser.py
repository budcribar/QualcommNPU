# ==============================================================================
#
#  Copyright (c) 2019-2020, 2022-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.utils.argparser_util import *
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.op_package_generator.generator import QnnOpPackageGenerator


class CodegenParser(ArgParserWrapper):
    def __init__(self):
        super(CodegenParser, self).__init__(description="This tool generates a "
                                                        "QNN Op package from a "
                                                        "config file that describes the attributes of the package.")
        self.add_required_argument("--config_path", '-p', action='append',
                                   help="The path to your config file that defines your QNN Op package(s).")
        self.add_optional_argument("--debug", action="store_true", help="Returns debugging information from generating"
                                                                        " the package")
        self.add_optional_argument("--output_path", "-o",  help="Path where the package should be saved")
        self.add_optional_argument("-f", "--force-generation", action="store_true",
                                   help="This option will delete the entire existing package "
                                        "Note appropriate file permissions must be set to use this option.")
        self.add_optional_argument("--gen_cmakelists", action="store_true", help=argparse.SUPPRESS)
        self.add_optional_argument("--converter_op_package", "-cop", action="store_true",
                                   help="Generates the skeleton code for shape inference feature ")


class CodeGenerator(object):
    def __init__(self, parser=None, generator=None):
        self.__parser = CodegenParser() if parser is None else parser
        self.__args = self.__parser.parse_args()
        self.__generator = generator if generator else QnnOpPackageGenerator()
        self.config_path = self.__args.config_path
        self.debug = self.__args.debug
        self.output_path = self.__args.output_path
        self.force_generation = self.__args.force_generation
        self.gen_cmakelists = self.__args.gen_cmakelists
        try:
            self.converter_op_package = self.__args.converter_op_package
        except:
            self.converter_op_package = False
        if self.debug:
            setup_logging(0)
        else:
            setup_logging(-1)

    def setup(self):
        self.__generator.parse_config(self.config_path, self.output_path, self.converter_op_package)
        self.__generator.setup_file_paths(self.force_generation, self.gen_cmakelists)

    def finalize(self):
        self.__generator.implement_packages()
        if not self.__generator.generation_is_complete():
            log_error('One or more package(s) failed to finalize. Please run --debug for more detailed information.')
