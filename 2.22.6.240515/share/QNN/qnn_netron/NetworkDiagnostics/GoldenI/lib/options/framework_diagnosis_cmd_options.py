# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.utils.nd_path_utility import get_absolute_path
from lib.options.cmd_options import CmdOptions
from lib.utils.nd_constants import Framework
from lib.utils.nd_exceptions import ParameterError

import argparse
import os
import json
from datetime import datetime


class FrameworkDiagnosisCmdOptions(CmdOptions):

    def __init__(self, args):
        super().__init__('framework_diagnosis', args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to generate intermediate tensors from an ML Framework.")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-f', '--framework', nargs='+', type=str.lower, required=True,
                            help='Framework type and version, version is optional. For example "caffe2 0.8.0".'
                            'currently supported frameworks are ["tensorflow","caffe","onnx","tflite"] '
                            'case insensitive but spelling sensitive')
        # TODO: add nargs='+' for model_path instead of taking ','
        required.add_argument('-m', '--model_path', type=str, required=True,
                            help='Path to the model file(s). caffe would require two inputs ie:'
                                '-m model.prototext,model.caffemodel')
        required.add_argument('-i', '--input_tensor', nargs="+", action='append', required=True,
                            help='The name, dimensions, raw data, and optionally data type of the network '
                                'input tensor(s) specified'
                                'in the format "input_name" comma-separated-dimensions path-to-raw-file, '
                                'for example: "data" 1,224,224,3 data.raw float32. Note that the quotes '
                                'should always be included in order to handle special characters, '
                                'spaces, etc. For multiple inputs specify multiple --input_tensor on '
                                'the command line like: --input_tensor "data1" 1,224,224,3 data1.raw '
                                '--input_tensor "data2" 1,50,100,3 data2.raw float32.')
        required.add_argument('-o', '--output_tensor', type=str, required=True, action='append',
                            help='Name of the graph\'s specified output tensor(s).')

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does '
                                    'not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                            help="Verbose printing")
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        if parsed_args.framework[0] == Framework.caffe.value or parsed_args.framework[0] == Framework.caffe2.value:
            paths = parsed_args.model_path.split(',', 2)
            if len(paths) != 2:
                raise ParameterError("caffe/caffe2 needs two inputs. error model: " + parsed_args.model_path)
            parsed_args.model_path = get_absolute_path(paths[0]) + "," + get_absolute_path(paths[1])
        else:
            parsed_args.model_path = get_absolute_path(parsed_args.model_path)
        # get framework and framework version if possible
        parsed_args.version = None
        if len(parsed_args.framework) > 2:
            raise ParameterError("Maximum two arguments required for framework.")
        elif len(parsed_args.framework) == 2:
            parsed_args.version = parsed_args.framework[1]

        parsed_args.framework = parsed_args.framework[0]

        # Parse input_tensor with default data type of float32
        for tensor in parsed_args.input_tensor:

            if len(tensor) < 3 or len(tensor) > 4:
                raise ParameterError("Invalid format for input_tensor, format as "
                                                "'--input_tensor \"INPUT_NAME\" INPUT_DIM INPUT_DATA [INPUT_TYPE].")
            tensor[2] = get_absolute_path(tensor[2])
            #TODO: add input_type checks
            input_type = tensor[3] if len(tensor) is 4 else "float32"
            del tensor[3:]
            tensor.append(input_type)
        return parsed_args
