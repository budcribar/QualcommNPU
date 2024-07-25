# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================



import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from lib.utils.nd_namespace import Namespace
from lib.utils.nd_path_utility import get_absolute_path
from lib.utils.nd_exceptions import ParameterError

class CmdOptions(ABC):

    def __init__(self, component, args, engine=None):
        self.component = component
        self.args = args
        self.engine = engine
        self.initialized = False

    def _get_configs(self,config_path):
        path = get_absolute_path(config_path)
        with open(path, 'r') as f:
            config_data = json.load(f)
        data = Namespace(config_data)
        return data

    def print_options(self, opts):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [output_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opts).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        use_config = False
        if not self.initialized: self.initialize()
        if len(self.args) == 1 and self.args[0] not in ['-h','--help'] :
            opts = self._get_configs(self.args[0])
            use_config = True
        else:
            opts, _ = self.parser.parse_known_args(self.args)
            self.print_options(opts)
        #TODO: parsing from configs may result in required argument not passed it and not catched
        # for example, output_tensor and if working_dir is empty
        # set working_dir and output directories
        if not hasattr(opts,'output_dirname') or opts.output_dirname == self.parser.get_default("output_dirname"):
            opts.output_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not hasattr(opts,'working_dir'): opts.working_dir = self.parser.get_default("working_dir")
        opts.working_dir = os.path.join(os.getcwd(), opts.working_dir)
        if os.path.isabs(opts.output_dirname):
            raise ParameterError("output_dirname should not be a absolute path.")
        output_dir = os.path.join(opts.working_dir, self.component, opts.output_dirname)
        if not os.path.isdir(opts.working_dir):
            os.makedirs(opts.working_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not use_config:
            self.save_options(opts, os.path.join(output_dir, self.component+'_options.json'))
        opts.output_dir=output_dir
        return self.verify_update_parsed_args(opts)

    def initialize_ann(self):
        pass

    @staticmethod
    def save_options(opts, jsonFile):
        """
        saves the opts into a config.json
        """
        with open(jsonFile, 'w') as json_file:
            json.dump(vars(opts), json_file, indent=4)

    @abstractmethod
    def verify_update_parsed_args(self, parsed_args):
        pass

    @abstractmethod
    def initialize(self):
        pass
