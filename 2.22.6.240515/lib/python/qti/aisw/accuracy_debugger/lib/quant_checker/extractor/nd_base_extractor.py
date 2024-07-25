# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import re
import json
import numpy as np
from pathlib import Path

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, Framework


class BaseExtractor:

    def __init__(self, args, quant_schemes_dir_map, logger=None):
        self._args = args
        self._quant_schemes_dir_map = quant_schemes_dir_map
        self._logger = setup_logger(args.verbose, args.output_dir) if logger is None else logger
        self._engine = self._engine_type = Engine(args.engine)
        self._input_file_names = []
        self._opMap = {}
        self._input_data = {}

    def getAllOps(self):
        return self._opMap

    def _extract_input_data(self):
        try:
            with open(self._args.input_list) as file:
                for line in file.readlines():
                    file_paths = line.rstrip()
                    file_names = []
                    for file_path in file_paths.split():
                        if file_path:
                            #input_list in case of multi input nodes contain ":=" string
                            #while single input model may not contain them
                            file_path = Path(file_path.split(":=")[1]) if ":=" in file_path else Path(file_path)
                            file_name = file_path.name.split('.')[0]
                            file_names.append(file_name)
                            self._input_data[file_name] = np.fromfile(file_path, dtype=np.float32)
                    if file_names:
                        self._input_file_names.append("_".join(file_names))
        except Exception as e:
            self._logger.info(
                get_message(
                    "Unable to open input list file, please check the file path! Exiting..."))
            exit(-1)

    def get_input_files(self):
        return self._input_file_names

    def getInputData(self):
        return self._input_data
