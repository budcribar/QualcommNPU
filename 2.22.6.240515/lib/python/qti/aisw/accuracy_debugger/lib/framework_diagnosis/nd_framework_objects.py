# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import os

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message


def get_available_frameworks():
    config_files = []
    frameworks = {}

    for dirpath, _, files in os.walk(os.path.join(os.path.dirname(__file__), "configs")):
        for file in files:
            if file.endswith(".json"):
                config_files.append(os.path.join(dirpath, file))

    if not config_files:
        raise FrameworkError(get_message("ERROR_FRAMEWORK_NO_VALID_CONFIGURATIONS"))

    for config_file in config_files:
        with open(config_file, 'r') as config_json:
            config = json.load(config_json)
            frameworks.update(config)

    return frameworks
