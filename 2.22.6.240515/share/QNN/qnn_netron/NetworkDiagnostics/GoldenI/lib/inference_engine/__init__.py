# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import importlib
import os

from .nd_registry import Registry

inference_engine_repository = Registry()

__all__ = ['inference_engine_repository', 'nd_registry']

packages = ["converters",
           "executors",
           "inference_engines"]

for package in packages:
    for module in [module_file.replace(".py", "")
        for module_file in os.listdir(os.path.join(os.path.dirname(__file__), package))
            if module_file.endswith('.py')]:
        importlib.import_module("." + package + "." + module, package=__package__)
