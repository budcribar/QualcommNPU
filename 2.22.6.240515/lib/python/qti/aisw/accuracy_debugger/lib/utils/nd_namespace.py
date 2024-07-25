# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
from argparse import Namespace
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger


class Namespace(Namespace):

    def __init__(self, data=None, **kwargs):
        if data is not None:
            kwargs.update(data)
        super(Namespace, self).__init__(**kwargs)


def update_layer_options(parsed_args):
    """
    Returns string of comma separated layer output name for provided
    layer types.
    """
    s_utility = SnooperUtils.getInstance(parsed_args)
    logger = setup_logger(False, parsed_args.output_dir, disable_console_logging=True)
    args_copy = copy.deepcopy(parsed_args)
    add_layer_types = args_copy.__dict__.pop('add_layer_types')
    add_layer_outputs = args_copy.__dict__.pop('add_layer_outputs')
    skip_layer_types = args_copy.__dict__.pop('skip_layer_types')
    skip_layer_outputs = args_copy.__dict__.pop('skip_layer_outputs')
    model_traverser = s_utility.setModelTraverserInstance(logger, args_copy,
                                                            add_layer_outputs=add_layer_outputs,
                                                            add_layer_types=add_layer_types,
                                                            skip_layer_outputs=skip_layer_outputs,
                                                            skip_layer_types=skip_layer_types)
    add_layer_outputs = model_traverser.get_all_layers()
    return ','.join(add_layer_outputs)