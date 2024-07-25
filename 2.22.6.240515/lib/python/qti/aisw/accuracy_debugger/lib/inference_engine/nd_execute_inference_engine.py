# =============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import traceback

from qti.aisw.accuracy_debugger.lib.inference_engine.nd_inference_engine_manager import InferenceEngineManager
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_symlink import symlink
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions


def exec_inference_engine(engine_type, sys_args, logger=None):

    args = InferenceEngineCmdOptions(engine_type, sys_args).parse()
    if logger is None:
        logger = setup_logger(args.verbose, args.output_dir)
    logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_STARTING'))
    symlink('latest', args.output_dir, logger)

    try:
        inference_engine_manager = InferenceEngineManager(args, logger=logger)
        inference_engine_manager.run_inference_engine()
        get_mapping_arg = Namespace(None, framework=args.framework, version=args.framework_version,
                                    model_path=args.model_path, output_dir=args.output_dir,
                                    engine=args.engine,
                                    golden_dir_for_mapping=args.golden_output_reference_directory)
        TensorMapping(get_mapping_arg, logger)
        logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_FINISHED'))
    except InferenceEngineError as e:
        raise InferenceEngineError("Inference failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))
