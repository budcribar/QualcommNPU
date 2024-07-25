# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import *


def run_simplifier(shared_queue, model, input_dims: dict = None, skip_optimization: bool = False):
    try:
        import onnxsim
        if input_dims:
            optimized_model, check_ok = onnxsim.simplify(model,
                                                         input_shapes=input_dims,
                                                         perform_optimization=not skip_optimization)
        else:
            optimized_model, check_ok = onnxsim.simplify(model,
                                                         perform_optimization=not skip_optimization)
        if check_ok:
            log_info("Successfully simplified the onnx model in child process")
            shared_queue.put(optimized_model)
        else:
            log_warning("Check failed. Couldn't simplify the onnx model")
    except ImportError as e:
        log_warning("Couldn't import onnx-simplifier. ({}: {})", type(e), str(e))
        log_warning("Install the onnx-simplifier for better model compatibility: \"pip3 install onnx-simplifier\"")
    except:
        log_warning("Model simplification failed with unexpected exception")


def run_shape_inference(shared_queue, model):
    try:
        from onnx import shape_inference
        from onnx import checker
        inferred_model = shape_inference.infer_shapes(model)
        checker.check_model(inferred_model)
        log_info("Successfully run shape inference in child process")
        shared_queue.put(inferred_model)
    except ImportError as e:
        log_warning("Couldn't import shape_inference from onnx. ({}: {})", type(e), str(e))
    except:
        log_warning("Onnx shape inference failed with unexpected exception")

