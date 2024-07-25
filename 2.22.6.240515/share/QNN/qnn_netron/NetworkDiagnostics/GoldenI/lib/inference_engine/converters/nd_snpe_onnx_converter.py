# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.inference_engine import inference_engine_repository
from lib.utils.nd_constants import ComponentType, Framework, Engine
from lib.inference_engine.converters.nd_SNPE_converter import SNPEConverter


@inference_engine_repository.register(cls_type=ComponentType.converter,
                                      framework=Framework.onnx,
                                      engine=Engine.SNPE,
                                      engine_version="1.51.0")
class SNPEOnnxConverter(SNPEConverter):
    def __init__(self, context):
        super(SNPEOnnxConverter, self).__init__(context)

    def format_input_tensors(self, input_tensors):  # type: (Dict[str][str]) -> Dict[str][str]
        return input_tensors

    def format_output_tensors(self, output_tensors):  # type: (List[str]) -> List[str]
        return output_tensors