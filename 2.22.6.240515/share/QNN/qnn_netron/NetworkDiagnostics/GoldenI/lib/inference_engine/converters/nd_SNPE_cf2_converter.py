# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from lib.inference_engine import inference_engine_repository
from lib.utils.nd_constants import ComponentType, Framework, Engine
from lib.inference_engine.converters.nd_SNPE_converter import SNPEConverter


@inference_engine_repository.register(cls_type=ComponentType.converter,
                                      framework=Framework.caffe2,
                                      engine=Engine.SNPE,
                                      engine_version="1.22.2.233")
class SNPECaffe2Converter(SNPEConverter):
    def __init__(self, context):
        super(SNPECaffe2Converter, self).__init__(context)

    def format_input_tensors(self, input_tensors):
        return input_tensors

    def format_output_tensors(self, output_tensors):
        return output_tensors
