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

from typing import Dict, List

import copy


@inference_engine_repository.register(cls_type=ComponentType.converter,
                                      framework=Framework.tensorflow,
                                      engine=Engine.SNPE,
                                      engine_version="1.22.2.233")
class SNPETensorflowConverter(SNPEConverter):
    def __init__(self, context):
        super(SNPETensorflowConverter, self).__init__(context)

    def format_input_tensors(self, input_tensors):  # type: (Dict[str][str]) -> Dict[str][str]
        formatted_input_tensors = {}
        sep = ':'
        for tensor, dim in input_tensors.items():
            formatted_input_tensors[tensor.split(sep)[0]] = dim

        return formatted_input_tensors

    def format_output_tensors(self, output_tensors):  # type: (List[str]) -> List[str]
        new_output_tensors = copy.deepcopy(output_tensors)
        sep = ':'
        new_output_tensors = [tensor.split(sep)[0] for tensor in new_output_tensors]
        return new_output_tensors
