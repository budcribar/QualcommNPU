# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os

from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.inference_engine.configs.nd_inference_engine_config import InferenceEngineConfig
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.utils.nd_graph_structure import GraphStructure
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError


class InferenceEngineManager(object):

    def __init__(self, args, logger):
        self.inference_engine = args.engine
        self.output_dir = args.output_dir
        self.logger = logger
        self.logger.debug(args)
        self.config = InferenceEngineConfig(args, inference_engine_repository, logger)
        self.inference_engine = self.config.load_inference_engine_from_config()
        if not hasattr(args, 'model_name'):
            args.model_name = "model_graph_struct"
        self.graph_struct_name = args.model_name + "_graph_struct"

    def validation_run(self, predetermined_dlc_or_binary):
        pass

    def compute_intermediate_tensors(self, model, input_tensors, output_tensors, fine_grained=True):
        pass

    def run_inference_engine(self):
        self.inference_engine.run()
        if self.inference_engine.engine_type == Engine.SNPE.value or self.inference_engine.engine_type == Engine.QNN.value:
            try:
                graph_structure_json = os.path.join(self.output_dir,
                                                    self.graph_struct_name + '.json')
                GraphStructure.save_graph_structure(graph_structure_json,
                                                    self.inference_engine.get_graph_structure())
            except Exception as e:
                self.logger.error("Graph Structure Generation failed! Encountered Error: {}".format(
                    str(e)))
