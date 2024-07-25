# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.arch_checker.modifier import ModelModifier
from qti.aisw.dlc_utils import modeltools
from qti.aisw.converters.qnn_backend.ir_to_dlc import DLCBackend as NativeBackend
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter
import qti.aisw.arch_checker.constants as const


class SnpeModifier(ModelModifier):
    def __init__(self, c_ir_graph, constraints_json, logger, query, df):
        super(SnpeModifier, self).__init__(c_ir_graph, constraints_json, logger, query, df)

    def get_header_name(self):
        layer_name = const.O_C_GRAPH_LAYERNAME
        return layer_name

    def save_modified_graph(self, out_file_dlc, converter_command=""):
        if len(self.query) == 0 or self.query == "show":
            # no need to save dlc in case of show
            return True

        try:
            # arguments for IrDlcSerializer: output_path, copyright_str, model_version, converter_command
            dlc_serializer = modeltools.IrDlcSerializer(out_file_dlc, "", "", "", converter_command)
            dlc_serializer.initialize()
            dlc_serializer.serialize(self.c_ir_graph)
            dlc_serializer.finish()
            self.logger.info("Saved modified dlc at: " + out_file_dlc)
            self.logger.info("Successfully applied all possible modifications!")
            self.logger.warning("Note: The modified model is to help visualize the modifications applied for specific rule and the graph would require retraining.")
            return True
        except:
            return False