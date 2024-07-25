# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import traceback
import os
import pandas as pd

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_tensor_paths
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import load_inputs, read_json
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import permute_tensor_data_axis_order
from qti.aisw.accuracy_debugger.lib.tensor_inspection.tensor_inspection_runner import TensorInspectionRunner
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_warning_message


class TensorInspector(object):

    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self._logger = logger
        self.args = args

    def run(self):

        try:
            golden_tensor_paths = get_tensor_paths(self.args.golden_output_reference_directory)
            inference_tensor_paths = get_tensor_paths(self.args.inference_results)

            inference_tensors = list(inference_tensor_paths.keys())

            mapping = {}
            if self.args.tensor_mapping and os.path.exists(self.args.tensor_mapping):
                with open(self.args.tensor_mapping) as tensor_mapping:
                    mapping = json.load(tensor_mapping)
            inference_to_golden_tensor_map = {
                inference:
                mapping[inference]
                if inference in mapping and mapping[inference] is not None else inference
                for inference in inference_tensors
            }

            qnn_model_tensors_json = None
            if self.args.qnn_model_json_path is not None:
                qnn_model_json = read_json(self.args.qnn_model_json_path)
                qnn_model_tensors_json = qnn_model_json['graph']['tensors']

            tensor_inspector = TensorInspectionRunner(self._logger)
            tensor_inspector_dir = os.path.join(self.args.output_dir, 'tensor_inspection')

            fields = ['Name', 'golden_min', 'golden_max', 'target_min', 'target_max']
            if self.args.target_encodings:
                fields.extend([
                    'calibrated_min', 'calibrated_max', '(target_min-calibrated_min)',
                    '(target_max-calibrated_max)'
                ])

            summary_df = pd.DataFrame(columns=fields)
            for inference_tensor in inference_tensors:
                golden_tensor_name = inference_to_golden_tensor_map[inference_tensor]
                if golden_tensor_name not in golden_tensor_paths.keys():
                    self._logger.warning(
                        get_warning_message("WARNING_VERIFIER_MISSING_GOLDEN_TENSOR_DATA")(
                            str(golden_tensor_name)))
                    continue

                golden_data = load_inputs(golden_tensor_paths[golden_tensor_name], "float32")
                inference_data = load_inputs(inference_tensor_paths[inference_tensor], "float32")

                # permute tensor if axis order is different
                if qnn_model_tensors_json is not None:
                    if golden_tensor_name in qnn_model_tensors_json:
                        tensor_dict = qnn_model_tensors_json[golden_tensor_name]
                        src_axis_format = tensor_dict['src_axis_format']
                        axis_format = tensor_dict['axis_format']
                        tensor_dims = tensor_dict['dims']
                        golden_data = permute_tensor_data_axis_order(src_axis_format, axis_format,
                                                                     tensor_dims, golden_data)

                result = tensor_inspector.run(inference_tensor, golden_data, inference_data,
                                              tensor_inspector_dir,
                                              target_encodings=self.args.target_encodings)
                summary_df = pd.concat([summary_df, pd.DataFrame([result])], ignore_index=True,
                                       sort=False)
            return summary_df

        except Exception as excinfo:
            traceback.print_exc()
            raise Exception("Encountered error: {}".format(str(excinfo)))
