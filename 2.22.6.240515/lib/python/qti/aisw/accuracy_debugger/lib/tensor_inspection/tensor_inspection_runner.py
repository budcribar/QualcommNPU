# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import sys
import traceback
import numpy as np
import pandas as pd
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python/'))
from qti.aisw.accuracy_debugger.lib.visualizer.nd_visualizers import Visualizers
from qti.aisw.accuracy_debugger.lib.verifier.nd_verifier_factory import VerifierFactory
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import VerifierError


class TensorInspectionRunner(object):

    def __init__(self, logger):
        # type: (Logger, namespace) -> None

        self._logger = logger

    def run(self, filename, golden_data, target_data, output_dir, target_encodings=None,
            verifiers=None):

        try:
            output_dir = os.path.join(output_dir, filename.replace('.raw', ''))
            os.makedirs(output_dir)

            if golden_data.shape != target_data.shape:
                self._logger.error(
                    f"Shapes of Golden tensor={golden_data.shape} and Target tensor={target_data.shape} are expected to be same."
                )
                return

            self._logger.debug(f"Inspecting tensor {filename} with shape {target_data.shape}")

            # flatten numpy arrays to 1D array
            golden_data, target_data = golden_data.flatten(), target_data.flatten()

            # dump raw data into csv files
            np.savetxt(os.path.join(output_dir, "golden_data.csv"), golden_data)
            np.savetxt(os.path.join(output_dir, "target_data.csv"), target_data)

            # dump plots
            dump_data = pd.DataFrame([])
            Visualizers.histogram_visualizer(golden_data, target_data,
                                             os.path.join(output_dir, "Histograms.png"))
            Visualizers.diff_visualizer(golden_data, target_data,
                                        os.path.join(output_dir, "Diff_plots.png"))
            Visualizers.cdf_visualizer(golden_data, target_data,
                                       os.path.join(output_dir, "CDF_plots.png"))

            result = {
                "Name": filename,
                "golden_min": golden_data.min(),
                "golden_max": golden_data.max(),
                "target_min": target_data.min(),
                "target_max": target_data.max(),
            }

            if verifiers:
                for verifier in verifiers:
                    verifier = verifier[0].split(',')
                    verifier_name = verifier[0]

                    verifier_config = {}
                    ret, verifier_config = VerifierFactory().validate_configs(
                        verifier_name, verifier[1:])
                    if not ret:
                        errormsg = str(
                            verifier_config['error']) if 'error' in verifier_config else ''
                        raise VerifierError("VerifierFactory config_verify error: " + errormsg)

                    verifier_obj = VerifierFactory().factory(verifier_name, verifier_config)
                    if verifier_obj is None:
                        raise VerifierError(
                            get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(verifier_name))
                    result[verifier_name] = verifier_obj.verify(filename, target_data.shape,
                                                                [golden_data], [target_data]).Metric

            # Analyze calibrated min/max when target encodings file is passed
            if target_encodings:
                # Load encoding file
                with open(target_encodings) as data:
                    encodings = json.loads(data.read())

                calibrated_min = calibrated_max = None
                encoding_name = filename.replace('.raw', '')
                if encoding_name not in encodings['activation_encodings'].keys():
                    self._logger.warning(f"Encoding not found for {encoding_name}")
                else:
                    calibrated_min = encodings['activation_encodings'][encoding_name][0].get('min')
                    calibrated_max = encodings['activation_encodings'][encoding_name][0].get('max')

                diff_min = diff_max = None
                if calibrated_min:
                    diff_min = result['target_min'] - calibrated_min
                if calibrated_max:
                    diff_max = result['target_max'] - calibrated_max

                result['calibrated_min'] = calibrated_min
                result['calibrated_max'] = calibrated_max
                result['(target_min-calibrated_min)'] = diff_min
                result['(target_max-calibrated_max)'] = diff_max

                # plot density graph highlighting calibrated min/max and target min/max
                if calibrated_min and calibrated_max:
                    Visualizers.distribution_visualizer(
                        target_data, os.path.join(output_dir,
                                                  "Distribution_min-max.png"), result['target_min'],
                        result['target_max'], result['calibrated_min'], result['calibrated_max'])

            return result
        except Exception as excinfo:
            traceback.print_exc()
            raise Exception("Encountered error: {}".format(str(excinfo)))
