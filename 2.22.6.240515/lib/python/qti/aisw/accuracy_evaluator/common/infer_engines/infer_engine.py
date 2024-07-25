##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
from abc import ABC
import os
import numpy as np


class InferenceEngine(ABC):
    """InferenceEngine class is an abstract class with implemented by different
    ML frameworks to do inference on a set of inputs."""

    def __init__(self, model, inputlistfile, output_path, multithread=False, input_info=None,
                 output_info=None, gen_out_file=None, extra_params=None, binary_path=None):
        self.model_path = model
        self.input_path = inputlistfile
        self.input_info = input_info
        self.output_info = output_info
        self.output_path = output_path
        self.multithread = multithread
        self.extra_params = extra_params
        self.gen_out_file = gen_out_file
        if binary_path:
            self.binary_path = binary_path
        else:
            self.binary_path = self.output_path + '/temp' if self.output_path else None

    def save_outputs_and_profile(self, output_names, outputs, iter, save_outputs, do_profile):
        """
        This method saves the output arrays in numpy format and do profiling
        Args:
            output_names              : list of model output names
            outputs                   : list of output numpy arrays
            iter                      : inference iteration
            save_outputs              : bool flag to save outputs
            do_profile                : bool flag to do profiling

        Returns:
            profile_data : dict mapping output names and its corresponding tuple of datatype,
            shape,min,max and median
        """
        profile_data = {}
        _paths = []
        for i, name in enumerate(output_names):
            if save_outputs:
                out_path = os.path.join(self.output_path, str(name) + '_' + str(iter) + '.raw')
                _paths.append(out_path)
                outputs[i].tofile(out_path)

            if do_profile:
                if (not outputs[i].size or outputs[i].dtype == bool):
                    profile_data[name] = (outputs[i].dtype, outputs[i].shape, outputs[i],
                                          outputs[i], outputs[i])
                else:
                    profile_data[name] = (outputs[i].dtype, outputs[i].shape,
                                          round(np.min(outputs[i]),
                                                3), round(np.max(outputs[i]),
                                                          3), round(np.median(outputs[i]), 3))
        return profile_data, _paths

    def execute(self):
        pass

    def validate(self):
        pass
