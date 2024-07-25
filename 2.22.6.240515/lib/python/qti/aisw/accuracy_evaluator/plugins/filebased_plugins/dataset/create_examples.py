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

import os
import pickle
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_dataset, DatasetPluginInputInfo, DatasetPluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class create_squad_examples(qacc_dataset):
    """Used for creating examples from squad json file."""

    def __init__(self):
        transformers = Helper.safe_import_package("transformers", "4.31.0")

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):
        transformers = Helper.safe_import_package("transformers", "4.31.0")
        input_path = d_in.get_annotation_file()
        calib_path = d_in.get_calibration_file()
        outdir = d_out.get_out_dir()

        version = d_in.get_param('squad_version', 1)
        if version == 1:
            processor = transformers.data.processors.squad.SquadV1Processor()
            print('Loaded Squadv1 processor')
        elif version == 2:
            processor = transformers.data.processors.squad.SquadV2Processor()
            print('Loaded Squadv2 processor')
        else:
            print('Only squad versions 1 and 2 are supported')
            d_out.set_status(qcc.STATUS_ERROR)
            return

        if input_path:
            data_file_path = create_squad_examples.save_examples(processor, input_path, outdir,
                                                                 'input_data')
            d_out.set_inputlist_file(data_file_path)
            d_out.inputlist_path_modified = True
        if calib_path:
            if calib_path.endswith('.txt'):
                json_path = open(calib_path, 'r').read().strip()
                calib_path = os.path.join(os.path.dirname(calib_path), json_path)
            data_file_path = create_squad_examples.save_examples(processor, calib_path, outdir,
                                                                 'calib_data')
            d_out.set_calibration_file(data_file_path)
            d_out.calibration_path_modified = True

        # Set output
        d_out.set_status(qcc.STATUS_SUCCESS)

    def save_examples(processor, data_path, outdir, data_type):
        outdir = os.path.join(outdir, data_type)

        print('Loading examples from ' + str(data_path))
        examples = processor.get_dev_examples('', filename=data_path)

        os.makedirs(outdir, exist_ok=True)
        print('Saving examples to ' + str(outdir))
        data_file_path = os.path.join(outdir, 'datafile.txt')
        txt = open(data_file_path, 'w')
        for idx, example in enumerate(examples):
            filename = 'example_' + str(idx) + '.pickle'
            output_path = os.path.join(outdir, filename)
            with open(output_path, 'wb') as example_file:
                pickle.dump(example, example_file)
            txt.write(os.path.join(data_type, filename) + '\n')
        txt.close()
        return data_file_path
