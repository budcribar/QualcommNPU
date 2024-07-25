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
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_dataset, DatasetPluginInputInfo, DatasetPluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class split_txt_data(qacc_dataset):
    """Used for saving individual txt files for each line present in the given
    input txt file."""

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):

        input_path = d_in.get_inputlist_file()
        outdir = d_out.get_out_dir()

        data_file_path = split_txt_data.save_data(input_path, outdir, 'input_data')
        d_out.set_inputlist_file(data_file_path)
        d_out.inputlist_path_modified = True

        # Set output
        d_out.set_status(qcc.STATUS_SUCCESS)

    def save_data(data_path, outdir, data_type):
        outdir = os.path.join(outdir, data_type)
        os.makedirs(outdir, exist_ok=True)

        qacc_file_logger.info('Loading data from ' + str(data_path))
        qacc_file_logger.info('Saving examples to ' + str(outdir))
        data_file_path = os.path.join(outdir, 'datafile.txt')
        txt = open(data_file_path, 'w')
        for idx, line in enumerate(open(data_path).readlines()):
            filename = str(idx) + '.txt'
            output_path = os.path.join(outdir, filename)
            with open(output_path, 'w') as f:
                f.write(line.strip())
            txt.write(os.path.join(data_type, filename) + '\n')
        txt.close()
        return data_file_path
