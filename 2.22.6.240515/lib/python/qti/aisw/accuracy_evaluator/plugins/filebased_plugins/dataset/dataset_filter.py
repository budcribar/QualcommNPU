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
import random
import errno

from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_dataset, DatasetPluginInputInfo, DatasetPluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class filter_dataset(qacc_dataset):

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):

        # local processed files
        processed_inputlist = None
        processed_calib = None

        # shuffle
        if d_in.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_RANDOM, False):
            processed_inputlist, processed_calib = self.shuffle_inputs(
                d_in, d_out, processed_inputlist, processed_calib)

        # limit inputs
        max_inputs = d_in.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_MAX_INPUTS, -1)

        if 0 == max_inputs:
            qacc_logger.error(
                "Plugin-[filter_dataset] has an invalid configuration [max_inputs = 0]")
            qacc_logger.error("please correct the config's yaml and run the tool again.")
            qacc_logger.info("qacc pipeline ended with failures")
            raise RuntimeError(
                "Plugin-[filter_dataset] has an invalid configuration [max_inputs = 0].please "
                "correct the config's yaml and run the tool again.")

        if -1 != max_inputs:
            self.limit_file(d_in, d_out, qcc.INPUT_LIST_FILE, max_inputs, processed_inputlist)

        # limit calib inputs
        max_calib = d_in.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_MAX_CALIB, -1)
        if -1 != max_calib:
            self.limit_file(d_in, d_out, qcc.CALIB_FILE, max_calib, processed_calib)

    def shuffle_inputs(self, d_in, d_out, processed_inputlist, processed_calib):

        if d_in.get_calibration_type() == qcc.CALIBRATION_TYPE_INDEX:
            qacc_file_logger.error('Input shuffling is skipped for calibration type {}'.format(
                qcc.CALIBRATION_TYPE_INDEX))
            return

        type = 'shuffle'  # set file generation type

        # shuffle input list file
        if d_in.get_inputlist_file():
            inputlist_file = self.get_updated_file(d_in, d_out, qcc.INPUT_LIST_FILE,
                                                   processed_inputlist)
            processed_inputlist = self.get_out_file_path(qcc.INPUT_LIST_FILE, d_out)
            self.generate_processed_file(inputlist_file, processed_inputlist, type)
            d_out.set_inputlist_file(processed_inputlist)
            d_out.set_status(qcc.STATUS_SUCCESS)  # set status success

        # shuffle calibration file
        if d_in.get_calibration_file():
            calib_file = self.get_updated_file(d_in, d_out, qcc.CALIB_FILE, processed_calib)
            processed_calib = self.get_out_file_path(qcc.CALIB_FILE, d_out)
            self.generate_processed_file(calib_file, processed_calib, type)
            d_out.set_calibration_file(processed_calib)
            d_out.set_status(qcc.STATUS_SUCCESS)  # set status success

        return processed_inputlist, processed_calib

    def generate_processed_file(self, original_file, processed_file, type, limit=None):
        try:
            current_file = original_file
            with open(original_file) as f:
                lines = f.readlines()

            if type == 'shuffle':
                random.shuffle(lines)
            elif type == 'limit':
                lines = lines[0:limit]

            current_file = processed_file
            with open(processed_file, "w") as f:
                f.writelines(lines)
        except FileNotFoundError as e:
            qacc_file_logger.error(f'{current_file} file not found')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), current_file)
        except Exception as e:
            raise ce.ConfigurationException(
                f'Error while reading the file: {current_file} error: {e}')

    def limit_file(self, d_in, d_out, file_type, limit, processed_file):
        original_file = self.get_updated_file(d_in, d_out, file_type, processed_file)
        processed_file = self.get_out_file_path(file_type, d_out)
        if file_type == qcc.INPUT_LIST_FILE:
            d_out.set_inputlist_file(processed_file)
        else:
            d_out.set_calibration_file(processed_file)

        if original_file:
            self.generate_processed_file(original_file, processed_file, 'limit', limit)
        d_out.set_status(qcc.STATUS_SUCCESS)  # set status success

    def get_out_file_path(self, type, d_out):
        if type == qcc.CALIB_FILE:
            return os.path.join(d_out.get_out_dir(), type)
        elif type == qcc.INPUT_LIST_FILE:
            return os.path.join(d_out.get_out_dir(), type)
        else:
            qacc_file_logger.error('Can not generate path for file type {}'.format(type))
            return None

    def get_updated_file(self, d_in, d_out, file_type, processed_file):
        if file_type == qcc.INPUT_LIST_FILE:
            din_file = d_in.get_inputlist_file()
        elif file_type == qcc.CALIB_FILE:
            din_file = d_in.get_calibration_file()
        return processed_file if (not d_out.get_status()) and processed_file else din_file
