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
import glob
import logging
import os

from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc


class DataSet:
    """QACC Dataset interface."""

    def __init__(self, dataset_config=None, input_list_file=None, caching=False):
        self.config = dataset_config
        if self.config:
            # input_list_file is updated with batched input file.
            self.input_list_file = self.config._inputlist_file
            # orig_input_list_file always points to the configured dataset inputlist_file
            self.orig_input_list_file = self.config._inputlist_file
            self.base_path = self.config._path
            self.base_path_inputlist = self.config._inputlist_path
            self.base_path_calibration = self.config._calibration_path
            self.total_entries = self.config._max_inputs
        elif input_list_file:
            # Used for stages other than preprocessing where dataset in not original dataset.
            self.input_list_file = input_list_file
            self.base_path = os.getcwd()
            self.base_path_inputlist = os.getcwd()
            self.base_path_calibration = os.getcwd()
            self.total_entries = sum(1 for input in open(input_list_file))

        self.cur_index = 0
        self.read_complete = False
        self._ds_generator = self.input_generator()
        self.in_mem_records = []
        self.caching = caching

    def input_generator(self):
        """Get generator."""

        # Read from cache if completed one iteration.
        if self.caching and self.read_complete:
            for idx in range(self.total_entries):
                record = self.in_mem_records[idx]
                self.cur_index = idx
                yield record
            return

        # First time read.
        for row in open(self.input_list_file, "r"):
            if self.config and (self.cur_index == self.config._max_inputs):
                # only for preprocessing datasets
                break

            self.cur_index += 1
            inputs = row.split(',')
            inputs = [x.strip() for x in inputs]
            # Ignore commented lines
            if inputs[0].startswith('#'):
                continue
            record = [os.path.join(self.base_path_inputlist, s) for s in inputs]
            if self.caching:
                self.in_mem_records.append(record)

            yield record

        # mark read completed.
        self.read_complete = True
        self.total_entries = self.cur_index
        self.cur_index = 0

    def load_dataset(self):
        """Caches the complete dataset input file in memory."""
        self.caching = True
        for _ in self._ds_generator:
            pass

    def reset(self):
        self.in_mem_records = []
        self.cur_index = 0
        self.read_complete = False

    def get_record(self, idx, num_records=1, last_batch=qcc.LAST_BATCH_NO_CHANGE):
        """Returns the record in nested list format.

        num_records governs the number of records to return starting
        from the idx.
        example:
        if num_records=2:
            [[input row at index idx],
            [input row at index idx + 1]]
        """
        if float(num_records).is_integer() and 0 < num_records and self.read_complete:
            # fetch records
            records = self.in_mem_records[idx * num_records:(idx * num_records) + num_records]

            is_last_batch_full = (len(records) == num_records)
            if is_last_batch_full or (last_batch == qcc.LAST_BATCH_NO_CHANGE):
                return records
            elif last_batch == qcc.LAST_BATCH_TRUNCATE:
                return []  #return empty list as last batch is truncated
            elif last_batch == qcc.LAST_BATCH_REPEAT_LAST_RECORD:
                # repeat last record
                repeated_record = [records[-1]] * (num_records - len(records))
                return records + repeated_record
        else:
            raise RuntimeError(
                'Number of records not a number or less than zero or dataset not ready to be '
                'accessed by record index.')

    def get_all_records(self, group=1, last_batch=qcc.LAST_BATCH_NO_CHANGE):
        """Returns the records in a list of nested list format.

        group governs the number of records in a group.
        example:
            if group=2 (each group has 2 elements):
            [
            [[input row at index 0], [input row at index 1]],
            [[input row at index 2], [input row at index 3]],
            ...
            [[input row at index N-1], [input row at index N]],
            ]
        """
        if self.read_complete:
            all_record_groups = []

            # iterate over records
            for idx in range(0, len(self.in_mem_records), group):

                # getting the desired record group
                record_group = self.in_mem_records[idx:idx + group]

                # if len is less means last batch
                if len(record_group) < group:
                    if last_batch == qcc.LAST_BATCH_NO_CHANGE:
                        all_record_groups.append(record_group)
                    elif last_batch == qcc.LAST_BATCH_TRUNCATE:
                        # ignoring last record group
                        pass
                    elif last_batch == qcc.LAST_BATCH_REPEAT_LAST_RECORD:
                        # repeat last record
                        repeated_record = [record_group[-1]] * (group - len(record_group))
                        # add repeated record to pad the record group
                        all_record_groups.append(record_group + repeated_record)
                else:
                    # add record group
                    all_record_groups.append(record_group)
            return all_record_groups
        else:
            raise RuntimeError('Dataset not ready to be accessed as record list.')

    def get_total_entries(self):
        return self.total_entries

    def get_cur_index(self):
        """Get current index in the dataset.

        Returns:
            cur_index: current index in dataset being processed.
        """
        return self.cur_index

    def get_file_list(self, path, extension):
        """Get the list of files from a given path based on a given extension.

        Returns:
            files: list of files
        """
        files = glob.glob(path + '/*.' + extension)
        self._total_entries = len(files)
        return files

    def get_dataset_annotation_file(self):
        return self.config._annotation_file

    def get_dataset_calibration(self):
        if self.config._calibration_file:
            return (self.config._calibration_type, self.config._calibration_file)
        else:
            return None

    def get_input_list_file(self):
        return self.input_list_file

    def get_orig_input_list_file(self):
        return self.orig_input_list_file
