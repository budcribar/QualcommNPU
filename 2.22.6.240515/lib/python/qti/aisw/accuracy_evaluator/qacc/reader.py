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
import cv2
from PIL import Image
import numpy as np

from itertools import count
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
import copy
import os

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.qacc._data_transport import PipelineItem


class Reader:

    def __init__(self):
        pass

    def read(self, input_path, dtype, format):
        """Use a specific reader to read the input.

        The read methods needs to be thread safe.
        """
        if format == qcc.FMT_CV2:
            return CV2Reader.read(input_path)
        elif format == qcc.FMT_PIL:
            return PILReader.read(input_path)
        elif format == qcc.FMT_NPY:
            return RawReader.read(input_path, dtype)

        raise ce.UnsupportedException('Invalid Reader type : ' + format)


class CV2Reader:

    @classmethod
    def read(self, input_path):
        image = cv2.imread(input_path)
        if image is None:
            raise RuntimeError('CV2 failed to read image :' + input_path)

        return image


class PILReader:

    @classmethod
    def read(self, input_path):
        image = Image.open(input_path)
        if image is None:
            raise RuntimeError('PIL failed to read image :' + input_path)

        return image


class RawReader:

    @classmethod
    def read(self, inp_path, dtype):
        inp = np.fromfile(inp_path, dtype=np.float32)
        return inp


# DataReader Class
class DataReader:

    def __init__(self, file_queue, num_readers=1, path_obj=None, iter_obj=None, batchsize=1,
                 relative_path=False) -> None:
        self.file_queue = file_queue
        self.path_obj = path_obj
        self.iter_obj = iter_obj
        self.num_readers = num_readers
        self.batchsize = batchsize
        self.relative_path = relative_path
        self.read_workers = ThreadPoolExecutor(num_readers, thread_name_prefix=f'DataReader')
        if self.path_obj:
            # convert path_obj to iter_obj
            self.updated_paths = self.handle_path_objects()
            try:
                self.iter_obj = iter(self.updated_paths)
            except Exception as e:
                qacc_logger.error(f"Failed to create DataReader using {self.path_obj}. Reason: {e}")
        self.iter_obj = zip(count(), self.iter_obj)
        self._sentinel = qcc.DATAREADER_SENTINEL_OBJ  #'END'

    def handle_path_objects(self):
        base_path = Path(self.path_obj).parent
        updated_paths = []

        # Find out whether the paths present in the file list is absolute or relative to file list
        for paths in open(str(self.path_obj)).readlines():
            temp_path = []
            for path in paths.strip().split(','):
                if len(path) > 0 and os.path.exists(path) and not self.relative_path:
                    temp_path.append(path)
                elif len(path) > 0 and os.path.exists(os.path.join(base_path, path)):
                    # Relative path case: Append with file path with file list directory path
                    temp_path.append(os.path.join(base_path, path))
            updated_paths.append(temp_path)
        return updated_paths

    def start(self):
        futures = []
        for worker_id in range(self.num_readers):
            futures.append(self.read_workers.submit(self.handle_iterator))
        return futures

    def handle_iterator(self):
        keep_running = True
        while keep_running:
            try:
                if not self.file_queue.full():
                    pipeline_items = []
                    for it in range(self.batchsize):
                        try:
                            inp_idx, data = next(self.iter_obj)
                            if isinstance(data, list):
                                # Handle case when there are multiple items in data. ie model has more than 1 input nodes
                                # pipeline_item = {"meta": [], 'input_idx': inp_idx, "data": data}
                                pipeline_item = PipelineItem(data=data, meta={}, input_idx=inp_idx)
                            else:
                                if isinstance(data, str):
                                    # each line in file list: handle multiple inputs
                                    data = [d for d in data.strip().split(',') if len(d)]
                                    pipeline_item = PipelineItem(data=data, meta={},
                                                                 input_idx=inp_idx)
                                else:
                                    # each line in the file list has only one input
                                    pipeline_item = PipelineItem(data=[data], meta={},
                                                                 input_idx=inp_idx)
                        except StopIteration:
                            # check if batch is fully filled. Handle the last batch
                            if len(pipeline_items) != 0 and len(pipeline_items) < self.batchsize:
                                pipeline_items = self.handle_batching(pipeline_items)
                            keep_running = False
                            break
                        else:
                            pipeline_items.append(pipeline_item)
            except Exception as e:
                qacc_logger.error(f"Data Reader failed to fetch items from input cache: {e}")
                break

            if pipeline_items is not None and len(pipeline_items) > 0:
                self.file_queue.put(pipeline_items)
        if not keep_running:
            self.file_queue.put(self._sentinel)

        qacc_logger.debug(f"DataReader complete {self.reader_name}")
        self.read_workers.shutdown(wait=True)

    def handle_batching(self, pipeline_items, mode=qcc.LAST_BATCH_REPEAT_LAST_RECORD):
        if mode == qcc.LAST_BATCH_NO_CHANGE or mode == qcc.LAST_BATCH_TRUNCATE:
            # drop the last few items as they dont meet the batching requirements
            return None
        elif mode == qcc.LAST_BATCH_REPEAT_LAST_RECORD:
            for i in range(self.batchsize - len(pipeline_items)):
                # Need perform deepcopy so that same object is not updating during preprocessing
                pipeline_items.append(copy.deepcopy(pipeline_items[-1]))
            return pipeline_items
