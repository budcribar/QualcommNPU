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
from typing import List
import os
from itertools import zip_longest
import contextlib
import shutil
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger


class create_batch(qacc_plugin):

    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_DIR,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_DIR,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        # Dir plugins always have a single entry.
        self.execute_index(pin_list[0], pout_list[0])

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        if not pin.is_directory_input():
            print('Only directory based input supported for CreateBatch')
            return

        # The inputs are the paths of the files we need to concatenate and create a batch input.
        # Below is a generic e.g. of the inputs
        # [[p11,None,p13], [p21, None, p23], ...]
        # For batch size 2, we need to concat p11+p21, p13+p23

        batch_size = pin.read_pipeline_cache_val(qcc.PIPELINE_BATCH_SIZE)
        is_truncate = pin.get_param('truncate', default=False)
        delete_prior = pin.get_param('delete_prior', default=True)

        if batch_size < 1:
            print('Batch size must be greater than 0 !')
            pout.set_status(qcc.STATUS_ERROR)
            return

        out_file_list = []  # final batched inputs
        input_list = pin.get_input()

        iters = [iter(input_list)] * batch_size
        if is_truncate:
            print('Truncating incomplete batches.')
            num_inputs = len(input_list)
            if num_inputs < batch_size:
                qacc_logger.error(
                    'Number of inputs must be greater than or equal to batch size when using truncate'
                )
                pout.set_status(qcc.STATUS_ERROR)
                return
            # Ignore the last incomplete batch
            to_iterate = zip(*iters)
        else:
            print('Padding incomplete batches.')
            # Use last element of input_list to fill the incomplete batch
            to_iterate = zip_longest(*iters, fillvalue=input_list[-1])

        record_counter = 0
        for batch_list in to_iterate:
            # batch_list has 'batch_size' number of lists.
            #     [ ['p11', None, 'p13'], ['p21', None, 'p23'] ]
            # concat_batches groups the inputs in same index into a list
            #     [ [p11,p21], [None,None], [p13, p23] ]
            concat_batches = [list(a) for a in zip(*batch_list)]
            record_paths = []  # single set of batched inputs
            for inp_idx, batch in enumerate(concat_batches):

                # skip any indexes which has no paths.
                # Applicable when specific input indexes are processed.
                if None in batch:
                    record_paths.append(None)
                    continue

                _out_path = os.path.join(pout.get_out_dir(),
                                         f'batched-inp-{record_counter}-{inp_idx}.raw')
                record_paths.append(_out_path)

                # if bs=1, rename and continue
                if batch_size == 1:
                    if delete_prior:
                        os.rename(batch[0], _out_path)
                    else:
                        shutil.copyfile(batch[0], _out_path)
                    continue

                inputs = [open(path, 'rb').read() for path in batch]
                with open(_out_path, 'wb') as f:
                    for i in inputs:
                        f.write(i)

                if not delete_prior:
                    continue
                # delete prior unbatched data to save space
                for path in batch:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(path)

            out_file_list.append(record_paths)
            record_counter += 1

        pout.set_dir_outputs(out_file_list)
        pout.set_status(qcc.STATUS_SUCCESS)
