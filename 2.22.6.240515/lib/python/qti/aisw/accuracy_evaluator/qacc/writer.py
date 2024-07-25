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

from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
import qti.aisw.accuracy_evaluator.common.exceptions as ce


class Writer:

    def __init__(self):
        pass

    def write(self, output_path, mem_obj, dtype, write_format='npy'):
        """Use a specific writer to write the mem_obj The write methods needs
        to be thread safe."""

        if write_format == qcc.FMT_CV2:
            return CV2Writer.write(output_path, mem_obj)
        elif write_format == qcc.FMT_PIL:
            return PILWriter.write(output_path, mem_obj)
        elif write_format == qcc.FMT_NPY:
            return RawWriter.write(output_path, mem_obj, dtype)

        raise ce.UnsupportedException('Invalid Writer type : ' + write_format)


class CV2Writer:

    @classmethod
    def write(self, output_path, mem_obj):
        cv2.imwrite(output_path, mem_obj)


class PILWriter:

    @classmethod
    def write(self, output_path, mem_obj):
        mem_obj.save(output_path)


class RawWriter:

    @classmethod
    def write(self, output_path, mem_obj, dtype):
        mem_obj.astype(dtype).tofile(output_path)
