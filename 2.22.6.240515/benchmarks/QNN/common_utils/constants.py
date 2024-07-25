#
# Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'

RUNTIME_STR_MAP = {
    'CPU': 'cpu_float32',
    'GPU': 'gpu_float32_16_hybrid',
    'GPU_FP16': 'gpu_float16',
    'DSP': 'dsp_fixed8_tf',
    'AIP': 'aip_fixed8_tf'
}
