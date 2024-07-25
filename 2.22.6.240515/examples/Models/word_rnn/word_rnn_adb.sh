#!/system/bin/sh
#==============================================================================
#
#  Copyright (c) 2020, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

export SNPE_TARGET_ARCH=aarch64-android
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/lib
export PATH=$PATH:/data/local/tmp/snpeexample/$SNPE_TARGET_ARCH/bin
export ADSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH:/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp;/data/local/tmp/snpeexample/dsp/lib"

if [ "$1" = "cpu" ]; then
    snpe-net-run --container word_rnn.dlc --input_list input_list.txt
elif [ "$1" = "gpu" ]; then
    snpe-net-run --container word_rnn.dlc --input_list input_list.txt --use_gpu
elif [ "$1" = "dsp" ]; then
    snpe-net-run --container word_rnn.dlc --input_list input_list.txt --use_dsp
elif [ "$1" = "aip" ]; then
    snpe-net-run --container word_rnn.dlc --input_list input_list.txt --use_aip
else
    echo "Invalid tag!"
    exit 1
fi
