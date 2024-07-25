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
import qti.aisw.accuracy_evaluator.common.defaults as df
import logging
import os

defaults = df.Defaults.getInstance(app='qacc')
qacc_logger = logging.getLogger('qacc')
qacc_file_logger = logging.getLogger('qacc_logfile')

# limiting logging for tensor flow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
