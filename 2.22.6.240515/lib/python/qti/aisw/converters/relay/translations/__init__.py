# ==============================================================================
#
#  Copyright (c) 2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from ...common.converter_ir import translation
RelayTranslations = translation.TranslationBank()

# these need to be imported so they are evaluated,
# so that the translations are registered in the Bank
from . import nn_translations
from . import data_translations
from . import math_translations
from . import custom_op_translations
