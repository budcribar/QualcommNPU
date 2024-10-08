# ==============================================================================
#
#  Copyright (c) 2020, 2022-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

APP_ABI      := arm64-v8a
APP_STL      := c++_static
APP_PLATFORM := android-21
APP_CPPFLAGS += -std=c++11 -O3 -Wall -Werror -fvisibility=hidden -DQNN_API="__attribute__((visibility(\"default\")))"
APP_LDFLAGS  += -lc -lm -ldl
