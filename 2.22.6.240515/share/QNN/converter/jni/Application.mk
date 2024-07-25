# Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

NDK_TOOLCHAIN_VERSION := clang++

APP_PLATFORM := android-16
APP_ABI := $(QNN_ANDROID_APP_ABIS)
APP_STL := c++_static
APP_CPPFLAGS += -std=c++11 -O3 -Wno-write-strings -DQNN_API="__attribute__((visibility(\"default\")))" \
                -fvisibility=hidden
APP_LDFLAGS = -lc -lm -ldl