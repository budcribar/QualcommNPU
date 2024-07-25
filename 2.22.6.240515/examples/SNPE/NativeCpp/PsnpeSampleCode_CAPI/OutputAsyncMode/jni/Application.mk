# Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

NDK_TOOLCHAIN_VERSION := clang

APP_PLATFORM := android-21
APP_ABI := arm64-v8a
APP_STL := c++_static
APP_CPPFLAGS += -std=c++11 -fexceptions -frtti
APP_LDFLAGS = -lc -lm -ldl
