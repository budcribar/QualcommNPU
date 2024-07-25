# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

LOCAL_PATH := $(call my-dir)

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
   ifeq ($(APP_STL), c++_static)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else ifeq ($(TARGET_ARCH_ABI), )
   ifeq ($(APP_STL), c++_static)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else
   $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

SNPE_INCLUDE_DIR := $(SNPE_ROOT)/include/SNPE

include $(CLEAR_VARS)
LOCAL_MODULE := psnpe-csample_sync
LOCAL_SRC_FILES := main.cpp ProcessDataType.cpp ProcessInputList.cpp ProcessUserBuffer.cpp BuildPSNPE.cpp Util.cpp
LOCAL_CFLAGS := -DENABLE_GL_BUFFER
LOCAL_SHARED_LIBRARIES := libSNPE
LOCAL_LDLIBS     := -lGLESv2 -lEGL
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

