# Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

LOCAL_PATH := $(call my-dir)

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
   ifeq ($(APP_STL), c++_static)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else
   $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

SNPE_INCLUDE_DIR := $(SNPE_ROOT)/include/SNPE

include $(CLEAR_VARS)
LOCAL_MODULE := snpe-sample
LOCAL_SRC_FILES := main.cpp CheckRuntime.cpp LoadContainer.cpp LoadUDOPackage.cpp LoadInputTensor.cpp SetBuilderOptions.cpp Util.cpp NV21Load.cpp CreateUserBuffer.cpp PreprocessInput.cpp SaveOutputTensor.cpp CreateGLBuffer.cpp CreateGLContext.cpp
LOCAL_CFLAGS := -DENABLE_GL_BUFFER
LOCAL_SHARED_LIBRARIES := libSNPE
LOCAL_LDLIBS     := -lGLESv2 -lEGL
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_INCLUDE_DIR)
include $(PREBUILT_SHARED_LIBRARY)

