# ===================================================================================================
#
# Copyright (c) 2020-2021, 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===================================================================================================

LOCAL_PATH := $(call my-dir)
SUPPORTED_TARGET_ABI := arm64-v8a x86 x86_64

include $(CLEAR_VARS)

#============================ Verify Target Info and Application Variables ===========================
ifneq ($(filter $(TARGET_ARCH_ABI),$(SUPPORTED_TARGET_ABI)),)
    ifneq ($(APP_STL), c++_static)
        $(error Unsupported APP_STL: "$(APP_STL)")
    endif
else
    $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

#==================================Define Common Variable=============================================

# ensure QNN_SDK_ROOT points to correct path

ifdef QNN_SDK_ROOT
PACKAGE_C_INCLUDES := $(QNN_SDK_ROOT)/include/QNN
PACKAGE_C_INCLUDES += $(QNN_SDK_ROOT)/include/QNN/GPU
else
$(error QNN_SDK_ROOT: Please set QNN_SDK_ROOT)
endif

SRC_FILES = $(wildcard $(LOCAL_PATH)/../src/*.cpp)
SRC_FILES += $(wildcard $(LOCAL_PATH)/../src/*.c)

#=================================Define GPU Runtime Variables========================================

LOCAL_MODULE     := libQnnGpuOpPackageExample
LOCAL_C_INCLUDES := $(PACKAGE_C_INCLUDES)
LOCAL_SRC_FILES  :=  $(subst make/,,$(SRC_FILES))
LOCAL_LDLIBS     := -lGLESv2 -lEGL

include $(BUILD_SHARED_LIBRARY)
