# ==============================================================================
#
#  Copyright (c) 2020-2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

LOCAL_PATH := $(call my-dir)
SUPPORTED_TARGET_ABI := arm64-v8a x86 x86_64

#============================ Verify Target Info and Application Variables =========================================
ifneq ($(filter $(TARGET_ARCH_ABI),$(SUPPORTED_TARGET_ABI)),)
    ifneq ($(APP_STL), c++_static)
        $(error Unsupported APP_STL: "$(APP_STL)")
    endif
else
    $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

#============================ Define Common Variables ===============================================================
# Include paths
# QNN_SDK_ROOT should be set and points to the SDK path, it will be used.
ifdef QNN_SDK_ROOT
PACKAGE_C_INCLUDES += -I $(QNN_SDK_ROOT)/examples/QNN/OpPackage/CPU/include -I $(QNN_SDK_ROOT)/include/QNN -I $(LOCAL_PATH)/../include/
else
$(error QNN_SDK_ROOT: Please set QNN_SDK_ROOT)
endif

#========================== Define OpPackage Library Build Variables =============================================
include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := $(PACKAGE_C_INCLUDES)
MY_SRC_FILES                    = $(wildcard $(LOCAL_PATH)/../src/*.cpp) $(wildcard $(LOCAL_PATH)/../src/ops/*.cpp)
LOCAL_MODULE                   := libQnnCpuOpPackageExample
LOCAL_SRC_FILES                := $(subst make/,,$(MY_SRC_FILES))
LOCAL_LDLIBS                   := -lGLESv2 -lEGL
include $(BUILD_SHARED_LIBRARY)
