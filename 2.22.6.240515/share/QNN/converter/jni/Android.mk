# Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# setup variables for compilation
ifndef QNN_SDK_ROOT
QNN_SDK_ROOT := $(abspath $(abspath $(lastword $(LOCAL_PATH)))/../../../../..)
ifneq ($(shell test -d $(QNN_SDK_ROOT) && echo 0),0)
$(error Unable to determine QNN_SDK_ROOT)
endif
endif

SRC_FILES := $(subst $(LOCAL_PATH)/,,$(wildcard $(LOCAL_PATH)/*.cpp))
# support any filename for .bin tarfile
# Note: Allow only one .bin per compilation for added simplicty and this also matches converter output spec
BINARY_FILE := $(wildcard $(LOCAL_PATH)/*.bin)
ifeq ($(BINARY_FILE),)
$(warning WARNING No Binary File provided. If params are required for model, this will potentially result in undefined reference when running.)
else
BINARY_FILE_CNT := $(words $(BINARY_FILE))
ifneq ($(BINARY_FILE_CNT),1)
$(error Only one Binary File can be provided. Got: $(BINARY_FILE) at $(abspath $(LOCAL_PATH)))
endif
endif

# User provided name takes precendence for created model
ifneq ($(QNN_MODEL_LIB_NAME),)
# if provided, remove prefix and suffix for naming shared library since ndk-build step will add it
LIBRARY := $(subst lib,,$(subst .so,,$(notdir $(QNN_MODEL_LIB_NAME))))
else
ifneq ($(BINARY_FILE),)
# if provided, use binary name for library as next option
LIBRARY := $(subst .bin,,$(notdir $(BINARY_FILE)))
else
# unable to determine name, hence default to generic name
LIBRARY := qnn_model
endif
endif

BINARY_DIR      := $(NDK_APP_OUT)/binary
BINARY_OBJ_DIR  := $(NDK_APP_OUT)/local/$(TARGET_ARCH_ABI)/objs/$(LIBRARY)
$(shell mkdir -p $(BINARY_DIR) $(BINARY_OBJ_DIR))
ifneq ($(BINARY_FILE),)
$(shell tar xf $(BINARY_FILE) -C $(BINARY_DIR) >/dev/null)
endif
BINARY := $(wildcard $(BINARY_DIR)/*.raw)
BINARY_OBJS := $(subst $(BINARY_DIR),,$(subst .raw,.o,$(BINARY)))

$(BINARY_OBJ_DIR)/%.o: $(BINARY_DIR)%.raw $(BINARY_FILE)
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
	$(eval NDK_OBJCOPY_CMD := $(NDK_ROOT)/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-objcopy \
	-I binary -O elf32-littlearm -B arm)
else
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
	$(eval NDK_OBJCOPY_CMD := $(NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objcopy \
	-I binary -O elf64-littleaarch64 -B aarch64)
else
	$(error "Unsupported target $(TARGET_ARCH_ABI)")
endif
endif
	$(NDK_OBJCOPY_CMD) $< $@

LOCAL_MODULE := $(LIBRARY)
LOCAL_C_INCLUDES := ./ $(QNN_SDK_ROOT)/include/QNN
LOCAL_SRC_FILES :=  $(SRC_FILES) $(BINARY_OBJS)
LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
include $(BUILD_SHARED_LIBRARY)
