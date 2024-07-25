#
# Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

# define default
default: all

# define package name
PACKAGE_NAME := $(notdir $(shell pwd))

# define library prerequisites list
sample_app := src
make_dir := make
EXE_SOURCES = $(sample_app)

# define target_architecture
export TARGET_AARCH_VARS:= -march=x86-64

# define target name
export TARGET = linux-x86_64

# specify compiler
export CXX := clang++

.PHONY: all $(EXE_SOURCES) all_x86 all_android hexagon

all: $(EXE_SOURCES) all_x86 all_android all_linux_oe_aarch64_gcc82 all_linux_oe_aarch64_gcc93 all_linux_oe_aarch64_gcc112 all_ubuntu_aarch64_gcc75 all_ubuntu_aarch64_gcc94

# Combined Targets
clean: clean_x86 clean_android clean_qnx

all_x86: clean_x86
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.linux-x86_64)

clean_x86:
	@rm -rf bin obj include

# Android Targets

all_android: aarch64-android

aarch64-android: check_ndk clean_aarch64-android
	$(call build_if_exists,$(sample_app),$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk)
	@$(rename_target_dirs)

clean_android: check_ndk clean_aarch64-android

clean_aarch64-android:
	@rm -rf bin/aarch64-android
	@rm -rf obj/local/aarch64-android

# QNX Target

all_qnx: check_qnx
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.qnx-aarch64)

clean_qnx:
	@rm -rf bin obj include

all_linux_oe_aarch64_gcc112: check_linux_oe_aarch64_gcc112
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc11.2)

clean_linux_oe_aarch64_gcc112:
	@rm -rf bin/aarch64-oe-linux-gcc11.2 obj/aarch64-oe-linux-gcc11.2

check_linux_oe_aarch64_gcc112:
ifeq ($(QNN_AARCH64_LINUX_OE_GCC_112),)
	$(error ERROR: QNN_AARCH64_LINUX_OE_GCC_112 not set, skipping compilation for Linux OE platform.)
endif


all_linux_oe_aarch64_gcc93: check_linux_oe_aarch64_gcc93
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc9.3)

clean_linux_oe_aarch64_gcc93:
	@rm -rf bin/aarch64-oe-linux-gcc9.3 obj/aarch64-oe-linux-gcc9.3

check_linux_oe_aarch64_gcc93:
ifeq ($(QNN_AARCH64_LINUX_OE_GCC_93),)
	$(error ERROR: QNN_AARCH64_LINUX_OE_GCC_93 not set, skipping compilation for Linux OE platform.)
endif


all_linux_oe_aarch64_gcc82: check_linux_oe_aarch64_gcc82
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc8.2)

clean_linux_oe_aarch64_gcc82:
	@rm -rf bin/aarch64-oe-linux-gcc8.2 obj/aarch64-oe-linux-gcc8.2

check_linux_oe_aarch64_gcc82:
ifeq ($(QNN_AARCH64_LINUX_OE_GCC_82),)
	$(error ERROR: QNN_AARCH64_LINUX_OE_GCC_82 not set, skipping compilation for Linux OE platform.)
endif


all_ubuntu_aarch64_gcc75: check_ubuntu_aarch64_gcc75
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc7.5)

clean_ubuntu_aarch64_gcc75:
	@rm -rf bin/aarch64-ubuntu-gcc7.5 obj/aarch64-ubuntu-gcc7.5

check_ubuntu_aarch64_gcc75:
ifeq ($(QNN_AARCH64_UBUNTU_GCC_75),)
	$(error ERROR: QNN_AARCH64_UBUNTU_GCC_75 not set, skipping compilation for Ubuntu platform.)
endif

all_ubuntu_aarch64_gcc94: check_ubuntu_aarch64_gcc94
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc9.4)

clean_ubuntu_aarch64_gcc94:
	@rm -rf bin/aarch64-ubuntu-gcc9.4 obj/aarch64-ubuntu-gcc9.4

check_ubuntu_aarch64_gcc94:
ifeq ($(QNN_AARCH64_UBUNTU_GCC_94),)
	$(error ERROR: QNN_AARCH64_UBUNTU_GCC_94 not set, skipping compilation for Ubuntu platform.)
endif


# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))
rename_target_dirs = find . -type d -execdir rename 's/arm64-v8a/aarch64-android/' '{}' \+ \
                     && mv libs/* bin/  \
                     && rm -rf libs \

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif

check_qnx:
ifeq ($(QNX_HOST),)
	$(error ERROR: QNX_HOST not set, skipping compilation for QNX platform.)
endif
ifeq ($(QNX_TARGET),)
	$(error ERROR: QNX_TARGET not set, skipping compilation for QNX platform.)
endif

# Hexagon Target
hexagon: clean_hexagon
	$(call build_if_exists,$(sample_app),-$(MAKE) -f $(make_dir)/Makefile.hexagon)

clean_hexagon:
	@rm -rf bin/hexagon obj/hexagon
