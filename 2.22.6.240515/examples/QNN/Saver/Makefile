# ===============================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

# define default
default: all

make_dir := make

# specify compiler
export CC := clang

# specify saver output file to compile
export SAVER_OUTPUT_FILENAME := saver_output.c

# combined targets

all: x86 android linux_oe_aarch64_gcc93 linux_oe_aarch64_gcc82 linux_oe_aarch64_gcc112 ubuntu_aarch64_gcc75 ubuntu_aarch64_gcc94

x86: cpu_x86 htp_x86

android: aarch64

aarch64: cpu_aarch64 gpu_aarch64 dsp_aarch64 htp_aarch64 hta_aarch64

qnx: cpu_qnx

clean: clean_x86 clean_aarch64

# x86 targets
cpu_x86:
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.linux-x86_64 cpu)

htp_x86:
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.linux-x86_64 htp)

clean_x86:
	@rm -rf x86_64-linux-clang

# qnx targets
cpu_qnx: check_qnx
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.qnx-aarch64 cpu)

clean_qnx:
	@rm -rf aarch64-qnx

# Linux OE 11.2 targets

linux_oe_aarch64_gcc112: cpu_linux_oe_aarch64_gcc112 gpu_linux_oe_aarch64_gcc112 htp_linux_oe_aarch64_gcc112 hta_linux_oe_aarch64_gcc112

cpu_linux_oe_aarch64_gcc112: check_linux_oe_aarch64_gcc112
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc11.2 cpu)

gpu_linux_oe_aarch64_gcc112: check_linux_oe_aarch64_gcc112
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc11.2 gpu)

htp_linux_oe_aarch64_gcc112: check_linux_oe_aarch64_gcc112
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc11.2 htp)

hta_linux_oe_aarch64_gcc112: check_linux_oe_aarch64_gcc112
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc11.2 hta)

clean_linux_oe_aarch64_gcc112:
	@rm -rf aarch64-oe-linux-gcc11.2

check_linux_oe_aarch64_gcc112:
ifeq ($(QNN_AARCH64_LINUX_OE_GCC_112),)
	$(error ERROR: QNN_AARCH64_LINUX_OE_GCC_112 not set, skipping compilation for Linux OE platform.)
endif


# Linux OE 9.3 targets

linux_oe_aarch64_gcc93: cpu_linux_oe_aarch64_gcc93 gpu_linux_oe_aarch64_gcc93 htp_linux_oe_aarch64_gcc93 hta_linux_oe_aarch64_gcc93

cpu_linux_oe_aarch64_gcc93: check_linux_oe_aarch64_gcc93
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc9.3 cpu)

gpu_linux_oe_aarch64_gcc93: check_linux_oe_aarch64_gcc93
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc9.3 gpu)

htp_linux_oe_aarch64_gcc93: check_linux_oe_aarch64_gcc93
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc9.3 htp)

hta_linux_oe_aarch64_gcc93: check_linux_oe_aarch64_gcc93
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc9.3 hta)

clean_linux_oe_aarch64_gcc93:
	@rm -rf aarch64-oe-linux-gcc9.3

check_linux_oe_aarch64_gcc93:
ifeq ($(QNN_AARCH64_LINUX_OE_GCC_93),)
	$(error ERROR: QNN_AARCH64_LINUX_OE_GCC_93 not set, skipping compilation for Linux OE platform.)
endif


# Linux OE 8.2 targets

linux_oe_aarch64_gcc82: cpu_linux_oe_aarch64_gcc82 gpu_linux_oe_aarch64_gcc82 htp_linux_oe_aarch64_gcc82 hta_linux_oe_aarch64_gcc82

cpu_linux_oe_aarch64_gcc82: check_linux_oe_aarch64_gcc82
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc8.2 cpu)

gpu_linux_oe_aarch64_gcc82: check_linux_oe_aarch64_gcc82
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc8.2 gpu)

htp_linux_oe_aarch64_gcc82: check_linux_oe_aarch64_gcc82
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc8.2 htp)

hta_linux_oe_aarch64_gcc82: check_linux_oe_aarch64_gcc82
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.oe-linux-aarch64-gcc8.2 hta)

clean_linux_oe_aarch64_gcc82:
	@rm -rf aarch64-oe-linux-gcc8.2

check_linux_oe_aarch64_gcc82:
ifeq ($(QNN_AARCH64_LINUX_OE_GCC_82),)
	$(error ERROR: QNN_AARCH64_LINUX_OE_GCC_82 not set, skipping compilation for Linux OE platform.)
endif


# Ubuntu 7.5 targets

ubuntu_aarch64_gcc75: cpu_ubuntu_aarch64_gcc75 gpu_ubuntu_aarch64_gcc75 htp_ubuntu_aarch64_gcc75 hta_ubuntu_aarch64_gcc75

cpu_ubuntu_aarch64_gcc75: check_ubuntu_aarch64_gcc75
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc7.5 cpu)

gpu_ubuntu_aarch64_gcc75: check_ubuntu_aarch64_gcc75
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc7.5 gpu)

htp_ubuntu_aarch64_gcc75: check_ubuntu_aarch64_gcc75
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc7.5 htp)

hta_ubuntu_aarch64_gcc75: check_ubuntu_aarch64_gcc75
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc7.5 hta)

clean_ubuntu_aarch64_gcc75:
	@rm -rf aarch64-ubuntu-gcc7.5

check_ubuntu_aarch64_gcc75:
ifeq ($(QNN_AARCH64_UBUNTU_GCC_75),)
	$(error ERROR: QNN_AARCH64_UBUNTU_GCC_75 not set, skipping compilation for Ubuntu platform.)
endif

# Ubuntu 9.4 targets

ubuntu_aarch64_gcc94: cpu_ubuntu_aarch64_gcc94 gpu_ubuntu_aarch64_gcc94 htp_ubuntu_aarch64_gcc94 hta_ubuntu_aarch64_gcc94

cpu_ubuntu_aarch64_gcc94: check_ubuntu_aarch64_gcc94
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc9.4 cpu)

gpu_ubuntu_aarch64_gcc94: check_ubuntu_aarch64_gcc94
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc9.4 gpu)

htp_ubuntu_aarch64_gcc94: check_ubuntu_aarch64_gcc94
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc9.4 htp)

hta_ubuntu_aarch64_gcc94: check_ubuntu_aarch64_gcc94
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),-$(MAKE) -f $(make_dir)/Makefile.ubuntu-aarch64-gcc9.4 hta)

clean_ubuntu_aarch64_gcc94:
	@rm -rf aarch64-ubuntu-gcc9.4

check_ubuntu_aarch64_gcc94:
ifeq ($(QNN_AARCH64_UBUNTU_GCC_94),)
	$(error ERROR: QNN_AARCH64_UBUNTU_GCC_94 not set, skipping compilation for Ubuntu platform.)
endif


# aarch64 targets
cpu_aarch64: check_ndk
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),$(ANDROID_NDK_ROOT)/ndk-build APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk QNN_BACKEND="QnnCpu")
	@$(rename_target_dirs)

gpu_aarch64: check_ndk
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),$(ANDROID_NDK_ROOT)/ndk-build APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk QNN_BACKEND="QnnGpu")
	@$(rename_target_dirs)

dsp_aarch64: check_ndk check_hexagon_sdk
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),$(ANDROID_NDK_ROOT)/ndk-build APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk QNN_BACKEND="QnnDsp")
	@$(rename_target_dirs)

htp_aarch64: check_ndk check_hexagon_sdk
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),$(ANDROID_NDK_ROOT)/ndk-build APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk QNN_BACKEND="QnnHtp")
	@$(rename_target_dirs)

hta_aarch64: check_ndk
	$(call build_if_exists,$(SAVER_OUTPUT_FILENAME),$(ANDROID_NDK_ROOT)/ndk-build APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk QNN_BACKEND="QnnHta")
	@$(rename_target_dirs)

clean_aarch64:
	@rm -rf aarch64-android

# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))
rename_target_dirs = find . -type d -execdir rename 's/arm64-v8a/aarch64-android/' '{}' \+ \
                     && rsync -a libs/* ./  \
                     && rm -rf libs obj

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif

check_hexagon_sdk:
ifeq ($(HEXAGON_SDK_ROOT),)
	$(error ERROR: HEXAGON_SDK_ROOT not set, skipping compilation for DSP/HTP backends on aarch64.)
endif

check_qnx:
ifeq ($(QNX_HOST),)
	$(error ERROR: QNX_HOST not set, skipping compilation for QNX platform.)
endif
ifeq ($(QNX_TARGET),)
	$(error ERROR: QNX_TARGET not set, skipping compilation for QNX platform.)
endif
