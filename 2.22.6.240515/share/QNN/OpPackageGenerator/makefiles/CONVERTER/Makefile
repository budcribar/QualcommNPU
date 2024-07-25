# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

# define default
default: all

# define package name
export PACKAGE_NAME := $(notdir $(shell pwd))

# define library prerequisites list
lib_cpu := ConverterOpPackage
LIB_SOURCES = $(lib_cpu))

# define target name
export TARGET = linux-x86_64

# specify compiler
export CXX ?= clang++

.PHONY: all $(LIB_SOURCES) cpu_x86
all: $(LIB_SOURCES) all_x86

# Combined Targets
cpu: cpu_x86
clean: clean_x86

# x86 Targets
all_x86: cpu_x86

cpu_x86:
	$(call build_if_exists,$(lib_cpu),-$(MAKE) -f $(shell pwd)/$(lib_cpu)/Makefile.linux-x86_64)

clean_x86:
	@rm -rf libs obj

# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))
