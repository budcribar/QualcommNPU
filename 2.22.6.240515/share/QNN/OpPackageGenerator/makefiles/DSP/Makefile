# ==============================================================================
#
#  Copyright (c) 2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# define relevant directories
HOME_DIR := ./

# NOTE:
# - this Makefile is going to be used only to create DSP skels, so no need for android.min

ifndef HEXAGON_SDK_ROOT
$(error "HEXAGON_SDK_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_SDK_ROOT to hexagon sdk installation.")
endif

# define variant as V=hexagon_Release_dynamic_toolv83_v60 - it can be hardcoded too
ifndef V
V = hexagon_Release_dynamic_toolv83_v60
endif

V_TARGET = $(word 1,$(subst _, ,$(V)))
ifneq ($(V_TARGET),hexagon)
$(error Unsupported target '$(V_TARGET)' in variant '$(V)')
endif

# must list all variants supported by this project
SUPPORTED_VS = $(default_VS)

# must list all the dependencies of this project
DEPENDENCIES = ATOMIC RPCMEM TEST_MAIN TEST_UTIL

ATOMIC_DIR = $(HEXAGON_SDK_ROOT)/libs/common/atomic
RPCMEM_DIR = $(HEXAGON_SDK_ROOT)/libs/common/rpcmem
TEST_MAIN_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_main
TEST_UTIL_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_util

include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min

ifndef QNN_INCLUDE
$(info "INFO: Qnn include not explicitly defined, attempting to use QNN_SDK_ROOT if it is valid")
QNN_INCLUDE := $(QNN_SDK_ROOT)/include/QNN
endif
ifeq ($(wildcard $(QNN_INCLUDE)),)
$(error "ERROR: QNN_INCLUDE path is not set. QNN include paths must be set to obtain BE headers necessary to compile the package")
endif

ifndef PACKAGE_NAME
PACKAGE_NAME := $(notdir $(shell pwd))
$(info "INFO: No package name defined. Using current directory name: $(PACKAGE_NAME) as the package name")
endif

# set include paths as compiler flags
CC_FLAGS += -I $(HOME_DIR)/include
CC_FLAGS += -I $(QNN_INCLUDE)/

# only build the shared object if dynamic option specified in the variant
ifeq (1,$(V_dynamic))
BUILD_DLLS = libQnn${PACKAGE_NAME}
endif

OP_SOURCES = $(wildcard $(HOME_DIR)/src/ops/*.cpp)
OTHER_SOURCES = $(wildcard $(HOME_DIR)/src/*.cpp)

# sources for the DSP implementation library in src directory
libQnn${PACKAGE_NAME}.C_SRCS := $(wildcard $(OP_SOURCES)) $(OTHER_SOURCES)

# copy final build products to the ship directory
BUILD_COPIES = $(DLLS) $(EXES) $(LIBS) build/DSP/

# always last
include $(RULES_MIN)

# define destination library directory, and copy files into lib/dsp
# this code will create it
SHIP_LIBS_DIR   := src/DSP/$(V)
LIB_DIR         := build/DSP
OBJ_DIR         := hexagon_Release_dynamic_toolv83_v60

.PHONY: dsp

dsp: tree
	mkdir -p ${"${OBJ_DIR}"};  ${"\\"}
	cp -Rf ${"${SHIP_LIBS_DIR}"}/. ${"${OBJ_DIR}"} ;${"\\"}
	rm -rf ${"${SHIP_LIBS_DIR}"};
