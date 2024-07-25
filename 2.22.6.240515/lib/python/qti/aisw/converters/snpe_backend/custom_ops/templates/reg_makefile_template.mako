<%doc>
# ==============================================================================
#
#  Copyright (c) 2020, 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
#==============================================================================
# Auto Generated Code for ${package_info.name}
#==============================================================================

# define relevant directories
SRC_DIR := ./

# define library name and corresponding directory
export LIB_DIR := ../../../libs/$(TARGET)
library := $(LIB_DIR)/libUdo${package_info.name}Reg.so

# define target architecture if not previously defined, default is x86
ifndef TARGET_AARCH_VARS
TARGET_AARCH_VARS:= -march=x86-64
endif
<% snpe_udo_root = package_info.SNPE_UDO_ROOT %>
# set snpe_udo_root if it was defined in the config
ifndef SNPE_UDO_ROOT
SNPE_UDO_ROOT := ${snpe_udo_root if snpe_udo_root else "NOT_DEFINED"}
endif

# specify package paths, should be able to override via command line?
UDO_PACKAGE_ROOT =${package_info.root}/${package_info.name}

include ../../../common.mk
