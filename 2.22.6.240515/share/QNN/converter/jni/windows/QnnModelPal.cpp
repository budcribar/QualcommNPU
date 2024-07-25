//==============================================================================
//
// Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// clang-format off
#include <windows.h>
#include <psapi.h>
#include <stdlib.h>
#include <winevt.h>
// clang-format on

#include <set>
#include <string>

#include "QnnModelPal.hpp"

#define STRINGIFY(x) #x
#define TOSTRING(x)  STRINGIFY(x)

static std::set<HMODULE> mod_handles;
static thread_local char *sg_lastErrMsg = const_cast<char *>("");

void *qnn_wrapper_api::dlSym(void *handle, const char *symbol) {
  FARPROC sym_addr = NULL;
  HANDLE cur_proc;
  DWORD size, size_needed;
  HMODULE *mod_list;
  HMODULE mod = 0;

  if ((!handle) || (!symbol)) {
    return NULL;
  }

  cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, NULL, 0, &size) == 0) {
    sg_lastErrMsg = const_cast<char *>("enumerate modules failed before memory allocation");
    return NULL;
  }

  mod_list = static_cast<HMODULE *>(malloc(size));
  if (!mod_list) {
    sg_lastErrMsg = const_cast<char *>("malloc failed");
    return NULL;
  }

  if (EnumProcessModules(cur_proc, mod_list, size, &size_needed) == 0) {
    sg_lastErrMsg = const_cast<char *>("enumerate modules failed after memory allocation");
    free(mod_list);
    return NULL;
  }

  // DL_DEFAULT needs to bypass those modules with DL_LOCAL flag
  if (handle == DL_DEFAULT) {
    for (size_t i = 0; i < (size / sizeof(HMODULE)); i++) {
      auto iter = mod_handles.find(mod_list[i]);
      if (iter != mod_handles.end()) {
        continue;
      }
      // once find the first non-local module with symbol
      // return its address here to avoid unnecessary looping
      sym_addr = GetProcAddress(mod_list[i], symbol);
      if (sym_addr) {
        free(mod_list);
        return *(void **)(&sym_addr);
      }
    }
  } else {
    mod = static_cast<HMODULE>(handle);
  }

  free(mod_list);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    sg_lastErrMsg = const_cast<char *>("can't resolve symbol");
    return NULL;
  }

  return *(void **)(&sym_addr);
}

char *qnn_wrapper_api::dlError(void) {
  char *retStr = sg_lastErrMsg;

  sg_lastErrMsg = const_cast<char *>("");

  return retStr;
}

char *qnn_wrapper_api::strnDup(const char *source, size_t maxlen) {
  size_t length = strnlen(source, maxlen);

  char *destination = (char *)malloc((length + 1) * sizeof(char));
  if (destination == nullptr) return nullptr;

  // copy length bytes to destination and leave destination[length] to be
  // null terminator
  strncpy_s(destination, length + 1, source, length);

  return destination;
}