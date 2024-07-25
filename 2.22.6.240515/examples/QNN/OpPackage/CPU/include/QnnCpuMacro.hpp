//============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#define QNN_CPU_BE_ENSURE(value, return_error) \
  do {                                         \
    if (!(value)) {                            \
      return return_error;                     \
    }                                          \
  } while (0)

#define QNN_CPU_BE_ENSURE_STATUS(status, return_error) \
  do {                                                 \
    if ((status) != QNN_SUCCESS) {                     \
      return return_error;                             \
    }                                                  \
  } while (0)

#define QNN_CPU_BE_ENSURE_EQ(a, b, return_error) \
  do {                                           \
    if ((a) != (b)) {                            \
      return return_error;                       \
    }                                            \
  } while (0)
