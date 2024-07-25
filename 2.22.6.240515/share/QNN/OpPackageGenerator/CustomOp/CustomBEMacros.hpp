//============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

namespace qnn {

namespace custom {

namespace macros {

#define QNN_CUSTOM_BE_ENSURE(value, return_error) \
  do {                                            \
    if (!(value)) {                               \
      return return_error;                        \
    }                                             \
  } while (0);

#define QNN_CUSTOM_BE_ENSURE_STATUS(status) \
  do {                                      \
    if ((status) != QNN_SUCCESS) {          \
      return status;                        \
    }                                       \
  } while (0);

#define QNN_CUSTOM_BE_ENSURE_EQ(a, b, return_error) \
  do {                                              \
    if ((a) != (b)) {                               \
      return return_error;                          \
    }                                               \
  } while (0);

}  // namespace macros
}  // namespace custom
}  // namespace qnn