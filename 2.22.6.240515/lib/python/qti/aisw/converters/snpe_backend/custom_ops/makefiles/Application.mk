APP_CPPFLAGS += -std=c++11
APP_ABI      := arm64-v8a
APP_STL      := c++_shared
APP_PLATFORM := android-21
APP_LDFLAGS  += -lc -lm -ldl
APP_ALLOW_MISSING_DEPS := false