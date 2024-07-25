 #=============================================================================
 #
 #  Copyright (c) 2023 Qualcomm Technologies, Inc.
 #  All Rights Reserved.
 #  Confidential and Proprietary - Qualcomm Technologies, Inc.
 #
 #=============================================================================

This is a sample app to demonstrate how to skip delegation ops that has floating point.

Please follow the steps below to setup your environment:

1. cd $QNN_SDK_ROOT/examples/QNN/TFLiteDelegate/SkipNodeExample

2. mkdir -p _build && cd _build

3. Ensure you have the required variables set: $ANDROID_NDK_ROOT, $QNN_SDK_ROOT, $TENSORFLOW_SOURCE_DIR

4. Run the following cmake command
```
cmake -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Debug\
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_NATIVE_API_LEVEL=23 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\
    -DQNN_SDK_ROOT=${QNN_SDK_ROOT}\
    -DTENSORFLOW_SOURCE_DIR=${TENSORFLOW_SOURCE_DIR} ..

make -j4
```

5. After the build is complete, push all necessary files to the device and execute the program.
Please refer to Qualcomm AI Engine Direct Delegate documentation for more information on the execution process.
For this sample, some necessary files include:
  - libQnnHtp.so
  - libQnnHtpPrepare.so
  - libQnnTFLiteDelegate.so
  - libQnnHtpV*Stub.so
  - libQnnHtpV*Skel.so
  - main
  - mix_precision_sample.tflite
