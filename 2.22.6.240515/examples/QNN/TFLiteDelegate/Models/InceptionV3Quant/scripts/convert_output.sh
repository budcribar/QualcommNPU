#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
cd ${QNN_SDK_ROOT}/examples/QNN/TFLiteDelegate/Models/InceptionV3Quant
mkdir -p output_android/Result_0
mkdir -p output_android/Result_1
mkdir -p output_android/Result_2
mkdir -p output_android/Result_3

cp output/Result_0/InceptionV3/Predictions/Reshape_1.raw output_android/Result_0/InceptionV3_Predictions_Reshape_1_0.raw
cp output/Result_1/InceptionV3/Predictions/Reshape_1.raw output_android/Result_1/InceptionV3_Predictions_Reshape_1_0.raw
cp output/Result_2/InceptionV3/Predictions/Reshape_1.raw output_android/Result_2/InceptionV3_Predictions_Reshape_1_0.raw
cp output/Result_3/InceptionV3/Predictions/Reshape_1.raw output_android/Result_3/InceptionV3_Predictions_Reshape_1_0.raw