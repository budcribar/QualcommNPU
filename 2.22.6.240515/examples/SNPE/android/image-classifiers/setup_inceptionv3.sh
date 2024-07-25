#
# Copyright (c) 2018, 2019, 2023-2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

#############################################################
# Inception V3 setup
#############################################################

mkdir -p inception_v3
mkdir -p inception_v3/images

cd inception_v3

cp -R ../../../../Models/InceptionV3/data/cropped/*.jpg images
FLOAT_DLC="../../../../Models/InceptionV3/dlc/inception_v3.dlc"
QUANTIZED_DLC="../../../../Models/InceptionV3/dlc/inception_v3_quantized.dlc"
UDO_DLC="../../../../Models/InceptionV3/dlc/inception_v3_udo.dlc"
UDO_QUANTIZED_DLC="../../../../Models/InceptionV3/dlc/inception_v3_udo_quantized.dlc"
UDO_PACKAGE_PATH="../../../../Models/InceptionV3/SoftmaxUdoPackage/libs/arm64-v8a/"
UDO_DSP_PACKAGE_PATH="../../../../../Models/InceptionV3/SoftmaxUdoPackage/libs/dsp_v60/"

if [ -d ${UDO_PACKAGE_PATH} ]; then
    if [ -f ${UDO_QUANTIZED_DLC} ] ; then
        cp -R ${UDO_QUANTIZED_DLC} model.dlc
    elif [ -f ${UDO_DLC} ]; then
          cp -R ${UDO_DLC} model.dlc
    fi
else
    if [ -f ${QUANTIZED_DLC} ]; then
        cp -R ${QUANTIZED_DLC} model.dlc
    else
        cp -R ${FLOAT_DLC} model.dlc
    fi
fi
if [ -d ${UDO_PACKAGE_PATH} ]; then
    mkdir udo
    cd udo
    mkdir arm64-v8a
    mkdir dsp
    cp -R ../${UDO_PACKAGE_PATH}/* ./arm64-v8a/
    mv ./arm64-v8a/libUdoSoftmaxUdoPackageReg.so ./arm64-v8a/UdoPackageReg.so
    if [ -d ${UDO_DSP_PACKAGE_PATH} ]; then
        cp -R ${UDO_DSP_PACKAGE_PATH}/* ./dsp/
    fi
    rm -rf ./arm64-v8a/libc++_shared.so
    rm -rf ./arm64-v8a/libOpenCL.so
    cd ../
fi

cp -R ../../../../Models/InceptionV3/data/imagenet_slim_labels.txt labels.txt

zip -r inception_v3.zip ./*
mkdir -p ../app/src/main/res/raw/
cp inception_v3.zip ../app/src/main/res/raw/

cd ..
rm -rf ./inception_v3
