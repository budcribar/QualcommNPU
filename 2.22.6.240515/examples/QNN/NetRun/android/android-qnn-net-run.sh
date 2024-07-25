#!/usr/bin/env bash

#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
set -e
BACKEND_LIST="cpu;gpu;htp-v68;htp-v69;htp-v73;dsp-v66;dsp-v65;saver;htp-v75"

function helpMessage(){
cat << EOF
Usage: $(basename -- $0) [-h] [-b BACKEND_TO_USE] [-o OUTPUT_DIR]

Required argument(s):
  -b BACKEND_NAME                       Either ${BACKEND_LIST[*]}

optional argument(s):
  -o OUTPUT_DIR                         Location for saving output files. Default: Current Directory
EOF
}

while getopts "h?b:o:" opt;
do
  case "$opt" in
    h )
      helpMessage;
      exit 1
      ;;
    b )
      backend=$OPTARG
      retVal=0
      ;;
    o )
      OUTPUT_DIR=`readlink -f $OPTARG`
      ;;
    ? )
      helpMessage;
      exit 1
      ;;
  esac
done

if [[ -z "$backend" ]]
  then
    echo "ERROR: No Backend Specified.";
    helpMessage;
    retVal=1
fi

found=0
function exists_in_list() {
  IFS=";"
  for item in ${BACKEND_LIST}; do
    if [[ $1 == "$item" ]]
    then
      found=1
      return 
    fi
  done
  return
}

exists_in_list "$backend"
if [[ $found -eq 0 ]]
then
  echo "ERROR: Invalid Backend Specified."
  helpMessage
  exit 1
fi

if [[ "$backend" == "cpu" ||"$backend" == "gpu" || "$backend" == "saver" ]]
then
  MODEL=qnn_model_float
else
  MODEL=qnn_model_8bit_quantized
fi

SCRIPT_LOCATION=`dirname $(readlink -f ${0})`
QNN_SDK_ROOT=${SCRIPT_LOCATION}/../../../..
OUTPUT_DIR=${SCRIPT_LOCATION}

set --
source ${QNN_SDK_ROOT}/bin/envsetup.sh
unset --

MODEL_ROOT=${QNN_SDK_ROOT}/examples/QNN/converter/models
MODEL_LIBS=${SCRIPT_LOCATION}/model_libs
QNN_MODEL_LIB_GENERATOR=${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator
QNN_NET_RUN=${QNN_SDK_ROOT}/bin/aarch64-android/qnn-net-run
QNN_LIB_ROOT=${QNN_SDK_ROOT}/lib/aarch64-android
HEXAGON_V68_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so
HEXAGON_V69_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so
HEXAGON_V73_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so
HEXAGON_V75_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
HEXAGON_V66_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v66/unsigned/libQnnDspV66Skel.so
HEXAGON_V65_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v65/unsigned/libQnnDspV65Skel.so

echo "INFO: Running QNN Model Lib Generator."
export PATH=${ANDROID_NDK_ROOT}:${PATH}
${QNN_MODEL_LIB_GENERATOR} -c ${MODEL_ROOT}/${MODEL}.cpp -b ${MODEL_ROOT}/${MODEL}.bin -o ${MODEL_LIBS}

TARGET_ROOT=/data/local/tmp/qnn/${MODEL}
# Push inputs, generated model, and executable
echo "INFO: Pushing required files onto device."
adb shell "mkdir -p "${TARGET_ROOT}
adb push ${QNN_NET_RUN} ${TARGET_ROOT}
adb push ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so ${TARGET_ROOT}
adb push ${MODEL_LIBS}/aarch64-android/lib${MODEL}.so ${TARGET_ROOT}
adb push ${MODEL_ROOT}/input_list_float.txt ${TARGET_ROOT}
adb push ${MODEL_ROOT}/input_data_float ${TARGET_ROOT}

case "$backend" in
  cpu )
    adb push ${QNN_LIB_ROOT}/libQnnCpu.so ${TARGET_ROOT}
    QNN_NET_RUN_CMD="./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnCpu.so"
    ;;
  saver )
    adb push ${QNN_LIB_ROOT}/libQnnSaver.so ${TARGET_ROOT}
    QNN_NET_RUN_CMD="./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnSaver.so"
    ;;
  gpu )
    adb push ${QNN_LIB_ROOT}/libQnnGpu.so ${TARGET_ROOT}
    QNN_NET_RUN_CMD="./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnGpu.so"
    ;;
  htp-v68 )
    adb push ${QNN_LIB_ROOT}/libQnnHtp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpPrepare.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpV68Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V68_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH=/vendor/dsp/cdsp:/vendor/lib64/:\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so"
    ;;
  htp-v69 )
    adb push ${QNN_LIB_ROOT}/libQnnHtp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpPrepare.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpV69Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V69_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH="${TARGET_ROOT}":/vendor/dsp/cdsp:/vendor/lib64/:\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so"
    ;;
  htp-v73 )
    adb push ${QNN_LIB_ROOT}/libQnnHtp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpPrepare.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpV73Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V73_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH="${TARGET_ROOT}":/vendor/dsp/cdsp:/vendor/lib64/:\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so"
    ;;
  htp-v75 )
    adb push ${QNN_LIB_ROOT}/libQnnHtp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpPrepare.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpV75Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V75_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH="${TARGET_ROOT}":/vendor/dsp/cdsp:/vendor/lib64/:\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so"
    ;;
  
  dsp-v66 )
    adb push ${QNN_LIB_ROOT}/libQnnDspV66Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V66_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH=/vendor/dsp/cdsp:/vendor/lib64/:\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnDspV66Stub.so"
    ;;
  dsp-v65 )
    adb push ${QNN_LIB_ROOT}/libQnnDspV65Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V65_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH=/vendor/dsp/cdsp:/vendor/lib64/:\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnDspV65Stub.so"
    ;;
esac

echo "INFO: Executing qnn-net-run."
ADB_SHELL_CMD="export LD_LIBRARY_PATH=/data/local/tmp/qnn/"${MODEL}" && cd /data/local/tmp/qnn/"${MODEL}" && "${QNN_NET_RUN_CMD}
adb shell ${ADB_SHELL_CMD}


echo "INFO: Pulling results from device."
adb pull ${TARGET_ROOT}/output ${SCRIPT_LOCATION}

#move output files to OUTPUT_DIR
if [[ "${OUTPUT_DIR}" != "${SCRIPT_LOCATION}" ]];
then
  if [[ ! -d ${OUTPUT_DIR} ]]
  then
    echo "INFO: Creating Output dir ${OUTPUT_DIR}."
    mkdir -p ${OUTPUT_DIR}
  fi
  mv -f output ${OUTPUT_DIR}/.
fi

#cleaning up generated files
trap "rm -rf ${MODEL_LIBS}" EXIT

echo "INFO: Done."

set +e
