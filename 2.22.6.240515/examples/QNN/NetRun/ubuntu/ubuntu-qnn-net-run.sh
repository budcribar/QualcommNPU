#!/usr/bin/env bash

#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#

# This script makes the assumption that there is only one Android device connected at a time.
# If the user would like to execute this script in an environment containing multiple
# devices, ANDROID_SERIAL environment variable will have to be set accordingly before
# executing this script, or any other method that lets the adb server to appropriately
# recognize the intended device.

set -e
#set -x

helpMessage(){
cat << EOF
Usage: $(basename -- $0) [-h] [-b BACKEND_TO_USE] [-o OUTPUT_DIR] [-a ARCHITECTURE]

Required argument(s):
 -b BACKEND_NAME                       Either cpu, gpu, htp-v68,htp-v69, dsp-v66, dsp-v65, or saver

optional argument(s):
 -o OUTPUT_DIR                         Location for saving output files. Default: Current Directory
 -a ARCHITECTURE                       Architecture to use. Possible options are 'aarch64-gcc75' for 'aarch64-gcc7.5'.
                                       Default is 'aarch64-gcc75'.
EOF
}

architecture="aarch64-gcc75"
while getopts "h?b:o:a:" opt;
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
    a )
      architecture=$OPTARG
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

if [[ "$backend" != "cpu" && "$backend" != "gpu" && "$backend" != "htp-v68" && "$backend" != "htp-v69" && "$backend" != "dsp-v66" && "$backend" != "dsp-v65" && "$backend" != "saver" && "$backend" != "hta" ]];
then
  echo "ERROR: Invalid Backend Specified."
  helpMessage
  exit 1
fi

if [[ "$architecture" == "aarch64-gcc75" ]]
then
  arch_variant="aarch64-ubuntu-gcc7.5"
  VENDOR_LIB="/usr/lib/"
else
  echo "ERROR: Wrong Architecture Provided."
  helpMessage
  exit 1
fi

SCRIPT_LOCATION=`dirname $(readlink -f ${0})`
QNN_SDK_ROOT=${SCRIPT_LOCATION}/../../../..
OUTPUT_DIR=${SCRIPT_LOCATION}

set --
source ${QNN_SDK_ROOT}/bin/envsetup.sh
unset --

MODEL_ROOT=${QNN_SDK_ROOT}/examples/QNN/converter/models
QNN_MODEL_LIB_GENERATOR=${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator
QNN_NET_RUN=${QNN_SDK_ROOT}/bin/${arch_variant}/qnn-net-run
QNN_LIB_ROOT=${QNN_SDK_ROOT}/lib/${arch_variant}
HEXAGON_V68_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so
HEXAGON_V69_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so
HEXAGON_V66_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v66/unsigned/libQnnDspV66Skel.so
HEXAGON_V65_SKEL_PATH=${QNN_SDK_ROOT}/lib/hexagon-v65/unsigned/libQnnDspV65Skel.so

if [[ "$backend" == "htp-v68" ||"$backend" == "htp-v69" || "$backend" == "dsp-v66" || "$backend" == "dsp-v65" || "$backend" == "hta" ]]
then
  MODEL=qnn_model_8bit_quantized
else
  MODEL=qnn_model_float
fi

MODEL_LIBS=${SCRIPT_LOCATION}/model_libs

echo "INFO: Running QNN Model Lib Generator."
${QNN_MODEL_LIB_GENERATOR} -c ${MODEL_ROOT}/${MODEL}.cpp -b ${MODEL_ROOT}/${MODEL}.bin -o ${MODEL_LIBS} -t ${arch_variant}

TARGET_ROOT=/data/local/tmp/qnn/${MODEL}
# Push inputs, generated model, and executable
echo "INFO: Pushing required files onto device."
adb shell "mkdir -p "${TARGET_ROOT}
adb push ${QNN_NET_RUN} ${TARGET_ROOT}
adb push ${MODEL_LIBS}/${arch_variant}/lib${MODEL}.so ${TARGET_ROOT}
adb push ${MODEL_ROOT}/input_list_float.txt ${TARGET_ROOT}
adb push ${MODEL_ROOT}/input_data_float ${TARGET_ROOT}

case "$backend" in
  cpu )
    adb push ${QNN_LIB_ROOT}/libQnnCpu.so ${TARGET_ROOT}
    QNN_NET_RUN_CMD="./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnCpu.so"
    ;;
  hta )
    adb push ${QNN_LIB_ROOT}/libhta_hexagon_runtime_qnn.so ${TARGET_ROOT}
    adb push ${QNN_LIB_ROOT}/libQnnHta.so ${TARGET_ROOT}
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH="${VENDOR_LIB}":"${TARGET_ROOT}":\$LD_LIBRARY_PATH"
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/dsp/cdsp;/usr/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHta.so"
    ;;
  saver )
    adb push ${QNN_LIB_ROOT}/libQnnSaver.so ${TARGET_ROOT}
    QNN_NET_RUN_CMD="./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnSaver.so"
    ;;
  gpu )
    adb push ${QNN_LIB_ROOT}/libQnnGpu.so ${TARGET_ROOT}
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH="${VENDOR_LIB}":\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnGpu.so"
    ;;
  htp-v68 )
    adb push ${QNN_LIB_ROOT}/libQnnHtp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpPrepare.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpV68Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V68_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}":/dsp/cdsp:/usr/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH=/dsp/cdsp:"${VENDOR_LIB}":\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so"
    ;;
  htp-v69 )
    adb push ${QNN_LIB_ROOT}/libQnnHtp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpPrepare.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnHtpV69Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V69_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}":/dsp/cdsp:/usr/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH="${TARGET_ROOT}":/dsp/cdsp:"${VENDOR_LIB}":\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ""./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so"
    ;;
  dsp-v66 )
    adb push ${QNN_LIB_ROOT}/libQnnDsp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnDspV66Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V66_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/dsp/cdsp;/usr/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH=/dsp/cdsp:"${VENDOR_LIB}":\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnDsp.so"
    ;;
  dsp-v65 )
    adb push ${QNN_LIB_ROOT}/libQnnDsp.so ${TARGET_ROOT}
    adb wait-for-device push ${QNN_LIB_ROOT}/libQnnDspV65Stub.so ${TARGET_ROOT}
    adb wait-for-device push ${HEXAGON_V65_SKEL_PATH} ${TARGET_ROOT}
    ADSP_LIBRARY_PATH="export ADSP_LIBRARY_PATH=\""${TARGET_ROOT}";/dsp/cdsp;/usr/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\""
    LD_LIBRARY_PATH="export LD_LIBRARY_PATH=/dsp/cdsp:"${VENDOR_LIB}":\$LD_LIBRARY_PATH"
    QNN_NET_RUN_CMD="${ADSP_LIBRARY_PATH} && ${LD_LIBRARY_PATH} && ./qnn-net-run --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnDsp.so"
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

#set +x
set +e
