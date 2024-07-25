#!/usr/bin/env bash

#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#

set -e
#set -x

helpMessage(){
cat << EOF
Usage: $(basename -- $0) [-h] [-b BACKEND_TO_USE] [-o OUTPUT_DIR]

Required argument(s):
 -b BACKEND_NAME                       Either cpu, htp or saver

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

if [[ "$backend" != "cpu" && "$backend" != "htp"  && "$backend" != "saver" ]]
then
  echo "ERROR: Invalid Backend Specified."
  helpMessage
  exit 1
fi

SCRIPT_LOCATION=`dirname $(readlink -f ${0})`
QNN_SDK_ROOT=${SCRIPT_LOCATION}/../../../..
OUTPUT_DIR=${SCRIPT_LOCATION}

set --
source ${QNN_SDK_ROOT}/bin/envsetup.sh

MODEL_ROOT=${QNN_SDK_ROOT}/examples/QNN/converter/models
MODEL_LIBS=${SCRIPT_LOCATION}/model_libs

QNN_MODEL_LIB_GENERATOR=${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator
QNN_NET_RUN=${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-net-run
export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang/:${MODEL_LIBS}/x86_64-linux-clang:${LD_LIBRARY_PATH}

# copy input files and input_list to SCRIPT directory
cp -r ${MODEL_ROOT}/input_data_float ${SCRIPT_LOCATION}
cp ${MODEL_ROOT}/input_list_float.txt ${SCRIPT_LOCATION}

case "$backend" in
  cpu )
    MODEL=qnn_model_float
    echo "INFO: Running QNN Model Lib Generator."
    ${QNN_MODEL_LIB_GENERATOR} -c "${MODEL_ROOT}"/"${MODEL}".cpp -b "${MODEL_ROOT}"/"${MODEL}".bin -o "${MODEL_LIBS}" -t x86_64-linux-clang
    echo "INFO: Running qnn-net-run"
    ${QNN_NET_RUN} --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnCpu.so
    ;;
  saver )
    MODEL=qnn_model_float
    echo "INFO: Running QNN Model Lib Generator."
    ${QNN_MODEL_LIB_GENERATOR} -c "${MODEL_ROOT}"/"${MODEL}".cpp -b "${MODEL_ROOT}"/"${MODEL}".bin -o "${MODEL_LIBS}" -t x86_64-linux-clang
    echo "INFO: Running qnn-net-run"
    ${QNN_NET_RUN} --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnSaver.so
    mkdir -p saver_output/non_quantized
    mv -f saver_output/saver_output.c saver_output/non_quantized/
    mv -f saver_output/params.bin saver_output/non_quantized/

    MODEL=qnn_model_8bit_quantized
    echo "INFO: Running QNN Model Lib Generator."
    ${QNN_MODEL_LIB_GENERATOR} -c "${MODEL_ROOT}"/"${MODEL}".cpp -b "${MODEL_ROOT}"/"${MODEL}".bin -o "${MODEL_LIBS}" -t x86_64-linux-clang
    echo "INFO: Running qnn-net-run"
    ${QNN_NET_RUN} --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnSaver.so
    mkdir -p saver_output/quantized
    mv -f saver_output/saver_output.c saver_output/quantized/
    mv -f saver_output/params.bin saver_output/quantized/
    ;;
  htp )
    MODEL=qnn_model_8bit_quantized
    echo "INFO: Running QNN Model Lib Generator."
    ${QNN_MODEL_LIB_GENERATOR} -c "${MODEL_ROOT}"/"${MODEL}".cpp -b "${MODEL_ROOT}"/"${MODEL}".bin -o "${MODEL_LIBS}" -t x86_64-linux-clang
    echo "INFO: Running qnn-net-run"
    ${QNN_NET_RUN} --model lib"${MODEL}".so --input_list input_list_float.txt --backend libQnnHtp.so
    ;;
esac

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

# cleaning up copied and generated files
cleanUp() {
  rm -rf ${SCRIPT_LOCATION}/input_data_float
  rm -rf ${SCRIPT_LOCATION}/input_list_float.txt
  rm -rf ${MODEL_LIBS}
}

trap cleanUp EXIT

echo "INFO: Done."

#set +x
set +e
