#!/usr/bin/env bash

#
#  Copyright (c) 2022, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#

#set -e
#set -x

helpMessage(){
cat << EOF
Usage: $(basename -- $0) [-h] [-q QNN_SDK_ROOT] [-c CERT_PATH] [-p PROJECT NAME] [-d DOCKER_NAME] [-r REGION] [-a ANDROID_NDK] [-o OUTPUT_DIR]

Required argument(s):
 -n NAS_CODE_ROOT           NAS_CODE, pass the path to the sync'd nas code
 -p PROJECT_NAME            GCS PROJECT name, must be same as search
 -d DOCKER_NAME             Local HIL DOCKER name
 -r REGION                  GCS Region name, must be same as search+GCS bucket

optional argument(s):
 -q QNN_SDK_ROOT            QNN SDK ROOT, either provided explicitly or set to the QNN_SDK_ROOT environment variable.
 -a ANDROID_NDK             ANDROID NDK PATH
 -c CERT_PATH               Any SSL certificate that must be registered as part of Google NAS setup
 -o OUTPUT_DIR              Location for saving output files. Default: Current Directory

EOF
}

errorhdlr () {
  rm -rf /tmp/nas_build
  rm -rf ./nas_build
}
trap errorhdlr ERR EXIT

QNN_SDK_ROOT=${QNN_SDK_ROOT}
copy_sdk=0
OUTPUT_DIR="`pwd`/."
CERT_CMD=""
ANDROID_NDK_DIR=""
NAS_DIR=""
REGION=""
PROJECT_NAME=""
DOCKER_NAME="qc_hil"
while getopts "h?a:c:d:p:r:n:o:q:" opt;
do
  case "$opt" in
    h )
      helpMessage;
      exit 1
      ;;
    a )
      ANDROID_SDK_DIR=`readlink -f $OPTARG`
      ;;
    c )
      if [[ -f "$OPTARG" ]]; then
        #CERT_PATH=`dirname $(readlink -f $OPTARG)`
        CERT_FILE=$OPTARG
        #CERT_FILE=`basename $OPTARG`
        #NAS_CERT="/qnn/cert/$CERT_FILE"
        #CERT_CMD="-v $OPTARG:$NAS_CERT -e NAS_CERT=$NAS_CERT"
        echo "Setting QNN cert to $OPTARG"
      fi

      ;;
    n )
      NAS_DIR=$OPTARG

      ;;
    o )
      OUTPUT_DIR=$OPTARG
      ;;
    d )
      DOCKER_NAME=$OPTARG
      ;;
    p )
      PROJECT_NAME=$OPTARG
      ;;
    r )
      REGION=$OPTARG
      ;;
    q )
      QNN_SDK_ROOT=$OPTARG
      copy_sdk=1
      echo "Mapping QNN SDK Root: $QNN_SDK_ROOT to /qnn/sdk"
      ;;
    ? )
      helpMessage;
      exit 1
      ;;
  esac
done

if [[ ! -d "$NAS_DIR" ]]
then
  echo "Invalid NAS directory '$NAS_DIR' provided. Please sync the NAS code and pass the path using -n <NAS_DIR>"
  helpMessage
  exit 1
fi

#move output files to OUTPUT_DIR
if [[ ! -d ${OUTPUT_DIR} ]]
then
  echo "INFO: Creating Output dir ${OUTPUT_DIR}."
  mkdir -p ${OUTPUT_DIR}
fi

rm -rf /tmp/nas_build
mkdir -p /tmp/nas_build
if [[ -d "$QNN_SDK_ROOT" ]]; then
  if [[ $copy_sdk -eq 1 ]]; then
    echo "Copying $QNN_SDK_ROOT to 'sdk'"
    cp -r -L $QNN_SDK_ROOT /tmp/nas_build/sdk
  fi
  #cp -r `dirname $(readlink -f $QNN_SDK_ROOT)` qnn_tmp/sdk
fi

echo "Copying $NAS_DIR to 'nas'"
cp -r $NAS_DIR /tmp/nas_build/nas
if [[ -f "$ANDROID_NDK_DIR" ]]; then
  echo "Copying $ANDROID_NDK_DIR to 'android_sdk'"
  cp -r $ANDROID_NDK_DIR /tmp/nas_build/android_ndk
fi
if [[ -f "$CERT_FILE" ]]; then
  mkdir /tmp/nas_build/cert
  cp -r $CERT_FILE /tmp/nas_build/cert/ca-certificates.crt
fi
mv /tmp/nas_build ./

REGION_CMD=""
PROJECT_CMD=""
if [[ ! -z $REGION ]]
then
  REGION_CMD="--build-arg REGION=$REGION"
fi
if [[ ! -z $PROJECT_NAME ]]
then
  PROJECT_CMD="--build-arg PROJECT=$PROJECT_NAME"
fi

# Build the docker
DOCKER_BUILD_CMD="docker build -f $QNN_SDK_ROOT/examples/QNN/GoogleNAS/Dockerfile \
                         --build-arg USER_ID=$(id -u ${USER}) \
                         --build-arg GROUP_ID=$(id -g ${USER}) \
                         $REGION_CMD $PROJECT_CMD \
                         -t "gcr.io/$PROJECT_NAME/$DOCKER_NAME" ."

export PROJECT_NAME=$PROJECT_NAME
export DOCKER_NAME=$DOCKER_NAME
export REGION=$REGION
echo $DOCKER_BUILD_CMD
$DOCKER_BUILD_CMD

rm -rf /tmp/nas_build
rm -rf ./nas_build

#set +x
#set +e
