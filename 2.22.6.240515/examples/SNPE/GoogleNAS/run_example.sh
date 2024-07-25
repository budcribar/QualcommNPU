#!/bin/bash
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

origdir=`pwd`

basedir=/tmp/google_nas

# Fill in the following variables based on your installation and configuration
echo
echo "********************************************************************************"
echo "* 1. Setting up paths                                                          *"
echo "********************************************************************************"
SDK_ROOT="" # This is the path to the root of the extracted SNPE SDK
NAS_CODE_LOCATION="" # Location of the downloaded Google NAS code as per the instructions in the readme
TARGET_COMPILER="" # Location of the compiler, eg Android/android-ndk-r19c"
CERT_LOCATION="" # SSL Certificate if necessary
HIL_OUTPUT="$basedir/output" # Output location
USER_HIL_DATA="$basedir/user_data"

# NAS Configuration keys
PROJECT="" # This is your Google Project name
REGION="" # This should be the GCS region you will be running your experiments (eg us-central1) https://cloud.google.com/ai-platform/training/docs/regions
GCS_ROOT_DIR="" # Choose a GCS bucket for the output directory, eg gs://my-project-bucket
SERVICE_ACCOUNT_KEY="" # Google service account json key (optional)

# Setup the temp directory for input/outputs
mkdir -p $basedir
mkdir -p $HIL_OUTPUT
mkdir -p $USER_HIL_DATA

# Setup the input data for the example
cd $USER_HIL_DATA
python $SDK_ROOT/examples/SNPE/GoogleNAS/mnist_example_data.py

# May or may not need this based on authentication requirements (eg if you're using a service account)
echo
echo "********************************************************************************"
echo "* 2. Authenticate Google NAS                                                   *"
echo "********************************************************************************"
SERVICE_ACCOUNT_CMD=""
GAUTH_CMD=""
if [ -f "$SERVICE_ACCOUNT_KEY" ]; then
    echo "Authenticating with service account key: $SERVICE_ACCOUNT_KEY"
    gcloud auth activate-service-account --key-file $SERVICE_ACCOUNT_KEY
    SERVICE_ACCOUNT_CMD="-v $SERVICE_ACCOUNT_KEY:/etc/nas_secrets/sa-key.json"
else
    echo "Authenticating with standard authentication"
    gcloud auth login --no-browser
    gcloud auth application-default login --no-browser
    GAUTH_CMD="-v ~/.config:/home/nas/.config"
fi

echo "Setting GoogleNAS project $PROJECT"
gcloud config set project $PROJECT
echo "Configuring GoogleNAS for docker"
gcloud auth configure-docker --quiet

echo
echo "********************************************************************************"
echo "* 3. Building trainer code                                                     *"
echo "********************************************************************************"
# Set a unique docker-id below. It is a good practice to add your user-name
# to prevent overwriting another user's docker image.
TUTORIAL_DOCKER_ID=${USER}_tutorial4

# Setting a unique job-id so that subsequent job-runs
# do not have naming conflict.
DATE="$(date '+%Y%m%d_%H%M%S')"
JOB_NAME="tutorial4_${DATE}"

cd $NAS_CODE_LOCATION

# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=$PROJECT \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial4.Dockerfile \

echo
echo "********************************************************************************"
echo "* 4. Running NAS search                                                        *"
echo "********************************************************************************"
# Run search algo
NAS_JOB_OUTPUT=$(python3 vertex_nas_cli.py search \
--project_id=$PROJECT \
--region=$REGION \
--job_name="${JOB_NAME}" \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--search_space_module=tutorial.search_spaces.mnist_list_of_dictionary_search_space \
--accelerator_type="" \
--nas_target_reward_metric="nas_reward" \
--root_output_dir=${GCS_ROOT_DIR} \
--max_nas_trial=5 \
--max_parallel_nas_trial=1 \
--max_failed_nas_trial=1 \
--target_device_type=CPU \
--search_docker_flags \
num_epochs=2 \
target_latency_millisec=3.0)

echo "$NAS_JOB_OUTPUT"

NAS_JOB_ID=`echo "$NAS_JOB_OUTPUT" | grep -oP 'NAS Search job ID: \K\w+'`

echo "Found NAS job ID: $NAS_JOB_ID. If this looks wrong, please correct in the command below."

echo
echo "********************************************************************************"
echo "* 5. Setup SNPE and build the HIL (hardware in the loop) docker                 *"
echo "********************************************************************************"
cd $basedir
#. $SDK_ROOT/bin/envsetup.sh -t .

echo "$SDK_ROOT/examples/SNPE/GoogleNAS/build.sh -c $CERT_LOCATION -n $NAS_CODE_LOCATION -p $PROJECT -d qc_hil -r $REGION"
$SDK_ROOT/examples/SNPE/GoogleNAS/build.sh -c $CERT_LOCATION -n $NAS_CODE_LOCATION -p $PROJECT -d qc_hil -r $REGION

PROJECT_ID=`gcloud projects describe $PROJECT --format="value(projectNumber)"`
echo
echo "********************************************************************************"
echo "* 6. Run the SNPE HIL example                                                    *"
echo "********************************************************************************"

echo "To run your experiment in the docker:"
echo "source /snpe/sdk/examples/GoogleNAS/setup.sh && python3 /snpe/sdk/lib/python/qti/aisw/nas/execute_hil.py --service_endpoint=$REGION-aiplatform.googleapis.com --project_id=$PROJECT  --nas_job_id=projects/$PROJECT_ID/locations/$REGION/nasJobs/$NAS_JOB_ID --latency_calculator_config  /snpe/sdk/examples/GoogleNAS/config.json"

docker run -it --net host --privileged -u $(id -u):$(id -g) \
$GAUTH_CMD \
$SERVICE_ACCOUNT_CMD \
-v /dev/bus/usb:/dev/bus/usb \
-v $TARGET_COMPILER:/snpe/compiler \
-v $SDK_ROOT:/snpe/sdk \
-v $HIL_OUTPUT:/snpe/output \
-v $USER_HIL_DATA/:/snpe/user_data  \
gcr.io/$PROJECT/qc_hil

cd $origdir
