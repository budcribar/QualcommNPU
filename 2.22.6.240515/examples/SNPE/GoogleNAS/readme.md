# Using SNPE with GoogleNAS

This guide will walk through using the Qualcomm SNPE SDK with
Google NAS. This should enable training models while taking into account
HW specific accuracy and latency measurements and thus producing a model
better suited for running on HW.

## What is provided
1. Client example docker file
```<SDK_ROOT>/examples/GoogleNAS/Dockerfile```
2. Docker build script
```<SDK_ROOT>/examples/GoogleNAS/build.sh```
3. HIL framework and device automation framework
```<SDK_ROOT>/target/x86_64-linux-clang/python/qti/aisw/nas/```
4. End to end example
```<SDK_ROOT>/examples/GoogleNAS/run_example.sh```


## Client Provided

1. Model Training Scripts (for running on Google NAS)
2. Optional python module which implements SNPE client interface
3. Appropriate version of Google NAS code for your project
4. Qnn SDK
5. Android SDK
6. Optional certificates

## Setup

### GCloud Authentication
```sh
gcloud config set project $PROJECT_ID
gcloud auth login --no-launch-browser
gcloud auth application-default login --no-launch-browser
```

First get your Google Auth Library setup with the Google account which has access to GoogleNAS
https://source.developers.google.com/auth/start?scopes=https://www.googleapis.com/auth/cloud-platform&state=

Make sure any required certificates are obtained as they will be setup in the docker during build.

Note that there are other ways to authenticate such as service account credentials. Please check with Google for setting these up.

### Syncing the Google NAS repository

Open a new local Shell Terminal.
Click this link to generate and store your Git credentials https://source.developers.google.com/auth/start?scopes=https://www.googleapis.com/auth/cloud-platform&state=
Copy and paste the Git credentials into the Terminal.
Run one of the following git clone commands:

For Tensorflow:
git clone https://source.developers.google.com/p/cloud-nas-260507/r/nas-codes-release

For Pytorch:
git clone https://source.developers.google.com/p/cloud-nas-260507/r/cloud-nas-pytorch

### Android NDK

When leveraging a device based on android you will need the appropriate android NDK present in order to compile and run the model.

You can obtain the NDK here: https://developer.android.com/ndk/downloads

Please check the NDK version to use in the SNPE documentation for the SNPE SDK release.

### Config file
The config file drives the configuration of the NAS HIL process. This is where clients indicate how the model should be converted, quantization data/parameters to use, inference data to use, etc

```sh

    "Name":"<Model Name>",
    "HostRootPath": "<Root path for user data in the docker: eg /snpe/user_data>",
    "HostResultsDir":"<Result Directory name in the docker, eg /snpe/output>",
    "DevicePath":"<Path to store results on device e.g: /data/local/tmp/nas_artifacts>",
    "Devices":["List of <device id as output by 'adb devices'>"],
    "Runs":<No. of runs, default=1>,

    "Model": {
        "Name": "<Model name>",
        "Data": ["<Host path in the docker to input images folder to push to device. ; example: /snpe/user_data/images, or just 'images'"],
        "InputList": "<Path to imagelist in the user_data directory, eg /snpe/user_data/input_list.txt or just 'input_list.txt>",
        "Conversion": {
          "Inputs": [ { <name>: "1,227,227,3"} ]
          "Outputs": [ <model output tensor names ]
          "Command": "<Full converter command. Model name and data input, etc  will overwritten w/above options. Eg "snpe-tensorflow-to-dlc -i saved_model -d x 1,28,28 --out_node Identity">
        },
        "Quantization": {
          "Command": "<Quantization command specifying all required and desired quantization options. Eg snpe-dlc-quantize --input_dlc model.dlc --input_list input_list.txt --axis_quant --use_encoding_optimizations --bias_bitwidth=32 --act_bitwidth=16>"
        }i
        "CustomProcessing": "<Path to optional python script for custom model and stat processing"
    },

    "Backends":"<One of the supported runtimes: "CPU","GPU", or "DSP>",
    "Chipset": "<Targeted chipset, eg "8550">"
}

```

Note that the inputs listed in "InputList" should use relative paths. Both the InputList and Data should ultimately be stored in the same directory so the paths to the inputs should be relative to that same directory.
## The Example

To run the end to end HIL example simply fill in the required parameters in the example script here:
```<SDK_ROOT>/examples/GoogleNAS/run_example.sh```

1. The path to the extracted SNPE SDK:
```SDK_ROOT="<PATH_TO_SDK>"```

2. The location of the downloaded Google NAS code. See above regarding syncing the Google NAS code:
```NAS_CODE_LOCATION=""```

3. Default location of the compiler to be used:
```TARGET_COMPILER="<eg Toolchain/Android/android-ndk-r19c>"```

4. Location of the SSL certificate, if required:
```CERT_LOCATION="<eg /etc/ssl/certs/ca-certificates.crt>"```

5. Default location for the output of the experiment:
```HIL_OUTPUT="$basedir/output" # Output location```

6. Location of the user data (images, input_list, etc):
```USER_HIL_DATA="$basedir/user_data"```

The following keys should be setup according to your Google NAS project information

7. Google NAS project name
```PROJECT=""```

8. This should be the GCS region you will be running your experiments (eg us-central1 from the list located here: https://cloud.google.com/ai-platform/training/docs/regions)
```REGION=""```

9. Choose a GCS bucket for the output directory. Make sure this GCS bucket exists in your Google Cloud project and that your user has permission to read/write
```GCS_ROOT_DIR="gs://my-gcs-bucket"```

10. Service Account credentials (if using a Google service account)
```SERVICE_ACCOUNT_KEY=""```

### Example breakdown

1. Setting up the required package paths, input/output paths, and Google NAS configuration information
2. Authenticating google nas. The example uses the standard login and application login to authenticate but this could also be accomplished using a service account (preferred method for >8 hour training)
3. Building the trainer code. Build the training docker and push to your GCS project
4. Running NAS search. This kick starts the Google NAS trainer/search space example. This runs on Google cloud.
5. Setup SNPE and build the hardware in the loop (HIL) docker that will run on your local machine
6. Run the HIL example end to end by first kickstarting the docker, setting up SNPE, and then running the HIL script

### Running the example

First make sure you've installed Google Cloud on your local machine.
Simply run the example by executing:
```bash
source <SDK_ROOT>/examples/GoogleNAS/run_example.sh
```

## Adding HW in the loop

For descriptions about the Google NAS Vertex API please refer to the
Google documentation. This should provide you with the appropriate references
for setting up the google cloud client, configuring the service, network and login
information. In addition, they provide examples of how to implement the training
side scripts.

To override model or statistic processing functions simply refer to the examples/GoogleNAS/custom_processing.py example. There are two functions which can be overridden:

1. A model processing function
```py
# (optional) Process the model to make it compatible with SNPE (Eg Keras saved_model -> frozen graph.pb)
def process_model(model_path: str) -> str:
```
2. A function to customize the statistics returned to the NAS service
```py
# (optional) Modify the overrides:
def process_stats(stats) -> map:
```
