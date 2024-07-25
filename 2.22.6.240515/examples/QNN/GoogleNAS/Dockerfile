# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

# Replace with some other base image if you require a specific tensorflow/pytorch/etc version
FROM tensorflow/tensorflow:2.6.0

# Update the apt configuration
# Need to add a real version of Python, so that crad-docker works.
RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y && \
    apt-get install -y --no-install-recommends \
    python python-pip \
    sudo

# Add sudo support
RUN echo "%users ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers

# TZ
RUN TZ="America/Los_Angeles" DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get install -y --no-install-recommends \

# Editors
        vim \
        wget \
        curl \
        zip unzip \
        ssh \
        g++ lib32gcc1 gcc-multilib \
        gdb ddd valgrind \
        python-virtualenv \
        virtualenvwrapper \
        environment-modules \
        python3-dev \
        python3-pip \
        python3-tk \
# Gcloud deps
        apt-transport-https ca-certificates gnupg \
# Debugging/device
        usbutils \
        adb \
# Finally, clean up any intermediate files
      && \
       apt-get autoremove -y \
# clean up any core files
      && \
       rm -f /core*

#ENV DEBIAN_FRONTEND=noninteractive

# Install google-cloud-sdk to enable "gcloud" cli.
# The install command is documented here:
# https://cloud.google.com/sdk/docs/install#deb under step-3.
RUN echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
apt-get update && sudo apt-get install google-cloud-cli

RUN pip install --upgrade pip==21.3.1 && \
    pip install Pillow==8.4.0 && \
    pip install fsspec==2021.10.1 && \
    pip install gcsfs==2021.10.1 && \
    pip install pyyaml && \
    pip install pandas && \
    pip install invoke

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ARG USER_ID
ARG GROUP_ID

# creating the group and user based on two args
RUN groupadd -g $GROUP_ID -o nas && useradd -m -u $USER_ID -g $GROUP_ID -o -s /bin/bash nas

# Prepare the environment for mapped in artifacts
ADD nas_build /qnn

# changing the work dir to the specified user id and group id
RUN chown -R $USER_ID:$GROUP_ID /qnn

# running the below command in container using the user
USER nas

# Optional arguments to setup the default Google NAS environment
ARG REGION
ARG PROJECT
ENV REGION=$REGION
ENV PROJECT=$PROJECT

ENTRYPOINT ["/bin/bash"]
