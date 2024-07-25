# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

source /snpe/sdk/bin/envsetup.sh -t .
if [[ -f "/snpe/cert/ca-certificates.crt" ]]; then
  export CURL_CA_BUNDLE=/snpe/cert/ca-certificates.crt
  export REQUESTS_CA_BUNDLE=/snpe/cert/ca-certificates.crt
  export HTTPLIB2_CA_CERTS=/snpe/cert/ca-certificates.crt

  # Enter your login info and register the certificates, if required
  gcloud config set core/custom_ca_certs_file /snpe/cert/ca-certificates.crt
fi

# Setup the default project, if required
gcloud config set project ${PROJECT}
if [[ -f "/etc/nas_secrets/sa-key.json" ]]; then
  gcloud auth activate-service-account --key-file "/etc/nas_secrets/sa-key.json"
  export GOOGLE_APPLICATION_CREDENTIALS="/etc/nas_secrets/sa-key.json"
else
  export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
fi

if [[ -d "/snpe/compiler" ]]; then
  export PATH=/snpe/compiler:${PATH}
fi

export PYTHONPATH=/snpe/nas:${PYTHONPATH}
