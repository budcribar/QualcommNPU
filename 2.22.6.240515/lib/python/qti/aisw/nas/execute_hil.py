# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import argparse
import logging
import os
import sys

from google.cloud import storage
import cloud_nas_utils

import model_metrics_evaluator

import json
import re
import importlib
import multiprocessing as mp
import numpy as np
from pathlib import Path

import qc_utils

def get_gcs_objs(model_uri, output_path):
    #matches = re.match("gs://(.*?)/(.*)", model_uri)
    reg = re.compile("gs://(.*?)/(.*)")
    matches = reg.match(model_uri)
    if not matches:
        logging.error("Couldn't parse GCS model uri %s!", model_uri)
        return None

    bucket, obj = matches.groups()
    logging.info("Got model %s in bucket %s", obj, bucket)

    client = storage.Client()
    bucket = client.bucket(bucket)
    blobs = bucket.list_blobs(prefix=obj)
    for blob in blobs:
         logging.debug('Got blob %s', blob.name)
         local_path = os.path.join(output_path, blob.name)
         os.makedirs(os.path.dirname(local_path), exist_ok = True)
         if not os.path.isdir(local_path):
             blob.download_to_filename(local_path)

    # download to a local file (with same folder structure as blob storage)
    model_path = os.path.join(output_path, obj)
    logging.info('Saving model to: %s\n',model_path)
    #os.makedirs(os.path.dirname(model_path), exist_ok = True)
    return model_path

class LatencyEvaluator(model_metrics_evaluator.ModelMetricsEvaluator):
  """Implements the process which evaluates and saves model-latency."""

  def __init__(self, service_endpoint, project_id, nas_job_id,
               latency_calculator_config,
               latency_worker_id = 0, num_latency_workers = 1):
    super(LatencyEvaluator, self).__init__(
        service_endpoint=service_endpoint,
        project_id=project_id,
        nas_job_id=nas_job_id,
        latency_worker_id=latency_worker_id,
        num_latency_workers=num_latency_workers)
    logging.info("Starting HIL:")
    logging.info("  Endpoint = %s", service_endpoint)
    logging.info("  Project  = %s", project_id)
    logging.info("  NAS Job  = %s", nas_job_id)
    logging.info("  Worker   = %s", latency_worker_id)
    logging.info("  Latency cfg = %s", latency_calculator_config)

    # Read the config
    self.config = json.load(open(latency_calculator_config, 'r'))
    logging.info("Received latency config: \n%s\n", self.config)

    if num_latency_workers != len(self.config['Devices']):
      logging.error("Num latency devices %d doesn't match # of provided HIL devices %d", num_latency_workers, self.config['Devices'])
      exit(1)

    self.custom_processing = None
    if "CustomProcessing" in self.config['Model']:
        sys.path.append(os.path.dirname(latency_calculator_config))
        logging.info('Found module %s', str(Path(self.config['Model']["CustomProcessing"]).stem))
        self.custom_processing = importlib.import_module(Path(self.config['Model']["CustomProcessing"]).stem)
    self.evaluator = qc_utils.RunBenchmark(self.config, latency_worker_id)

    logging.info("************************************************")
    logging.info("Waiting for initial model")


  def evaluate_saved_model(self, trial_id, saved_model_path):
    """Returns model latency."""
    logging.info("Job output directory is %s", self.job_output_dir)
    logging.info("Received model path: %s", saved_model_path)
    output_path = os.path.join('/qnn/output', self.config["HostResultsDir"])
    model_path = get_gcs_objs(saved_model_path, output_path)
    if self.custom_processing is not None and callable(getattr(self.custom_processing,'process_model', None)):
        logging.info("Calling custom model processing function")
        model_path = self.custom_processing.process_model(model_path)
    hil_artifact_path = os.path.join(os.path.dirname(model_path),'hil_artifacts')
    os.makedirs(os.path.dirname(hil_artifact_path), exist_ok = True)
    compiled_model = self.evaluator.convert(model_path, hil_artifact_path)
    if not compiled_model:
        logging.error("No valid model generated from conversion, failing.")
        return {"latency_in_seconds": 0, "model_memory": 0}

    # Run the model and generate the stats
    stats = self.evaluator.runner(compiled_model)

    ###############################################################
    # Modify, if needed
    ###############################################################
    stats.update({"latency_in_seconds": stats["latency_in_us"]/(10**6)})
    stats.update({"model_memory": stats['memory'] if 'memory' in stats else 0})
    if self.custom_processing is not None and callable(getattr(self.custom_processing,'process_stats', None)):
        logging.info("Calling custom stats processing function")
        stats = self.custom_processing.process_stats(stats)

    logging.info("Trial %d latency is %f us", trial_id, stats["latency_in_us"])
    logging.info("************************************************\n")

    logging.info("\n************************************************")
    logging.info("Waiting for next model")
    return stats

def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--latency_calculator_config", type=str, help="The config file "
      "to pass required HIL parameters.")
  # FLAGS for multi-device latency calculation.
  parser.add_argument(
      "--latency_worker_id",
      type=int,
      default=0,
      required=False,
      help="Latency calculation worker ID to start. Should be an integer in "
      "[0, num_latency_workers - 1]. If num_latency_workers > 1, each worker "
      "will only handle a subset of the parallel training trials based on "
      "their trial-ids.")
  parser.add_argument(
      "--num_latency_workers",
      type=int,
      default=1,
      required=False,
      help="The total number of parallel latency calculator workers. If "
      "num_latency_workers > 1, it is used to select a subset of the parallel "
      "training trials based on their trial-ids.")
  ######################################################
  ######## These FLAGS are set automatically by the nas-client.
  ######################################################
  parser.add_argument(
      "--project_id",
      type=str,
      default=os.getenv('PROJECT'),
      help="The project ID to check for NAS job.")
  parser.add_argument(
      "--nas_job_id", type=str, default="", help="The NAS job id.")
  parser.add_argument(
      "--service_endpoint",
      type=str,
      default="https://ml.googleapis.com/v1",
      help="The end point of the service. Default is https://ml.googleapis.com/v1."
  )
  ######################################################
  ######################################################

  return parser


def compute_latency(argv):
  """Computes latency."""
  latency_evaluator = LatencyEvaluator(
      service_endpoint=argv.service_endpoint,
      project_id=argv.project_id,
      nas_job_id=argv.nas_job_id,
      latency_worker_id=argv.latency_worker_id,
      num_latency_workers=argv.num_latency_workers,
      latency_calculator_config=argv.latency_calculator_config)
  latency_evaluator.run_continuous_evaluation_loop()


if __name__ == "__main__":
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()

  if flags.project_id is None:
    logging.info('Invalid project id!')
    exit(1)

  # Allocate and start a process for each HIL instance, then wait for each instance
  # to finish
  jobs = []
  for i in range(0, flags.num_latency_workers):
      logging.info('Setting up HIL worker %d',i)
      flags.latency_worker_id = i
      p = mp.Process(target=compute_latency, args=(flags,))
      jobs.append(p)

  for j in jobs:
      print("Starting worker")
      j.start()

  for j in jobs:
      print("Waiting for worker")
      j.join()

#  compute_latency(flags)
  logging.info("************************************************");
  logging.info("* NAS Sim complete                             *");
  logging.info("************************************************");

