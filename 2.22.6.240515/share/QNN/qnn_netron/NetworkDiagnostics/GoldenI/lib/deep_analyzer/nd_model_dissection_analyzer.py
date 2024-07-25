# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import os
import logging

import pandas as pd
from tabulate import tabulate

from lib.utils.nd_constants import Framework
from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import DeepAnalyzerError
from lib.utils.nd_graph_structure import GraphStructure
from lib.deep_analyzer.partitioner.nd_partitioner import PartitionedModel, Partitioner
from lib.verifier.nd_verification import Verification
from lib.utils.nd_path_utility import get_tensor_paths

class ModelDissectionAnalyzer:
    """Class that runs Model Dissection Analysis"""
    ITERATION_TIMEOUT_LIMIT = 99
    def __init__(self, summary_df, args, logger, envToolConfig):


        def generate_inference_to_golden_map(inference_tensors, mapping):
            return {inference: mapping[inference] if inference in mapping else
                    inference for inference in inference_tensors}

        self.summary_df = summary_df
        self.verifier = args.default_verifier
        self.engine = args.engine
        self.engine_version = args.engine_version
        self.engine_path = args.engine_path
        self.framework = args.framework
        self.framework_version = args.framework_version
        self.ndk_path = args.ndk_path
        self.model_path = args.model_path
        self.working_dir = args.working_dir
        self.output_dir = args.output_dir
        self.framework_results = args.framework_results
        self.inference_results = args.inference_results
        self.tensor_mapping_path = args.tensor_mapping
        if os.path.exists(self.tensor_mapping_path):
            with open(self.tensor_mapping_path) as tensor_mapping:
                self.tensor_mapping = json.load(tensor_mapping)
        else:
            #Shouldn't ever reach here as tensor_mapping_path should be always provided here or auto generated
            raise DeepAnalyzerError("while running MDA, we detected tensor_mapping_path is missing or doesn't exist.")
        self.graph_struct_path = args.graph_struct
        if os.path.exists(self.graph_struct_path):
            self.graph_struct = GraphStructure.load_graph_structure(args.graph_struct)
            self.inference_tensors = self.graph_struct.get_all_tensors()
        else:
            raise DeepAnalyzerError("while running MDA, we detected graph_struct is missing or doesn't exist.")

        self.auto_stop_iterations = args.auto_stop_iterations
        self.maximum_dissection_iterations = args.maximum_dissection_iterations
        self.golden_tensor_paths = get_tensor_paths(args.framework_results)
        self.inference_to_golden_tensor_map = generate_inference_to_golden_map(self.inference_tensors,
                                                                               self.tensor_mapping)
        self.golden_to_inference_tensor_map = {golden: inference for inference, golden in self.inference_to_golden_tensor_map.items()}
        self.problem_tensor_name = args.problem_inference_tensor
        self.verifier_threshold = args.verifier_threshold
        self.target_device = args.target_device
        self.runtime = args.runtime
        self.architecture = args.architecture
        self.logger = logger
        self.envToolConfig = envToolConfig
        # self.config_file_path = args.config_file_path
        if args.partition_override:
            with open(args.partition_override, "r") as f:
                self.partition_override_data = json.load(f)
        else:
            self.partition_override_data = None

        self.deviceId = args.deviceId

    def runModelDissection(self):
        partitioned_model_list = []

        # initialize the Partitioner, get PartitionedModel based on verifier or specified tensor
        partitioner = Partitioner(self.verifier, self.summary_df, self.graph_struct, self.golden_tensor_paths, self.golden_to_inference_tensor_map, self.inference_to_golden_tensor_map, self.framework, self.logger)

        if not self.partition_override_data:
            ExtendInput=False
            if not self.maximum_dissection_iterations: self.maximum_dissection_iterations=self.ITERATION_TIMEOUT_LIMIT
            for iter in range(0,self.maximum_dissection_iterations):
                if ExtendInput:
                    partitioned_model = partitioner.partition_analyze(extend_inputs=True, partitioned_model=partitioned_model, auto_stop=self.auto_stop_iterations)
                else:
                    partitioned_model = partitioner.partition_analyze(problem_tensor_name=self.problem_tensor_name, verifier_threshold=self.verifier_threshold)
                    ExtendInput=True
                if not partitioned_model:
                    self.logger.info("CANNOT EXTEND MODEL: No FURTHER DISSECTION")
                    break
                elif partitioned_model and not partitioned_model.input_tensors:
                    self.logger.info("CANNOT EXTEND MODEL: NO INPUT TENSORS")
                    break

                # Dissect and run model
                self.logger.info("Partitioned Model Inputs: {}".format(partitioned_model.input_tensors))
                self.logger.info("Partitioned Model Outputs: {}".format(partitioned_model.output_tensors))
                new_inference_dir = self.dissect_and_execute(partitioned_model)
                if not os.path.isdir(new_inference_dir):
                    raise DeepAnalyzerError(get_message("ERROR_DEEP_ANALYZER_NON_EXISTENT_PATH")(new_inference_dir))

                # Run verification on partitioned model outputs
                new_verification_dir = self.validate_dissected_models(new_inference_dir)
                verifier_summary_path = os.path.join(new_verification_dir, Verification.SUMMARY_NAME + '.csv')
                if not os.path.exists(verifier_summary_path):
                    raise DeepAnalyzerError(get_message("ERROR_DEEP_ANALYZER_NON_EXISTENT_PATH")(verifier_summary_path))

                df = pd.read_csv(verifier_summary_path)
                partitioned_model.actual_accuracy = df.iloc[-1][self.verifier]
                partitioned_model_list.append(partitioned_model)

            if iter == self.maximum_dissection_iterations-1:
                self.logger.info("Iteration Limit Reached")

        else: #going down the override path
            self.logger.info("Taking override inputs and outputs to dissect models")
            for partition in self.partition_override_data:
                partitioned_model = partitioner.create_partitioned_model(partition)

                # Dissect and run model
                self.logger.info("Partitioned Model Inputs: {}".format(partitioned_model.input_tensors))
                self.logger.info("Partitioned Model Outputs: {}".format(partitioned_model.output_tensors))
                new_inference_dir = self.dissect_and_execute(partitioned_model)
                if not os.path.isdir(new_inference_dir):
                    raise DeepAnalyzerError(get_message("ERROR_DEEP_ANALYZER_NON_EXISTENT_PATH")(new_inference_dir))

                # Run verification on partitioned model outputs
                new_verification_dir = self.validate_dissected_models(new_inference_dir)
                verifier_summary_path = os.path.join(new_verification_dir, Verification.SUMMARY_NAME + '.csv')
                if not os.path.exists(verifier_summary_path):
                    raise DeepAnalyzerError(get_message("ERROR_DEEP_ANALYZER_NON_EXISTENT_PATH")(verifier_summary_path))

                df = pd.read_csv(verifier_summary_path)
                partitioned_model.actual_accuracy = df.iloc[-1][self.verifier]
                partitioned_model_list.append(partitioned_model)


        self.logger.info("Finished Iterating.")
        self.generate_summary(partitioned_model_list, partitioned_model)

    def generate_summary(self, partitioned_model_list, partitioned_model):
        # type: (list[PartitionedModel], PartitionedModel) -> None
        """Generate, export, and log Model Dissection Summary"""

        if not partitioned_model_list:
            self.logger.info("No Partitioned Models were created.")
            return

        # Log and export model dissection summary
        dissection_summary_columns = ['Inputs', 'Outputs', self.verifier]
        dissection_summary_df = pd.DataFrame(columns=dissection_summary_columns)

        dissection_summary_initial_entry = pd.Series([
            'Original Inputs',
            ','.join([name for name,*_ in partitioned_model_list[0].output_tensors]),
            partitioned_model_list[0].original_accuracy
        ], index=dissection_summary_columns)
        dissection_summary_df = dissection_summary_df.append(dissection_summary_initial_entry, ignore_index=True)

        for pm in partitioned_model_list:
            df_new_entry = pd.Series([
                ','.join([name for name,_ in pm.input_tensors]),
                ','.join([name for name,*_ in pm.output_tensors]),
                pm.actual_accuracy
            ], index=dissection_summary_columns)
            dissection_summary_df = dissection_summary_df.append(df_new_entry, ignore_index=True)

        self.logger.info("Exporting dissection summary to: {}".format(self.output_dir))
        dissection_summary_df.to_csv(os.path.join(self.output_dir, 'dissection_summary.csv'))
        dissection_summary_df.to_html(os.path.join(self.output_dir, 'dissection_summary.html'))

        # Generate Dissection Summary
        dissection_summary_log = []
        dissection_summary_log.append((
            "\n=============================================================="
            "\nDISSECTION SUMMARY"
            "\n=============================================================="
        ))

        dissection_summary_log.append(tabulate([
            ['Outputs'],
            ['\n'.join(dissection_summary_df['Outputs'][0].split(','))]
        ], headers='firstrow'))

        for iter,row in dissection_summary_df.iterrows():
            dissection_summary_log.append(tabulate([
                ['Iteration', 'Inputs', self.verifier],
                [iter, '\n'.join(row['Inputs'].split(',')), row[self.verifier]]
            ], headers='firstrow'))

        if not partitioned_model:
            # Iterations stopped by algorithm
            dissection_summary_log.append('No further dissection based on partition analysis')
        elif len(partitioned_model_list) == self.maximum_dissection_iterations:
            dissection_summary_log.append('Maximum number of iterations reached')
        elif partitioned_model and not partitioned_model.input_tensors:
            dissection_summary_log.append('No input tensors available for iteration.')

        dissection_summary_log.append('The following QNN nodes may be sources of error and should be inspected further.')
        latest_inputs = [tensor_name for tensor_name,*_ in partitioned_model_list[-1].input_tensors]

        if not self.partition_override_data:
            relevant_nodes = Partitioner.get_relevant_nodes_from_inputs(self.graph_struct, self.golden_to_inference_tensor_map, latest_inputs)
        else:
            relevant_nodes = latest_inputs
        dissection_summary_log.append('\n'.join(['  - {}'.format(node) for node in relevant_nodes]))

        self.logger.info('\n\n'.join(dissection_summary_log))

    def dissect_and_execute(self, partitioned_model):
        # type: (PartitionedModel) -> str
        """Dissect and run new model using nd_run_qnn_inference_engine.

        Dissects model by opening subprocess to run nd_run_qnn_inference_engine. QNN converters
        helps dissect the model, and results are obtained via qnn-net-run. Inference uses
        HTP Backend, with floatonly input/output (converts quant to float). Returns path to
        inference engine outputs.

        :param partitioned_model: PartitionedModel object providing dissection details
        :return: path to directory containing inference engine output tensors
        """
        def swoppos(list):
            newlist=[list[0]]+[list[-1]]+list[1:3]
            return newlist

        strip_char = ""
        if self.framework == Framework.tensorflow.value:
            strip_char = ":0"
            # prepare tensor parameters
            output_tensors = [[partitioned_model.output_tensors[0][0].strip(strip_char)]]

            # Use golden tensors as input
            input_tensors = [
                [name.strip(strip_char), ",".join(str(dim) for dim in dims), self.golden_tensor_paths[name]]
                for name,dims in partitioned_model.input_tensors
            ]

        elif self.framework == Framework.onnx.value:
            output_tensors = [[partitioned_model.output_tensors[0][0]]]
            input_tensors = [
                [name.strip(strip_char), ",".join(str(dim) for dim in swoppos(dims)), self.golden_tensor_paths[name]]
                for name,dims in partitioned_model.input_tensors
            ]

        # save input_list.txt to working directory
        input_list_txt = os.path.join(self.output_dir, 'input_list.txt')
        with open(input_list_txt, "w") as input_file:
            raws = ['{}:={}'.format(name, path) for name, _, path in input_tensors]
            input_file.write(" ".join(raws))

        args = {
            'framework': '{} {}'.format(self.framework, (self.framework_version if self.framework_version else '')),
            'engine_path': self.engine_path,
            'runtime': self.runtime,
            'working_dir': self.output_dir,
            'input_list': input_list_txt,
            'deviceId': self.deviceId,
            'target_device': self.target_device,
            'model_path': self.model_path,
            'model_inputs': ''.join([' --input_tensor {} {} {}'.format(name,dim,path) for name,dim,path in input_tensors]),
            'model_outputs': ''.join([' --output_tensor {}'.format(name) for name,*_ in output_tensors]),
            'target_architecture': self.architecture,
            'ndk_path': self.ndk_path,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else '')
        }

        inference_args = (
            ' --framework {args[framework]}'
            ' --engine_path {args[engine_path]}'
            ' --runtime {args[runtime]}'
            ' --working_dir {args[working_dir]}'
            ' --input_list {args[input_list]}'
            ' --deviceId {args[deviceId]}'
            ' --target_device {args[target_device]}'
            ' --model_path {args[model_path]}'
            ' --architecture {args[target_architecture]}'
            ' --ndk_path {args[ndk_path]}'
            '{args[model_inputs]}'
            '{args[model_outputs]}'
            '{args[verbose]}'
        ).format(args=args)
        #TODO: disabled reference_code_analyzer ability in MDA
        # if self.engine==Engine.QNN.value and self.config_file_path:
        #     inference_args=inference_args+' --qnn_netrun_config_file {config_file_path}'.format(config_file_path=self.config_file_path)

        # configs and spawns run_inference_engine sub-process
        self.logger.debug("Running nd_run_qnn_inference_engine.py with parameters: {}".format(inference_args))
        ret_inference_engine = self.envToolConfig.run_qnn_inference_engine(inference_args.split())
        if ret_inference_engine != 0:
            raise DeepAnalyzerError("Subprocess finished with exit code {}".format(ret_inference_engine))

        return os.path.join(self.output_dir, 'inference_engine', 'latest', 'output', 'Result_0')

    def validate_dissected_models(self, new_inference_results):
        # type: (str) -> str
        """Run verification on partitioned model results.

        Verifies partitioned model results by running nd_run_verification.
        Compares results to original golden tensors in specified directories
        using specified verifiers. Input parameters are all used by nd_run_verification.
        Returns path to new verification results.

        :param new_inference_results: path to directory containing inference engine output tensors
        :return: path to directory containing verification results
        """

        # Get new accuracies run with golden input, compare to original accuracies
        args = {
            'default_verifier': self.verifier,
            'framework_results': self.framework_results,
            'inference_results': new_inference_results,
            'working_dir': self.output_dir,
            'tensor_mapping_path': self.tensor_mapping_path,
            'graph_struct': self.graph_struct_path,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else '')
        }

        verification_args = (
            ' --default_verifier {args[default_verifier]}'
            ' --framework_results {args[framework_results]}'
            ' --inference_results {args[inference_results]}'
            ' --working_dir {args[working_dir]}'
            ' --tensor_mapping {args[tensor_mapping_path]}'
            ' --graph_struct {args[graph_struct]}'
            '{args[verbose]}'
        ).format(args=args)

        if self.framework == 'onnx':
            verification_args=verification_args + ' --layout NCHW'

        self.logger.debug("Running nd_run_verification.py with parameters: {}".format(verification_args))
        ret_verifier = self.envToolConfig.run_verifier(verification_args.split())
        if ret_verifier != 0:
            raise DeepAnalyzerError("Subprocess finished with exit code {}".format(ret_verifier))

        return os.path.join(self.output_dir, 'verification', 'latest')
