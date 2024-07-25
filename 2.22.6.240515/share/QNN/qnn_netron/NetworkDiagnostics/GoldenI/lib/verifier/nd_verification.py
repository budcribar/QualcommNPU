# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import logging
import os

import numpy as np
import pandas as pd

from lib.utils.nd_errors import get_message, get_warning_message
from lib.utils.nd_exceptions import VerifierError, VerifierInputError
from lib.utils.nd_framework_utility import load_inputs
from lib.utils.nd_graph_structure import GraphStructure
from lib.verifier.models import VerifyResult
from lib.verifier.nd_verifier_factory import VerifierFactory
from lib.verifier.verifiers import ScaledDiffVerifier
from lib.utils.nd_path_utility import get_tensor_paths

class Verification:
    SUMMARY_NAME = "summary"

    def __init__(self, default_verifier, logger, args, default_verifier_config = None):
        def generate_inference_to_golden_map(inference_tensors, mapping):
            return {inference: mapping[inference] if inference in mapping and mapping[inference] is not None else
                    inference for inference in inference_tensors}

        def load_verifiers(verifier_configs):
            # type: (dict) -> dict
            verifier_objects = {}
            for verifier, config in verifier_configs.items():
                verifier_object = VerifierFactory().factory(verifier, config.get("parameters", {}))
                if verifier_config is None:
                    raise VerifierError(get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(verifier))
                verifier_objects[verifier] = verifier_object, config
            return verifier_objects

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.verifier_configs = {}
        if args.verifier_config and os.path.exists(args.verifier_config):
            with open(args.verifier_config) as verifier_config:
                self.verifier_configs = json.load(verifier_config)

        self.tensor_mapping = {}
        if args.tensor_mapping and os.path.exists(args.tensor_mapping):
            with open(args.tensor_mapping) as tensor_mapping:
                self.tensor_mapping = json.load(tensor_mapping)

        self.golden_tensor_paths = get_tensor_paths(args.framework_results)
        self.inference_tensor_paths = get_tensor_paths(args.inference_results)

        self.inference_tensors = tuple(self.inference_tensor_paths.keys())
        self.inference_types = None
        self.tensor_dimensions = None
        self.output_tensor_encodings = None
        if args.graph_struct and os.path.exists(args.graph_struct):
            graph_struct = GraphStructure.load_graph_structure(args.graph_struct)
            self.inference_tensors = graph_struct.get_all_tensors()
            self.inference_types = graph_struct.get_all_types()
            self.tensor_dimensions = graph_struct.get_tensor_dimension_dict()
            self.output_tensor_encodings = graph_struct.get_all_output_tensor_encodings_dict()

        if ScaledDiffVerifier.NAME in self.verifier_configs or default_verifier == ScaledDiffVerifier.NAME:
            if not self.output_tensor_encodings:
                raise VerifierError(get_message('ERROR_VERIFIER_SCALED_DIFF_MISSING_OUTPUT_ENCODING'))
            o_encodings = list(self.output_tensor_encodings.items())
            if o_encodings[0][1]['min'] == o_encodings[0][1]['max'] == o_encodings[0][1]['scale'] \
                 == o_encodings[0][1]['offset'] == 0.0:
                raise VerifierError(get_message('ERROR_VERIFIER_CANNOT_USE_SCALEDDIFF_VERIFIR'))

        self.inference_to_golden_tensor_map = generate_inference_to_golden_map(self.inference_tensors,
                                                                               self.tensor_mapping)

        self.specific_verifiers = load_verifiers(self.verifier_configs)

        self.default_verifier = default_verifier
        default_config={}
        if default_verifier in self.verifier_configs:

            #TODO: what if default_config is not configed properly need to be able to validate
            default_config = self.verifier_configs[default_verifier]["parameters"]
        elif default_verifier_config:
            ret, default_config = VerifierFactory().validate_configs(default_verifier, default_verifier_config)
            if not ret:
                errormsg= str(default_config['error']) if 'error' in default_config else ''
                raise VerifierError("VerifierFactory config_verify error: " + errormsg)
        self.default_verifier_obj = VerifierFactory().factory(default_verifier, default_config)
        if self.default_verifier_obj is None:
            raise VerifierError(get_message('ERROR_VERIFIER_INVALID_VERIFIER_NAME')(default_verifier))

        self.output_dir = args.output_dir
        self.layout = args.layout

    def has_specific_verifier(self):
        for _, verifier_data in self.specific_verifiers.items():
            _, config = verifier_data
            tensors = config.get("tensors", list())
            for tensor in tensors:
                if tensor in self.inference_tensors:
                    return True
        return False

    def verify_tensors(self):
        """
        Runs the verifiers on their corresponding tensors
        :return: Pandas dataframe of the verifier return data
        """

        def get_tensor_data(inference_tensor_names):
            # type: (list[str]) -> (list[np.array], list[np.array])
            """
            Given a list inference tensors, find the corresponding golden tensors and then retrieve the data for both
            :param inference_tensor_names: a list of tensor names
            :return: pair of lists containing tensor data
            """
            missing = [tensor for tensor in inference_tensor_names if tensor not in self.inference_tensor_paths.keys()]
            if missing:
                self.logger.warning(get_warning_message("WARNING_VERIFIER_MISSING_INFERENCE_TENSOR_DATA")(str(missing)))
                return None, None

            golden_tensors = [self.inference_to_golden_tensor_map[tensor] for tensor in inference_tensor_names]

            missing = [tensor for tensor in golden_tensors if tensor not in self.golden_tensor_paths.keys()]
            if missing:
                self.logger.warning(get_warning_message("WARNING_VERIFIER_MISSING_GOLDEN_TENSOR_DATA")(str(missing)))
                return None, None

            golden_data = [load_inputs(self.golden_tensor_paths[tensor], "float32") for tensor in golden_tensors]

            if self.layout == 'NCHW':
                shape = self.tensor_dimensions[inference_tensor_names[0]]
                if len(shape) ==4 :
                    golden_data = np.reshape(golden_data[0], (shape[0], shape[3], shape[1], shape[2]))
                    golden_data = np.transpose(golden_data, (0,2,3,1))
                    golden_data = [golden_data.flatten()]

            inference_data = [load_inputs(self.inference_tensor_paths[tensor], "float32")
                              for tensor in inference_tensor_names]
            return golden_data, inference_data

        def generate_tensor_save_path(tensor_dir, tensor_name):
            save_path = os.path.join(tensor_dir, tensor_name)
            save_path_dir = os.path.dirname(save_path)

            # create the folder to save the tensor in if it doesn't already exist
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir, exist_ok=True)
            return save_path

        def verify(name, layer_type, tensor_dimensions, inference_tensors, verifier):
            """
            Find corresponding golden tensors for inference tensors, loads their data, and then performs verification
            :param name: Name for this verification result
            :param inference_tensors: list of tensors
            :param verifier: verifier to perform verification with
            :return: dictionary of verification result with a 'Name' field
            """
            golden_data, inference_data = get_tensor_data(inference_tensors)

            if golden_data is None or inference_data is None:
                return None

            if type(verifier) == ScaledDiffVerifier:
                verifier.set_output_encoding(self.output_tensor_encodings[inference_tensors[0]])

            result = verifier.verify(layer_type, tensor_dimensions, golden_data, inference_data)
            return dict(result._asdict(), Name=name)

        summary_df = pd.DataFrame(columns=["Name"] + list(VerifyResult._fields))

        verifier_config_tensors = []
        for verifier_config in self.verifier_configs.values():
            verifier_config_tensors.extend(verifier_config["tensors"])
        # collect all the tensors used by specific verifiers
        verifier_config_tensors = set([item for sublist in verifier_config_tensors for item in sublist])

        # ignore cases where the tensor will be verifier by a specific verifier
        default_verifiable_tensors = list(filter(lambda t: t not in verifier_config_tensors, self.inference_tensors))

        default_verifier_path = os.path.join(self.output_dir, self.default_verifier)

        if not os.path.isdir(default_verifier_path):
            os.makedirs(default_verifier_path)

        # We run our default verifier on all remaining tensors that have not been verified yet by any specified
        # verifier, we may consider adding support for more default verifiers in the future
        for inference_tensor_name in default_verifiable_tensors:
            try:
                # verify one tensor at a time
                verification_data = verify(inference_tensor_name,
                                           self.inference_types[inference_tensor_name]
                                           if self.inference_types and inference_tensor_name in self.inference_types else " ",
                                           str(self.tensor_dimensions[inference_tensor_name])
                                           if self.tensor_dimensions and inference_tensor_name in self.tensor_dimensions else " ",
                                           [inference_tensor_name], self.default_verifier_obj)
            except VerifierInputError as e:
                self.logger.warning(str(e))
                continue

            if verification_data is None:
                continue

            summary_df = summary_df.append(verification_data, ignore_index=True)

            tensor_path = generate_tensor_save_path(default_verifier_path, inference_tensor_name)
            self.default_verifier_obj.save_data(tensor_path)

        for verifier_name, verifier_data in self.specific_verifiers.items():
            specific_verifier, config = verifier_data
            verifier_path = os.path.join(self.output_dir, verifier_name)

            for tensors in config.get("tensors", list()):
                try:
                    verification_data = verify(str(tensors), self.inference_types[tensors]
                                               if self.inference_types and tensors in self.inference_types else " ",
                                               str(self.tensor_dimensions[tensors])
                                               if self.tensor_dimensions and tensors in self.tensor_dimensions else " ",
                                               [tensors], specific_verifier)
                except VerifierInputError as e:
                    self.logger.warning(str(e))
                    continue

                if verification_data is None:
                    continue

                summary_df = summary_df.append(verification_data, ignore_index=True)

                tensor_path = generate_tensor_save_path(verifier_path, tensors[0])
                specific_verifier.save_data(tensor_path)

        # Don't truncate columns
        with pd.option_context('display.max_colwidth', -1):
            summary_df.to_csv(os.path.join(default_verifier_path, self.SUMMARY_NAME + ".csv"), index=False, encoding="utf-8")
            summary_df.to_html(os.path.join(default_verifier_path, self.SUMMARY_NAME + ".html"), index=False, classes='table')

        return summary_df
