# =============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import sys
from importlib import import_module
from logging import Logger

import numpy as np

if sys.version_info < (3, 8):
    # distutils deprecated for Python 3.8 and up
    from distutils.version import StrictVersion as Version
else:
    # packaging requires Python 3.8 and up
    from packaging.version import Version as Version

from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_objects import get_available_frameworks
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message, get_warning_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError, UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import save_outputs, load_inputs, dump_json
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import FrameworkExtension
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name


class FrameworkRunner(object):

    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self.parsed_args = args
        self.framework = args.framework.lower()
        self.version = args.version
        self.version_original = args.version
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        self.available_frameworks = get_available_frameworks()
        self.framework_instance = None
        self._logger = logger

    def _validate_framework(self):  # type: () -> None

        def max_version(framework):
            versions = self.available_frameworks.get(framework, {})
            return max(versions.keys(), key=lambda x: Version(x))

        self._logger.debug(
            get_progress_message('PROGRESS_FRAMEWORK_VERSION_VALIDATION')(self.framework,
                                                                          self.version))

        if self.framework not in self.available_frameworks:
            raise FrameworkError(
                get_message('ERROR_FRAMEWORK_FAILED_CONFIGURATION')(self.framework))

        if self.version is None:
            self.version = max_version(self.framework)
            self._logger.debug(
                get_progress_message('PROGRESS_FRAMEWORK_VERSION_AUTOMATIC')(self.framework,
                                                                             self.version))

        if self.version not in self.available_frameworks[self.framework]:
            raise FrameworkError(
                get_message('ERROR_FRAMEWORK_FAILED_CONFIGURATION')(self.framework))

    def _validate_framework_instance(self):
        self._logger.debug(get_progress_message('PROGRESS_FRAMEWORK_INSTANCE_VALIDATION'))

        if not self.version == self.framework_instance.get_version():
            self._logger.warning(
                get_warning_message("WARNING_FRAMEWORK_API_VERSION_VS_ENV_LOADED_LIB_MISMATCH")\
                (self.version, self.framework_instance.get_version()))

    def load_framework_instance(self):
        module, framework = self.available_frameworks[self.framework][self.version]

        self._logger.debug(get_progress_message('PROGRESS_FRAMEWORK_INSTANTIATION')(framework))
        try:
            framework_type = getattr(import_module(module), framework)
        except ImportError as exc:
            self._logger.exception(exc)
            raise FrameworkError(
                get_message('ERROR_FRAMEWORK_FAILED_CONFIGURATION')(self.framework))

        if hasattr(
                self.parsed_args, "onnx_custom_op_lib"
        ) and self.parsed_args.onnx_custom_op_lib is not None and self.framework == "onnx":
            self.framework_instance = framework_type(
                self._logger, custom_op_lib=self.parsed_args.onnx_custom_op_lib)
        else:
            self.framework_instance = framework_type(self._logger)

    def load_framework(self):  # type: () -> None
        self._validate_framework()
        self.load_framework_instance()

        if not self.version == self.framework_instance.get_version(
        ) and self.version_original is None and self.framework not in ["tflite","onnx"]:
            self.version = self.framework_instance.get_version()
            self._logger.debug(
                get_progress_message('PROGRESS_FRAMEWORK_VERSION_AUTOMATIC')(self.framework,
                                                                             self.version))
            self.load_framework_instance()

        #tflite's get_version() is the tensorflow's version, so tflite'version is alway check failed
        if self.framework != "tflite":
            self._validate_framework_instance()

        if hasattr(
                self.parsed_args, "disable_graph_optimization"
        ) and not self.parsed_args.disable_graph_optimization and self.framework == "onnx":
            optimized_model_path = os.path.join(
                self.output_dir,
                "optimized_model" + FrameworkExtension.framework_extension_mapping[self.framework])
            self.framework_instance.optimize(self.model_path, optimized_model_path)
            self.model_path = optimized_model_path
        self.framework_instance.load_model(self.model_path)

    def generate_intermediate_outputs(self, output_dir, input=None,
                                      output=None):  # type: (str) -> None
        data_path = os.path.join(output_dir, '{}{}')

        #validation check for input_tensor/output_tensor
        try:
            if not input and not output:
                output_names = self.parsed_args.output_tensor
                input_tensor = self.parsed_args.input_tensor
            else:
                output_names = output
                input_tensor = input
        except Exception:
            raise FrameworkError(
                get_message('ERROR_FRAMEWORK_RUNNER_NO_INPUT_TENSOR_OR_NO_OUTPUT_TENSOR'))

        if output_names == []:
            raise FrameworkError(
                get_message('ERROR_FRAMEWORK_RUNNER_NO_INPUT_TENSOR_OR_NO_OUTPUT_TENSOR'))

        #get the input tensor
        in_list = list(zip(*input_tensor))
        if len(in_list) == 4:
            (in_names, in_dims, in_data_paths, in_types) = in_list
        elif len(in_list) == 3:
            (in_names, in_dims, in_data_paths) = in_list
            in_types = None
        else:
            raise FrameworkError(get_message('ERROR_FRAMEWORK_RUNNER_INPUT_TENSOR_LENGHT_ERROR'))
        input_names = list(in_names)
        input_dims = [[int(x) for x in dim.split(',')] for dim in in_dims]
        input_data_paths = in_data_paths
        input_types = in_types
        if self.framework == 'onnx':
            self.framework_instance.add_outputs()
        tensor_pairs = self.framework_instance.get_intermediate_tensors(input_names, output_names)

        self._logger.info(
            get_progress_message('PROGRESS_FRAMEWORK_GENERATE_OUTPUTS')(data_path.format('', '')))

        if not input_types:
            self._logger.error('ERROR_PROGRESS_FRAMEWORK_GENERATE_OUTPUTS_NO_INPUT_TYPES')

        input_data = [
            load_inputs(file, data_type, dim)
            for file, data_type, dim in zip(input_data_paths, input_types, input_dims)
        ]
        tensor_info = {}
        for _, output_tensor_names in tensor_pairs:
            result = self.framework_instance.run_inference(input_data, input_names,
                                                           output_tensor_names)
            for output_tensor, data in result.items():
                # Sanitize output tensor name to avoid dumping outputs in sub-folders.
                output_tensor = santize_node_name(output_tensor)
                file_path = data_path.format(output_tensor, '.raw')
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                save_outputs(data, file_path, "float32")
                try:
                    if type(data) is list:
                        data = np.array(data, dtype=np.float32)
                except Exception as e:
                    raise Exception("Encountered Error: {}".format(str(e)))

                if (not data.size or data.dtype == bool):
                    if data.size == 0:
                        tensor_info[output_tensor] = (
                            '-',
                            '-',
                            '-',
                            '-',
                            '-',
                        )
                    else:
                        tensor_info[output_tensor] = (str(data.dtype), data.shape, data.tolist(),
                                                      data.tolist(), data.tolist())
                else:
                    tensor_info[output_tensor] = (str(data.dtype), data.shape,
                                                  str(round(np.min(data),
                                                            3)), str(round(np.max(data), 3)),
                                                  str(round(np.median(data), 3)))
        tensor_info_json = data_path.format('profile_info', '.json')
        dump_json(tensor_info, tensor_info_json)

        self._logger.info(
            get_progress_message('PROGRESS_FRAMEWORK_GENERATED_INTERMEDIATE_TENSORS')(
                self.framework, self.framework_instance.get_version()))

    def load_framework_for_tensor_mapping(self):
        self.load_framework()

    def run(self):
        self.load_framework()
        self.generate_intermediate_outputs(self.output_dir)

    def extract_sub_graph(self, start_layer_output_name, end_layer_output_name=None,
                          out_model_path=None):
        self.load_framework()
        status, new_model, new_inputs = self.framework_instance.extract(
            start_layer_output_name, end_layer_output_name, out_model_path)
        return status, new_model, new_inputs

    def fetch_input_info(self):
        self.load_framework()
        input_info = self.framework_instance.get_input_layers_info()
        # convert input_info from tuple to map
        input_info_map = {}
        for item in input_info:
            input_info_map[item[0]] = list(item[1:])
        return input_info_map


class ModelTraverser(FrameworkRunner):

    def __init__(self, logger, args, add_layer_outputs=[], add_layer_types=[],
                 skip_layer_outputs=[], skip_layer_types=[]):
        self._current_pos = 0
        self._layerlist = None
        super().__init__(logger, args)
        self.add_layer_outputs = add_layer_outputs
        self.add_layer_types = add_layer_types
        self.skip_layer_types = skip_layer_types
        self.skip_layer_outputs = skip_layer_outputs
        self.prepare_layerlist()

    def get_next_layer(self):
        """
        This method returns the tuple(layer_name,output_name) for the next layer in sequence
        Returns:
            layer : tuple(layer_name,output_name) for next layer in sequence.
        """
        if self._layerlist is None:
            self._logger.error('Internal error - layer list cannot be None')
            raise UnsupportedError('Internal error - layer list cannot be None')

        if len(self._layerlist) == self._current_pos:
            # No more layers
            return (1, None, None, None)
        else:
            status = 0
            layer_info = self._layerlist[self._current_pos]
            self._current_pos += 1
            if layer_info[2] in self.skip_layer_types or layer_info[1] in self.skip_layer_outputs:
                # Skipped layer
                status = 2

            return (status, layer_info[0], layer_info[1], layer_info[2])

    def get_layer_count(self):
        """This method returns the count of layers present in model."""
        if self._layerlist:
            return len(self._layerlist)
        else:
            return 0

    def filter_layerlist(self, skip_layer_patterns):
        """This method eliminates the user supplied layer patterns from the
        layerlist."""
        temp_layerlist = self._layerlist.copy()
        # Check and filter out each layer pattern from layerslist to avoid adding ouputs in
        # between pattern
        for seq in skip_layer_patterns:
            filtered_layerlist = []
            i = 0
            n = len(seq)
            while (i <= len(temp_layerlist)):
                try:
                    _, _, layertypes = zip(*temp_layerlist[i:i + n])
                except:
                    layertypes = ()
                if list(layertypes) != seq and len(layertypes) > 0:
                    filtered_layerlist.append(temp_layerlist[i])
                    i += 1
                else:
                    i += n - 1
            temp_layerlist = filtered_layerlist
        self._layerlist = filtered_layerlist

    def prepare_layerlist(self):
        """This method prepares the list containing all layers of model."""
        self._validate_framework()
        self.load_framework_instance()
        self.framework_instance.load_model(self.model_path)

        self._layerlist = self.framework_instance.get_layer_identifiers()
        self._current_pos = 0

        layer_types = [it[2] for it in self._layerlist]
        layer_outputs = [it[1] for it in self._layerlist]

        if not set(self.add_layer_types).issubset(set(layer_types)):
            invalid_layers = list(set(self.add_layer_types) - set(layer_types))
            self._logger.info('Valid layer_types : {}'.format(list(set(layer_types))))
            raise UnsupportedError(
                'Provided invalid add_layers :{}. Please check the log for valid layer_types and provide proper layer type'
                .format(invalid_layers))
        # validation check for add_layer_outputs
        if isinstance(self.add_layer_outputs, str):
            self.add_layer_outputs = [s for s in self.add_layer_outputs.split(',')]
        if not set(self.add_layer_outputs).issubset(set(layer_outputs)):
            invalid_layer_outputs = list(set(self.add_layer_outputs) - set(layer_outputs))
            self._logger.info('Valid layer_outputs : {}'.format(layer_outputs))
            raise UnsupportedError(
                'Provided invalid add_layer_outputs :{} . Please check the log for valid layer_outputs and provide proper layer output'
                .format(invalid_layer_outputs))
        if self.add_layer_types:
            if self.skip_layer_types:
                raise UnsupportedError('--skip-layer_types can not be used with --add_layer_types')
            self.skip_layer_types = [
                layer_item[2] for layer_item in self._layerlist
                if layer_item[2] not in self.add_layer_types
            ]
        if self.add_layer_outputs:
            if self.skip_layer_outputs:
                raise UnsupportedError(
                    '--skip_layer_outputs can not be used with --add_layer_outputs')
            self.skip_layer_outputs = [
                layer_item[1] for layer_item in self._layerlist
                if layer_item[1] not in self.add_layer_outputs
            ]

    def get_all_layers(self, names_only=True):
        """
        Returns the output names for all the layers in current model after skipping the configured
        nodes
        """
        if self._layerlist is None:
            self._logger.error('Internal error - layer list cannot be None')
            raise UnsupportedError('Internal error - layer list cannot be None')
        layer_out_names = []
        for layer_info in self._layerlist:
            if layer_info[2] in self.skip_layer_types or layer_info[1] in self.skip_layer_outputs:
                continue
            else:
                if names_only:
                    layer_out_names.append(layer_info[1])
                else:
                    layer_out_names.append((layer_info[1], None, layer_info[2]))
        return layer_out_names
