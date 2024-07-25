##############################################################################
#
# Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import ast
import os
import re
import shutil
import sys
import yaml
import copy
import json
import inspect

import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.plugin as pl
from qti.aisw.accuracy_evaluator.qacc.utils import convert_npi_to_json, cleanup_quantization_overrides
from qti.aisw.accuracy_evaluator.common.transforms import ModelHelper
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc import defaults
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc


class Configuration:
    """QACC configuration class having all the configurations supplied by the
    user.

    To use:
    >>> config = Configuration(config_path, work_dir, set_global=None, batchsize=None, use_memory_plugins=False)
    """
    __instance = None

    def __init__(self, config_path, work_dir, set_global=None, batchsize=None, model_path=None,
                 use_memory_plugins=False):
        """Creates Configuration object using the supplied arguments.
        Args:
        config_path: path to yaml configuration
        work_dir: Work directory to be used to store results and other artifacts
        set_global: Global constants values supplied via cli
        batchsize: batchsize to be updated from value passed from cli
        use_memory_plugins: Flag to enable memory plugins usage
        """
        # Copy config file in work dir.
        self._config_path = config_path
        self._work_dir = work_dir
        self.set_global = set_global
        self.batchsize = batchsize
        self.model_path = model_path
        self.use_memory_plugins = use_memory_plugins
        self.load_config_from_yaml()

    def load_config_from_yaml(self):
        """Loads the config from yaml file.
        Raises:
            ConfigurationException:
                - if incorrect configuration file provided
                - configuration file empty
        """
        if not os.path.exists(self._work_dir):
            os.makedirs(self._work_dir)

        _, config_file_name = os.path.split(self._config_path)
        cur_path = os.path.join(self._work_dir, config_file_name)
        shutil.copyfile(self._config_path, cur_path)

        with open(cur_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ce.ConfigurationException('incorrect configuration file', exc)

        if config == None:
            raise ce.ConfigurationException('configuration file empty')
        elif config['model'] == None:
            raise ce.ConfigurationException('model key not found in configuration file')

        # Figure out batch dimension from the config: Assumes only one batch dim per input/output
        # node
        self.batching_dims = {}
        self.batching_dims['inputs_info'] = self.get_batch_dims_info(config, node_type='input')
        self.batching_dims['outputs_info'] = self.get_batch_dims_info(config, node_type='output')

        is_config_updated = False

        # Updating the model_path to dir
        if self.model_path is not None:
            config['model']['inference-engine']['model_path'] = self.model_path
        # Write back to the file
        with open(cur_path, 'w') as stream:
            yaml.dump(config, stream)

        if 'globals' in config['model'] and config['model']['globals']:
            gconfig = config['model']['globals']

            if len(gconfig) > 0:
                is_config_updated = True
                # replace globals with cmd line args -global
                cmd_gconfig = {}
                if self.set_global:
                    for g in self.set_global:
                        elems = g[0].split(':')
                        cmd_gconfig[elems[0]] = elems[1]
                    gconfig.update(cmd_gconfig)

                # update config file with globals.
                with open(cur_path, 'r') as stream:
                    file_data = stream.read()

                with open(cur_path, 'w') as stream:
                    for k, v in gconfig.items():
                        file_data = file_data.replace('$' + k, str(v))
                    stream.write(file_data)
        bs_key = qcc.MODEL_INFO_BATCH_SIZE
        if self.batchsize:
            bs = self.batchsize
        elif 'info' in config['model'] and 'batchsize' in config['model']['info']:
            bs = config['model']['info'][bs_key]
        else:
            bs = 1
        bs = str(bs)
        with open(cur_path, 'r') as stream:
            file_data = stream.read()
        # modify the batchsize in the config file
        file_data = re.sub('[\"\']\s*\*\s*[\"\']', bs, file_data)
        file_data = re.sub(bs_key + ':\s+\d+', bs_key + ': ' + bs, file_data)
        file_data = re.sub("-\s\'\*\'", '- ' + bs, file_data)

        # write again so that config can be reloaded
        with open(cur_path, 'w') as stream:
            stream.write(file_data)

        with open(cur_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ce.ConfigurationException('incorrect configuration file', exc)

        self.set_config_from_yaml(config['model'], use_memory_plugins=self.use_memory_plugins)
        if 'info' in config['model']:
            qacc_file_logger.info('Model Info : {}'.format(config['model']['info']))
        if 'globals' in config['model']:
            qacc_file_logger.info('Global vars : {}'.format(config['model']['globals']))

    @staticmethod
    def get_batch_dims_info(config, node_type):
        batch_node_type_key = f'{node_type}s_info'
        batch_node_info = {}
        if 'inference-engine' in config['model'] and batch_node_type_key in config['model'][
                'inference-engine'] and config['model']['inference-engine'][batch_node_type_key]:
            nodes_config_info = config['model']['inference-engine'][batch_node_type_key]
            for node_infos in nodes_config_info:
                for node_name, node_info in node_infos.items():
                    if '*' in node_info['shape']:
                        batch_node_info[node_name] = node_info['shape'].index('*')
                    else:
                        batch_node_info[node_name] = None
        return batch_node_info

    def set_config_from_yaml(self, config, use_memory_plugins=False):
        """Set dataset, processing, inference and evaluator config using yaml
        config.

        Args:
            config: config as dictionary from yaml
            use_memory_plugins: Flag to enable memory plugins usage

        Raises:
            ConfigurationException
        """

        if 'info' in config:
            info = InfoConfiguration(**config['info'])
        else:
            info = InfoConfiguration()
        if 'dataset' in config:
            dataset = DatasetConfiguration(**config['dataset'])
        else:
            dataset = None
            qacc_file_logger.info('No dataset section found')

        if self.is_sub_section_valid(config, 'processing', 'preprocessing'):
            preprocessing = ProcessingConfiguration(**config['processing']['preprocessing'],
                                                    use_memory_plugins=use_memory_plugins)
            # Squash is disabled for preprocessing.
            preprocessing.squash_results = False
        else:
            preprocessing = None
            qacc_file_logger.info('No preprocessing section found')

        if self.is_sub_section_valid(config, 'processing', 'postprocessing'):
            postprocessing = ProcessingConfiguration(**config['processing']['postprocessing'],
                                                     use_memory_plugins=use_memory_plugins)
        else:
            postprocessing = None
            qacc_file_logger.info('No postprocessing section found')

        if ('inference-engine' in config):
            infer_engines = InferenceEngineConfiguration(**config['inference-engine'],
                                                         work_dir=self._work_dir,
                                                         batching_dims=self.batching_dims)
        else:
            infer_engines = None
            qacc_file_logger.info('No inference-engine section found')

        if ('evaluator' in config):
            evaluator = EvaluatorConfiguration(**config['evaluator'])
        else:
            evaluator = None
            qacc_file_logger.info('No evaluator section found')

        self.set_config(info_config=info, dataset_config=dataset,
                        preprocessing_config=preprocessing, postprocessing_config=postprocessing,
                        inference_config=infer_engines, evaluator_config=evaluator)

    def is_sub_section_valid(self, config, section, subsection):
        """Returns true if the subsection is configured in the model config."""
        if (section in config) and (config[section]) and \
                (subsection in config[section]):
            return True
        else:
            return False

    def set_config(self, info_config=None, dataset_config=None, preprocessing_config=None,
                   postprocessing_config=None, inference_config=None, evaluator_config=None):
        """Setter for config.

        Args:
            info_config
            dataset_config
            preprocessing_config
            postprocessing_config
            inference_config
            evaluator_config
        """
        self._info_config = info_config
        self._dataset_config = dataset_config
        self._preprocessing_config = preprocessing_config
        self._postprocessing_config = postprocessing_config
        self._inference_config = inference_config
        self._evaluator_config = evaluator_config

    def get_ref_inference_schema(self):
        ref_found = False
        ref_inference_schemas = []
        for inference_schema in self._inference_config._inference_schemas:
            if inference_schema._is_ref:
                ref_found = True
                qacc_file_logger.info('[configuration] schema' + str(inference_schema._idx) + '_' +
                                      inference_schema._name + '[is_ref=True]')
                ref_inference_schemas.append(inference_schema)

        if not ref_found:
            qacc_file_logger.info(
                'is_ref is not set for any inference schema, tool is using first inference schema as reference inference schema'
            )
            qacc_file_logger.info('reference inference schema name=schema0_' +
                                  self._inference_config._inference_schemas[0]._name)
            return self._inference_config._inference_schemas[0]
        else:
            if len(ref_inference_schemas) > 1:
                qacc_file_logger.info('is_ref is set to True for multiple inference schemas')
                qacc_file_logger.info(
                    'tool is using first configured inference schema as reference inference schema')

            qacc_file_logger.info('reference inference schema name=schema' +
                                  str(ref_inference_schemas[0]._idx) + '_' +
                                  ref_inference_schemas[0]._name)
            return ref_inference_schemas[0]


class InfoConfiguration:

    def __init__(self, desc=None, batchsize=1):
        self._desc = desc
        self._batchsize = batchsize if batchsize is not None else 1
        if batchsize is None:
            qacc_file_logger.error(
                '{} not present in info section of model config. Using {} = 1.'.format(
                    qcc.MODEL_INFO_BATCH_SIZE, qcc.MODEL_INFO_BATCH_SIZE))


class DatasetConfiguration:
    """QACC dataset configuration class.

    To use:
    >>> dataset_config = DatasetConfiguration(name, path, inputlist_file,
                                                    annotation_file, calibration, max_inputs)
    """

    def __init__(self, name="Unnamed", path=None, inputlist_file=None, annotation_file=None,
                 calibration=None, max_inputs=None, transformations=None):
        self._name = name

        if path:
            self._path = path
            if not os.path.exists(self._path):
                raise ce.ConfigurationException('Invalid dataset path {} in model config'.format(
                    self._path))
            self._inputlist_path = path
            self._calibration_path = path
        else:
            raise ce.ConfigurationException('Dataset path not provided in model config')

        if inputlist_file:
            self._inputlist_file = os.path.join(self._inputlist_path, inputlist_file)
            if not os.path.exists(self._inputlist_file):
                raise ce.ConfigurationException(
                    'Invalid inputlist_file path {} in model config'.format(self._inputlist_file))
        else:
            raise ce.ConfigurationException('Inputfile path not provided in model config')

        if annotation_file:
            self._annotation_file = os.path.join(self._path, annotation_file)
            if not os.path.exists(self._annotation_file):
                raise ce.ConfigurationException(
                    'Invalid annotation_file path {} in model config'.format(self._annotation_file))
        else:
            self._annotation_file = None

        if calibration:
            self._calibration_file = os.path.join(self._calibration_path, calibration['file'])
            if not os.path.exists(self._calibration_file):
                raise ce.ConfigurationException(
                    'Invalid calibration file path {} in model config'.format(
                        self._calibration_file))
            self._calibration_type = calibration['type']
            if self._calibration_type not in [
                    qcc.CALIBRATION_TYPE_INDEX, qcc.CALIBRATION_TYPE_RAW,
                    qcc.CALIBRATION_TYPE_DATASET
            ]:
                raise ce.ConfigurationException(
                    'Invalid calibration type {}. Can be index|raw|dataset'.format(
                        self.calibration_type))
        else:
            self._calibration_file = None
            self._calibration_type = None

        self._max_inputs = max_inputs
        self._update_max_inputs()

        # instantiate transformations
        self._transformations = TransformationsConfiguration(transformations)

    def _update_max_inputs(self):
        max_count = sum(1 for input in open(self._inputlist_file))
        # if self._max_inputs is None or -1 == self._max_inputs:
        self._max_inputs = max_count

    def __str__(self):
        return self._name

    def validate(self):
        # TODO: check file exists or not
        """Validates the dataset config.

        Returns:
            path: true if the dataset config is valid and false otherwise.
        """
        pass


class ProcessingConfiguration:
    """QACC processing configuration class handling both preprocessing and
    postprocessing.

    Note: Set use_memory_plugins Flag=True to enable memory plugins usage
    To use:
    >>> preprocessing_config = ProcessingConfiguration(name, path, generate_annotation,
                                                        inputlist_file,
                                                        annotation_file, calibration_file,
                                                        max_inputs, use_memory_plugins=False)
    """

    def __init__(self, transformations, path=None, target=None, enable=True, squash_results=False,
                 save_outputs=False, defaults=None, use_memory_plugins=False):
        self._transformations = TransformationsConfiguration(transformations,
                                                             use_memory_plugins=use_memory_plugins)
        self._path = path
        # squash only used for post processors.
        self._squash_results = squash_results
        if target == None:
            # TODO set target from default config
            pass
        else:
            self._target = target
        self._enable = enable


class PluginConfiguration:
    """QACC plugin configuration class."""

    def __init__(self, name, input_info=None, output_info=None, env=None, indexes=None,
                 params=None):
        self._name = name
        if name in pl.PluginManager.registered_plugins:
            self._cls = pl.PluginManager.registered_plugins[name]
            if inspect.isclass(self._cls) and issubclass(self._cls, pl.qacc_plugin):
                self._input_info = self.get_info_dict(input_info, type='in')
                self._output_info = self.get_info_dict(output_info, type='out')
        elif name in pl.PluginManager.registered_metric_plugins:
            # metric plugins dont need input and output info.
            self._cls = pl.PluginManager.registered_metric_plugins[name]
            if inspect.isclass(self._cls) and issubclass(self._cls, pl.qacc_metric):
                self._input_info = None
                self._output_info = None
        elif name in pl.PluginManager.registered_dataset_plugins:
            # dataset plugins don't need input and output info.
            self._cls = pl.PluginManager.registered_dataset_plugins[name]
            self._input_info = None
            self._output_info = None
        else:
            raise ce.ConfigurationException('Configured plugin {} is not registered'.format(name))

        self._env = env
        self._indexes = indexes.split(',') if indexes else None
        self._params = params

    def get_info_dict(self, info, type):
        """Type=mem|path|dir, dtype=float32, format=cv2."""
        info_dict = {}
        if info:
            info = info.split(',')
            for i in info:
                kv = i.strip().split('=')
                info_dict[kv[0]] = kv[1]
        else:
            # use default defined in Plugin class
            if type == 'in':
                info_dict = self._cls.default_inp_info
            else:
                info_dict = self._cls.default_out_info

        return info_dict

    def __str__(self):
        return '\nPlugin Info::\nName: {}\nInput:{}\nOutput:{}\nEnv:{}\nIndex:{}\n' \
               '\nParams:{}' \
            .format(self._name, self._input_info, self._output_info, self._env, self._indexes,
                    self._params)


class TransformationsConfiguration:
    """QACC transformations configuration class."""

    def __init__(self, transformations, use_memory_plugins=False):
        self._plugin_config_list = self.get_plugin_list_from_dict(
            transformations, use_memory_plugins=use_memory_plugins) if transformations else []

    def get_plugin_list_from_dict(self, transformations, use_memory_plugins=False):
        """Returns a list of plugin objects from the dictionary of plugin
        objects.

        Args:
            transformations: transformations as dictionary
            use_memory_plugins : Flag to enable memory plugins usage
        Returns:
            plugin_config_list: list of plugin config objects
        """
        plugin_config_list = []
        for plugin in transformations:
            plugin_config_list.append(PluginConfiguration(**plugin['plugin']))

        # validate plugin list
        # Skip validation for Memory Plugins
        if not use_memory_plugins:
            self.validate_plugin_config_list(plugin_config_list)
        return plugin_config_list

    def validate_plugin_config_list(self, plugin_config_list):
        """Directory plugins can't have output configured as path/mem."""
        for idx, plugin in enumerate(plugin_config_list):
            if plugin._input_info and plugin._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR:
                if plugin._output_info[qcc.IO_TYPE] != qcc.PLUG_INFO_TYPE_DIR:
                    raise ce.ConfigurationException(
                        '{} : Directory plugin with Path or Mem output type '
                        'not supported'.format(plugin._name))


class InferenceEngineConfiguration:

    def __init__(self, model_path, batching_dims, clean_model=True, inference_schemas=None,
                 inputs_info=None, outputs_info=None, device_ids=None, max_calib=-1,
                 simplify_model=True, check_model=True, onnx_define_symbol=None, export_format=None,
                 auto_inference_schema=None, inference_schema_common_params=None, work_dir=None):
        # note: batching_dims is supplied from config based on '*' index in model config.
        self._model_object = False  # this flag is to check and skip model cleaning if model_path is a tf.session or nn.module
        self._inference_schemas = []
        self._clean_model = True  #Setting this to true every time
        self._simplify_model = simplify_model
        # Skip framework specific Validations if any
        self._check_model = check_model
        model_zoo_path = os.environ.get('MODEL_ZOO_PATH', qcc.DEFAULT_MODEL_ZOO_PATH)
        # Prepend Model Zoo path with the Assumption that model path is relative to MODEL_ZOO_PATH Environment variable
        model_path_relative_model_zoo = os.path.join(model_zoo_path, model_path)
        # Check if model path provided exist if yes do not prepend model_zoo_path
        if os.path.exists(model_path):
            self._model_path = model_path
        elif os.path.exists(model_path_relative_model_zoo):
            self._model_path = model_path_relative_model_zoo
        else:
            raise ce.ConfigurationException(
                f"Model not found in the configured path: {model_path}.  \
                                        Model path must be either absolute or relative to the MODEL_ZOO_PATH Environment variable."
            )
        self._cleaned_only_model_path = None
        self._max_calib = max_calib
        self._device_ids = device_ids if isinstance(device_ids, list) else [device_ids]
        self._converter_export_format = export_format if export_format is not None else qcc.DEFAULT_CONVERTER_EXPORT_FORMAT
        qacc_file_logger.debug(f"Evaluating using export format: {self._converter_export_format }")
        # used to perform calibration if int8 inference schema available
        self._is_calib_req = False
        # Skip framework specific Validations if any
        self._check_model = check_model
        # Any Symbols used internally by onnx model
        self._onnx_define_symbol = onnx_define_symbol
        #  params that are common across all inference schemas: eg custom-op
        self._inference_schema_common_params = inference_schema_common_params
        # Save input names in order
        self._input_names = []

        # Formatting the input_info and output_info as needed by inference engine.
        self._input_info = self.format_model_node_info(node_type='input',
                                                       configured_info=inputs_info,
                                                       batching_dims=batching_dims)
        self._output_info = self.format_model_node_info(node_type='output',
                                                        configured_info=outputs_info,
                                                        batching_dims=batching_dims)
        # Making the output info to float32 in qnn inference schema
        inference_schema_names = []
        #  Auto inference schema
        if isinstance(auto_inference_schema, dict):
            self._auto_inference_schema = auto_inference_schema.get('enabled', False)
            run_native_inference_schema = auto_inference_schema.get('run_native_inference_schema',
                                                                    True)
        else:  # for bool and None Case
            self._auto_inference_schema = auto_inference_schema
            run_native_inference_schema = True
        if self._auto_inference_schema:
            inference_schemas = self.add_auto_inference_schema(
                run_native_inference_schema=run_native_inference_schema)
            for idx, schema in enumerate(inference_schemas):
                schema._input_info = self._input_info
                schema._output_info = self._output_info
                schema._model_path = self._model_path
                schema._idx = idx
                inference_schema_names.append(schema._name)
                self._inference_schemas.append(schema)
        elif inference_schemas and len(inference_schemas) > 0:
            auto_quantization_flag = False
            # TODO: Enable auto quantization for QNN
            for idx, inference_schema in enumerate(inference_schemas):
                if (inference_schema['inference_schema']['precision'] == qcc.PRECISION_QUANT
                        and 'converter_params' not in inference_schema['inference_schema']):
                    inference_schema['inference_schema']['converter_params'] = defaults.get_value(
                        'qacc.auto_quantization')
                    auto_quantization_flag = True
                if self._inference_schema_common_params:
                    inference_schema = self.update_inference_schema_common_params(inference_schema)
                schema = QnnInferenceSchemaConfiguration(**inference_schema['inference_schema'],
                                                         work_dir=work_dir,
                                                         model_path=self._model_path)
                schema._input_info = self._input_info
                schema._output_info = self._output_info
                schema._idx = idx
                self._inference_schemas.append(schema)
                if auto_quantization_flag:
                    break  # Stop adding other combinations
                qacc_file_logger.debug(f"Inference schema name : {schema._name}")
                inference_schema_names.append(schema._name)
        else:
            raise ce.ConfigurationException("No inference schemas to run. Check the config file.")

        # Add int64 flags if input type is int64
        if self._input_info:
            is_inp_int64 = False
            for inp, info in self._input_info.items():
                if info[0] == "int64":
                    is_inp_int64 = True

            if is_inp_int64:
                for idx, schema in enumerate(self._inference_schemas):
                    if schema._name == "qnn" and schema._backend != "htp" and "dspv" not in schema._backend:
                        converter_params = copy.deepcopy(schema._converter_params)
                        converter_params["keep_int64_inputs"] = "True"
                        converter_params["use_native_dtype"] = "True"
                        schema._converter_params = converter_params

    def format_model_node_info(self, node_type, configured_info, batching_dims):
        fmt_info = None
        if configured_info:
            fmt_info = {}
            for out_info in configured_info:
                for k, m in out_info.items():
                    if node_type == 'input':
                        assert len(m.values(
                        )) == 2, 'Invalid format for input info. Should have type and shape keys'
                    if node_type == 'output':
                        assert len(m.values(
                        )) >= 2, 'Invalid format for output info. Must have type and shape keys'
                    if isinstance(m['shape'], list):
                        node_name = Helper.sanitize_node_names(str(k))
                        fmt_info[node_name] = [
                            m['type'], m['shape'], batching_dims[f'{node_type}s_info'][k]
                        ]
                        if node_type == 'input':
                            self._input_names.append(node_name)
                        if node_type == 'output' and 'comparator' in m:
                            fmt_info[node_name].append(m['comparator'])
                    else:
                        raise ce.ConfigurationException('Invalid shape in {} info :{}.'
                                                        ' usage e.g: [1,224,224,3]'.format(
                                                            node_type, m['shape']))
        return fmt_info

    def update_inference_schema_common_params(self, inference_schema):
        for inference_schema_param in self._inference_schema_common_params:
            for inference_schema_type, params in inference_schema_param.items():
                if inference_schema_type == inference_schema['inference_schema']['name']:
                    # add params section if not already present
                    if 'params' not in inference_schema['inference_schema']:
                        inference_schema['inference_schema']['params'] = {}

                    for k, v in params.items():
                        if k in inference_schema['inference_schema'][
                                'params']:  # if key:value given within inference schema ignore
                            continue
                        else:  # If not given update
                            inference_schema['inference_schema']['params'].update({k: v})
        return inference_schema

    def add_auto_inference_schema(self, run_native_inference_schema=True):
        """Adds a Reference(Native) inference schema, fp16 and int8 aic
        inference schemas."""
        inference_schemas = []
        if run_native_inference_schema:
            reference_inference_schemas = [
                qcc.INFER_ENGINE_ONNXRT, qcc.INFER_ENGINE_TFRT, qcc.INFER_ENGINE_TORCHSCRIPTRT
            ]
            ref_inference_schema_type_idx = Helper.get_model_type(self._model_path).value
            ref_name = reference_inference_schemas[ref_inference_schema_type_idx]
            reference_inference_schema = InferenceSchemaConfiguration(name=ref_name,
                                                                      precision='fp32', is_ref=True)
            inference_schemas.append(reference_inference_schema)

        aic_common_params = {}
        if self._inference_schema_common_params:
            aic_common_params = [
                inference_schema_param
                for inference_schema_param in self._inference_schema_common_params
                for inference_schema_type, params in inference_schema_param.items()
                if inference_schema_type == 'aic'
            ]
            if len(aic_common_params) != 0:
                aic_common_params = aic_common_params[0][
                    'aic']  # Always the 1st item [Only Distinct inference schema types]

        if self._inference_schema_common_params is None or len(aic_common_params) == 0:
            aic_common_params = {}

        # fp16 aic inference schema
        fp16_inference_schema = InferenceSchemaConfiguration(name='aic', precision='fp16',
                                                             params=aic_common_params)
        inference_schemas.append(fp16_inference_schema)

        auto_quant_params = defaults.get_value('qacc.auto_quantization')
        auto_quant_params.update(aic_common_params)

        auto_quant_inference_schema = InferenceSchemaConfiguration(name='aic', precision='int8',
                                                                   params=auto_quant_params)
        inference_schemas.append(auto_quant_inference_schema)

        return inference_schemas

    def set_inference_schema_names(self):
        inference_schema_names = {}
        for inference_schema in self._inference_schemas:
            if inference_schema._idx not in inference_schema_names:
                inference_schema_names[inference_schema._idx] = []
            schema_name, converter_options_name = inference_schema.get_inference_schema_name_with_params(
            )
            if converter_options_name not in inference_schema_names[inference_schema._idx]:
                inference_schema_names[inference_schema._idx].append(converter_options_name)
            schema_id = inference_schema_names[inference_schema._idx].index(converter_options_name)
            schema_name += "_" + str(schema_id)
            inference_schema._inference_schema_name = schema_name


class InferenceSchemaConfiguration:

    def __init__(self, name, env=None, params={}, binary_path=None, multithreaded=True,
                 precision='fp32', use_precompiled=None, reuse_pgq=True, pgq_group=None, tag='',
                 is_ref=False):

        self._name = name
        self._idx = None
        self._env = env
        self._params = params if params else {}
        # batchsize is added in manager based on exec support of providing
        # batchsize with a model having multiple inputs.
        # This filed is added while setting pipeline cache with key INTERNAL_EXEC_BATCH_SIZE

        # onnxrt specific
        self._multithreaded = multithreaded

        # AIC specific params
        self._use_precompiled = use_precompiled
        self._binary_path = binary_path
        if precision == 'default':
            self._precision = qcc.PRECISION_FP32
        else:
            self._precision = precision
        self._input_info = None
        self._output_info = None
        self._precompiled_path = use_precompiled
        self._is_ref = is_ref

        # to specify if PGQ profile to be reused across
        # the generated inference schemas after search space scan
        self._reuse_pgq = reuse_pgq

        # to specify if PGQ profile can be used across multiple
        # inference schemas. All the inference schemas having the pgq_group tag will
        # use the same generated PGQ profile. If reuse_pgq is False the
        # PGQ profile will be regenerated for all the configured inference schemas
        # in one inference schema section.
        self._pgq_group = pgq_group

        # inference schema tag used to filter an inference schema from multiple same
        # type of inference schemas.
        # Example: Used to distinguish between multiple inference schemas
        self._tag = [t.strip() for t in tag.split(',')]
        self.validate()

    def validate(self):
        if self._name is None:
            raise ce.ConfigurationException('inference-engine inference_schema name is mandatory!')
        if self._name not in [
                qcc.INFER_ENGINE_AIC, qcc.INFER_ENGINE_ONNXRT, qcc.INFER_ENGINE_TFRT,
                qcc.INFER_ENGINE_TFRT_SESSION, qcc.INFER_ENGINE_TORCHSCRIPTRT
        ]:
            raise ce.ConfigurationException('Invalid or Unsupported inference schema: {}'.format(
                self._name))

        if self._precision not in [qcc.PRECISION_FP16, qcc.PRECISION_FP32, qcc.PRECISION_INT8]:
            raise ce.ConfigurationException('Invalid precision in inference engine: {}'.format(
                self._precision))

        if 'external-quantization' in self._params:
            profile_path = self._params.get('external-quantization', None)
            if profile_path is None or not os.path.exists(profile_path):
                raise ce.ConfigurationException(
                    f"External Quantization profile supplied is invalid or doesn't exists: {profile_path}"
                )

        if self._is_ref is True and self._params is not None:
            is_permutational = False
            for key, val in self._params.items():
                val = str(val)
                vals = [v.strip() for v in val.split(qcc.SEARCH_SPACE_DELIMITER)]
                if len(vals) > 1:
                    is_permutational = True
                    break
                for v_idx, v in enumerate(vals):
                    if v.startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and v.endswith(')'):
                        is_permutational = True
            if is_permutational:
                qacc_file_logger.error(
                    "is_ref is set to True for a configuration which generates multiple inference schemas."
                )
                qacc_file_logger.error(
                    "Please set a single inference schema as reference inference schema")
                raise ce.ConfigurationException(
                    'inference_schema={}, is_ref is set to True for a configuration which generates multiple inference schemas.'
                    .format(self._name))

    def __str__(self):
        print('name: {}, params: {}'.format(self._name, self._params))

    def __eq__(self, other_inference_schema):

        if self._name == other_inference_schema._name and self._precision == other_inference_schema._precision \
                and self._precompiled_path == other_inference_schema._precompiled_path and self._params \
                == other_inference_schema._params:
            return True
        else:
            return False


class QnnInferenceSchemaConfiguration:

    def __init__(self, name, target_arch="x86_64-linux-clang", backend="cpu", env=None,
                 backend_extensions={}, netrun_params={}, converter_params={}, contextbin_params={},
                 binary_path=None, multithreaded=True, precision='fp32', use_precompiled=None,
                 reuse_pgq=True, pgq_group=None, tag='', is_ref=False, convert_nchw=False,
                 model_path=None, context_backend_extensions_json=None,
                 netrun_backend_extensions_json=None, work_dir=None):

        self._name = name
        self._idx = None
        self._env = env
        self._model_path = model_path

        self._target_arch = target_arch
        self._backend = backend
        self._backend_extensions = backend_extensions
        self._netrun_params = netrun_params
        self._converter_params = converter_params
        self._contextbin_params = contextbin_params
        self._context_backend_extensions_json = context_backend_extensions_json
        self._netrun_backend_extensions_json = netrun_backend_extensions_json
        # qacc_file_logger.debug(f"{self._backend_extensions}\n {self._netrun_params}\n" +
        #                        f"{self._converter_params}")
        # batchsize is added in manager based on exec support of providing
        # batchsize with a model having multiple inputs.
        # This filed is added while setting pipeline cache with key INTERNAL_EXEC_BATCH_SIZE

        # onnxrt specific
        self._multithreaded = multithreaded

        # AIC specific params
        self._use_precompiled = use_precompiled
        self._binary_path = binary_path
        if precision == 'default':
            self._precision = qcc.PRECISION_FP32
        else:
            self._precision = precision
        self._input_info = None
        self._output_info = None
        self._precompiled_path = use_precompiled
        self._is_ref = is_ref

        # to specify if PGQ profile to be reused across
        # the generated inference schemas after search space scan
        self._reuse_pgq = reuse_pgq

        # to specify if PGQ profile can be used across multiple
        # inference schemas. All the inference schemas having the pgq_group tag will
        # use the same generated PGQ profile. If reuse_pgq is False the
        # PGQ profile will be regenerated for all the configured inference schemas
        # in one inference schema section.
        self._pgq_group = pgq_group

        # inference schema tag used to filter a inference schema from multiple same
        # type of inference schemas.
        # Example: Used to distinguish between multiple inference schemas
        self._tag = [t.strip() for t in tag.split(',')]

        # Modify input layout from NHWC to NCHW for onnxrt inference schema,
        # Since we pass NHWC inputs to the QNN.
        self._convert_nchw = convert_nchw

        self._inference_schema_name = None

        # Convert npi yaml to json
        if "quantization_overrides" in self._converter_params:
            if os.path.exists(self._converter_params['quantization_overrides']):
                filepath, extn = os.path.splitext(self._converter_params['quantization_overrides'])
                if extn not in ['.yaml', '.json']:
                    raise ValueError(
                        "quantization_overrides must be in either of YAML or JSON file format")
                else:
                    filename = os.path.basename(filepath)
                    output_json = os.path.join(work_dir, f"{filename}.json")
                    output_json_cleaned = os.path.join(work_dir, f"{filename}_cleaned.json")

                    # convert YAML to JSON format update quantization_overrides in _converter_params
                    if extn == ".yaml":
                        convert_npi_to_json(self._converter_params['quantization_overrides'],
                                            output_json)
                        self._converter_params['quantization_overrides'] = output_json

                    # Cleanup the JSON with appropriate cleaned up node names
                    output_json_cleaned = cleanup_quantization_overrides(
                        self._converter_params['quantization_overrides'], self._model_path,
                        outpath=output_json_cleaned)
                    self._converter_params['quantization_overrides'] = output_json_cleaned
            else:
                raise ValueError(
                    f"quantization_overrides file {self._converter_params['quantization_overrides']} configured doesn't exist. Please provide a valid quantization_overrides file"
                )
        if self._backend == qcc.BACKEND_HTP_MCP:
            if self._backend_extensions.get("elf_path") is None:
                qacc_file_logger.warning("Param elf_path not provided, using default path")
                self._backend_extensions["elf_path"] = qcc.DEFAULT_MCP_ELF_PATH
        if "dsp" in self._backend:
            if self._backend_extensions.get("dsp_arch") is None:
                dsp_version = 'v' + re.findall(r'\d+', self._backend)[0]
                self._backend_extensions["dsp_arch"] = dsp_version

        self.validate()

        if self._context_backend_extensions_json:
            self._backend_extensions = self._read_config_json(self._context_backend_extensions_json)
        if self._netrun_backend_extensions_json:
            self._backend_extensions = self._backend_extensions.update(
                self._read_config_json(self._netrun_backend_extensions_json))

    def _read_config_json(self, config_json):
        """Read param values from json and returns a dict."""
        with open(config_json, "r") as f:
            params = json.load(f)
        return params

    def get_inference_schema_name_with_params(self):
        """Returns an inference schema name based on its params."""
        inference_schema_name = 'schema' + str(self._idx) + '_' + self._name
        inference_schema_name += "_" + self._precision
        converter_options_name = ""
        for param in qcc.PIPE_SUPPORTED_CONVERTER_PARAMS:
            if param in self._converter_params:
                if param == "algorithms" and self._converter_params["algorithms"] == "default":
                    continue
                if self._converter_params[param] != "False":
                    converter_options_name += "_" + qcc.PIPE_SUPPORTED_CONVERTER_PARAMS[param]
                    if self._converter_params[param] != "True":
                        converter_options_name += self._converter_params[param]

        return inference_schema_name, converter_options_name

    def get_inference_schema_name(self):
        return self._inference_schema_name

    def validate(self):
        if self._name is None:
            raise ce.ConfigurationException('inference-engine inference_schema name is mandatory!')
        if self._name not in [
                qcc.INFER_ENGINE_QNN, qcc.INFER_ENGINE_AIC, qcc.INFER_ENGINE_ONNXRT,
                qcc.INFER_ENGINE_TFRT, qcc.INFER_ENGINE_TFRT_SESSION, qcc.INFER_ENGINE_TORCHSCRIPTRT
        ]:
            raise ce.ConfigurationException('Invalid or Unsupported inference schema: {}'.format(
                self._name))

        if self._precision not in [
                qcc.PRECISION_FP16, qcc.PRECISION_FP32, qcc.PRECISION_INT8, qcc.PRECISION_QUANT
        ]:
            raise ce.ConfigurationException('Invalid precision in inference engine: {}'.format(
                self._precision))

        android_backends = [
            qcc.BACKEND_CPU, qcc.BACKEND_DSPV69, qcc.BACKEND_DSPV73, qcc.BACKEND_DSPV75,
            qcc.BACKEND_GPU
        ]
        x86_backends = [qcc.BACKEND_CPU, qcc.BACKEND_HTP, qcc.BACKEND_AIC, qcc.BACKEND_HTP_MCP]
        if (self._target_arch == 'x86_64-linux-clang' and self._backend not in x86_backends) or (
                self._target_arch == 'aarch64-android' and self._backend not in android_backends):
            raise ce.ConfigurationException(
                'Target architecture {} not supported for backend {}'.format(
                    self._target_arch, self._backend))

        if 'quantization_overrides' in self._converter_params:
            profile_path = self._converter_params.get('quantization_overrides', None)
            if profile_path is None or not os.path.exists(profile_path):
                raise ce.ConfigurationException(
                    f"External Quantization profile supplied is invalid or doesn't exists: {profile_path}"
                )

        if self._converter_params:
            invalid_params = set(self._converter_params.keys()).difference(
                set(qcc.SUPPORTED_CONVERTER_PARAMS))
            if invalid_params:
                raise ce.ConfigurationException(
                    f"Invalid converter parameters provided: {invalid_params}")

        if self._contextbin_params:
            invalid_params = set(self._contextbin_params.keys()).difference(
                set(qcc.SUPPORTED_CONTEXT_BINARY_PARAMS))
            if invalid_params:
                raise ce.ConfigurationException(
                    f"Invalid context-binary parameters provided: {invalid_params}")
            if "enable_intermediate_outputs" in self._contextbin_params and "set_output_tensors" in self._contextbin_params:
                raise ce.ConfigurationException(
                    "Options enable_intermediate_outputs and set_output_tensors are mutually exclusive. Only one of them can be specified at a time"
                )
            if "extra_args" in self._contextbin_params:
                overlap_params = [
                    key for key in self._contextbin_params
                    if key != "extra_args" and key in self._contextbin_params["extra_args"]
                ]
                if overlap_params:
                    raise ce.ConfigurationException(
                        f"{overlap_params} provided as both context-binary params and extra args")

        if self._netrun_params:
            invalid_params = set(self._netrun_params.keys()).difference(
                set(qcc.SUPPORTED_NETRUN_PARAMS))
            if invalid_params:
                raise ce.ConfigurationException(
                    f"Invalid netrun parameters provided: {invalid_params}")
            if "use_native_input_files" in self._netrun_params and "native_input_tensor_names" in self._netrun_params:
                raise ce.ConfigurationException(
                    "Options use_native_input_files and native_input_tensor_names are mutually exclusive. Only one of them can be specified at a time"
                )
            if "max_input_cache_tensor_sets" in self._netrun_params and "max_input_cache_size_mb" in self._netrun_params:
                raise ce.ConfigurationException(
                    "Options max_input_cache_tensor_sets and max_input_cache_size_mb are mutually exclusive. Only one of them can be specified at a time"
                )
            if "extra_args" in self._netrun_params:
                overlap_params = [
                    key for key in self._netrun_params
                    if key != "extra_args" and key in self._netrun_params["extra_args"]
                ]
                if overlap_params:
                    raise ce.ConfigurationException(
                        f"{overlap_params} provided as both netrun params and extra args")

        if self._backend_extensions:
            backend = "htp" if "dspv" in self._backend else self._backend
            context_backend_ext = set(self._backend_extensions.keys()).difference(
                set(qcc.SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS[backend]))
            if context_backend_ext and self._context_backend_extensions_json is not None:
                raise ce.ConfigurationException(
                    "context_backend_extensions_json and backend_extensions cannot be used together"
                )
            invalid_context_backend_ext = set(context_backend_ext).difference(
                set(qcc.SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS[backend]))
            if invalid_context_backend_ext:
                raise ce.ConfigurationException(
                    f"Invalid context backend extensions provided: {invalid_context_backend_ext}")

            netrun_backend_ext = set(self._backend_extensions.keys()).difference(
                set(qcc.SUPPORTED_CONTEXT_BACKEND_EXTENSION_PARAMS[backend]))
            if netrun_backend_ext and self._netrun_backend_extensions_json is not None:
                raise ce.ConfigurationException(
                    "netrun_backend_extensions_json and backend_extensions cannot be used together")
            invalid_netrun_backend_ext = set(netrun_backend_ext).difference(
                set(qcc.SUPPORTED_NETRUN_BACKEND_EXTENSION_PARAMS[backend]))
            if invalid_netrun_backend_ext:
                raise ce.ConfigurationException(
                    f"Invalid netrun backend extensions provided: {invalid_netrun_backend_ext}")

        if self._is_ref is True and self._converter_params is not None:
            is_permutational = False
            for key, val in self._converter_params.items():
                val = str(val)
                vals = [v.strip() for v in val.split(qcc.SEARCH_SPACE_DELIMITER)]
                if len(vals) > 1:
                    is_permutational = True
                    break
                for v_idx, v in enumerate(vals):
                    if v.startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and v.endswith(')'):
                        is_permutational = True
            if is_permutational:
                qacc_file_logger.error(
                    "is_ref is set to True for a configuration which generates multiple inference schemas."
                )
                qacc_file_logger.error(
                    "Please set a single inference schema as reference inference schema")
                raise ce.ConfigurationException(
                    'inference_schema={}, is_ref is set to True for a configuration which generates multiple inference schemas.'
                    .format(self._name))
        # Check for ADB binary
        if self._backend == 'htp' or self._target_arch == 'aarch64-android':
            # check for adb binary path from env variable exist or adb from default path exist
            adb_path = os.environ.get("ADB_PATH", qcc.DEFAULT_ADB_PATH)
            if not os.path.exists(adb_path):
                raise ce.ConfigurationException("Environment variable ADB_PATH not set. \
                                                ADB_PATH is required while evaluating htp backend or target_arch is aarch64-android. \
                                                Set the path to adb binary.")

    def __str__(self):
        print('name: {}, params: {}'.format(self._name, self._converter_params))

    def __eq__(self, other_inference_schema):

        if (self._name == other_inference_schema._name
                and self._precision == other_inference_schema._precision
                and self._precompiled_path == other_inference_schema._precompiled_path
                and self._backend_extensions == other_inference_schema._backend_extensions
                and self._netrun_params == other_inference_schema._netrun_params
                and self._converter_params == other_inference_schema._converter_params
                and self._contextbin_params == other_inference_schema._contextbin_params):
            return True
        else:
            return False


class EvaluatorConfiguration:
    """QACC Evaluator configuration class."""

    def __init__(self, comparator=None, metrics=None):
        # comparator is enabled by default. If not provided, create default config.
        if comparator is None:
            self._comparator = {'enabled': True, 'fetch_top': 1, 'type': 'avg', 'tol': 0.001}
        else:
            self._comparator = {}
            self._comparator['enabled'] = comparator['enabled'] if 'enabled' in comparator else True
            self._comparator[
                'fetch_top'] = comparator['fetch-top'] if 'fetch-top' in comparator else 1
            self._comparator['type'] = comparator['type'] if 'type' in comparator else 'avg'
            self._comparator['tol'] = comparator['tol'] if 'tol' in comparator else 0.001
            self._comparator['box_input_json'] = comparator[
                'box_input_json'] if 'box_input_json' in comparator else None

        self._metrics_plugin_list = None
        if metrics:
            self._metrics_plugin_list = self.get_plugin_list_from_dict(metrics)

    def get_plugin_list_from_dict(self, metrics):
        """Returns a list of plugin objects from the dictionary of plugin
        objects.

        Args:
            transformations: metrics as dictionary

        Returns:
            plugin_config_list: list of plugin config objects
        """
        plugin_config_list = []
        for plugin in metrics:
            plugin_config_list.append(PluginConfiguration(**plugin['plugin']))
        return plugin_config_list


class PipelineCache:
    """Class acting as global pipeline_cache for the entire pipeline and
    plugins to share relevant information between the plugins or stages of the
    E2E pipeline."""
    __instance = None

    def __init__(self):
        if PipelineCache.__instance != None:
            pass
        else:
            PipelineCache.__instance = self
            self._pipeline_cache = {}
            self._nested_keys = [
                qcc.PIPELINE_POSTPROC_DIR, qcc.PIPELINE_POSTPROC_FILE, qcc.PIPELINE_INFER_DIR,
                qcc.PIPELINE_INFER_FILE, qcc.PIPELINE_NETWORK_DESC, qcc.PIPELINE_NETWORK_BIN_DIR,
                qcc.PIPELINE_PROGRAM_QPC, qcc.INTERNAL_INFER_TIME, qcc.INTERNAL_POSTPROC_TIME,
                qcc.INTERNAL_METRIC_TIME, qcc.INTERNAL_QUANTIZATION_TIME,
                qcc.INTERNAL_COMPILATION_TIME
            ]
            # init empty nested keys
            for key in self._nested_keys:
                self._pipeline_cache[key] = {}

    @classmethod
    def getInstance(cls):
        if PipelineCache.__instance == None:
            PipelineCache()
        return cls.__instance

    def set_val(self, key, val, nested_key=None):
        """Stores the key and value in the global dictionary."""
        if nested_key is None:
            self._pipeline_cache[key] = val
            qacc_file_logger.debug('Pipeline pipeline_cache - storing key {} value {}'.format(
                key, val))
        else:
            self._pipeline_cache[key][nested_key] = val
            qacc_file_logger.debug('Pipeline pipeline_cache - storing key {}:{}  value {}'.format(
                key, nested_key, val))

    def get_val(self, key, nested_key=None):
        """Returns value from information stored in dictionary during various
        stages of the pipeline.

        Args:
            key_string: nested keys in string format eg key.key.key

        Returns:
            value: value associated to the key, None otherwise
        """
        if key not in self._nested_keys:
            if key in self._pipeline_cache:
                return self._pipeline_cache[key]
            else:
                qacc_file_logger.warning('Pipeline pipeline_cache key {} incorrect'.format(key))
        else:
            if key in self._pipeline_cache and nested_key in self._pipeline_cache[key]:
                return self._pipeline_cache[key][nested_key]
            else:
                qacc_file_logger.warning('Pipeline pipeline_cache key {}:{} incorrect'.format(
                    key, nested_key))
        return None
