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
import os
import sys
import yaml
import json
import qti.aisw.accuracy_evaluator.common.defaults as df
import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.utilities import Helper, ModelType
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger

defaults = df.Defaults.getInstance()


def check_model_dir(config_file):
    """Checks if model path or dir is given."""
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ce.ConfigurationException('incorrect configuration file', exc)

    is_model_dir = False
    try:
        model_path = config['model']['inference-engine']['model_path']
        if os.path.isdir(model_path):
            is_model_dir = True
    except Exception as e:
        raise ce.ConfigurationException(f'Failed to read the config file', exc)

    return is_model_dir, model_path


def process_model_dir(model_dir_path, parent_work_dir):
    models = []
    work_dirs = []
    preproc_files = []
    for root, dirs, files in os.walk(model_dir_path):
        for f in files:
            if (f.endswith(".onnx") or f.endswith(".pb") or f.endswith(".tflite")
                    or f.endswith(".pt")):
                models.append(os.path.join(root, f))
                model_dir_name = root.split("/")[-2]
                model_work_dir = os.path.join(parent_work_dir, model_dir_name)
                work_dirs.append(model_work_dir)

    for model_path in models:
        curr_dir = os.path.dirname(os.path.dirname(model_path))  # Going to the parent directory
        data_list = os.path.join(curr_dir, "inputs", "input_list.txt")
        if os.path.isfile(data_list):
            preproc_files.append(data_list)
        else:
            raise ce.ConfigurationException(f'input_list.txt not present in the {curr_dir} -', exc)

    return models, preproc_files, work_dirs


def create_default_config(work_dir, model_path, backend, target_arch, comparator, tol_thresh,
                          input_info=None, act_bw=None, bias_bw=None, box_input=None):
    """Create temp config file from the command line parameters."""
    temp_dir = os.path.join(work_dir, "temp")
    os.makedirs(temp_dir)
    temp_config_file = os.path.join(temp_dir, "temp_config.yaml")
    temp_config = {
        "model": {
            "info": {
                "desc": "Default Config"
            },
            "inference-engine": None,
            "evaluator": None
        }
    }
    temp_config["model"]["inference-engine"] = {"model_path": None, "inference_schemas": None}
    temp_config["model"]["evaluator"] = {
        "comparator": None,
    }

    if input_info is not None:
        temp_config["model"]["inference-engine"]["inputs_info"] = []
        for inp in input_info:
            inp_name = inp[0]
            inp_shape = [int(i) for i in inp[1].split(',')]
            info_dict = {inp_name: {"shape": inp_shape, "type": "float32"}}
            temp_config["model"]["inference-engine"]["inputs_info"].append(info_dict)

    temp_config["model"]["inference-engine"]["inference_schemas"] = []
    temp_config["model"]["inference-engine"]["inference_schemas"].append(
        defaults.get_value("qacc.default_inference_schemas.cpu"))

    allowed_backends = [
        qcc.BACKEND_HTP, qcc.BACKEND_AIC, qcc.BACKEND_HTP_MCP, qcc.BACKEND_DSPV69,
        qcc.BACKEND_DSPV73, qcc.BACKEND_DSPV75, qcc.BACKEND_CPU, qcc.BACKEND_GPU
    ]
    if backend in allowed_backends:

        if target_arch == "x86_64-linux-clang":
            if backend == "htp":
                temp_config["model"]["inference-engine"]["inference_schemas"].append(
                    defaults.get_value("qacc.default_inference_schemas.htp_x86"))
            elif backend == "aic":
                temp_config["model"]["inference-engine"]["inference_schemas"].append(
                    defaults.get_value("qacc.default_inference_schemas.aic_x86"))
            elif backend == "htp_mcp":
                temp_config["model"]["inference-engine"]["inference_schemas"].append(
                    defaults.get_value("qacc.default_inference_schemas.htp_mcp_x86"))
            elif backend == "cpu":
                raise ce.ConfigurationException(
                    'Backend cpu on x86_64-linux-clang architecture is used as reference in minimal mode. Please choose a different backend or architecture.'
                )
            else:
                raise ce.ConfigurationException(
                    f'Target architecture {target_arch} not supported for backend {backend}')
        elif target_arch == "aarch64-android":
            if "dspv" in backend or backend == "gpu":
                temp_config["model"]["inference-engine"]["inference_schemas"].append(
                    defaults.get_value("qacc.default_inference_schemas." + backend))
            elif backend == "cpu":
                temp_config["model"]["inference-engine"]["inference_schemas"].append(
                    defaults.get_value("qacc.default_inference_schemas.cpu_android"))
            else:
                raise ce.ConfigurationException(
                    f'Target architecture {target_arch} not supported for backend {backend}')

        temp_config["model"]["inference-engine"]["model_path"] = model_path
    else:
        raise ce.ConfigurationException(f'Backend {backend} is not supported')

    #Update the comparator config
    temp_config["model"]["evaluator"]["comparator"] = defaults.get_value("qacc.comparator")
    temp_config["model"]["evaluator"]["comparator"]["type"] = comparator
    temp_config["model"]["evaluator"]["comparator"]["tol"] = tol_thresh
    if comparator == "box":
        temp_config["model"]["evaluator"]["comparator"]["box_input_json"] = box_input

    with open(temp_config_file, "w") as stream:
        yaml.dump(temp_config, stream, default_flow_style=False)

    return temp_config_file


def convert_npi_to_json(npi_yaml_file, output_json):
    """Converts npi yaml file to output json in qnn quantization_overrides
    format and saves it at output_json location."""

    with open(npi_yaml_file) as F:
        data = yaml.safe_load(F)

    list_of_tensors_in_fp16 = []
    list_of_tensors_in_fp32 = []

    for key in data:
        if key == 'FP16NodeInstanceNames':
            list_of_tensors_in_fp16.extend(data[key])
        elif key == 'FP32NodeInstanceNames':
            list_of_tensors_in_fp32.extend(data[key])
        else:
            print('Incorrect entry in YAML file: ', key)
            exit(1)

    overrides_dict = {"activation_encodings": {}, "param_encodings": {}}

    activation_encodings_dict = {}
    for tensor in list_of_tensors_in_fp32:
        value = [{"bitwidth": 32, "dtype": "float"}]
        activation_encodings_dict[tensor] = value

    for tensor in list_of_tensors_in_fp16:
        value = [{"bitwidth": 16, "dtype": "float"}]
        activation_encodings_dict[tensor] = value

    overrides_dict["activation_encodings"] = activation_encodings_dict

    with open(output_json, 'w') as fp:
        json.dump(overrides_dict, fp)


def process_encoding(encoding_dict, output_name_to_elem_type):
    new_encodings = {}
    for name, enc in encoding_dict.items():
        # Required to make sure node names in encoding provided is santized
        name = Helper.sanitize_node_names(name)
        if (name in output_name_to_elem_type.keys()
                and output_name_to_elem_type[name] not in ["INT64", "INT32", "UINT64", "UINT32"]):
            new_encodings[name] = enc
        elif (name not in output_name_to_elem_type.keys()):
            qacc_file_logger.warning(f"Did not find tensor info for: {name} ")
            new_encodings[name] = enc
        else:
            print("Skipping: ", name)
    return new_encodings


def cleanup_quantization_overrides(quant_overrides_json_path: str, model_path: str,
                                   outpath: str) -> str:
    """
    Cleans up quantization_overrides json based on the input model supplied and saves it to outpath location.
    Using the given model, quantization_overrides encodings are filtered to retain only the valid nodes.
    Note: Clean up is applicable only for ONNX Models.
    For other model types, we return the same quantization_overrides json without any changes
    params:
    quant_overrides_json_path: Absolute path to the quantization_overrides json file
    model_path: Absolute path to the model file.
    outpath: Path to store the cleaned up json file.
    """
    onnx = Helper.safe_import_package("onnx")
    if Helper.get_model_type(model_path) == ModelType.ONNX:
        # Cleanup performed only for ONNX Models
        with open(quant_overrides_json_path, "r") as stream:
            try:
                raw_json_dict = json.load(stream)
                activation_encodings = raw_json_dict["activation_encodings"]
                param_encodings = raw_json_dict["param_encodings"]
            except Exception as e:
                qacc_file_logger.error(f"Error parsing quantization_overrides json. Reason: {e}")
        #Load Model
        original_model = onnx.load(model_path)
        inferred_model = onnx.shape_inference.infer_shapes(original_model)

        # dict to store node_name and node_type mappings
        output_name_to_elem_type = {}
        for elem in inferred_model.graph.value_info:
            # Node Name Sanitization logic applied.
            output_name = Helper.sanitize_node_names(elem.name)
            elem_type = onnx.TensorProto.DataType.Name(elem.type.tensor_type.elem_type)
            output_name_to_elem_type[output_name] = elem_type

        new_activation_encodings = process_encoding(activation_encodings, output_name_to_elem_type)
        new_param_encodings = process_encoding(param_encodings, output_name_to_elem_type)

        cleaned_json_dict = {}
        cleaned_json_dict["activation_encodings"] = new_activation_encodings
        cleaned_json_dict["param_encodings"] = new_param_encodings

        # Write the cleaned json into outpath file
        with open(f"{outpath}", "w") as out_file:
            json.dump(cleaned_json_dict, out_file, indent=4)
        qacc_file_logger.info(f"Cleaned Quantization Overrides JSON dumped at: {outpath}")
    else:
        qacc_file_logger.warning("Quantization Overrides JSON are cleaned only for ONNX Models")
        outpath = quant_overrides_json_path
    return outpath
