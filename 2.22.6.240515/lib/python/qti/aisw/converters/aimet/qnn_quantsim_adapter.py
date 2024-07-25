# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================
import json
import logging
import os
import shutil
import re
import sys
import tempfile
import traceback
import itertools
from typing import Callable, List, Tuple, Union
from tqdm import tqdm

from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrGraph = ir_graph_lib.IrGraph
logger = logging.getLogger('AIMET Quantizer')

try:
    import numpy as np
    import torch
    from aimet_common.defs import QuantizationDataType, QuantScheme
    from aimet_common.quantsim_config.config_utils import \
        get_path_for_target_config
    from aimet_torch import utils
    from aimet_torch.cross_layer_equalization import equalize_model
    from aimet_torch.pro.model_preparer import prepare_model_from_ir_graph
    from aimet_torch.pro.quantsim import QuantizationSimModel
    from aimet_torch.pro.quant_sim.dlc_quantsim_exporter import DlcQuantSimExporter
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as e:
    traceback.print_exc()
    logger.error("Unable to import required modules to run AIMET Quantizer. "
                 "'--use_aimet_quantizer' option can't be used")
    sys.exit()


# Mapping from QNN calibration to AIMET QuantScheme
QNN_TO_AIMET_QUANT_SCHEME_MAP = {
    "tf": QuantScheme.post_training_tf,
    "min-max": QuantScheme.post_training_tf,
    "enhanced": QuantScheme.post_training_tf_enhanced,
    "sqnr": QuantScheme.post_training_tf_enhanced,
    "percentile": QuantScheme.post_training_percentile
}

def _modify_quantizers_if_necessary(sim: QuantizationSimModel,
                                    param_quant_scheme: Union[str, QuantScheme],
                                    act_quant_scheme: Union[str, QuantScheme],
                                    percentile_calibration_value: float,
                                    is_unsigned_symmetric_param: bool,
                                    is_unsigned_symmetric_act: bool):
    """
    :param sim: Quantsim Model object
    :param param_quant_scheme: quant scheme to be set for all param quantizers in provided Quantsim model
    :param act_quant_scheme: quant scheme to be set for all activation quantizers in provided Quantsim model
    :return: Modified quantsim model
    """

    # Get all quantizers from QSim Model
    param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

    # Set input/output quantizers' quant schemes
    for quantizer in itertools.chain(input_quantizers, output_quantizers):
        quantizer.quant_scheme = act_quant_scheme
        if is_unsigned_symmetric_act:
            # Set the following for unsigned_symmetric
            quantizer.use_unsigned_symmetric = True
            quantizer.use_symmetric_encodings = True
        if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                percentile_calibration_value is not None:
            quantizer.set_percentile_value(percentile_calibration_value)

    # Set param quantizers' quant schemes
    for quantizer in param_quantizers:
        quantizer.quant_scheme = param_quant_scheme
        if is_unsigned_symmetric_param:
            # Set the following for unsigned_symmetric
            quantizer.use_unsigned_symmetric = True
            quantizer.use_symmetric_encodings = True
        if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                percentile_calibration_value is not None:
            quantizer.set_percentile_value(percentile_calibration_value)

    return sim


def _replace_invalid_chars_for_variable(name: str) -> str:
    """
    Replace invalid chars such as dot, slash, ...
    :param name: name to replace invalid chars
    :return: cleansed string can be used Python variable name
    """
    if not name.isidentifier():
        found_list = re.findall(r"\w+", name)
        if found_list:
            name = "_".join(found_list)
        if name[0].isdigit():
            name = '_' + name
        if not name.isidentifier():
            error_str = f"Unable to produce a valid model name name from {name}"
            logger.error(error_str)
            raise RuntimeError(error_str)
    return name


class InputListDataset(Dataset):

    def __init__(self, input_list_path: str, input_shapes: dict, input_dtypes: dict, use_native_input_files: bool) -> None:
        super().__init__()
        self._input_list = open(input_list_path).readlines()
        self._input_shapes = input_shapes
        self._is_input_list_formatted = True
        self._use_native_input_files = use_native_input_files
        # Creates a list of ordered input file list for each input based on input ordering in Ir graph
        # This ordering is inferred from the input_shapes dict as dictionaries are ordered in python>=3.7
        self._ordered_input_list = list(map(self._order_input_files, self._input_list))

        self._input_dtypes = input_dtypes

    def __len__(self):
        return len(self._input_list)

    def _read_raw_data(self, file_name: str, dtype: str) -> np.ndarray:
        """ Read data from the .raw files into a numpy array """
        with open(file_name, "rb") as file:
            raw_data = file.read()
            if self._use_native_input_files:
                numpy_from_raw_data = np.frombuffer(raw_data, dtype=dtype)
            else:
                numpy_from_raw_data = np.frombuffer(raw_data, dtype=np.float32)
                numpy_from_raw_data = numpy_from_raw_data.astype(dtype, copy=False)
            return numpy_from_raw_data

    def _order_input_files(self, input_files):
        """ Order input files based on IR graph input name(s) """
        input_files = input_files.split() # Inputs separated by space
        is_formatted = [':=' in x for x in input_files]
        assert all(is_formatted) or not any(is_formatted), ("Input list is not well formatted")
        if all(is_formatted):
            input_files_dict = {y[0]:y[1] for y in [x.split(':=') for x in input_files]}
            input_files = [input_files_dict[input_name] for input_name in self._input_shapes.keys()]
        else:
            # Print warning message only once
            if len(input_files) > 1 and self._is_input_list_formatted:
                self._is_input_list_formatted = False
                logger.warning("Input list is not properly formatted, may result in errors. "
                               "Write input list with input_name appended before file path "
                               "for each input, input_name:=<filepath> ..")
        return input_files

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get the input tensor(s) at a given index of the input_list file
        Input files can be specified with the below format for three sets of inputs for two input layers

        Input_1:=Placeholder_1/real_input_inputs_1/0-0#67c965.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/1-1#54f1ff.rawtensor
        Input_1:=Placeholder_1/real_input_inputs_1/1-0#b42dc6.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/2-1#346a0e.rawtensor
        Input_1:=Placeholder_1/real_input_inputs_1/2-0#e6fb51.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/0-1#8a171b.rawtensor
        """
        ordered_input_files = self._ordered_input_list[index]

        tensors: List[torch.Tensor] = []
        for n, file_name in enumerate(ordered_input_files):
            tensor_name, tensor_dim = list(self._input_shapes.items())[n]
            tensor_dtype=self._input_dtypes[tensor_name]
            raw_data_numpy_array = self._read_raw_data(file_name, dtype=tensor_dtype)
            assert raw_data_numpy_array.shape[0] == np.prod(tensor_dim), (f'Could not reshape input tensor "{tensor_name}" '
                                                                          f'as required. Raw Numpy Data Shape: {raw_data_numpy_array.shape}; '
                                                                          f'Required tensor shape: {tensor_dim}')
            reshaped_numpy_array = raw_data_numpy_array.reshape(tensor_dim)
            tensor = torch.tensor(reshaped_numpy_array)
            tensors.append(tensor)

        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)


class QnnToAimetAdapter:

    def __init__(self, ir_graph, opts, datadir: str = "", use_cuda: bool = True):
        self.opts = opts
        # Get activation and param quantizer schemes
        self.param_quant_scheme, self.act_quant_scheme = self._get_act_param_quant_schemes()
        self._valid_opts = self._validate_opts()
        if not self._valid_opts:
            logger.warning("Invalid argument provided. '--use_aimet_quantizer' option can't be used")
        else:
            self._device = self._set_device(use_cuda)

            # Take input shapes from IR Graph
            self._input_shapes = {input_tensor.name():input_tensor.dims() for input_tensor in ir_graph.get_input_tensors_to_graph()}

            if not self._input_shapes:
                logger.error("Could not infer model input shapes from the model. Please specify --input_dim")
                self._valid_opts = False
            else:
                self._input_dtypes = self._infer_input_dtypes(ir_graph)
                if datadir:
                    self._datadir = datadir
                else:
                    self._temp_datadir = tempfile.TemporaryDirectory()
                    self._datadir = self._temp_datadir.name

                self._input_list = self._get_input_list_abs()
                self._input_list_dataset = InputListDataset(self._input_list, self._input_shapes, self._input_dtypes, self.opts.use_native_input_files)
                # While iterating through the dataloader, the tensors have an additional pseudo dimensions (added by Torch DataLoader class).
                # So, while using it, we squeeze the additional pseudo-dimension. For example, if the input_tensor has dimensions [1, 224, 224, 3],
                # while iterating through the dataloader, it will be [1, 1, 224, 224, 3]. So, we squeeze the pseudo-dimension by using input_tensor[0].
                self._input_list_dataloader = DataLoader(self._input_list_dataset, batch_size=1, shuffle=False)

                self._filename = 'input_model'

                if os.path.isfile(self.opts.input_network):
                    self._filename, _ = os.path.splitext(os.path.basename(self.opts.input_network))

                self._model_name = _replace_invalid_chars_for_variable(self._filename + "_model") # Same as filename +'_model' for simplicity
                if self.opts.output_path is not None:
                    self._output_name, _ = os.path.splitext(os.path.basename(self.opts.output_path))
                else:
                    self._output_name = self._filename
                keep_linear_without_bias = False
                ir_graph_output_names = [output_tensor.name() for output_tensor in ir_graph.get_output_tensors_of_graph()]
                # Prepare model
                self._prepared_model = prepare_model_from_ir_graph(ir_graph,
                                                                   self._datadir,
                                                                   self._filename,
                                                                   self._model_name,
                                                                   keep_linear_without_bias,
                                                                   ir_graph_output_names).to(self._device)
                self._converted_model_info_file = os.path.join(self._datadir, f"{self._filename}_prepared_model_info.pkl")
                if self.should_run_cle():
                    input_shapes = list(self._input_shapes.values())
                    equalize_model(self._prepared_model, input_shapes)

    def _set_device(self, use_cuda):
        """ Set device for Quantsim """
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if use_cuda and not torch.cuda.is_available():
            logger.warning("Using 'cpu' for Quantization by AIMET Quantizer, as torch is not compiled with CUDA")
        else:
            logger.info(f"Using '{device}' for Quantization by AIMET Quantizer")
        return device

    def is_valid_opts(self):
        """ Returns whether aimet quantizer can be used with the given arguments """
        return self._valid_opts

    def _get_input_list_abs(self):
        """ Get the absolute path for modified input list """
        with open(self.opts.input_list, "r") as file:
            input_0 = file.readline()

        path_example = input_0.split()[0]
        if ":=" in path_example:
            path_example = path_example.split(":=")[1]

        if os.path.exists(path_example):
            return self.opts.input_list

        else:
            # Re-write input paths
            new_lines = []
            input_file_dir = os.path.dirname(self.opts.input_list)
            with open(self.opts.input_list, "r") as file:
                lines = file.readlines()
                if lines[0][0] == '#' or lines[0][0] == '%':
                    self._output_names = lines[0][1:].split(' ')
                    lines = lines[1:]
                for line in lines:
                    new_line = []
                    for input_path in line.split():
                        if ":=" in input_path:
                            input_name, path = input_path.split(":=")
                            new_path = f"{input_name}:={os.path.join(input_file_dir, path)}"
                        else:
                            new_path = os.path.join(input_file_dir, line)
                        new_line.append(new_path)
                    new_line = " ".join(new_line)
                    new_lines.append(new_line)

            temp_input_list = os.path.join(self._datadir, 'temp_input_list.txt')
            with open(temp_input_list, 'w') as f:
                for line in new_lines:
                    f.write(line)

            return temp_input_list

    def _infer_input_dtypes(self, ir_graph: IrGraph):
        """ Infer the input dtypes of the model from IR graph """
        input_dtypes = {tensor.name(): tensor.data_type_as_numpy_type()
                        for tensor in ir_graph.get_input_tensors_to_graph()}

        return input_dtypes

    def _validate_opts(self) -> bool:
        """ Validate the command-line opts to check if aimet quantizer can be used """
        valid_opts = True
        if not self.opts.input_list:
            logger.error("'--input_list' not specified")
            valid_opts = False
        if self.param_quant_scheme not in QNN_TO_AIMET_QUANT_SCHEME_MAP.keys():
            param_quantizer_arg_key = "--param_quantizer" if not self.opts.disable_legacy_quant_scheme_opts else "--param_quantizer_calibration"
            logger.error(f"invalid value '{self.param_quant_scheme}' for {param_quantizer_arg_key}")
            valid_opts = False
        if self.act_quant_scheme not in QNN_TO_AIMET_QUANT_SCHEME_MAP.keys():
            act_quantizer_arg_key = "--act_quantizer" if not self.opts.disable_legacy_quant_scheme_opts else "--act_quantizer_calibration"
            logger.error(f"invalid value '{self.act_quant_scheme}' for {act_quantizer_arg_key}")
            valid_opts = False
        return valid_opts

    def _get_act_param_quant_schemes(self):
        if self.opts.disable_legacy_quant_scheme_opts:
            param_quant_scheme = self.opts.quant_schemes['param_quant']["calibration"]
            act_quant_scheme = self.opts.quant_schemes['act_quant']["calibration"]
        else:
            param_quant_scheme = self.opts.quant_schemes['param_quant']
            act_quant_scheme = self.opts.quant_schemes['act_quant']
        return param_quant_scheme, act_quant_scheme

    def _get_config_file(self) -> str:
        """ Get path to config file """
        # TODO: Add backend awareness
        config_file = get_path_for_target_config("htp_quantsim_config_v75")
        config_dict = json.load(open(config_file))

        if self.opts.use_per_channel_quantization:
            logger.info(f'Per Channel Quantization is enabled')
        if self.opts.use_per_row_quantization:
            logger.info(f'Per Row Quantizaton is enabled')

        def add_to_config(op_type, flag):
            if op_type in config_dict["op_type"].keys():
                config_dict["op_type"][op_type]["per_channel_quantization"] = flag
            else:
                config_dict["op_type"][op_type] = {"per_channel_quantization": flag}

        config_dict["defaults"]["per_channel_quantization"] = str(self.opts.use_per_channel_quantization)
        # per_row_ops = ["Gemm", "MatMul"]
        add_to_config("Gemm", str(self.opts.use_per_row_quantization))
        add_to_config("MatMul", str(self.opts.use_per_row_quantization))

        quantizer_type_to_config_key = {
            'act_quant': "ops",
            'param_quant': "params"
        }

        if self.opts.disable_legacy_quant_scheme_opts:
            for quantizer in ['act_quant', 'param_quant']:
                if self.opts.quant_schemes[quantizer]['schema'] in ["asymmetric"]:
                    config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "False"
                elif self.opts.quant_schemes[quantizer]['schema'] in ["symmetric"]:
                    config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "True"

        temp_config_file = tempfile.NamedTemporaryFile(delete=False)

        with open(temp_config_file.name, "w") as file:
            json.dump(config_dict, file)

        return temp_config_file.name

    def _get_data_dir(self) -> str:
        """ Returns the path to the directory storing converter artifacts """
        return self._datadir

    def get_sample_data(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get n random samples from the dataset
        :return: return random samples or  entire dataset if input list provided
        """
        if self.opts.input_list:
            sample = next(iter(self._input_list_dataloader))
            if isinstance(sample, (tuple, list)):
                sample = tuple([tensor[0].to(self._device) for tensor in sample])
                logger.info(f'Sample input data shape, {[x.shape for x in sample]}')
            else:
                sample = sample[0]
                sample = sample.to(self._device)
                logger.info(f'Sample input data shape, {sample.shape}')
            return sample

        else:
            dummy_input = tuple(
                torch.randn(shape).to(self._device)
                for shape in self._input_shapes.keys()
            )
            if len(dummy_input) == 1:
                dummy_input = dummy_input[0]
            logger.info(f'Sample input data shape, {[x.shape for x in dummy_input]}')

            return dummy_input

    def get_input_shape(self) -> dict:
        """
        Get the shape of the input tensor to the model
        """
        return self._input_shapes

    def should_run_cle(self) -> bool:
        """
        Returns true if cle should be run on the model before quantization
        """
        return "cle" in self.opts.algorithms

    def _get_quant_scheme(self) -> QuantScheme:
        """
        Get the quantization scheme from qnn arguments
        """
        if self.param_quant_scheme != self.act_quant_scheme:
            logger.warning("Quantization schemes for parameter quantizers and activations quantizers are different")
            logger.info(f"Using '{self.param_quant_scheme}' as quantization scheme for params, "
                        f"'{self.act_quant_scheme}' as quantization scheme for activations")
            # Set default Quantization scheme to "tf", and modify after QuantSim instantiation
            quant_scheme = "tf"
        else:
            quant_scheme = self.param_quant_scheme
            logger.info(f"Using '{quant_scheme}' quantization scheme")
        return QNN_TO_AIMET_QUANT_SCHEME_MAP[quant_scheme]

    def get_prepare_model(self):
        return self._prepared_model

    def _get_quantsim_args(self) -> dict:
        """
        Get the arguments to quantsim as kwargs dict
        """
        quantsim_args_dict = {
            "model": self._prepared_model,
            "dummy_input": self.get_sample_data(),
            "quant_scheme": self._get_quant_scheme(),
            "rounding_mode": "nearest",
            "default_output_bw": self.opts.act_bitwidth,
            "default_param_bw": self.opts.weights_bitwidth,
            "in_place": True,
            "config_file": self._get_config_file(),
            "default_data_type": QuantizationDataType.int,
        }
        return quantsim_args_dict

    def get_compute_encodings_calibration_cb(self) -> Callable:
        """Get the calibration callback needed for computing encodings"""

        def pass_calibration_data(sim_model, *args):

            sim_model.to(self._device)
            sim_model.eval()

            with torch.no_grad():
                for input_data in tqdm(self._input_list_dataloader):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data[0].to(self._device)
                        sim_model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d[0].to(self._device), input_data))
                        sim_model(*input_batch)

        return pass_calibration_data

    def generate_quantized_dlc(self, converted_dlc_path, for_qairt) -> IrGraph:
        """
        Return the DLC reader object for the converted, quantized IR graph
        The DLC reader needs to persist to prevent garbage collection of IRGraph static tensor data
        """
        quantsim_args = self._get_quantsim_args()

        # Initialize Quantsim
        sim = QuantizationSimModel(**quantsim_args)

        # Modify Quantsim when Quant schemes are different for Params and Activations
        if self.param_quant_scheme != self.act_quant_scheme:
            # Set Quant schemes for act quantizers and param quantizers when they are not equal
            is_unisgned_symmetric_param=self.opts.quant_schemes['param_quant']['schema']=="unsignedsymmetric"
            is_unsigned_symmetric_act=self.opts.quant_schemes['act_quant']['schema']=="unsignedsymmetric"
            sim = _modify_quantizers_if_necessary(sim=sim,
                                                  param_quant_scheme=QNN_TO_AIMET_QUANT_SCHEME_MAP[self.param_quant_scheme],
                                                  act_quant_scheme=QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme],
                                                  percentile_calibration_value=self.opts.percentile_calibration_value,
                                                  is_unsigned_symmetric_param=is_unisgned_symmetric_param,
                                                  is_unsigned_symmetric_act=is_unsigned_symmetric_act)

        # Check for any quantization overrides
        mpp_aligned_torch_enc_file = os.path.join(self._datadir, self._filename+'_torch.encoding')
        if not self.opts.ignore_encodings:
            if os.path.exists(mpp_aligned_torch_enc_file):
                logger.info('Quantization overrides provided, AIMET will compute any missing encodings')
                sim.load_and_freeze_encodings(mpp_aligned_torch_enc_file, ignore_when_quantizer_disabled=True)
            else:
                logger.info('No quantization overrides provided, AIMET will compute the full encodings')
        else:
            logger.info('--ignore_encodings flag is provided, AIMET will ignore any encodings provided')

        # Compute Encodings
        sim.compute_encodings(self.get_compute_encodings_calibration_cb(), None)

        # Export to DLC
        DlcQuantSimExporter.export(sim, self._datadir, self._output_name + '_quantized', converted_dlc_path,
                                   self._converted_model_info_file, quantize_dlc=True)

        # Post-processing
        quantized_dlc_path = os.path.join(self._datadir, f"{self._output_name}_quantized.dlc")
        if for_qairt:
            shutil.move(quantized_dlc_path, self.opts.output_path)
            quantized_dlc_path = self.opts.output_path
        logger.info('Quantization using AIMET Quantizer is done!')

        return quantized_dlc_path