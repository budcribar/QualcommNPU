# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================
import sys
import multiprocessing as mp
import os
import tempfile
import traceback
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning, log_error


dlc_import_error = False
try:
    from qti.aisw.converters.common import modeltools
except ImportError:
    try:
        # May need this import for QAIRT
        from qti.aisw.dlc_utils import modeltools
    except ImportError:
        dlc_import_error = True
        log_info("Unable to import DLC utilties")

from qti.aisw.converters.common import ir_graph as ir_graph_lib
IrGraph = ir_graph_lib.IrGraph


def setup_environment():
    """
    Set up the environment to run AIMET-specific code by modifying site packages
    and python path list to place virtual environment's site packages at the beginning
    """
    import site
    import sys

    base_dir = os.path.dirname(os.path.dirname(sys.executable))
    log_info(f"Using virtual environment at {base_dir} to run AIMET quantizer")
    if sys.platform == "win32":
        site_packages = os.path.join(base_dir, "Lib", "site-packages")
    else:
        major_v = sys.version_info.major
        minor_v = sys.version_info.minor
        site_packages = os.path.join(base_dir, "lib", f"python{major_v}.{minor_v}", "site-packages")

    site.addsitedir(site_packages)

    # Order the python path so that the site-package of the virtual environment is at the beginning
    # of the python path list. This prioritizes the version of a package in the virtual environment
    sys.path = [path for path in sys.path if base_dir in path] + \
               [path for path in sys.path if base_dir not in path]


def serialize_ir_graph_to_dlc(ir_graph, path: str, filename: str):
    """
    Serialize IrGraph to dlc and save it to the specified path with the provided filename
    :param ir_graph: IrGraph to be serialized
    :param path: Path to save exported dlc model
    :param filename: filename to save exported dlc model
    """
    if not dlc_import_error:
        _serialize_ir_graph_to_dlc(ir_graph, path, filename)
    else:
        raise Exception('Unable to serialize IR graph to DLC')


def _serialize_ir_graph_to_dlc(ir_graph, path: str, filename: str):
    """
    Serialize IrGraph to dlc and save it to the specified path with the provided filename
    :param ir_graph: IrGraph to be serialized
    :param path: Path to save exported dlc model
    :param filename: filename to save exported dlc model
    """
    dlc_serializer = modeltools.IrDlcSerializer(os.path.join(path, filename + ".dlc"))
    dlc_serializer.initialize()
    dlc_serializer.serialize(ir_graph)
    dlc_serializer.finish()


def get_ir_graph_from_dlc(dlc_path: str):
    """
    Obtain IR Graph from DLC (non quantized).
    :param dlc_path: Path where dlc is located
    """
    if not dlc_import_error:
        dlc_reader_obj = _get_dlc_reader(dlc_path)
        ir_graph = dlc_reader_obj.get_ir_graph()
        return ir_graph, dlc_reader_obj
    else:
        raise Exception('Unable to obtain IR graph from dlc as relevant utils are not imported')


def get_dlc_reader(dlc_path: str):
    if not dlc_import_error:
        dlc_reader_obj = _get_dlc_reader(dlc_path)
        return dlc_reader_obj
    else:
        raise Exception('Unable to obtain IR graph from dlc as relevant utils are not imported')



def _get_dlc_reader(dlc_path: str):
    """
    Obtain IR Graph from DLC (non quantized).
    :param dlc_path: Path where dlc is located
    """
    dlc_reader = modeltools.IrDlcReader()
    dlc_reader.open(dlc_path)
    return dlc_reader


def get_python_executable():
    aimet_env_python_exec = os.environ.get("AIMET_ENV_PYTHON")
    error_msg = ('Provided python executable at $AIMET_ENV_PYTHON is invalid. Please run '
                 'aimet_env_setup.sh to ensure AIMET_ENV_PYTHON is set to <aimet_venv>/lib/python')
    if os.path.exists(aimet_env_python_exec):
        # This returns python version, must contain 'python' in version string,
        # if it is a valid python interpreter
        try:
            python_version = os.popen(f'{aimet_env_python_exec} --version').read().strip()
            assert 'python' in python_version.lower(), error_msg
            log_info('Validated environment variable, AIMET_ENV_PYTHON')
            return aimet_env_python_exec
        except Exception:
            raise EnvironmentError(error_msg)
    else:
        raise EnvironmentError(error_msg)


def quantize_model_with_aimet(dlc_path, conn, tmpdir, opts, for_qairt):
    """
    Call this function within a subprocess to execute aimet specific code in a separate virtual environment
    """
    # Set up the virtual environment before importing any packages from aimet-specific modules
    setup_environment()

    quantized_dlc_path = None
    try:
        # Import this only after adding the virtual environment's site package to python path
        from qti.aisw.converters.aimet.qnn_quantsim_adapter import QnnToAimetAdapter
        ir_graph, dlc_reader = get_ir_graph_from_dlc(dlc_path)
        qnn_adapter = QnnToAimetAdapter(ir_graph, opts, datadir=tmpdir, use_cuda=True)
        if qnn_adapter.is_valid_opts():
            # Persist DLC reader to avoid DLC shared memory being freed. Read IR graph from DLC reader
            quantized_dlc_path = qnn_adapter.generate_quantized_dlc(dlc_path, for_qairt)
    except Exception as e:
        traceback.print_exc()
    finally:
        conn.send([quantized_dlc_path,])
        conn.close()


def aimet_quantizer(ir_graph, opts):
    aimet_env_python_exec = get_python_executable()
    if not aimet_env_python_exec:
        raise EnvironmentError(
            """Environment variable 'AIMET_ENV_PYTHON' not set.
            Please run  'source $QNN_SRC/QTI/scripts/aimet_env_setup.sh --env-path <PATH>' if you want to use aimet quantizer
            or omit the '--use_aimet_quantizer' flag to use the default quantizer"""
        )
    # Create a multiprocessing context with start method 'spawn' and set the python executable path
    with tempfile.TemporaryDirectory() as tmpdir:
        unquantized_dlc_filename = 'model_fp'
        unquantized_dlc_path = tmpdir
        serialize_ir_graph_to_dlc(ir_graph, unquantized_dlc_path, unquantized_dlc_filename)
        ctx = mp.get_context("spawn")
        ctx.set_executable(aimet_env_python_exec)
        # Create a process and run aimet-specific code within the context of that process
        parent_conn, child_conn = mp.Pipe()
        fullpath = os.path.join(unquantized_dlc_path, unquantized_dlc_filename + '.dlc')
        process = ctx.Process(target=quantize_model_with_aimet, args=(fullpath , child_conn, tmpdir, opts, False))
        process.start()
        retval = parent_conn.recv()
        quantized_dlc_path = retval[0]
        process.join()
        process.terminate()
        if quantized_dlc_path is not None and os.path.exists(quantized_dlc_path):
            reader = get_dlc_reader(quantized_dlc_path)
            return reader
        else:
            log_error('Exception occured in Spawned AIMET Process, Unable to proceed with Quantization')
            sys.exit()


class AimetQuantizerOpts:
    def __init__(self,
                 input_network,
                 output_path,
                 input_list,
                 quant_schemes,
                 disable_legacy_quant_scheme_opts,
                 algorithms,
                 act_bitwidth,
                 weights_bitwidth,
                 bias_bitwidth,
                 percentile_calibration_value,
                 ignore_encodings,
                 use_per_channel_quantization,
                 use_per_row_quantization,
                 use_native_input_files,
                 use_native_output_files):
        self.input_network = input_network
        self.output_path = output_path
        self.input_list = input_list
        self.quant_schemes = quant_schemes
        # Flag to detect whether to use --act_quantizer and --param_quantizer (or) [--act_quantizer_calibration, --act_quantizer_schema]
        # [--param_quantizer_calibration, --param_quantizer_schema] to resolve AIMET Quant Scheme
        self.disable_legacy_quant_scheme_opts = disable_legacy_quant_scheme_opts
        self.algorithms = algorithms
        self.act_bitwidth = act_bitwidth
        self.weights_bitwidth = weights_bitwidth
        self.bias_bitwidth = bias_bitwidth
        self.percentile_calibration_value = percentile_calibration_value
        self.ignore_encodings = ignore_encodings
        self.use_per_channel_quantization = use_per_channel_quantization
        self.use_per_row_quantization = use_per_row_quantization
        self.use_native_input_files = use_native_input_files
        self.use_native_output_files = use_native_output_files
        self.validate_aimet_quant_opts()

    def validate_aimet_quant_opts(self):
        # TODO: Support --use_native_output_files if required
        if self.use_native_output_files:
            raise Exception("AIMET Quantizer doesn't support --use_native_output_files")
        # TODO: Support --bias_bitwidth 8
        if self.bias_bitwidth != 32:
            # TODO: raise Exception once the default is changed to 32
            log_warning(f"AIMET Quantizer doesn't support {self.bias_bitwidth} for --bias_bitwidth or --bias_bw, using 32")
            self.bias_bitwidth = 32


def aimet_dlc_quantizer(opts):
    aimet_env_python_exec = get_python_executable()
    if not aimet_env_python_exec:
        raise EnvironmentError(
            """Environment variable 'AIMET_ENV_PYTHON' not set.
            Please run  'source $QNN_SRC/QTI/scripts/aimet_env_setup.sh --env-path <PATH>' if you want to use aimet quantizer
            or omit the '--use_aimet_quantizer' flag to use the default quantizer"""
        )
    # Create a multiprocessing context with start method 'spawn' and set the python executable path
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context("spawn")
        ctx.set_executable(aimet_env_python_exec)
        # Create a process and run aimet-specific code within the context of that process
        parent_conn, child_conn = mp.Pipe()
        process = ctx.Process(target=quantize_model_with_aimet, args=(opts.input_network, child_conn, tmpdir, opts, True))
        process.start()
        retval = parent_conn.recv()
        quantized_dlc_path = retval[0]
        process.join()
        process.terminate()
        if quantized_dlc_path is None or not os.path.exists(quantized_dlc_path):
            log_error('Exception occured in Spawned AIMET Process, Unable to proceed with Quantization')
            sys.exit()