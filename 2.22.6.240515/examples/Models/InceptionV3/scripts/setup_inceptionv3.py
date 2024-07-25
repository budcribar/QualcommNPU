#
# Copyright (c) 2017-2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run inception_v3 model with QNN SDK.
'''

from dataclasses import dataclass, field
import os
import subprocess
import shutil
import hashlib
import argparse
import sys
from pathlib import Path
import platform
import requests
import tarfile

INCEPTION_V3_ARCHIVE_CHECKSUM = 'a904ddf15593d03c7dd786d552e22d73'
INCEPTION_V3_ARCHIVE_FILE = 'inception_v3_2016_08_28_frozen.pb.tar.gz'
INCEPTION_V3_ARCHIVE_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/' + INCEPTION_V3_ARCHIVE_FILE
INCEPTION_V3_PB_FILENAME = 'inception_v3_2016_08_28_frozen.pb'
INCEPTION_V3_PB_OPT_FILENAME = 'inception_v3_2016_08_28_frozen_opt.pb'
INCEPTION_V3_LBL_FILENAME = 'imagenet_slim_labels.txt'
OPT_4_INFERENCE_SCRIPT = 'optimize_for_inference.py'
RAW_LIST_FILE = 'raw_list.txt'
TARGET_RAW_LIST_FILE = 'target_raw_list.txt'


@dataclass
class OpPackage:
    name: str = field(init=False)
    src_file: str = field(init=False)
    xml_file_dict: dict = field(init=False)
    package_name: str = field(init=False)
    symbol: str = field(init=False)
    lib: str = field(init=False)


@dataclass
class RELU(OpPackage):
    def __post_init__(self):
        self.name = self.__class__.__name__
        self.src_file = f"{self.name.capitalize()}.cpp"
        self.package_name = f"{self.name.capitalize()}OpPackage"
        self.xml_file_dict = {b: f"{self.package_name}{b.__name__.capitalize()}.xml" for b in Backend.__subclasses__()}
        self.symbol = "ReluOpPackageInterfaceProvider"
        # FIXME: determine by platform once OpPackage is supported on Windows
        self.lib = "libReluOpPackage.so"


class SDK:
    ROOT = Path()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(WindowsSDK if platform.system() == "Windows" else LinuxSDK)
        return cls._instance

    def __init__(self):
        self.is_windows = isinstance(self, WindowsSDK)
        self.op_package_gen_dir = SDK.ROOT / "examples" / "QNN" / "OpPackageGenerator"
        self.op_package_dir = self.op_package_gen_dir / "generated"
        self.model_dir = SDK.ROOT / "examples" / "Models" / "InceptionV3"
        self.op_package_out_dir = self.model_dir / "InceptionV3OpPackage"

    def host_arch(self):
        raise NotImplementedError

    def model_lib_gen_target_arch(self):
        raise NotImplementedError

    def bin(self):
        return self.__class__.ROOT / 'bin' / self.host_arch()


class WindowsSDK(SDK):
    def host_arch(self):
        return "x86_64-windows-msvc"

    def model_lib_gen_target_arch(self):
        return ('windows-x86_64', 'windows-aarch64')


class LinuxSDK(SDK):
    def host_arch(self):
        return "x86_64-linux-clang"

    def model_lib_gen_target_arch(self):
        return (self.host_arch(), "aarch64-android")


class Backend:
    _instances = dict()
    _sdk = SDK()

    def __new__(cls):
        if cls == Backend:
            raise ValueError("Cannot construct instance of base class Backend")
        if cls not in Backend._instances:
            Backend._instances[cls] = super().__new__(cls)
        return Backend._instances[cls]

    def __init__(self):
        _cls = self.__class__
        self._name = self.__class__.__name__
        self.op_package_dir = _cls._sdk.op_package_dir / self._name
        self.op_package_out_dir = _cls._sdk.op_package_out_dir / self._name
        if not self.op_package_dir.is_dir():
            raise FileNotFoundError(f"{self._name} example package directory does not exist: {self.op_package_dir}")

    def __call__(self, op: OpPackage):
        op_src = self.op_package_dir / op.src_file
        if not op_src.is_file():
            raise FileNotFoundError(f"Cannot retrieve {self._name} example source code needed for compilation"
                                    f", does {op_src} exist ?")
        op_config = self.__class__._sdk.op_package_gen_dir / op.xml_file_dict[self.__class__]
        if not op_config.is_file():
            raise FileNotFoundError(op_config)
        op_package_path = self.op_package_out_dir / op.package_name
        print(f"INFO: Generating {op.name} package for {self._name}")
        cmd = [self.__class__._sdk.bin() / "qnn-op-package-generator",
               "-p", op_config,
               "-o", self.op_package_out_dir]
        subprocess.check_call(cmd)

        self._copy_essential_files(op)
        print(f"INFO: Compiling {op.name} Op Package for {self._name} targets")
        self._compile("cpu" if self._name == "CPU" else "all", cwd=op_package_path)
        print(f"INFO: {self._name} Op Package compilation done")

    def _copy_essential_files(self, op: OpPackage):
        print(f"INFO: Replacing skeleton {self._name} op package "
              f"source code: {op.src_file} with completed example")
        shutil.copy(self.op_package_dir / op.src_file, 
                    self.op_package_out_dir / op.package_name / "src" / "ops")

    def _compile(self, tgt, cwd):
        subprocess.check_call(["make", tgt],
                              cwd=cwd,
                              stdout=subprocess.DEVNULL)

    def _check_hexagon_sdk(self, ver: str = "4.2.0"):
        if 'HEXAGON_SDK_ROOT' not in os.environ:
            raise RuntimeError('HEXAGON_SDK_ROOT is not set. Please see HTP/DSP example README for details on hexagon sdk setup')
        sdk_root = os.environ.get('HEXAGON_SDK_ROOT')
        if not sdk_root or not ver in sdk_root:
            raise RuntimeError('Hexagon sdk root is set to the wrong version. '
                            'Expected: hexagon-sdk-{}, instead got {}'.format(ver, sdk_root))


# TODO: Add generation capability for GPU
class CPU(Backend):
    pass


class DSP(Backend):
    def __init__(self):
        super().__init__()
        self.op_header = "DspOps.hpp"
        self._check_hexagon_sdk("3.5.2")

    def _copy_essential_files(self, op: OpPackage):
        super()._copy_essential_files(op)
        example_include = self.op_package_dir / self.op_header
        if not example_include.is_file():
            raise FileNotFoundError('Cannot retrieve DSP example include file needed for compilation')
        print(f"INFO: Replacing skeleton DSP op package include: {self.op_header} with completed example")
        shutil.copy(example_include, self.op_package_out_dir / op.package_name / "include")


class HTP(Backend):
    def __init__(self):
        super().__init__()
        self._check_hexagon_sdk()


def wget(download_dir, file_url):
    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)
    dst_file = download_dir.joinpath(os.path.basename(file_url))
    req = requests.get(file_url, verify=False)
    dst_file.write_bytes(req.content)


def generateMd5(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as data_file:
        while True:
            block = data_file.read(checksum.block_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()


def checkResource(inception_v3_data_dir, filename, md5):
    filepath = os.path.join(inception_v3_data_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(filename + ' not found at the location specified by ' \
                           + inception_v3_data_dir + '. Re-run with download option.')
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + \
                               ' does not match checksum of file ' + md5)


def find_optimize_for_inference():
    tensorflow_root = os.path.abspath(os.environ['TENSORFLOW_HOME'])
    for root, dirs, files in os.walk(tensorflow_root):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)


def optimize_for_inference(tensorflow_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    pb_filename = ""

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + \
              " script. Skipping inference optimization.\n")
        pb_filename = INCEPTION_V3_PB_FILENAME
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        cmd = [sys.executable, opt_4_inference_file,
               '--input', tensorflow_dir / INCEPTION_V3_PB_FILENAME,
               '--output', tensorflow_dir / INCEPTION_V3_PB_OPT_FILENAME,
               '--input_names', 'input',
               '--output_names', 'InceptionV3/Predictions/Reshape_1']
        subprocess.call(cmd, stdout=subprocess.DEVNULL)
        pb_filename = INCEPTION_V3_PB_OPT_FILENAME

    return pb_filename


def prepare_data_images(model_dir, tensorflow_dir):
    data_dir, scripts_dir = map(model_dir.joinpath, ("data", "scripts"))
    data_cropped_dir = data_dir / 'cropped'
    data_cropped_dir.mkdir(exist_ok=True)

    # copy the labels file to the data directory
    src_label_file = tensorflow_dir / INCEPTION_V3_LBL_FILENAME
    shutil.copy(src_label_file, data_dir)

    print('INFO: Creating QNN inception_v3 raw data')
    create_raws_script = scripts_dir / 'create_inceptionv3_raws.py'
    cmd = [sys.executable, create_raws_script,
           '-i', data_dir,
           '-d', data_cropped_dir]
    subprocess.call(cmd)

    print('INFO: Creating image list data files')
    create_file_list_script = scripts_dir / 'create_file_list.py'
    cmd = [sys.executable, create_file_list_script,
           '-i', data_cropped_dir,
           '-o', data_cropped_dir / RAW_LIST_FILE,
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = [sys.executable, create_file_list_script,
           '-i', data_cropped_dir,
           '-o', data_dir / TARGET_RAW_LIST_FILE,
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)


def generate_compile_packages(backend="CPU"):
    sdk = SDK()
    if not sdk.op_package_dir.is_dir():
        raise RuntimeError('{} does not exist.  Your SDK may be faulty.'.format(sdk.op_package_dir))
    sdk.op_package_out_dir.mkdir(exist_ok=True)
    try:
        be = next((be_cls for be_cls in Backend.__subclasses__() if be_cls.__name__ == backend.upper()))()
    except StopIteration:
        print(f"Unsupported backend type: {backend} specified for package generation")
        raise
    be(RELU())


def convert_model(pb_filename, model_dir, tensorflow_dir, quantize, custom):
    print('INFO: Converting ' + pb_filename + ' to QNN API calls')
    model = model_dir / "model" / ("Inception_v3_quantized" if quantize else "Inception_v3")
    model_cpp, model_bin = map(model.with_suffix, (".cpp", ".bin"))
    if not model_cpp.parent.is_dir():
        model_cpp.parent.mkdir()
    sdk = SDK()
    cmd = [sys.executable, sdk.bin() / 'qnn-tensorflow-converter',
           '--input_network', tensorflow_dir / pb_filename,
           '--input_dim', 'input', '1,299,299,3',
           '--out_node', 'InceptionV3/Predictions/Reshape_1',
           '--output_path', model_cpp]
    if quantize:
        cmd.append('--input_list')
        cmd.append(model_dir / "data" / "cropped" / "raw_list.txt")

    if custom:
        op, cpu_be = RELU(), CPU()
        op_config = op.xml_file_dict[cpu_be.__class__]
        print(f'INFO: Using custom op config: {op_config}')
        cmd.append('--op_package_config')
        cmd.append(sdk.op_package_gen_dir / op_config)

        if quantize:
            # sanity check op package lib path
            cpu_op_lib = cpu_be.op_package_out_dir / op.package_name / "libs" / sdk.host_arch() / op.lib
            if not cpu_op_lib.is_file():
                raise RuntimeError('Could not retrieve op package library: {}'.format(cpu_op_lib))
            print('INFO: Using custom op package library: {} for quantization'.format(cpu_op_lib))
            cmd.append('--op_package_lib')
            cmd.append(':'.join((cpu_op_lib.as_posix(), op.symbol)))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

    model_libs = model_dir / "model_libs"
    print('INFO: Compiling model artifacts into shared libraries at: {}'.format(model_libs))
    # qnn-model-lib-generator for linux is a bash script
    # and can be executed w/o specifying interpreter due to SHEBANG
    cmd = [sys.executable] if sdk.is_windows else []
    cmd += [sdk.bin() / 'qnn-model-lib-generator',
               '-c', model_cpp,
               '-b', model_bin,
               '-o', model_libs, '-t']
    if sdk.is_windows:
        cmd += sdk.model_lib_gen_target_arch()
    else:
        cmd.append(" ".join(sdk.model_lib_gen_target_arch()))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)  # only print errors or warnings


def setup_assets(inception_v3_data_dir, download, convert, quantize, custom, generate):
    if 'QNN_SDK_ROOT' not in os.environ:
        raise RuntimeError('QNN_SDK_ROOT not setup.  Please run the SDK env setup script.')
    SDK.ROOT = Path(os.environ["QNN_SDK_ROOT"])
    if not SDK.ROOT.is_dir():
        raise FileNotFoundError('QNN_SDK_ROOT (%s) is not a dir' % SDK.ROOT)

    sdk = SDK()
    if not sdk.model_dir.is_dir():
        raise RuntimeError('{} does not exist.  Your SDK may be faulty.'.format(sdk.model_dir))
    if convert:
        if not sdk.is_windows and 'ANDROID_NDK_ROOT' not in os.environ:
            raise RuntimeError('ANDROID_NDK_ROOT not setup.  Please run the SDK env setup script.')

    if generate is not None:
        generate_compile_packages(backend=generate)
    elif quantize and custom:
        print('INFO: Package generation is not enabled but CPU package will be generated due to quantize option being set')
        generate_compile_packages()

    if download:
        url_path = INCEPTION_V3_ARCHIVE_URL
        print("INFO: Downloading inception_v3 TensorFlow model...")
        wget(inception_v3_data_dir, url_path)

    try:
        checkResource(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE, INCEPTION_V3_ARCHIVE_CHECKSUM)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        sys.exit(0)

    print('INFO: Extracting TensorFlow model')
    tensorflow_dir = sdk.model_dir / 'tensorflow'
    if not tensorflow_dir.is_dir():
        tensorflow_dir.mkdir()
    with tarfile.open(os.path.join(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE)) as ivaf:
        ivaf.extractall(path=tensorflow_dir)

    pb_filename = optimize_for_inference(tensorflow_dir)

    prepare_data_images(sdk.model_dir, tensorflow_dir)

    if convert:
        convert_model(pb_filename, sdk.model_dir, tensorflow_dir, quantize, custom)

    print('INFO: Setup inception_v3 completed.')


def getArgs():
    parser = argparse.ArgumentParser(
        prog=str(__file__),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepares the inception_v3 assets for tutorial examples.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                          help='directory containing the inception_v3 assets')
    optional.add_argument('-c', '--convert_model', action="store_true", required=False,
                          help='Convert and compile model once acquired.')
    optional.add_argument('-cu', '--custom', action="store_true", required=False,
                          help='Convert the model using Relu as a custom operation. Only available if --c or --convert_model option is chosen')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                          help='Download inception_v3 assets to inception_v3 example directory')
    optional.add_argument('-g', '--generate_packages', type=str, choices=['cpu', 'dsp', 'htp'], required=False,
                          help='Generate and compile custom op packages for HTP, CPU and DSP')
    optional.add_argument('-q', '--quantize_model', action="store_true", required=False,
                          help='Quantize the model during conversion. Only available if --c or --convert_model option is chosen')

    args = parser.parse_args()
    # FIXME: support compiling custom op package on windows
    if (args.generate_packages or args.custom) and SDK().is_windows:
        parser.error("Compiling custom op package is currently unsupported on Windows.")
    if args.quantize_model and not args.convert_model:
        parser.error("ERROR: --quantize_model option must be run with --convert_model option.")

    return args


if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download, args.convert_model, args.quantize_model, args.custom, args.generate_packages)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
