#
# Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run inception_v3 model with SNPE SDK.
'''

import udo_setup_functions as UDO
import os
import subprocess
import shutil
import hashlib
import argparse
import sys
import requests
import platform
import tarfile
from pathlib import Path

INCEPTION_V3_ARCHIVE_CHECKSUM       = 'a904ddf15593d03c7dd786d552e22d73'
INCEPTION_V3_ARCHIVE_FILE           = 'inception_v3_2016_08_28_frozen.pb.tar.gz'
INCEPTION_V3_ARCHIVE_URL            = 'https://storage.googleapis.com/download.tensorflow.org/models/' + INCEPTION_V3_ARCHIVE_FILE
INCEPTION_V3_PB_FILENAME            = 'inception_v3_2016_08_28_frozen.pb'
INCEPTION_V3_PB_OPT_FILENAME        = 'inception_v3_2016_08_28_frozen_opt.pb'
INCEPTION_V3_DLC_FILENAME           = 'inception_v3.dlc'
INCEPTION_V3_DLC_QUANTIZED_FILENAME = 'inception_v3_quantized.dlc'
INCEPTION_V3_LBL_FILENAME           = 'imagenet_slim_labels.txt'
OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'
RAW_LIST_FILE                       = 'raw_list.txt'
TARGET_RAW_LIST_FILE                = 'target_raw_list.txt'

class SDK:
    ROOT = Path()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(WindowsSDK if platform.system() == "Windows" else LinuxSDK)
        return cls._instance

    def __init__(self):
        self.is_windows = isinstance(self, WindowsSDK)

    def host_arch(self):
        raise NotImplementedError

    def udo_package(self):
        raise NotImplementedError

    def bin(self):
        return self.__class__.ROOT / 'bin' / self.host_arch()

    def model(self):
        return self.__class__.ROOT / 'examples' / 'Models' / 'InceptionV3'

class WindowsSDK(SDK):
    def host_arch(self):
        return "x86_64-windows-msvc"

    def udo_package(self):
        return Path(self.model(), UDO.UDO_PACKAGE, "libs", "arm64_windows", "UdoSoftmaxUdoPackageReg.dll")

class LinuxSDK(SDK):
    def host_arch(self):
        return "x86_64-linux-clang"

    def udo_package(self):
        return Path(self.model(), UDO.UDO_PACKAGE, "libs", "x86-64_linux_clang", "libUdoSoftmaxUdoPackageReg.so")

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
        raise RuntimeError(filename + ' not found at the location specified by ' + inception_v3_data_dir + '. Re-run with download option.')
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + ' does not match checksum of file ' + md5)

def find_optimize_for_inference():
    tensorflow_root = os.path.abspath(os.environ['TENSORFLOW_HOME'])
    for root, dirs, files in os.walk(tensorflow_root):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)

def optimize_for_inference(model_dir, tensorflow_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    pb_filename = INCEPTION_V3_PB_FILENAME

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        dlc_dir = model_dir / 'dlc'
        if not dlc_dir.is_dir():
            dlc_dir.mkdir()
        cmd = [sys.executable, opt_4_inference_file,
               '--input', tensorflow_dir / INCEPTION_V3_PB_FILENAME,
               '--output', tensorflow_dir / INCEPTION_V3_PB_OPT_FILENAME,
               '--input_names', 'input',
               '--output_names', 'InceptionV3/Predictions/Reshape_1']
        try:
            subprocess.check_call(cmd)
            pb_filename = INCEPTION_V3_PB_OPT_FILENAME
            print('INFO: Optimizing for inference succeeded (%s).' %(INCEPTION_V3_PB_OPT_FILENAME))
        except subprocess.CalledProcessError as e:
            print(e)
            print("WARNING: Optimizing for inference failed (%d)." %(e.returncode))
            print("         Skipping inference optimization.")

    return pb_filename

def prepare_data_images(model_dir, tensorflow_dir):
    data_dir = model_dir / 'data'
    data_cropped_dir = data_dir / 'cropped'
    data_cropped_dir.mkdir(exist_ok=True)

    # copy the labels file to the data directory
    src_label_file = tensorflow_dir / INCEPTION_V3_LBL_FILENAME
    shutil.copy(src_label_file, data_dir)

    print('INFO: Creating SNPE inception_v3 raw data')
    create_raws_script = model_dir / 'scripts' / 'create_inceptionv3_raws.py'
    cmd = [sys.executable, create_raws_script,
           '-i', data_dir,
           '-d', data_cropped_dir]
    subprocess.call(cmd)

    print('INFO: Creating image list data files')
    create_file_list_script = model_dir / 'scripts' / 'create_file_list.py'
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


def convert_to_dlc(pb_filename, model_dir, tensorflow_dir, runtime, udo, htp_soc):
    print('INFO: Converting ' + pb_filename +' to SNPE DLC format')
    dlc_dir = model_dir / 'dlc'
    if not dlc_dir.is_dir():
        dlc_dir.mkdir(parents=True)
    dlc_name = UDO.INCEPTION_V3_UDO_DLC_FILENAME if udo else INCEPTION_V3_DLC_FILENAME
    sdk = SDK()
    snpe_tensorflow_to_dlc = sdk.bin().joinpath('snpe-tensorflow-to-dlc')
    cmd = [sys.executable, snpe_tensorflow_to_dlc,
           '--input_network', tensorflow_dir / pb_filename,
           '--input_dim', 'input', '1,299,299,3',
           '--out_node', 'InceptionV3/Predictions/Reshape_1',
           '--output_path', dlc_dir / dlc_name]
    if udo:
        cmd.append('--udo')
        if htp_soc in ["sm8650", "sm8550", "sm8450", "sm8350", "sm7325"]:
            cmd.append(os.path.join(UDO.SNPE_UDO_PATH, 'config', UDO.INCEPTION_V3_UDO_PLUGIN_HTP))
        else:
            cmd.append(os.path.join(UDO.SNPE_UDO_PATH, 'config', UDO.INCEPTION_V3_UDO_PLUGIN))

    subprocess.call(cmd)

    # Further optimize the model with quantization for fixed-point runtimes if required.
    if ('dsp' == runtime or 'aip' == runtime or 'all' == runtime):
        quant_dlc_name = UDO.INCEPTION_V3_UDO_QUANTIZED_DLC_FILENAME if udo else INCEPTION_V3_DLC_QUANTIZED_FILENAME
        print('INFO: Creating ' + quant_dlc_name + ' quantized model')
        # separate snpe-dlc-quantize
        data_cropped_dir = model_dir / 'data' / 'cropped'
        cmd = [sdk.bin() / 'snpe-dlc-quant', '--input_dlc', dlc_dir / dlc_name,
               '--input_list', data_cropped_dir / RAW_LIST_FILE,
               '--output_dlc', dlc_dir / quant_dlc_name]
        if udo:
            udo_package_path = sdk.udo_package()
            x86_lib_path = udo_package_path.parent
            os.environ['LD_LIBRARY_PATH'] = ":".join(map(os.path.abspath, filter(None, (os.environ.get('LD_LIBRARY_PATH'), x86_lib_path))))
            cmd += ("--udo_package_path", udo_package_path)
        print(f'INFO: Executing {cmd[0].name}')
        subprocess.call(cmd , env=os.environ)

        cmd = [sdk.bin() / 'snpe-dlc-graph-prepare', '--input_dlc', dlc_dir / quant_dlc_name,
               '--input_list', data_cropped_dir / RAW_LIST_FILE,
               '--output_dlc', dlc_dir / quant_dlc_name]

        if udo:
            udo_package_path = sdk.udo_package()
            x86_lib_path = udo_package_path.parent
            os.environ['LD_LIBRARY_PATH'] = ":".join(map(os.path.abspath, filter(None, (os.environ.get('LD_LIBRARY_PATH'), x86_lib_path))))
            cmd += ("--udo_package_path", udo_package_path)
        if (htp_soc and ('dsp' == runtime or "all" == runtime)):
            print ('INFO: Compiling HTP metadata for DSP runtime.')
            cmd += ('--htp_socs', htp_soc)
        elif ('aip' == runtime or "all" == runtime):
            print ("WARNING : The argument --enable_hta has been retired from SNPE version >= 2.x.")
        print(f'INFO: Executing {cmd[0].name}')
        subprocess.call(cmd , env=os.environ)

def setup_assets(inception_v3_data_dir, download, runtime, udo, htp_soc):

    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')
    SDK.ROOT = Path(os.environ['SNPE_ROOT'])
    if not SDK.ROOT.is_dir():
        raise FileNotFoundError('SNPE_ROOT (%s) is not a dir' % SDK.ROOT)
    if download:
        url_path = INCEPTION_V3_ARCHIVE_URL;
        print("INFO: Downloading inception_v3 TensorFlow model...")
        wget(inception_v3_data_dir, url_path)

    try:
        checkResource(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE, INCEPTION_V3_ARCHIVE_CHECKSUM)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        sys.exit(0)

    model_dir = SDK.ROOT / 'examples' / 'Models' / 'InceptionV3'
    if not model_dir.is_dir():
        raise FileNotFoundError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    print('INFO: Extracting TensorFlow model')
    tensorflow_dir = model_dir / 'tensorflow'
    if not tensorflow_dir.is_dir():
        tensorflow_dir.mkdir()
    with tarfile.open(os.path.join(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE)) as ivaf:
        ivaf.extractall(path=tensorflow_dir)

    pb_filename = optimize_for_inference(model_dir, tensorflow_dir)

    prepare_data_images(model_dir, tensorflow_dir)

    # If the UDO flag is set, two packages are generated with 'Softmax.json' and 'Softmax_Quant.json' for demo purposes
    # to demonstrate different types of config specifications. Only one of the packages is necessary for execution.
    if udo:
        is_quantized = False
        UDO.setup_udo(model_dir, runtime, is_quantized, htp_soc)

        udo_quantized_package_path = model_dir / "quant_package"
        if not udo_quantized_package_path.is_dir():
            udo_quantized_package_path.mkdir()
        UDO.setup_udo(udo_quantized_package_path, runtime, not is_quantized, htp_soc)


    convert_to_dlc(pb_filename, model_dir, tensorflow_dir, runtime, udo, htp_soc)


    print('INFO: Setup inception_v3 completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=
        '''Prepares the inception_v3 assets for tutorial examples.''')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                          help='directory containing the inception_v3 assets')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                          help='Download inception_v3 assets to inception_v3 example directory')
    optional.add_argument('-r', '--runtime', type=str, required=False,
                          default='cpu', choices=('cpu', 'gpu', 'dsp', 'aip', 'all'),
                          help='Choose a runtime to set up tutorial for. \'all\' option is only supported with --udo flag')
    optional.add_argument('-u', '--udo', action="store_true", required=False,
                          help='Generate and compile a user-defined operation package to be used with inception_v3. Softmax is simulated as a UDO for this script. '
                          'Currently NOT supported on Windows')
    optional.add_argument('-l', '--htp_soc', type=str, nargs='?', const='sm8650', required=False,
                          help='Specify SOC target for generating HTP Offline Cache. For example: "--htp_soc sm8450" for waipio, default value is sm8650 if no value specified')
    args = parser.parse_args()
    # FIXME: support UDO-related on windows
    if args.udo and SDK().is_windows:
        parser.error("Compiling UDO package is currently unsupported on Windows.")
    if args.runtime == "all" and not args.udo:
        parser.error('--udo flag needed to use \'all\' runtime selection')

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download, args.runtime, args.udo, args.htp_soc)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
