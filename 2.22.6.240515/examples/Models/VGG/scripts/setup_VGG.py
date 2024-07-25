#
# Copyright (c) 2021, 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run VGG model with SNPE SDK.
'''
import numpy as np
import os
import subprocess
import shutil
import argparse
import sys
import glob

VGG_MODEL_URL          = 'https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx'
VGG_LBL_URL            = 'https://s3.amazonaws.com/onnx-model-zoo/synset.txt'
VGG_IMG_URL            = 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
VGG_ONNX_FILENAME      = 'vgg16.onnx'
VGG_DLC_FILENAME       = 'vgg16.dlc'
VGG_LBL_FILENAME       = 'synset.txt'
RAW_LIST_FILE          = 'raw_list.txt'

def wget(download_dir, file_url):
    cmd = ['wget', '-N', '--directory-prefix={}'.format(download_dir), file_url]
    subprocess.call(cmd)

def prepare_data_images(snpe_root, model_dir, onnx_dir):
    '''Copy the image to the data directory'''
    src_img_files = os.path.join(onnx_dir, '*.jpg')
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir + '/cropped')
    for file in glob.glob(src_img_files):
        shutil.copy(file, data_dir)

    '''Copy the labels file to the data directory'''
    src_label_file = os.path.join(onnx_dir, VGG_LBL_FILENAME)
    shutil.copy(src_label_file, data_dir)

    print('INFO: Creating SNPE VGG raw data')
    scripts_dir = os.path.join(model_dir, 'scripts')
    create_raws_script = os.path.join(scripts_dir, 'create_VGG_raws.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    cmd = ['python', create_raws_script,
           '-i', data_dir,
           '-d', data_cropped_dir]
    subprocess.call(cmd)

    print('INFO: Creating image list data files')
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, RAW_LIST_FILE),
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, RAW_LIST_FILE),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)

def convert_to_dlc(onnx_filename, model_dir, onnx_dir):
    print('INFO: Converting ' + onnx_filename +' to SNPE DLC format')
    dlc_dir = os.path.join(model_dir, 'dlc')
    if not os.path.isdir(dlc_dir):
        os.makedirs(dlc_dir)
    dlc_name = VGG_DLC_FILENAME
    cmd = ['snpe-onnx-to-dlc',
           '--input_network', os.path.join(onnx_dir, onnx_filename),
           '--output_path', os.path.join(dlc_dir, dlc_name)]
    subprocess.call(cmd)

def setup_assets(VGG_data_dir, download):
    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')

    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)

    model_dir = os.path.join(snpe_root, 'examples', 'Models', 'VGG')
    if not os.path.isdir(model_dir):
        raise RuntimeError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    onnx_dir = os.path.join(model_dir, 'onnx')
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)

    if download:
        print("INFO: Downloading VGG ONNX model, label file, and image...")
        wget(onnx_dir, VGG_MODEL_URL)
        wget(onnx_dir, VGG_LBL_URL)
        wget(onnx_dir, VGG_IMG_URL)

    try:
        os.listdir(onnx_dir)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        sys.exit(0)

    prepare_data_images(snpe_root, model_dir, onnx_dir)

    onnx_filename = os.path.join(onnx_dir, VGG_ONNX_FILENAME)
    convert_to_dlc(onnx_filename, model_dir, onnx_dir)


    print('INFO: Setup VGG completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''Prepares the VGG assets for tutorial examples.''')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                        help='directory containing the VGG assets')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                        help='Download VGG assets to VGG example directory')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
