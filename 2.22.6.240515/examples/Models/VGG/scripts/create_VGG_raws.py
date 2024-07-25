#!/usr/bin/python3
#==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

import argparse
import numpy as np
import os

from PIL import Image

CROPPED_SIZE = 224

def __get_img_raw(img_filepath):
    img_filepath = os.path.abspath(img_filepath)
    img = Image.open(img_filepath)
    img_ndarray = np.array(img) # read it
    if len(img_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(img_ndarray.shape))
    if (img_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    return img_ndarray

def __create_raw_vgg(img_filepath):
    img_raw = __get_img_raw(img_filepath)
    if img_raw.shape[2] != 3:
        raise RuntimeError('Require image with rgb but channel is %d' % img_raw.shape[2])
    img_raw = np.transpose(img_raw, (2, 0, 1))
    img_raw = (img_raw - img_raw.min())/(img_raw.max()-img_raw.min())
    mean_rgb = [0.485, 0.456, 0.406] # RGB mean value
    std_rgb =  [0.229, 0.224, 0.225] # RGB std value
    mean = np.empty(img_raw.shape)
    std  = np.empty(img_raw.shape)
    norm = np.empty(img_raw.shape)
    for i in range(len(img_raw)):
        mean[i].fill(mean_rgb[i])
        std[i].fill(std_rgb[i])
        norm[i] = (img_raw[i] - mean[i]) /std[i]
    snpe_raw = np.transpose(norm, (1, 2, 0))
    snpe_raw = snpe_raw.astype('float32')

    img_filepath = os.path.abspath(img_filepath)
    filename, ext = os.path.splitext(img_filepath)
    snpe_raw_filename = filename
    snpe_raw_filename += '.raw'
    snpe_raw.tofile(snpe_raw_filename)
    return 0

def __resize_square_to_jpg(src, dst):
    src_img = Image.open(src)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2: src_img = src_img.convert(mode = 'RGB')
    src_img = src_img.resize((256, 256),resample=Image.BILINEAR)
    # center crop to 224x224
    width, height = src_img.size
    short_dim = CROPPED_SIZE
    crop_coord = (
        (width  - short_dim) / 2,
        (height - short_dim) / 2,
        (width  + short_dim) / 2,
        (height + short_dim) / 2
    )
    dst_img = src_img.crop(crop_coord)
    dst_img.save(dst)
    return 0

def convert_img(src,dest):
    print("Converting images for VGG network.")

    print("Scaling to square: " + src)
    for root,dirs,files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' in src_image):
                print(src_image)
                dest_image = os.path.join(dest, jpgs)
                __resize_square_to_jpg(src_image,dest_image)

    print("Image mean: " + dest)
    for root,dirs,files in os.walk(dest):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' in src_image):
                print(src_image)
                __create_raw_vgg(src_image)

def main():
    parser = argparse.ArgumentParser(description="Convert a jpg",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dest',type=str, required=True)
    parser.add_argument('-i','--img_folder',type=str, required=True)

    args = parser.parse_args()

    src = os.path.abspath(args.img_folder)
    dest = os.path.abspath(args.dest)

    convert_img(src,dest)
    print("Preprocessed successfully!" )




if __name__ == '__main__':
    exit(main())