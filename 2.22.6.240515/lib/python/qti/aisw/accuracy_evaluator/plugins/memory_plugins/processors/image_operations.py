##############################################################################
#
# Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
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
import cv2
from PIL import Image
import numpy as np
import os
import json
from pathlib import Path
from normalize import normalize
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_memory_preprocessor
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class expand_dims(qacc_memory_preprocessor):
    """Plugin to add N dimension,e.g., HWC to NHWC."""

    def execute(self, data, meta, input_idx, **kwargs):
        out_data = []
        for inp in data:
            inp = np.expand_dims(inp, axis=0)
            out_data.append(inp)
        return out_data, meta


class create_batch(qacc_memory_preprocessor):
    """Used in case when no preprocessing is required or all of preprocessing
    is done within dataset plugin itself."""

    def execute(self, data, meta, input_idx, **kwargs):
        input_dtypes = [inp_info[0] for inp_info in list(self.extra_params['input_info'].values())]
        out_data = []
        for inp_idx, inp in enumerate(data):
            if isinstance(inp, (str, Path)):
                out_data.append(np.fromfile(inp, dtype=input_dtypes[inp_idx]))
        if len(out_data) == 0:
            out_data = data  # Noop
        return out_data, meta


class convert_nchw(qacc_memory_preprocessor):
    """Plugin to swap the input from NHWC to NCHW."""

    def execute(self, data, meta, input_idx, expand_dims=True, **kwargs):
        out_data = []
        for inp in data:
            inp = inp.transpose([2, 0, 1])
            if expand_dims:
                inp = np.expand_dims(inp, axis=0)
            out_data.append(inp)
        return out_data, meta


class crop(qacc_memory_preprocessor):
    """Takes a image data as numpy array, crops it with the given configuration
    and returns the cropped image."""

    def execute(self, data, meta, input_idx, dims, library='numpy', typecasting_required=True,
                supported_libraries=['numpy', 'torchvision'], **kwargs):
        dims = [int(d.strip()) for d in dims.split(',')]
        out_data = []
        status = True
        for inp in data:
            out_height, out_width = dims[:2]

            if library == 'numpy':
                img = crop.crop_numpy(inp, out_height, out_width)
            elif library == 'torchvision':
                img = crop.crop_tv(inp, out_height, out_width,
                                   typecast_required=typecasting_required)
            out_data.append(img)
        return out_data, meta

    @staticmethod
    def crop_numpy(inp, out_height, out_width):
        """
        TODO: Need to add support for cropping images whose dimensions are smaller than crop dimensions
        """
        inp_dims = inp.ndim
        if inp_dims != 3:
            qacc_file_logger.error('input dim for crop image must be 3')
            return

        height, width = inp.shape[0], inp.shape[1]
        left = int(round((width - out_width) / 2))
        right = left + out_width
        top = int(round((height - out_height) / 2))
        bottom = top + out_height

        inp = inp[top:bottom, left:right]
        return inp

    @staticmethod
    def crop_tv(inp, out_height, out_width, typecast_required=True):
        if not isinstance(inp, Image.Image):  # To check if the input is valid PIL image or not
            qacc_file_logger.error(
                'This version(0.7.0) of torchvision supports only valid PIL images as input')
            return

        if out_height == out_width:
            crop_size = out_height
        else:
            crop_size = (out_height, out_width)
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        inp = torchvision.transforms.functional.center_crop(inp, crop_size)

        if typecast_required:
            inp = np.asarray(inp, dtype=np.float32)
        return inp

    @staticmethod
    def crop_tf(inp, out_height, out_width, typecast_required=True):
        assert out_height == out_width, "Support only square crop"
        img_h = inp.shape[0]
        img_w = inp.shape[1]
        tf = Helper.safe_import_package("tensorflow")
        tf_image = tf.convert_to_tensor(inp, dtype=tf.float32)
        pad_s = round(out_height * (1 / 0.875 - 1))
        crop_s = tf.cast(
            ((out_height / (out_height + pad_s)) * tf.cast(tf.minimum(img_h, img_w), tf.float32)),
            tf.int32)
        offset_height = ((img_h - crop_s) + 1) // 2
        offset_width = ((img_w - crop_s) + 1) // 2
        tf_image = tf.image.crop_to_bounding_box(tf_image, offset_height, offset_width, crop_s,
                                                 crop_s)
        if typecast_required:
            inp = tf_image.numpy()
        return inp


class resize(qacc_memory_preprocessor):
    """Takes a image path or image data, resizes it with the given
    configuration and returns the resized image."""
    supported_libraries = ['opencv', 'torchvision', 'pillow', 'tensorflow']
    interp_flags = {
        'opencv': {
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'area': cv2.INTER_AREA
        },
        'torchvision': {
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST,
            'bicubic': Image.BICUBIC
        },
        'pillow': {
            'bicubic': Image.BICUBIC,
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST,
            'box': Image.BOX,
            'hamming': Image.HAMMING,
            'lanczos': Image.LANCZOS
        }
    }

    def execute(self, data, meta, input_idx, dims=None, library='opencv', interp=None,
                typecasting_required=None, type=None, channel_order=None,
                resize_before_typecast=None, mean={
                    "R": 0,
                    "G": 0,
                    "B": 0
                }, std={
                    "R": 1,
                    "G": 1,
                    "B": 1
                }, norm=255.0, normalize_first=True, normalize_before_resize=False,
                crop_before_resize=False, **kwargs):

        if library not in resize.supported_libraries:
            # TODO: Error Handling
            qacc_file_logger.error(f'Resize plugin does not support library {library}')
            return 1

        dims = [int(d.strip()) for d in dims.split(',')]
        height, width = dims[:2]
        given_dims = [int(height), int(width)]
        if library == 'pillow':
            typecasting_required = True
        else:
            typecasting_required = typecasting_required
        out_data = []
        for idx, item in enumerate(data):
            if not isinstance(item, str):  # not Path: Assume image/tensor object
                img = item
            else:
                # Path
                item = item.strip()
                if library == 'opencv':
                    img = cv2.imread(item)
                elif library == 'pillow' or library == 'torchvision' or library == 'tensorflow':
                    img = Image.open(item)

                if img is None:
                    logging.error('Failed to read image :' + item)

            img_dims = resize.get_img_dims(img, library)
            if img_dims is None:
                # TODO: Error Handling
                logging.error('Failed to get image dimensions :' + item)

            resize_dims, letterbox_scale = resize.get_resize_dims(given_dims, img_dims, type)
            img = self.resize_libs(img, given_dims, resize_dims, img_dims, letterbox_scale, library,
                                   interp, resize_before_typecast, typecasting_required,
                                   channel_order, type, mean, std, normalize_before_resize,
                                   crop_before_resize, norm, normalize_first)

            out_data.append(img)
        return out_data, meta

    @staticmethod
    def get_resize_dims(given_dims, img_dims, resize_type):

        height, width = given_dims
        orig_height, orig_width = img_dims
        letterbox_scale = None
        # Resize based on type.
        if resize_type == 'letterbox':
            new_width, new_height, letterbox_scale = resize.letterbox_resize(
                height, width, orig_height, orig_width)
        elif resize_type == 'imagenet':
            new_height, new_width = resize.imagenet_resize(height, width, orig_height, orig_width)
        elif resize_type == "aspect_ratio":
            ratio = min(height / orig_height, width / orig_width)
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)
        else:
            new_height, new_width = height, width

        return [new_height, new_width], letterbox_scale

    def resize_libs(self, img, given_dims, resize_dims, img_dims, letterbox_scale, library_name,
                    interp, resize_before_typecast, typecasting_required, channel_order,
                    resize_type, mean, std, normalize_before_resize, crop_before_resize, norm,
                    normalize_first):

        height, width = resize_dims
        inp_height, inp_width = given_dims

        if library_name == 'pillow':
            interp_str = interp if interp else 'bicubic'
        else:
            interp_str = interp if interp else 'bilinear'

        # interpolation type for non tensorflow case
        if library_name != 'tensorflow':
            interp = resize.interp_flags[library_name].get(interp_str, None)
            if interp is None:
                qacc_file_logger.error(f'Invalid interpolation method {interp_str}')
                return
        if resize_before_typecast is None:
            resize_first = True
        else:
            resize_first = resize_before_typecast

        # self.resize_before_typecast = True = pin.get_param('resize_before_typecast', True)
        if channel_order is None:
            channel_order = 'RGB'
        else:
            channel_order = channel_order

        if library_name == 'torchvision':
            img = resize.resize_tv(img, height, width, interp, resize_first, channel_order,
                                   typecast_required=typecasting_required)
        elif library_name == 'opencv':
            img = resize.resize_cv(img, height, width, interp, resize_first, channel_order,
                                   img_dims, resize_dims, inp_height, inp_width, resize_type,
                                   letterbox_scale)
        elif library_name == 'pillow':
            img = resize.resize_pil(img, height, width, interp, resize_first, channel_order,
                                    resize_dims, inp_height, inp_width, resize_type, library_name)
        elif library_name == 'tensorflow':
            tf = Helper.safe_import_package("tensorflow")
            interp_flags = {
                'bicubic': tf.image.ResizeMethod.BICUBIC,
                'bilinear': tf.image.ResizeMethod.BILINEAR,
                'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                'area': tf.image.ResizeMethod.AREA,
                'gaussian': tf.image.ResizeMethod.GAUSSIAN,
                'lanczos3': tf.image.ResizeMethod.LANCZOS3,
                'lanczos5': tf.image.ResizeMethod.LANCZOS5,
                'mitchellcubic': tf.image.ResizeMethod.MITCHELLCUBIC
            }
            interp = interp_flags.get(interp_str, None)

            img = resize.resize_tf(img, height, width, interp, resize_first, channel_order,
                                   inp_height, inp_width, mean, std, normalize_before_resize,
                                   crop_before_resize, norm, normalize_first)

        return img

    @staticmethod
    def resize_tv(img, height, width, interp, resize_first, channel_order, typecast_required=True):
        if not resize_first:
            qacc_file_logger.error('For torchvision, typecasting before resizing is not supported')
            return

        if channel_order == 'RGB':
            img = img.convert('RGB')

        if height == width:
            resize_size = height
        else:
            resize_size = (height, width)

        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        img = torchvision.transforms.functional.resize(img, resize_size, interp)
        if typecast_required and resize_first:  # TODO : Need to find a way to handle this scenario
            img = np.asarray(img, dtype=np.float32)
        return img

    @staticmethod
    def resize_cv(img, height, width, interp, resize_first, channel_order, img_dims, resize_dims,
                  inp_height, inp_width, resize_type, letterbox_scale):
        if channel_order == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if letterbox_scale and img_dims != resize_dims:
            interp = cv2.INTER_AREA if letterbox_scale < 1 else cv2.INTER_LINEAR

        # Type-cast before resize
        if not resize_first:
            img = img.astype(np.float32)

        img = cv2.resize(img, (width, height), interpolation=interp)

        if resize_type == 'letterbox':
            img = pad.create_border(img, height=inp_height, width=inp_width,
                                    target_dims=resize_dims)

        # Type-cast after resize
        if resize_first:
            img = img.astype(np.float32)
        return img

    @staticmethod
    def resize_pil(img, height, width, interp, resize_first, channel_order, resize_dims, inp_height,
                   inp_width, resize_type, library_name):
        if not resize_first:
            qacc_file_logger.error('For pillow, typecasting before resizing is not supported')
            return

        img = img.convert('RGB')
        if channel_order == 'BGR' or channel_order == 'bgr':
            img = np.asarray(img)
            img = img[:, :, ::-1]
            img = Image.fromarray(img)

        img = img.resize((width, height), interp)

        if resize_type == 'letterbox':
            img = pad.create_border(img, height=inp_height, width=inp_width,
                                    library_name=library_name, target_dims=resize_dims,
                                    mode=channel_order)
        # Type-cast after resize
        if resize_first:
            img = np.asarray(img, dtype=np.float32)
        return img

    @staticmethod
    def resize_tf(img, height, width, interp, resize_first, channel_order, inp_height, inp_width,
                  mean, std, normalize_before_resize=False, crop_before_resize=False, norm=255.0,
                  normalize_first=True):
        tf = Helper.safe_import_package("tensorflow")
        if not resize_first:
            qacc_file_logger.error('Typecasting before resizing is not supported for PIL Image')
            return
        img = img.convert('RGB')
        if channel_order == 'BGR' or channel_order == 'bgr':
            img = np.asarray(img)
            img = img[:, :, ::-1]
            img = Image.fromarray(img)
        img = np.asarray(img)

        #normalize
        if normalize_before_resize:
            if channel_order == "BGR":
                mean = [mean["B"], mean["G"], mean["R"]]
                std = [std["B"], std["G"], std["R"]]
            else:
                mean = [mean["R"], mean["G"], mean["B"]]
                std = [std["R"], std["G"], std["B"]]

            img = normalize.norm_numpy(img, mean, std, norm, normalize_first)

        #crop image
        if crop_before_resize:
            img = crop.crop_tf(img, inp_height, inp_width)
        #resize
        tf_image = tf.convert_to_tensor(img, dtype=tf.float32)
        tf_image_r = tf.image.resize(tf_image, [height, width], method=interp)
        scaled_image = tf_image_r[0:inp_height, 0:inp_width, :]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, inp_height, inp_width)
        #Type-cast after resize
        if resize_first:
            img = output_image.numpy()
        return img

    @staticmethod
    def get_img_dims(img, library_name):

        if library_name == 'opencv':
            orig_height, orig_width = img.shape[:2]
        elif library_name == 'pillow':
            orig_width, orig_height = img.size
        elif library_name == 'torchvision':
            try:  # To check if the input is valid PIL image or not
                orig_width, orig_height = img.size
            except TypeError as e:
                qacc_file_logger.error(str(e))
                qacc_file_logger.error(
                    'This version(0.7.0) of torchvision supports only valid PIL images as input')
                return None
        elif library_name == 'tensorflow':
            orig_width, orig_height = img.size
        return [orig_height, orig_width]

    @staticmethod
    def imagenet_resize(height, width, orig_height, orig_width):
        if orig_height > orig_width:
            new_width = width
            new_height = int(height * orig_height / orig_width)
        else:
            new_height = height
            new_width = int(width * orig_width / orig_height)

        return new_height, new_width

    @staticmethod
    def letterbox_resize(height, width, orig_height, orig_width):
        # Scale ratio (new / old)
        scale = min(height / orig_height, width / orig_width)
        new_width, new_height = int(orig_width * scale), int(orig_height * scale)
        return new_width, new_height, scale


class pad(qacc_memory_preprocessor):
    """Image padding with constant size or based on target dimensions."""

    def execute(self, data, meta, input_idx, dims, type='constant', pad_size=None,
                img_position='center', color=114, **kwargs):
        dims = [int(d.strip()) for d in dims.split(',')]
        if type == "target_dims":
            color_dim1 = int(color)  #Padding value for all planes is same.
            color = (color_dim1, color_dim1, color_dim1)
        out_data = []

        for inp in data:
            if type == 'constant':
                img = pad.create_border(inp, constant_pad=pad_size)
            elif type == 'target_dims':
                new_height, new_width = dims[:2]
                img = pad.create_border(inp, height=new_height, \
                        width=new_width, color=color, img_position=img_position)
            else:
                raise Exception(f"Pad 'type' not in ['constant', 'target_dims']")
            out_data.append(img)
        return out_data, meta

    @staticmethod
    def create_border(img, constant_pad=None, height=None, width=None, color=(114, 114, 114),
                      library_name=None, target_dims=None, mode=None, img_position=None):
        if constant_pad:
            top = bottom = left = right = constant_pad
        else:
            if target_dims:
                orig_height, orig_width = target_dims
            else:
                orig_height, orig_width = img.shape[:2]
            if img_position == 'center':
                pad_w, pad_h = (width - orig_width) / 2, (height - orig_height) / 2
                top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
                left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
            elif img_position == 'corner':
                pad_w, pad_h = (width - orig_width), (height - orig_height)
                top, bottom = 0, pad_h
                left, right = 0, pad_w
        if library_name and mode and library_name == 'pillow':
            new_img = Image.new(mode, (width, height), color=color)
            new_img.paste(img, (left, top, width - right, height - bottom))
            img = new_img
        else:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=color)
        return img
