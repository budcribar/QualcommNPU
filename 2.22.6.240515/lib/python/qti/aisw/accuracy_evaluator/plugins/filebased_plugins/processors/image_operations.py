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
from typing import List
import cv2
from PIL import Image
import numpy as np
import os
import json
from qti.aisw.accuracy_evaluator.plugins.filebased_plugins.processors.normalization import normalize
from qti.aisw.accuracy_evaluator.qacc.plugin import qacc_plugin, PluginInputInfo, PluginOutputInfo
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
from qti.aisw.accuracy_evaluator.common.utilities import Helper


class expand_dims(qacc_plugin):
    """Plugin to add N dimension,e.g., HWC to NHWC."""
    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin, pout):
        if not pin.is_memory_input():
            print('Only in memory input supported for expand dims.')
            pout.set_status(qcc.STATUS_ERROR)
            return

        inp = pin.get_input()
        inp = np.expand_dims(inp, axis=0)

        pout.set_mem_output(inp)
        pout.set_status(qcc.STATUS_SUCCESS)


class convert_nchw(qacc_plugin):
    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin, pout):
        if not pin.is_memory_input():
            print('Only in memory input supported for crop.')
            pout.set_status(qcc.STATUS_ERROR)
            return

        inp = pin.get_input()

        inp = inp.transpose([2, 0, 1])

        if pin.get_param('expand-dims', True):
            inp = np.expand_dims(inp, axis=0)

        pout.set_mem_output(inp)
        pout.set_status(qcc.STATUS_SUCCESS)


class crop(qacc_plugin):
    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    supported_libraries = ['numpy', 'torchvision']

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        library_name = pin.get_param('library', 'numpy')
        if library_name not in self.supported_libraries:
            print('crop plugin does not support library ' + library_name)
            pout.set_status(qcc.STATUS_ERROR)
            return
        if not pin.is_memory_input():
            print('Only in memory input supported for crop.')
            pout.set_status(qcc.STATUS_ERROR)
            return

        inp = pin.get_input()
        out_height, out_width = map(int, pin.get_param('dims').split(',')[:2])

        if library_name == 'numpy':
            img = crop.crop_numpy(pin, pout, inp, out_height, out_width)
        elif library_name == 'torchvision':
            img = crop.crop_tv(pin, pout, inp, out_height, out_width)

        if not img is None:
            pout.set_mem_output(img)
            pout.set_status(qcc.STATUS_SUCCESS)

    @staticmethod
    def crop_numpy(pin: PluginInputInfo, pout: PluginOutputInfo, inp, out_height, out_width):
        """Need to add support for cropping images whose dimensions are smaller
        than crop dimensions."""
        inp_dims = inp.ndim
        if inp_dims != 3:
            print('input dim for crop image must be 3')
            pout.set_status(qcc.STATUS_ERROR)
            return

        height, width = inp.shape[0], inp.shape[1]
        left = int(round((width - out_width) / 2))
        right = left + out_width
        top = int(round((height - out_height) / 2))
        bottom = top + out_height

        inp = inp[top:bottom, left:right]
        return inp

    @staticmethod
    def crop_tv(pin: PluginInputInfo, pout: PluginOutputInfo, inp, out_height, out_width):
        if not isinstance(inp, Image.Image):  #To check if the input is valid PIL image or not
            print('This version(0.7.0) of torchvision supports only valid PIL images as input')
            pout.set_status(qcc.STATUS_ERROR)
            return

        if out_height == out_width:
            crop_size = out_height
        else:
            crop_size = (out_height, out_width)
        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        inp = torchvision.transforms.functional.center_crop(inp, crop_size)

        typecast_required = pin.get_param('typecasting_required', True)
        if typecast_required:
            inp = np.asarray(inp, dtype=pout.get_output_dtype())
        return inp

    @staticmethod
    def crop_tf(pin: PluginInputInfo, pout: PluginOutputInfo, inp, out_height, out_width):
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
        typecast_required = pin.get_param('typecasting_required', True)
        if typecast_required:
            inp = tf_image.numpy()
        return inp


class resize(qacc_plugin):
    """Takes a image path or image data, resizes it with the given
    configuration and returns the resized image."""
    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_PATH,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_CV2
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
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

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin: PluginInputInfo, pout: PluginOutputInfo):

        library_name = pin.get_param('library', 'opencv')
        if library_name not in self.supported_libraries:
            print('Resize plugin does not support library ' + library_name)
            pout.set_status(qcc.STATUS_ERROR)
            return

        height, width = map(int, pin.get_param('dims').split(',')[:2])
        given_dims = [height, width]

        if pin.is_memory_input():
            img = pin.get_input()
        else:
            if library_name == 'opencv':
                img = cv2.imread(pin.get_input())
            elif library_name == 'pillow' or library_name == 'torchvision' or library_name == 'tensorflow':
                img = Image.open(pin.get_input())

            if img is None:
                print('Failed to read image :' + pin.get_input())
                pout.set_status(qcc.STATUS_ERROR)
                return

        img_dims = resize.get_img_dims(img, library_name)
        if img_dims is None:
            pout.set_status(qcc.STATUS_ERROR)
            return

        resize_type = pin.get_param('type', None)
        resize_dims, letterbox_scale = resize.get_resize_dims(given_dims, img_dims, resize_type)
        self.resize_libs(pin, pout, img, given_dims, resize_dims, img_dims, library_name,
                         letterbox_scale)

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

    def resize_libs(self, pin: PluginInputInfo, pout: PluginOutputInfo, img, given_dims,
                    resize_dims, img_dims, library_name, letterbox_scale):

        height, width = resize_dims
        inp_height, inp_width = given_dims

        if library_name == 'pillow':
            interp_str = pin.get_param('interp', default='bicubic')
        else:
            interp_str = pin.get_param('interp', default='bilinear')

        # interpolation type for non tensorflow case
        if library_name != 'tensorflow':
            interp = self.interp_flags[library_name].get(interp_str, None)
            if interp is None:
                print('Invalid interpolation method ' + interp_str)
                pout.set_status(qcc.STATUS_ERROR)
                return
        resize_first = pin.get_param('resize_before_typecast', True)
        channel_order = pin.get_param('channel_order', 'RGB')

        if library_name == 'torchvision':
            img = resize.resize_tv(pin, pout, img, height, width, interp, resize_first,
                                   channel_order)
        elif library_name == 'opencv':
            img = resize.resize_cv(pin, pout, img, height, width, interp, resize_first,
                                   channel_order, img_dims, resize_dims, inp_height, inp_width,
                                   letterbox_scale)
        elif library_name == 'pillow':
            img = resize.resize_pil(pin, pout, img, height, width, interp, resize_first,
                                    channel_order, resize_dims, inp_height, inp_width, library_name)
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
            img = resize.resize_tf(pin, pout, img, height, width, interp, resize_first,
                                   channel_order, resize_dims, inp_height, inp_width, library_name)

        if not img is None:
            if pout.is_memory_output():
                pout.set_mem_output(img)
            else:
                img.tofile(pout.get_output_path())
            pout.set_status(qcc.STATUS_SUCCESS)

    @staticmethod
    def resize_tv(pin: PluginInputInfo, pout: PluginOutputInfo, img, height, width, interp,
                  resize_first, channel_order):
        if not resize_first:
            print('For torchvision, typecasting before resizing is not supported')
            pout.set_status(qcc.STATUS_ERROR)
            return

        if channel_order == 'RGB':
            img = img.convert('RGB')

        if height == width:
            resize_size = height
        else:
            resize_size = (height, width)

        torchvision = Helper.safe_import_package("torchvision", "0.14.1")
        img = torchvision.transforms.functional.resize(img, resize_size, interp)

        typecast_required = pin.get_param('typecasting_required', True)
        if typecast_required and resize_first:
            img = np.asarray(img, dtype=pout.get_output_dtype())
        return img

    @staticmethod
    def resize_cv(pin: PluginInputInfo, pout: PluginOutputInfo, img, height, width, interp,
                  resize_first, channel_order, img_dims, resize_dims, inp_height, inp_width,
                  letterbox_scale):
        if channel_order == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if letterbox_scale and img_dims != resize_dims:
            interp = cv2.INTER_AREA if letterbox_scale < 1 else cv2.INTER_LINEAR

        # Type-cast before resize
        if not resize_first:
            img = img.astype(pout.get_output_dtype())

        img = cv2.resize(img, (width, height), interpolation=interp)

        if pin.get_param('type') == 'letterbox':
            img = pad.create_border(img, height=inp_height, width=inp_width,
                                    target_dims=resize_dims)

        #Type-cast after resize
        if resize_first:
            img = img.astype(pout.get_output_dtype())
        return img

    @staticmethod
    def resize_pil(pin: PluginInputInfo, pout: PluginOutputInfo, img, height, width, interp,
                   resize_first, channel_order, resize_dims, inp_height, inp_width, library_name):
        if not resize_first:
            print('For pillow, typecasting before resizing is not supported')
            pout.set_status(qcc.STATUS_ERROR)
            return

        img = img.convert('RGB')
        if channel_order == 'BGR' or channel_order == 'bgr':
            img = np.asarray(img)
            img = img[:, :, ::-1]
            img = Image.fromarray(img)

        img = img.resize((width, height), interp)

        if pin.get_param('type') == 'letterbox':
            img = pad.create_border(img, height=inp_height, width=inp_width,
                                    library_name=library_name, target_dims=resize_dims,
                                    mode=channel_order)
        #Type-cast after resize
        if resize_first:
            img = np.asarray(img, dtype=pout.get_output_dtype())
        return img

    @staticmethod
    def get_img_dims(img, library_name):

        if library_name == 'opencv':
            orig_height, orig_width = img.shape[:2]
        elif library_name == 'pillow':
            orig_width, orig_height = img.size
        elif library_name == 'torchvision':
            try:  #To check if the input is valid PIL image or not
                orig_width, orig_height = img.size
            except TypeError as e:
                print('ERROR: ' + str(e))
                print('This version(0.7.0) of torchvision supports only valid PIL images as input')
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

    @staticmethod
    def resize_tf(pin: PluginInputInfo, pout: PluginOutputInfo, img, height, width, interp,
                  resize_first, channel_order, resize_dims, inp_height, inp_width, library_name):
        tf = Helper.safe_import_package("tensorflow")
        if not resize_first:
            print('Typecasting before resizing is not supported for PIL Image')
            pout.set_status(qcc.STATUS_ERROR)
            return

        img = img.convert('RGB')
        if channel_order == 'BGR' or channel_order == 'bgr':
            img = np.asarray(img)
            img = img[:, :, ::-1]
            img = Image.fromarray(img)

        img = np.asarray(img)
        norm_before_resize = pin.get_param("normalize_before_resize", False)
        crop_before_resize = pin.get_param("crop_before_resize", False)

        #normalize
        if norm_before_resize:
            mean = pin.get_param("mean")
            std = pin.get_param("std")
            if channel_order == "BGR":
                mean = [mean["B"], mean["G"], mean["R"]]
                std = [std["B"], std["G"], std["R"]]
            else:
                mean = [mean["R"], mean["G"], mean["B"]]
                std = [std["R"], std["G"], std["B"]]

            img = normalize.norm_numpy(pin, pout, img, mean, std)

        #crop image
        if crop_before_resize:
            img = crop.crop_tf(pin, pout, img, inp_height, inp_width)

        #resize
        tf_image = tf.convert_to_tensor(img, dtype=tf.float32)
        tf_image_r = tf.image.resize(tf_image, [height, width], method=interp)
        scaled_image = tf_image_r[0:inp_height, 0:inp_width, :]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, inp_height, inp_width)
        #Type-cast after resize
        if resize_first:
            img = output_image.numpy()
        return img


class pad(qacc_plugin):
    """Image padding with constant size or based on target dimensions."""
    default_inp_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }
    default_out_info = {
        qcc.IO_TYPE: qcc.PLUG_INFO_TYPE_MEM,
        qcc.IO_DTYPE: qcc.DTYPE_FLOAT32,
        qcc.IO_FORMAT: qcc.FMT_NPY
    }

    def execute(self, pin_list: List[PluginInputInfo], pout_list: List[PluginOutputInfo]):

        for pin, pout in zip(pin_list, pout_list):
            self.execute_index(pin, pout)

    def execute_index(self, pin, pout):
        if not pin.is_memory_input():
            print('Only in memory input supported for pad plugin.')
            pout.set_status(qcc.STATUS_ERROR)
            return

        inp = pin.get_input()
        pad_type = pin.get_param('type')

        if pad_type == 'constant':
            pad_size = pin.get_param('pad_size')
            inp = pad.create_border(inp, constant_pad=pad_size)
        elif pad_type == 'target_dims':
            img_position = pin.get_param('img_position', 'center')
            color_dim1 = int(pin.get_param('color', 114))  #Padding value for all planes is same.
            color = (color_dim1, color_dim1, color_dim1
                     )  #Provision to have 3 different values for each plane.
            new_height, new_width = map(int, pin.get_param('dims').split(',')[:2])
            inp = pad.create_border(inp, height=new_height, width=new_width, color=color,
                                    img_position=img_position)
        else:
            qacc_logger.error("Invalid pad_type provided")
            pout.set_status(qcc.STATUS_ERROR)
            return

        pout.set_mem_output(inp)
        pout.set_status(qcc.STATUS_SUCCESS)

    @staticmethod
    def create_border(img, constant_pad=None, height=None, width=None, color=(114, 114, 114),
                      library_name=None, target_dims=None, mode=None, img_position='center'):
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
