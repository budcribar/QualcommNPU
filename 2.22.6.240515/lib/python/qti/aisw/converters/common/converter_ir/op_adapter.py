# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import math
import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil, floor
from typing import List

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder, AxisOrders, AxisTracker
from qti.aisw.converters.common.custom_ops.utils.config_helpers import *
from qti.aisw.converters.qnn_backend.custom_ops.core import *
from qti.aisw.converters.common.converter_ir import translation


OpAdapterMap = translation.TranslationBank()

class IRPaddingStrategies(object):
    """ Padding size strategies support in IR."""

    # No padding
    PADDING_SIZE_IMPLICIT_VALID = 0
    # Pad input so that output spatial size matches input. In case of odd total
    # pad value across a spatial dimension, the extra padding is place at the beginning.
    PADDING_SIZE_IMPLICIT_SAME_BEGIN = 1
    # Pad input so that output spatial size matches input. In case of odd total
    # pad value across a spatial dimension, the extra padding is place at the end.
    PADDING_SIZE_IMPLICIT_SAME_END = 2
    # padding values are applied only to the right-hand side of the input and floor operation
    # is used to calculate output dims.
    PADDING_SIZE_EXPLICIT_RIGHTHANDED = 3
    # padding values are explicitly specified by source framework and ceil operation is used
    # to calculate output dims
    PADDING_SIZE_EXPLICIT = 4
    # same as explicit, but floor operation is used to calculate output dims
    PADDING_SIZE_EXPLICIT_FLOOR = 5

    py_to_c = {
        PADDING_SIZE_IMPLICIT_VALID: ir_graph.PADDING_SIZE_IMPLICIT_VALID,
        PADDING_SIZE_IMPLICIT_SAME_BEGIN: ir_graph.PADDING_SIZE_IMPLICIT_SAME_BEGIN,
        PADDING_SIZE_IMPLICIT_SAME_END: ir_graph.PADDING_SIZE_IMPLICIT_SAME_END,
        PADDING_SIZE_EXPLICIT_RIGHTHANDED: ir_graph.PADDING_SIZE_EXPLICIT_RIGHTHANDED,
        PADDING_SIZE_EXPLICIT: ir_graph.PADDING_SIZE_EXPLICIT,
        PADDING_SIZE_EXPLICIT_FLOOR: ir_graph.PADDING_SIZE_EXPLICIT_FLOOR
    }


class Op(ABC):

    @property
    def TRANSLATION_KEY(self):
        raise NotImplementedError

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__dict__['attrs'] = {}
        return instance

    def __init__(self, name, type, num_outputs=1, **kwargs):
        self.c_op = ir_graph.IrOp(name if name is not None else "placeholder_aisw", type)
        self.num_outputs = num_outputs

        # This tracks the input buffer formats for which the current params & output buffer format are valid
        # Any change in the actual input formats detected during optimizations should result in either Permute Op
        # inserted or modification of the params and the output buffer format
        self.data_axis_formats = []

        self.macs = 0
        self.params_count = 0  # i.e size of weights

        # for facade c++
        self.input_tensors = []
        self.output_tensors = []

    def __repr__(self):
        return self.c_op.name

    @property
    def name(self):
        return self.c_op.name if self.c_op.name != "placeholder_aisw" else None

    @name.setter
    def name(self, name):
        self.c_op.name = name

    @property
    def type(self):
        return self.c_op.type

    def addattr(self, key, source, default, use_default_type=True):
        attr = source.get(key, default)
        # Use the default's type when value/type is not None
        if attr is None or type(default) is type(None):
            self.attrs[key] = attr
        else:
            if type(default) is np.ndarray or type(attr) is np.ndarray:
                self.attrs[key] = np.array(attr)
            elif type(default) is type:
                self.attrs[key] = attr
            elif use_default_type:
                try:
                    self.attrs[key] = type(default)(attr)
                except:
                    self.attrs[key] = np.dtype(attr)
            else:
                self.attrs[key] = attr

    def assertattr(self, key, source):
        if key in source:
            self.attrs[key] = source[key]
        else:
            raise KeyError("Op %s missing required argument %s" % (self.name, key))

    def hasattr(self, key):
        return key in self.list_params()

    def get_attrs_keyvalue(self, dtype, key):
        g_dtype_getfunctions = {
            ir_graph.QNN_DATATYPE_BOOL_8  : self.c_op.attrs.get_bool,
            ir_graph.QNN_DATATYPE_UINT_8  : self.c_op.attrs.get_uint8,
            ir_graph.QNN_DATATYPE_UINT_16 : self.c_op.attrs.get_uint16,
            ir_graph.QNN_DATATYPE_UINT_32 : self.c_op.attrs.get_uint32,
            ir_graph.QNN_DATATYPE_INT_8   : self.c_op.attrs.get_int8,
            ir_graph.QNN_DATATYPE_INT_16  : self.c_op.attrs.get_int16,
            ir_graph.QNN_DATATYPE_INT_32  : self.c_op.attrs.get_int32,
            ir_graph.QNN_DATATYPE_FLOAT_16: self.c_op.attrs.get_float,
            ir_graph.QNN_DATATYPE_FLOAT_32: self.c_op.attrs.get_float
        }
        return g_dtype_getfunctions[dtype](key)

    def g_dtype_addfunction(self, dtype, attrs):
        g_dtype_addfunctions = {
            np.dtype('bool')   : attrs.addBool,
            np.dtype('uint8')  : attrs.addUint8,
            np.dtype('uint16') : attrs.addUint16,
            np.dtype('uint32') : attrs.addUint32,
            np.dtype('int8')   : attrs.addInt8,
            np.dtype('int16')  : attrs.addInt16,
            np.dtype('int32')  : attrs.addInt32,
            np.dtype('float32'): attrs.addFloat
        }
        return g_dtype_addfunctions[dtype]

    def qnn_to_numpy_datatype(self, dtype):
        qnn_to_numpy_datatype = {
            ir_graph.QNN_DATATYPE_INT_8: np.dtype('int8'),
            ir_graph.QNN_DATATYPE_INT_16: np.dtype('int16'),
            ir_graph.QNN_DATATYPE_INT_32: np.dtype('int32'),
            ir_graph.QNN_DATATYPE_INT_64: np.dtype('int64'),
            ir_graph.QNN_DATATYPE_UINT_8: np.dtype('uint8'),
            ir_graph.QNN_DATATYPE_UINT_16: np.dtype('uint16'),
            ir_graph.QNN_DATATYPE_UINT_32: np.dtype('uint32'),
            ir_graph.QNN_DATATYPE_UINT_64: np.dtype('uint64'),
            ir_graph.QNN_DATATYPE_FLOAT_16: np.dtype('float16'),
            ir_graph.QNN_DATATYPE_FLOAT_32: np.dtype('float32'),
            ir_graph.QNN_DATATYPE_BOOL_8: np.dtype('bool')
        }
        return qnn_to_numpy_datatype[dtype]

    def to_qnn_dtype(self, value):
        to_qnn_dtype = {
            QnnDatatype.QNN_DATATYPE_INT_8.value: ir_graph.QNN_DATATYPE_INT_8,
            QnnDatatype.QNN_DATATYPE_INT_16.value: ir_graph.QNN_DATATYPE_INT_16,
            QnnDatatype.QNN_DATATYPE_INT_32.value: ir_graph.QNN_DATATYPE_INT_32,
            QnnDatatype.QNN_DATATYPE_INT_64.value: ir_graph.QNN_DATATYPE_INT_64,
            QnnDatatype.QNN_DATATYPE_UINT_8.value: ir_graph.QNN_DATATYPE_UINT_8,
            QnnDatatype.QNN_DATATYPE_UINT_16.value: ir_graph.QNN_DATATYPE_UINT_16,
            QnnDatatype.QNN_DATATYPE_UINT_32.value: ir_graph.QNN_DATATYPE_UINT_32,
            QnnDatatype.QNN_DATATYPE_UINT_64.value: ir_graph.QNN_DATATYPE_UINT_64,
            QnnDatatype.QNN_DATATYPE_FLOAT_16.value: ir_graph.QNN_DATATYPE_FLOAT_16,
            QnnDatatype.QNN_DATATYPE_FLOAT_32.value: ir_graph.QNN_DATATYPE_FLOAT_32,
            QnnDatatype.QNN_DATATYPE_BOOL_8.value: ir_graph.QNN_DATATYPE_BOOL_8
        }
        return to_qnn_dtype[value]

    def snpe_to_qnn_dtype(self, value):
        snpe_to_qnn_dtype = {
            'SNPE_UDO_DATATYPE_INT_8': ir_graph.QNN_DATATYPE_INT_8,
            'SNPE_UDO_DATATYPE_INT_16': ir_graph.QNN_DATATYPE_INT_16,
            'SNPE_UDO_DATATYPE_INT_32': ir_graph.QNN_DATATYPE_INT_32,
            'SNPE_UDO_DATATYPE_INT_64': ir_graph.QNN_DATATYPE_INT_64,
            'SNPE_UDO_DATATYPE_UINT_8': ir_graph.QNN_DATATYPE_UINT_8,
            'SNPE_UDO_DATATYPE_FIXED_8': ir_graph.QNN_DATATYPE_SFIXED_POINT_8,
            'SNPE_UDO_DATATYPE_FIXED_16': ir_graph.QNN_DATATYPE_SFIXED_POINT_16,
            'SNPE_UDO_DATATYPE_UINT_16': ir_graph.QNN_DATATYPE_UINT_16,
            'SNPE_UDO_DATATYPE_UINT_32': ir_graph.QNN_DATATYPE_UINT_32,
            'SNPE_UDO_DATATYPE_UINT_64': ir_graph.QNN_DATATYPE_UINT_64,
            'SNPE_UDO_DATATYPE_FLOAT_16': ir_graph.QNN_DATATYPE_FLOAT_16,
            'SNPE_UDO_DATATYPE_FLOAT_32': ir_graph.QNN_DATATYPE_FLOAT_32,
            'SNPE_UDO_DATATYPE_UINT_8': ir_graph.QNN_DATATYPE_BOOL_8
        }
        return snpe_to_qnn_dtype[value]

    def __getattr__(self, key):
        try:
            if key in self.__dict__['attrs']:
                return self.__dict__['attrs'][key]
            elif "c_op" in self.__dict__ and hasattr(self.__dict__["c_op"], key):
                # This will only work if key is defined as a property in pybind for this op
                return ABC.__getattribute__(self.__dict__["c_op"], key)
            else:
                return self.__dict__[key]
        except KeyError:
            raise AttributeError("Op %s has no attribute %s" % (self.name, key))

    def __setattr__(self, key, value):
        # Name, type are stored in the c_op for all ops, also attributes set in the c_op should be set there
        if key in ["name", "type"]:
            super(Op, self).__setattr__(key, value)
        # This will only work if key is defined as a property in pybind for this op
        elif ("c_op" in self.__dict__
              and self.__dict__["c_op"].attrs is not None
              and self.__dict__["c_op"].attrs.has(key)):
            setattr(self.__dict__["c_op"], key, value)
        # Next, prefer attributes which are stored in the Python Op attrs dict
        elif key in self.__dict__['attrs']:
            self.__dict__['attrs'][key] = value
        # Finally, store in Python __dict__
        else:
            self.__dict__[key] = value

    def encode(self):
        return {}

    @abstractmethod
    def infer_shape(self, input_shapes: list, input_axis_formats: list, num_outputs: int, axis_order: AxisOrder) -> list:
        raise NotImplementedError(
            "infer_shape for {} not implemented ".format(str(self.__class__.__name__)))

    def infer_shape_c_op_wrapper(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        # TODO: at this point this is really just needed because our unit-tests are not migrated to properly construct
        #       an op and its input tensors before inferring
        if len(self.c_op.inputs()) == 0:
            for i, shape in enumerate(input_shapes):
                # Add tensor if input shape is not null
                if i < len(input_axis_formats) and shape:
                    if input_axis_formats[i] not in list(AxisTracker.AxisFormat.ir_to_c_axis_format.keys()):
                        raise ValueError("Got unsupported axis order: {}".format(input_axis_formats[i]))
                    self.input_tensors.append(ir_graph.IrTensor("dummy_%d" % i, shape,
                                         ir_graph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
                                         AxisTracker.AxisFormat.ir_to_c_axis_format[input_axis_formats[i]]))
                else:
                    self.input_tensors.append(ir_graph.IrTensor("dummy_%d" % i, shape))
            self.c_op.set_input_tensors(self.input_tensors)

        if not self.c_op.is_canonicalized():
            self.c_op.canonicalize_op(num_outputs)
        return self.c_op.infer_output_shapes(AxisOrders.python_to_c_axis_orders(axis_order), num_outputs)

    def infer_datatype_c_op_wrapper(self, input_names, input_shapes, input_datatypes,
                                    input_axis_formats, num_outputs):
        input_tensors = []
        for i, input_name in enumerate(input_names):
            input_tensors.append(ir_graph.IrTensor(input_name, input_shapes[i], input_datatypes[i],
                                                   AxisTracker.AxisFormat.ir_to_c_axis_format[input_axis_formats[i]]))
        self.c_op.set_input_tensors(input_tensors)
        return self.c_op.infer_output_datatypes(num_outputs)

    def get_default_output_dtypes_c_op_wrapper(self, num_outputs):
        return self.c_op.get_default_output_dtypes(num_outputs)

    def macs_c_op_wrapper(self, input_shapes, output_shapes, axis_order):
        # TODO: at this point this is really just needed because our unit-tests are not migrated to properly construct
        #       an op and its input tensors before inferring
        if len(self.c_op.inputs()) == 0:
            for i, shape in enumerate(input_shapes):
                self.input_tensors.append(ir_graph.IrTensor("dummy_%d" % i, shape))
            self.c_op.set_input_tensors(self.input_tensors)
        if len(self.c_op.outputs()) == 0:
            for i, shape in enumerate(output_shapes):
                self.output_tensors.append(ir_graph.IrTensor("dummy_%d" % i, shape))
            self.c_op.set_output_tensors(self.output_tensors)
        return self.c_op.macs(AxisOrders.python_to_c_axis_orders(axis_order))

    def params_count_c_op_wrapper(self, input_shapes, output_shapes, axis_order):
        # TODO: at this point this is really just needed because our unit-tests are not migrated to properly construct
        #       an op and its input tensors before inferring
        if len(self.c_op.inputs()) == 0:
            for i, shape in enumerate(input_shapes):
                self.input_tensors.append(ir_graph.IrTensor("dummy_%d" % i, shape))
            self.c_op.set_input_tensors(self.input_tensors)
        if len(self.c_op.outputs()) == 0:
            for i, shape in enumerate(output_shapes):
                self.output_tensors.append(ir_graph.IrTensor("dummy_%d" % i, shape))
            self.c_op.set_output_tensors(self.output_tensors)
        return self.c_op.params_count(AxisOrders.python_to_c_axis_orders(axis_order))

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        pass

    @staticmethod
    def get_general_macs_val(output_shapes, native_dim_size=3):
        """
        Calculates the macs(multiply and accumulates) value for given Op for the general case
        :param output_shapes: the inferred output shapes for Op
        :param native_dim_size: the dimension to start at for calculating macs value (note: negative of value is used
                                to index the output_dim)
        :return the macs value for Op
        """
        native_output_dims = output_shapes[0][-native_dim_size:]
        return np.prod(native_output_dims)

    def populate_data_axis_formats(self, graph, input_buffers):
        self.data_axis_formats = [in_buf.axis_format for in_buf in input_buffers]

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        """
        Sets the axis format of the buffer using the input axis formats

        :param graph: IROpGraph object
        :param buf: The op_graph.Buffer class object for assigning axis format
        :param src_axis_order: the src framework axis order
        :param encodings: the Encodings passed by user for inputs. Used to determine type of network
        :param input_buffers: Input buffers from the graph, to the Op represented by self
        """

        ####
        # e.g. Assume there is a model converted from Keras to ONNX. It has a Softmax Op that has NSC input.
        # Since, it is an ONNX model, the src_axis_order is ONNX and the 1st if-check will fail for NSC input buffer
        # Next, the check for NSC format in TF axis order passes, and the output buffer also gets assigned NSC
        ####
        in_formats = [in_format for in_format in self.data_axis_formats
                      if in_format not in [AxisTracker.AxisFormat.ANY, AxisTracker.AxisFormat.NF]]
        if not in_formats:
            in_formats = self.data_axis_formats
        if not self.data_axis_formats:
            buf.populate_axis_format(src_axis_order, encodings)
        elif any([in_format == AxisTracker.AxisFormat.NONTRIVIAL for in_format in in_formats]):
            buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)
        elif any([in_format in src_axis_order.axis_formats for in_format in in_formats]):
            buf.populate_axis_format(src_axis_order, encodings)
        elif any([in_format in AxisOrders.TF.axis_formats for in_format in in_formats]):
            buf.populate_axis_format(AxisOrders.TF, encodings)
        elif any([in_format in AxisOrders.ONNX.axis_formats for in_format in in_formats]):
            buf.populate_axis_format(AxisOrders.ONNX, encodings)

        else:
            raise ValueError("Unsupported input_axis_formats {} for Node {} of type {}".format(self.data_axis_formats,
                                                                                               self,
                                                                                               self.type))

    def list_params(self: type):
        """ This gets instance variables of this class as key/value"""

        instance_vars = dict(self.__dict__)
        # above will get the attrs as {'attrs': {name1:val1...} instead we want to expand that
        del instance_vars['attrs']

        # set op attrs
        op_attrs = self.attrs
        instance_vars.update(op_attrs)

        # set c_op attrs
        if "c_op" in self.__dict__ \
                and self.__dict__["c_op"].attrs is not None:
            for attr_name in self.c_op.attrs.list_names():
                instance_vars.update({attr_name: getattr(self, attr_name)})
            del instance_vars['c_op']
        del instance_vars['input_tensors']
        del instance_vars['output_tensors']

        # capture class properties
        property_vars = {}
        for key, val in self.__class__.__dict__.items():
            if isinstance(val, property):
                property_vars.update({key: getattr(self, key)})

        instance_vars.update(property_vars)

        return instance_vars

    def is_equal(self, other_op):
        """
        Compares another op instance to current one based on type and attribute matching
        :param other_op: an op_adapter object
        :return: bool, msg. True if type and attr/params match, False otherwise. Plus message detailing what was
                            different
        """
        # instance equality check
        if not isinstance(other_op, self.__class__):
            return False, "{} is not an instance of current Op {}".format(other_op, self.__class__)

        # attr/param list equality check
        other_op_params = other_op.list_params()
        current_op_params = self.list_params()
        if not other_op_params.keys() == current_op_params.keys():
            return False, "Op adapter for {} not set with same attribute as current Op. Expected keys: {}. Got {}". \
                format(type(other_op.type), current_op_params.keys(), other_op_params.keys())
        # loop through attributes. Since we verified above both objects are same instance and have same attrs/params
        # we can use one to list all
        for attr_ in list(current_op_params.keys()):
            if attr_ == "c_op":
                if not other_op_params[attr_].attrs == current_op_params[attr_].attrs:
                    return False, "Attribute match error for Op: {} Attribute: {}. Expected {}, Got {} ".format(
                        str(other_op.type), attr_, str(current_op_params[attr_]),
                        str(other_op_params[attr_]))
            elif not translation_utils.compare_values(other_op_params[attr_],
                                                    current_op_params[attr_]):
                return False, "Attribute match error for Op: {} Attribute: {}. Expected {}, Got {} ".format(
                    str(other_op.type), attr_, str(current_op_params[attr_]),
                    str(other_op_params[attr_]))

        return True, "Op {} is equal to current Op instance".format(other_op)

    def __eq__(self, other_op):
        return self.is_equal(other_op)[0]

    def update_param_quant_overrides(self, graph, node):
        return


class InputOp(Op):
    TRANSLATION_KEY = 'input'
    LEGACY_TRANSLATION_KEY = 'input'

    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.assertattr('input_encoding_in', kargs)
        self.assertattr('input_encoding_out', kargs)
        self.assertattr('input_type', kargs)
        self.addattr('input_dtype', kargs, np.dtype("float32"))

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.shape[:]]


class ArgOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_ARG

    ir_to_legacy_type = {
        ir_graph.QNN_OP_ARGMAX: 'argmax',
        ir_graph.QNN_OP_ARGMIN: 'argmin'
    }

    attr_key_map = {
        ir_graph.QNN_OP_ARGMAX: {
            "axis": ir_graph.QNN_OP_ARGMAX_PARAM_AXIS,
            "keep_dims": ir_graph.QNN_OP_ARGMAX_PARAM_KEEP_DIMS
        },
        ir_graph.QNN_OP_ARGMIN: {
            "axis": ir_graph.QNN_OP_ARGMIN_PARAM_AXIS,
            "keep_dims": ir_graph.QNN_OP_ARGMIN_PARAM_KEEP_DIMS
        }
    }

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        arg_type = kargs[ir_graph.IR_OP_ARG_TYPE]
        axis_key = self.attr_key_map[arg_type]["axis"]
        keep_dims_key = self.attr_key_map[arg_type]["keep_dims"]

        attrs = ir_graph.IrAttributes()
        attrs.addString(ir_graph.IR_OP_ARG_TYPE, arg_type, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addInt32(axis_key, kargs[axis_key])
        if keep_dims_key in kargs:
            attrs.addBool(keep_dims_key, kargs[keep_dims_key])
        self.c_op = ir_graph.ArgOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        if self.keep_dims:
            buf.set_axis_format(self.data_axis_formats[0])
        else:
            buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class AxisAlignedBboxTransformOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # weights
        weights = kargs.get(ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM_PARAM_WEIGHTS)
        if weights is not None:
            weights_data = np.array(weights, dtype=np.float32)
            weights_tensor = ir_graph.IrStaticTensor(ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM_PARAM_WEIGHTS,
                                                     [len(weights_data)],
                                                     weights_data,
                                                     ir_graph.QNN_DATATYPE_FLOAT_32)
            attrs.add(ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM_PARAM_WEIGHTS, weights_tensor)

        self.c_op = ir_graph.AxisAlignedBboxTransformOp(name, attrs)

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce input buffer axis format to NONTRIVIAL
        for input_buffer in input_buffers:
            input_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class BatchnormOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_BATCHNORM
    LEGACY_TRANSLATION_KEY = 'batchnorm'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.c_op = ir_graph.BatchnormOp(name)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    # TODO: update once axis tracking supported
    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].set_axis_format(graph.src_axis_order.get_axis_format(len(input_buffers[0].shape)))
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def set_macs_params(self, input_shapes: list, output_shapes, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class BatchPermutationOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_BATCH_PERMUTATION
    LEGACY_TRANSLATION_KEY = 'batch_permutation'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        self.c_op = ir_graph.BatchPermutationOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
         return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class BatchToSpaceOp(Op):
    TRANSLATION_KEY = 'batch_to_space'
    LEGACY_TRANSLATION_KEY = 'batch_to_space'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('block_shape', kargs)
        self.addattr('crops', kargs, [[0, 0], [0, 0]])

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        input_batch, input_height, input_width, input_channel = axis_order.extract_2d_spatial_dims(
            input_shapes[0])
        output_batch = input_batch // (self.block_shape[0] * self.block_shape[1])
        output_height = input_height * self.block_shape[0] - (self.crops[0][0] + self.crops[0][1])
        output_width = input_width * self.block_shape[1] - (self.crops[1][0] + self.crops[1][1])
        output_shape = axis_order.format_2d_spatial_output_shape(batch_size=output_batch,
                                                                 channel=input_channel,
                                                                 height=output_height,
                                                                 width=output_width)
        return [output_shape]


class BoxDecoderOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_BOX_DECODER_TYPE
    LEGACY_TRANSLATION_KEY = "ssd"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addFloat(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_Y, kwargs.get(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_Y))
        attrs.addFloat(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_X, kwargs.get(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_X))
        attrs.addFloat(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_H, kwargs.get(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_H))
        attrs.addFloat(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_W, kwargs.get(ir_graph.IR_OP_BOX_DECODER_PARAM_SCALE_W))
        self.c_op = ir_graph.BoxDecoderOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class BoxWithNmsLimitOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # nms_kernel_method
        nms_kernel_method = kargs.get(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_KERNEL_METHOD)
        if nms_kernel_method is None:
            raise ValueError("nms_kernel_method attribute must be specified for BoxWithNmsLimitOp {}".format(name))
        attrs.addInt32(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_KERNEL_METHOD, nms_kernel_method)

        # nms_score_threshold
        nms_score_threshold = kargs.get(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_SCORE_THRESHOLD)
        if nms_score_threshold is None:
            raise ValueError("nms_score_threshold attribute must be specified for BoxWithNmsLimitOp {}".format(name))
        attrs.addFloat(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_SCORE_THRESHOLD, nms_score_threshold)

        # score_threshold
        score_threshold = kargs.get(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SCORE_THRESHOLD, 0.0)
        attrs.addFloat(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SCORE_THRESHOLD, score_threshold)

        # pre_nms_limit
        pre_nms_limit = kargs.get(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_PRE_NMS_LIMIT, -1)
        attrs.addInt32(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_PRE_NMS_LIMIT, pre_nms_limit)

        # iou_threshold
        iou_threshold = kargs.get(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_IOU_THRESHOLD)
        if iou_threshold is None:
            raise ValueError("iou_threshold attribute must be specified for BoxWithNmsLimitOp {}".format(name))
        attrs.addFloat(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_IOU_THRESHOLD, iou_threshold)

        # sigma
        sigma = kargs.get(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SIGMA)
        if sigma is None:
            raise ValueError("sigma attribute must be specified for BoxWithNmsLimitOp {}".format(name))
        attrs.addFloat(ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SIGMA, sigma)

        self.c_op = ir_graph.BoxWithNmsLimitOp(name, attrs)

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce input buffer axis format to NONTRIVIAL
        for input_buffer in input_buffers:
            input_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class BufferOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_BUFFER
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        buffer_size = kargs.get(ir_graph.QNN_OP_BUFFER_PARAM_BUFFER_SIZE)
        if buffer_size is None:
            raise ValueError("buffer_size attribute must be specified for BufferOp {}".format(name))
        attrs.addUint32(ir_graph.QNN_OP_BUFFER_PARAM_BUFFER_SIZE, buffer_size)

        buffer_dim = kargs.get(ir_graph.QNN_OP_BUFFER_PARAM_BUFFER_DIM)
        if buffer_dim is None:
            raise ValueError("buffer_dim attribute must be specified for BufferOp {}".format(name))
        attrs.addUint32(ir_graph.QNN_OP_BUFFER_PARAM_BUFFER_DIM, buffer_dim)

        stride = kargs.get(ir_graph.QNN_OP_BUFFER_PARAM_STRIDE, 1)
        attrs.addUint32(ir_graph.QNN_OP_BUFFER_PARAM_BUFFER_STRIDE, stride)

        mode =  kargs.get(ir_graph.QNN_OP_BUFFER_PARAM_MODE, ir_graph.QNN_OP_BUFFER_MODE_BLOCKING)
        attrs.addUint32(ir_graph.QNN_OP_BUFFER_PARAM_BUFFER_MODE, mode)

        self.c_op = ir_graph.BufferOp(name, attrs)

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class CastOp(Op):
    TRANSLATION_KEY = 'cast'
    LEGACY_TRANSLATION_KEY = 'cast'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('to_type', kargs)
        # TODO: change default to previous input_tensor datatype once tensor datatype is tracked in IR
        # from_type if not present populate it with empty string, it will be populated correctly in the backend
        self.addattr('from_type', kargs, '')
        if not isinstance(self.to_type, str):
            raise TypeError("Cast to_type is expected to be a str, received {}".format(type(self.to_type)))
        if not isinstance(self.from_type, str):
            raise TypeError("Cast from_type is expected to be a str, received {}".format(type(self.from_type)))
        # Override to float32 as largest float bitwidth because float64 is not yet supported
        if self.from_type == 'float64':
            self.from_type = 'float32'
        if self.to_type == 'float64':
            self.to_type = 'float32'

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return input_shapes[:num_outputs]

    def encode(self):
        return {"from_type": self.from_type, "to_type" : self.to_type}


class ChannelShuffleOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CHANNEL_SHUFFLE
    LEGACY_TRANSLATION_KEY = "channel_shuffle"
    GROUPED = "CHANNEL_SHUFFLE_GROUPED"

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # num_groups
        num_groups = kargs.get(ir_graph.QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS)
        if num_groups is None:
            raise ValueError("num_groups attributes must be specified for ChannelShuffleOp {}".format(name))
        attrs.addUint32(ir_graph.QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS, num_groups)

        # axis
        axis = kargs.get(ir_graph.QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS, -1)
        attrs.addInt32(ir_graph.QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS, axis)

        # TODO: Re-evaluate support for these. QNN has no support currently
        # self.addattr('shuffle_mode', kargs, self.GROUPED)

        # TODO Remove name placeholder_aisw once name is read-only
        self.c_op = ir_graph.ChannelShuffleOp(name if name is not None else "placeholder_aisw", attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class CollectRpnProposalsOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS
    LEGACY_TRANSLATION_KEY = 'collect_rpn_proposals'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        rpn_min_level = kargs.get(ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MIN_LEVEL)
        if rpn_min_level is not None:
            attrs.addUint32(ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MIN_LEVEL, rpn_min_level)

        rpn_max_level = kargs.get(ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MAX_LEVEL)
        if rpn_max_level is not None:
            attrs.addUint32(ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MAX_LEVEL, rpn_max_level)

        post_nms_top = kargs.get(ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_POST_NMS_TOP)
        if post_nms_top is not None:
            attrs.addUint32(ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_POST_NMS_TOP, post_nms_top)

        self.c_op = ir_graph.CollectRpnProposalsOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce input buffer axis format to NONTRIVIAL
        for input_buffer in input_buffers:
            input_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class ColorTransformOp(Op):
    TRANSLATION_KEY = 'color_transform'
    LEGACY_TRANSLATION_KEY = 'color_transform'

    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.assertattr('input_encoding_in', kargs)
        self.assertattr('input_encoding_out', kargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.shape[:]]


class ConcatOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CONCAT
    LEGACY_TRANSLATION_KEY = 'concat'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addInt32(ir_graph.QNN_OP_CONCAT_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_CONCAT_PARAM_AXIS))
        self.c_op = ir_graph.ConcatOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Update shape and axis format of Constant nodes to match the other input
        input_nodes = [buf.producer for buf in input_buffers]
        if any([isinstance(node.op, ConstantOp) for node in input_nodes]):
            const_node_idx = {i for i,node in enumerate(input_nodes) if isinstance(node.op, ConstantOp)}
            non_const_node_idx = {i for i in range(0, len(input_buffers))}
            non_const_node_idx = non_const_node_idx - const_node_idx
            non_const_axis_formats = [input_buffers[i].axis_format for i in non_const_node_idx]
            # All non-constant axis-format must be same if we want to fix the axis-foramt of Constant
            if all([axis_format == non_const_axis_formats[0] for axis_format in non_const_axis_formats]) and \
                    non_const_axis_formats[0] not in [AxisTracker.AxisFormat.NONTRIVIAL, AxisTracker.AxisFormat.ANY, AxisTracker.AxisFormat.NOT_YET_DEFINED]:
                for i in const_node_idx:
                    # avoid the infinite loop due to the shared buffer
                    if input_buffers[i].axis_format == non_const_axis_formats[0]:
                        continue
                    input_buffers[i].axis_format = non_const_axis_formats[0]
                    # call populate_data_axis_format for consumer of any buffer that had the format overridden
                    for consumer in input_buffers[i].consumers:
                        if consumer.op != self:
                            consumer.op.populate_data_axis_formats(graph, graph.get_input_buffers(consumer))

        # comfirm that all input buffers have same rank
        in_ranks = [len(in_buf.shape) for in_buf in input_buffers]
        first_in_rank = in_ranks[0]
        if not all([first_in_rank == in_rank for in_rank in in_ranks]):
            raise ValueError("Input ranks for {} Op are not identical".format(self.name))

        in_axis_formats = [buf.axis_format for buf in input_buffers]
        input_nodes = [buf.producer for buf in input_buffers]
        NT_inds = {i for i, axis_format in enumerate(in_axis_formats) if axis_format == AxisTracker.AxisFormat.NONTRIVIAL}
        non_NT_axis_inds = {i for i in range(0, len(input_buffers))}
        non_NT_axis_inds = non_NT_axis_inds - NT_inds
        known_axis_formats = []
        for i in non_NT_axis_inds:
            known_axis_format = in_axis_formats[i]
            if not isinstance(input_nodes[i].op, ConstantOp) and \
                    known_axis_format not in [AxisTracker.AxisFormat.ANY, AxisTracker.AxisFormat.NOT_YET_DEFINED]:
                known_axis_formats.append(known_axis_format)

        # check whether it's suitable to do backward-propagation
        if NT_inds and known_axis_formats and \
                all([known_axis_formats[0] == axis_format for axis_format in known_axis_formats]):
            # Concat has multiple input buffers
            # One is NONTRIVIAL, others are known axis-format (neither ANY nor NOT_YET_DEFINED)
            # For example, assume in_axis_formats = [NCS, NONTRIVIAL, NONTRIVIAL]
            # In this case, we can backward-propagate NCS to its input buffers
            # and change axis_formats for all input buffers to NCS
            for NT_idx in NT_inds:
                input_buffers[NT_idx].axis_format = known_axis_formats[0]

        self.data_axis_formats = [in_buf.axis_format for in_buf in input_buffers]


class ConstantOp(Op):
    TRANSLATION_KEY = 'constant'
    LEGACY_TRANSLATION_KEY = 'constant'

    def __init__(self, name, tensor, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

        self.tensor = self.downcast_dtype_64bit_to_32bit(tensor, tensor.dtype)

        self.addattr('quantizable', kargs, None)
        # determine quantizable property from dtype if property was no explicitly set
        if self.quantizable is None:
            self.quantizable = False
            if self.tensor.dtype in [np.float32, np.float64]:
                self.quantizable = True

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [list(self.tensor.shape)]

    def update_param_quant_overrides(self, graph, node):
        if graph.has_user_quantization_overrides() and self.name in graph.user_quantization_overrides['param_encodings']:
            graph.user_quantization_overrides['activation_encodings'][self.name] = \
                graph.user_quantization_overrides['param_encodings'][self.name]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.params_count = np.prod(self.tensor.shape)

    def downcast_dtype_64bit_to_32bit(self, tensor, tensor_dtype):
        numpy_dtype_downcast = {
            np.dtype('int64'): np.int32,
            np.dtype('uint64'): np.uint32,
            np.dtype('float64'): np.float32,
        }
        if tensor_dtype in numpy_dtype_downcast:
            tensor_dtype = numpy_dtype_downcast[tensor_dtype]

        if tensor_dtype != tensor.dtype:
            return tensor.astype(tensor_dtype)
        else:
            return tensor

class ConvertOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CONVERT
    LEGACY_TRANSLATION_KEY = 'convert'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # Include bool indicating input data type can change between calls to execute
        attrs.addBool(ir_graph.QNN_OP_CONVERT_PARAM_DYNAMIC_INPUT_DATA, kwargs.get(ir_graph.QNN_OP_CONVERT_PARAM_DYNAMIC_INPUT_DATA, False))
        # Include bool indicating output data type can change between calls to execute
        attrs.addBool(ir_graph.QNN_OP_CONVERT_PARAM_DYNAMIC_OUTPUT_DATA, kwargs.get(ir_graph.QNN_OP_CONVERT_PARAM_DYNAMIC_OUTPUT_DATA, False))

        self.c_op = ir_graph.ConvertOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

class Conv1dOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_CONV_1D
    LEGACY_TRANSLATION_KEY = 'convolution1d'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridex = kwargs.get('stridex')
        if stridex is None:
            raise ValueError("Stride attributes must be specified for Conv1dOp {}".format(name))
        stride_data = np.array([stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_1D_PARAM_STRIDE,
                                         [1],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_CONV_1D_PARAM_STRIDE, stride)

        # Pad amount
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if padx_before is None or padx_after is None:
            raise ValueError("Pad amount attribute must be specified for Conv1dOp {}".format(name))
        pad_data = np.array([padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_1D_PARAM_PAD_AMOUNT,
                                      [2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_CONV_1D_PARAM_PAD_AMOUNT, pad)

        # Dilation
        dilation_data = np.array([kwargs.get('dilationx', 1)],
                                 dtype=np.uint32)
        dilation = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_1D_PARAM_DILATION,
                                           [1],
                                           dilation_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_CONV_1D_PARAM_DILATION, dilation)

        # Groups
        attrs.addUint32(ir_graph.IR_OP_CONV_1D_PARAM_GROUP, kwargs.get("groups", 1))

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_CONV_1D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        if padding_size_strategy:
            attrs.addUint8(ir_graph.IR_OP_CONV_1D_PARAM_PADDING_SIZE_STRATEGY,
                           padding_size_strategy,
                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_CONV_1D_PARAM_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_CONV_1D_PARAM_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.Conv1dOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        input_buffers[0].axis_format = src_axis_order.get_axis_format(len(input_buffers[0].shape))
        input_buffers[1].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        self.data_axis_formats = [in_buf.axis_format for in_buf in input_buffers]
        buf.populate_axis_format(src_axis_order, encodings)


class Conv2dOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CONV_2D
    LEGACY_TRANSLATION_KEY = 'convolution'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridey = kwargs.get('stridey')
        stridex = kwargs.get('stridex')
        if stridey is None or stridex is None:
            raise ValueError("Stride attributes must be specified for Conv2dOp {}".format(name))
        stride_data = np.array([stridey, stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CONV_2D_PARAM_STRIDE,
                                         [2],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CONV_2D_PARAM_STRIDE, stride)

        # Pad amount
        pady_before = kwargs.get('pady_before')
        pady_after = kwargs.get('pady_after')
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if pady_before is None or pady_after is None or padx_before is None or padx_after is None:
            raise ValueError("Pad amount attribute must be specified for Conv2dOp {}".format(name))
        pad_data = np.array([pady_before, pady_after, padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                                      [2, 2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, pad)

        # Dilation
        dilation_data = np.array([kwargs.get('dilationy', 1), kwargs.get('dilationx', 1)],
                                 dtype=np.uint32)
        dilation = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CONV_2D_PARAM_DILATION,
                                           [2],
                                           dilation_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CONV_2D_PARAM_DILATION, dilation)

        # Groups
        attrs.addUint32(ir_graph.QNN_OP_CONV_2D_PARAM_GROUP, kwargs.get("groups", 1))

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_CONV_2D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        attrs.addUint8(ir_graph.IR_OP_CONV_2D_PARAM_PADDING_SIZE_STRATEGY,
                       padding_size_strategy,
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_CONV_2D_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_CONV_2D_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.Conv2dOp(name, attrs)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        input_buffers[1].axis_format = graph.src_axis_order.conv2d_weights_format
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)

    def update_param_quant_overrides(self, graph, node):
        # Handle cases where FakeQuant encodings have been added directly to the quantization_params
        if graph.quantization_params and self.name in graph.quantization_params:
            param_encodings = graph.quantization_params[self.name]['param_encodings']
            for encoding in param_encodings:
                if encoding['name'] == 'weights':
                    encoding_producer = graph.get_input_buffers(node)[1].producer
                elif len(node.input_names) == 3 and encoding['name'] == 'bias':
                    encoding_producer = graph.get_input_buffers(node)[2].producer
                else:
                    raise ValueError("Encoding for node {} is unhandled.".format(node.op.name))

                output_encodings={"name": encoding_producer.op.name,
                                  "bw": encoding['bw'],
                                  "min": encoding['min'],
                                  "max": encoding['max'],
                                  "overridden": True}
                if "axis" in encoding:
                    output_encodings.update({"axis":encoding['axis']})
                if "is_symmetric" in encoding:
                    output_encodings.update({"is_symmetric":encoding['is_symmetric']})
                graph.add_quantization_params(encoding_producer.op.name, output_encodings=output_encodings)


class Conv3dOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CONV_3D
    LEGACY_TRANSLATION_KEY = 'convolution3d'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridez = kwargs.get('stridez')
        stridey = kwargs.get('stridey')
        stridex = kwargs.get('stridex')
        if stridey is None or stridex is None or stridez is None:
            raise ValueError("Stride attributes must be specified for Conv3dOp {}".format(name))
        stride_data = np.array([stridez, stridey, stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CONV_3D_PARAM_STRIDE,
                                         [3],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CONV_3D_PARAM_STRIDE, stride)

        # Pad amount
        padz_before = kwargs.get('padz_before')
        padz_after = kwargs.get('padz_after')
        pady_before = kwargs.get('pady_before')
        pady_after = kwargs.get('pady_after')
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if pady_before is None or pady_after is None or padx_before is None or padx_after is None or padz_before is None or padz_after is None:
            raise ValueError("Pad amount attribute must be specified for Conv3dOp {}".format(name))
        pad_data = np.array([padz_before, padz_after, pady_before, pady_after, padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CONV_3D_PARAM_PAD_AMOUNT,
                                      [3, 2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CONV_3D_PARAM_PAD_AMOUNT, pad)

        # Dilation
        dilation_data = np.array([kwargs.get('dilationz', 1), kwargs.get('dilationy', 1), kwargs.get('dilationx', 1)],
                                 dtype=np.uint32)
        dilation = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CONV_3D_PARAM_DILATION,
                                           [3],
                                           dilation_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CONV_3D_PARAM_DILATION, dilation)

        # Groups
        attrs.addUint32(ir_graph.QNN_OP_CONV_3D_PARAM_GROUP, kwargs.get("groups", 1))

        # Reuse sparse indices
        attrs.addBool(ir_graph.QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES, kwargs.get("reuse_sparse_indicies", False))

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_CONV_3D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        attrs.addUint8(ir_graph.IR_OP_CONV_3D_PARAM_PADDING_SIZE_STRATEGY,
                       padding_size_strategy,
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_CONV_3D_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_CONV_3D_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.Conv3dOp(name, attrs)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        input_buffers[1].axis_format = graph.src_axis_order.conv3d_weights_format
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def update_param_quant_overrides(self, graph, node):
        # Handle cases where FakeQuant encodings have been added directly to the quantization_params
        if graph.quantization_params and self.name in graph.quantization_params:
            param_encodings = graph.quantization_params[self.name]['param_encodings']
            for encoding in param_encodings:
                if encoding['name'] == 'weights':
                    encoding_producer = graph.get_input_buffers(node)[1].producer
                elif len(node.input_names) == 3 and encoding['name'] == 'bias':
                    encoding_producer = graph.get_input_buffers(node)[2].producer
                else:
                    raise ValueError("Encoding for node {} is unhandled.".format(node.op.name))

                output_encodings={"name": encoding_producer.op.name,
                                  "bw": encoding['bw'],
                                  "min": encoding['min'],
                                  "max": encoding['max'],
                                  "overridden": True}
                if "axis" in encoding:
                    output_encodings.update({"axis":encoding['axis']})
                if "is_symmetric" in encoding:
                    output_encodings.update({"is_symmetric":encoding['is_symmetric']})
                graph.add_quantization_params(encoding_producer.op.name, output_encodings=output_encodings)


class CreateSparseOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CREATE_SPARSE
    LEGACY_TRANSLATION_KEY = 'create_sparse'

    def __init__(self, name, shape, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        self.c_op = ir_graph.CreateSparseOp(name, attrs)
        self.shape = shape

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.shape]


class CropAndResizeOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CROP_AND_RESIZE
    LEGACY_TRANSLATION_KEY = 'crop_and_resize'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # resize_dims
        resize_dims = kwargs.get(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_RESIZE_DIMS)
        if resize_dims is None:
            raise ValueError("resize_dims attributes must be specified for CropAndResizeOp {}".format(name))
        resize_dims_data = np.array(resize_dims, dtype=np.uint32)
        resize_dims_tensor = ir_graph.IrStaticTensor(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_RESIZE_DIMS,
                                                     [len(resize_dims_data)],
                                                     resize_dims_data,
                                                     ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_RESIZE_DIMS, resize_dims_tensor)

        # interpolation_mode
        interpolation_mode = kwargs.get(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_INTERPOLATION_MODE,
                                        ir_graph.QNN_OP_CROP_AND_RESIZE_INTERPOLATION_MODE_BILINEAR)
        attrs.addUint32(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_INTERPOLATION_MODE, interpolation_mode)

        # extrapolation_value
        extrapolation_value = kwargs.get(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_EXTRAPOLATION_VALUE, 0.0)
        attrs.addFloat(ir_graph.QNN_OP_CROP_AND_RESIZE_PARAM_EXTRAPOLATION_VALUE, extrapolation_value)

        # TODO: Re-evaluate support for these. QNN has no support currently
        # self.assertattr("num_boxes", kwargs)

        self.c_op = ir_graph.CropAndResizeOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class CumSumOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_CUMULATIVE_SUM
    LEGACY_TRANSLATION_KEY = 'cumsum'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        # validate the inputs
        if kwargs.get(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_AXIS) < 0:
            raise ValueError("axis expected to be positive, received {}".format(type(self.axis)))

        attrs = ir_graph.IrAttributes()
        attrs.addUint32(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_AXIS))
        attrs.addBool(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE, kwargs.get(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE, False))
        attrs.addBool(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE, kwargs.get(ir_graph.QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE, False))

        self.c_op = ir_graph.CumSumOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class CustomOp(Op):
    TRANSLATION_KEY = 'custom'
    LEGACY_TRANSLATION_KEY = 'custom'

    def __init__(self,
                 name,
                 package_name,
                 custom_type,
                 inputs,
                 outputs,
                 axis_orders,
                 output_dims,
                 scalar_params,
                 tensor_params,
                 converter_op_package_lib=None):

        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis_orders = axis_orders
        self.outputs = outputs
        attrs = ir_graph.IrAttributes()
        self.converter_op_package_lib = converter_op_package_lib

        # output dims
        for index, out_dim in enumerate(output_dims):
            out_dim = np.array(out_dim, dtype=np.uint32)
            out_dim = ir_graph.IrStaticTensor("output_dim"+str(index),
                                              out_dim.shape,
                                              out_dim,
                                              ir_graph.QNN_DATATYPE_UINT_32)
            attrs.add("output_dim" + str(index), out_dim)

        # custom type
        attrs.addString("custom_type",
                        custom_type)

        # package name
        attrs.addString("package_name",
                        package_name)

        # converter_op_package_lib
        # TODO: We remove converter_op_package_lib and output_dim_i from the attrs
        #  in the CustomOp's inferOutputShapes method since we don't want
        #  extra unwanted params in OpConfig when we create one from IrOp
        #  figure out an alternative way to do this instead of adding the
        #  attributes in python and removing it in cpp function call
        if converter_op_package_lib:
            attrs.addString("converter_op_package_lib",
                            converter_op_package_lib)

        # tensor params
        for param_name, param_info in tensor_params.items():
            # TODO: revisit after SNPE-UDO and Custom Op alignment [AISW-48484]
            if isinstance(param_info["data_type"], str):
                dtype = self.qnn_to_numpy_datatype(self.snpe_to_qnn_dtype(param_info["data_type"]))
                data = np.array(param_info['data'], dtype=dtype)
                data = ir_graph.IrStaticTensor(param_name,
                                               param_info['dimensions'],
                                               data,
                                               self.snpe_to_qnn_dtype(param_info["data_type"]))
            else:
                dtype = self.qnn_to_numpy_datatype(self.to_qnn_dtype(param_info["data_type"].value))
                data = np.array(param_info['data'], dtype=dtype)
                data = ir_graph.IrStaticTensor(param_name,
                                               param_info['dimensions'],
                                               data,
                                               self.to_qnn_dtype(param_info["data_type"].value))
            attrs.add(param_name, data)

        # scalar params
        for param_name, param_info in  scalar_params.items():
            # TODO: revisit after SNPE-UDO and Custom Op alignment [AISW-48484]
            if isinstance(param_info["data_type"], str):
                dtype = self.qnn_to_numpy_datatype(self.snpe_to_qnn_dtype(param_info["data_type"]))
            else:
                dtype = self.qnn_to_numpy_datatype(self.to_qnn_dtype(param_info["data_type"].value))
            add_attr = self.g_dtype_addfunction(dtype, attrs)
            add_attr(param_name, param_info['data'])

        self.c_op = ir_graph.CustomOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def infer_datatype(self, input_names, input_shapes, input_datatypes, input_axis_formats, num_outputs):
        return self.infer_datatype_c_op_wrapper(input_names, input_shapes, input_datatypes,
                                                input_axis_formats, num_outputs)

    def __getattr__(self, key):
        if key in self.c_op.attrs.list_names() and key not in ['output_dims', 'package_name', 'custom_type']:
            if self.c_op.attrs.get_attr_type(key) == ir_graph.Qnn_ParamType_t.QNN_PARAMTYPE_SCALAR:
                dtype = self.c_op.attrs.get_data_type(key)
                return self.get_attrs_keyvalue(dtype, key)
            else:
                return np.array(self.c_op.attrs.get_static_tensor_data(key))
        else:
            return super(CustomOp, self).__getattr__(key)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        # if the axis order has been defined then we keep the format set by the CustomOp object.
        # Otherwise, the axis format will be set according to framework AxisOrder class using
        # the buffer rank when we call populate axis format.
        if self.axis_orders[buf.name] == 'NOT_YET_DEFINED':
            super().populate_axis_format(graph, buf, src_axis_order, encodings, input_buffers)
            self.axis_orders[buf.name] = buf.axis_format
        else:
            buf.axis_format = self.axis_orders[buf.name]


class TransposeConv1dOp(Op):
    TRANSLATION_KEY = "TransposeConv1d"
    LEGACY_TRANSLATION_KEY = 'TransposeConv1d'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridex = kwargs.get('stridex')
        if stridex is None:
            raise ValueError("Stride attributes must be specified for TransposeConv1dOp {}".format(name))
        stride_data = np.array([stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_STRIDE,
                                         [1],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_STRIDE, stride)

        # Pad amount
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if padx_before is None or padx_after is None:
            raise ValueError("Pad amount attribute must be specified for TransposeConv1dOp {}".format(name))
        pad_data = np.array([padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_PAD_AMOUNT,
                                      [2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_PAD_AMOUNT, pad)

        # Groups
        attrs.addUint32(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_GROUP, kwargs.get("groups", 1))

        # Output padding
        output_padding_data = np.array([kwargs.get('output_paddingx', 0)], dtype=np.uint32)
        output_padding = ir_graph.IrStaticTensor(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_OUTPUT_PADDING,
                                                 [1],
                                                 output_padding_data,
                                                 ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_OUTPUT_PADDING, output_padding)

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        attrs.addUint8(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_PADDING_SIZE_STRATEGY,
                       padding_size_strategy,
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Output size
        output_size_data = np.array([kwargs.get('output_width', 0)], dtype=np.uint32)
        output_size = ir_graph.IrStaticTensor(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_OUTPUT_SIZE,
                                              [1],
                                              output_size_data,
                                              ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_TRANSPOSE_CONV_1D_PARAM_OUTPUT_SIZE,
                  output_size,
                  ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_TRANSPOSE_CONV_1D_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_TRANSPOSE_CONV_1D_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.TransposeConv1dOp(name, attrs)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        input_buffers[1].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def update_param_quant_overrides(self, graph, node):
        # Handle cases where FakeQuant encodings have been added directly to the quantization_params
        if graph.quantization_params and self.name in graph.quantization_params:
            param_encodings = graph.quantization_params[self.name]['param_encodings']
            for encoding in param_encodings:
                if encoding['name'] == 'weights':
                    encoding_producer = graph.get_input_buffers(node)[1].producer
                elif len(node.input_names) == 3 and encoding['name'] == 'bias':
                    encoding_producer = graph.get_input_buffers(node)[2].producer
                else:
                    raise ValueError("Encoding for node {} is unhandled.".format(node.op.name))

                output_encodings={"name": encoding_producer.op.name,
                                  "bw": encoding['bw'],
                                  "min": encoding['min'],
                                  "max": encoding['max'],
                                  "overridden": True}
                if "axis" in encoding:
                    output_encodings.update({"axis":encoding['axis']})
                if "is_symmetric" in encoding:
                    output_encodings.update({"is_symmetric":encoding['is_symmetric']})
                graph.add_quantization_params(encoding_producer.op.name, output_encodings=output_encodings)


class TransposeConv2dOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_TRANSPOSE_CONV_2D
    LEGACY_TRANSLATION_KEY = 'deconvolution'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridey = kwargs.get('stridey')
        stridex = kwargs.get('stridex')
        if stridey is None or stridex is None:
            raise ValueError("Stride attributes must be specified for TransposeConv2dOp {}".format(name))
        stride_data = np.array([stridey, stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE,
                                         [2],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE, stride)

        # Pad amount
        pady_before = kwargs.get('pady_before')
        pady_after = kwargs.get('pady_after')
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if pady_before is None or pady_after is None or padx_before is None or padx_after is None:
            raise ValueError("Pad amount attribute must be specified for TransposeConv2dOp {}".format(name))
        pad_data = np.array([pady_before, pady_after, padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT,
                                      [2, 2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT, pad)

        # Groups
        attrs.addUint32(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP, kwargs.get("groups", 1))

        # Output padding
        output_padding_data = np.array([kwargs.get('output_paddingy', 0), kwargs.get('output_paddingx', 0)], dtype=np.uint32)
        output_padding = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING,
                                                 [2],
                                                 output_padding_data,
                                                 ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING, output_padding)

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        attrs.addUint8(ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_PADDING_SIZE_STRATEGY,
                       padding_size_strategy,
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Output size
        output_size_data = np.array([kwargs.get('output_height', 0), kwargs.get('output_width', 0)], dtype=np.uint32)
        output_size = ir_graph.IrStaticTensor(ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_SIZE,
                                              [2],
                                              output_size_data,
                                              ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_SIZE,
                  output_size,
                  ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_TRANSPOSE_CONV_2D_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_TRANSPOSE_CONV_2D_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.TransposeConv2dOp(name, attrs)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        input_buffers[1].axis_format = graph.src_axis_order.deconv2d_weights_format
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def update_param_quant_overrides(self, graph, node):
        # Handle cases where FakeQuant encodings have been added directly to the quantization_params
        if graph.quantization_params and self.name in graph.quantization_params:
            param_encodings = graph.quantization_params[self.name]['param_encodings']
            for encoding in param_encodings:
                if encoding['name'] == 'weights':
                    encoding_producer = graph.get_input_buffers(node)[1].producer
                elif len(node.input_names) == 3 and encoding['name'] == 'bias':
                    encoding_producer = graph.get_input_buffers(node)[2].producer
                else:
                    raise ValueError("Encoding for node {} is unhandled.".format(node.op.name))

                output_encodings={"name": encoding_producer.op.name,
                                  "bw": encoding['bw'],
                                  "min": encoding['min'],
                                  "max": encoding['max'],
                                  "overridden": True}
                if "axis" in encoding:
                    output_encodings.update({"axis":encoding['axis']})
                if "is_symmetric" in encoding:
                    output_encodings.update({"is_symmetric":encoding['is_symmetric']})
                graph.add_quantization_params(encoding_producer.op.name, output_encodings=output_encodings)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class TransposeConv3dOp(TransposeConv2dOp):
    TRANSLATION_KEY = ir_graph.QNN_OP_TRANSPOSE_CONV_3D
    LEGACY_TRANSLATION_KEY = "transposeconv3d"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridez = kwargs.get('stridez')
        stridey = kwargs.get('stridey')
        stridex = kwargs.get('stridex')
        stride_data = np.array([stridez, stridey, stridex], dtype=np.uint32)
        if any(stride is None for stride in stride_data):
            raise ValueError("Stride attributes must be specified for TransposeConv3dOp {}".format(name))
        stride = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE,
                                         [3],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE, stride)

        # Pad amount
        padz_before = kwargs.get('padz_before')
        padz_after = kwargs.get('padz_after')
        pady_before = kwargs.get('pady_before')
        pady_after = kwargs.get('pady_after')
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        pad_data = np.array([padz_before, padz_after, pady_before, pady_after, padx_before, padx_after], dtype=np.uint32)
        if any(pad is None for pad in pad_data):
            raise ValueError("Pad amount attribute must be specified for TransposeConv3dOp {}".format(name))
        pad = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT,
                                      [3, 2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT, pad)

        # Dilation
        dilation_data = np.array([kwargs.get('dilationz', 1), kwargs.get('dilationy', 1), kwargs.get('dilationx', 1)], dtype=np.uint32)
        dilation = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION,
                                           [3],
                                           dilation_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION, dilation)

        # Groups
        attrs.addUint32(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_GROUP, kwargs.get("groups", 1))

        # Output padding
        output_padding_data = np.array([kwargs.get('output_paddingz', 0), kwargs.get('output_paddingy', 0), kwargs.get('output_paddingx', 0)], dtype=np.uint32)
        output_padding = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_PADDING,
                                                 [3],
                                                 output_padding_data,
                                                 ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_PADDING, output_padding)

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        attrs.addUint8(ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_PADDING_SIZE_STRATEGY,
                       padding_size_strategy,
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Output size
        output_size_data = np.array([kwargs.get('output_depth', 0), kwargs.get('output_height', 0), kwargs.get('output_width', 0)], dtype=np.uint32)
        output_size = ir_graph.IrStaticTensor(ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_SIZE,
                                              [3],
                                              output_size_data,
                                              ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_SIZE,
                  output_size,
                  ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_TRANSPOSE_CONV_3D_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_TRANSPOSE_CONV_3D_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.TransposeConv3dOp(name, attrs)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        input_buffers[1].axis_format = graph.src_axis_order.deconv3d_weights_format
        self.data_axis_formats = [in_buf.axis_format for in_buf in input_buffers]


class DepthwiseConv1dOp(Conv1dOp):
    TRANSLATION_KEY = ir_graph.IR_OP_DEPTH_WISE_CONV_1D
    LEGACY_TRANSLATION_KEY = 'depthwise_conv1d'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridex = kwargs.get('stridex')
        if stridex is None :
            raise ValueError("Stride attributes must be specified for DepthwiseConv1dOp {}".format(name))
        stride_data = np.array([stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_STRIDE,
                                         [1],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_STRIDE, stride)

        # Pad amount
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if padx_before is None or padx_after is None :
            raise ValueError("Pad amount attribute must be specified for DepthwiseConv1dOp {}".format(name))
        pad_data = np.array([padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_PAD_AMOUNT,
                                      [2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_PAD_AMOUNT, pad)

        # Dilation
        dilation_data = np.array([kwargs.get('dilationx', 1)],
                                 dtype=np.uint32)
        dilation = ir_graph.IrStaticTensor(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_DILATION,
                                           [1],
                                           dilation_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_DILATION, dilation)

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        if padding_size_strategy:
            attrs.addUint8(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_PADDING_SIZE_STRATEGY,
                           padding_size_strategy,
                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_DEPTH_WISE_CONV_1D_PARAM_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.DepthwiseConv1dOp(name, attrs)


class DepthwiseConv2dOp(Conv2dOp):
    TRANSLATION_KEY = ir_graph.QNN_OP_DEPTH_WISE_CONV_2D
    LEGACY_TRANSLATION_KEY = 'depthwise_convolution'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()

        # Stride
        stridey = kwargs.get('stridey')
        stridex = kwargs.get('stridex')
        if stridey is None or stridex is None:
            raise ValueError("Stride attributes must be specified for DepthwiseConv2dOp {}".format(name))
        stride_data = np.array([stridey, stridex], dtype=np.uint32)
        stride = ir_graph.IrStaticTensor(ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE,
                                         [2],
                                         stride_data,
                                         ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE, stride)

        # Pad amount
        pady_before = kwargs.get('pady_before')
        pady_after = kwargs.get('pady_after')
        padx_before = kwargs.get('padx_before')
        padx_after = kwargs.get('padx_after')
        if pady_before is None or pady_after is None or padx_before is None or padx_after is None:
            raise ValueError("Pad amount attribute must be specified for DepthwiseConv2dOp {}".format(name))
        pad_data = np.array([pady_before, pady_after, padx_before, padx_after], dtype=np.uint32)
        pad = ir_graph.IrStaticTensor(ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT,
                                      [2, 2],
                                      pad_data,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT, pad)

        # Dilation
        dilation_data = np.array([kwargs.get('dilationy', 1), kwargs.get('dilationx', 1)],
                                 dtype=np.uint32)
        dilation = ir_graph.IrStaticTensor(ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_DILATION,
                                           [2],
                                           dilation_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_DILATION, dilation)

        # Padding size strategy
        padding_size_strategy = kwargs.get(ir_graph.IR_OP_CONV_2D_PARAM_PADDING_SIZE_STRATEGY,
                                           ir_graph.PADDING_SIZE_EXPLICIT_FLOOR)
        if padding_size_strategy:
            attrs.addUint8(ir_graph.IR_OP_CONV_2D_PARAM_PADDING_SIZE_STRATEGY,
                           padding_size_strategy,
                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Bias op name
        bias_op_name = kwargs.get(ir_graph.IR_OP_CONV_2D_BIAS_OP_NAME, None)
        if bias_op_name:
            attrs.addString(ir_graph.IR_OP_CONV_2D_BIAS_OP_NAME,
                            bias_op_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.DepthwiseConv2dOp(name, attrs)


class DepthToSpaceOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_DEPTH_TO_SPACE
    LEGACY_TRANSLATION_KEY = 'pixel_shuffle'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # block_size
        block_size_data = np.array(kwargs.get(ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE), dtype=np.uint32)
        block_size = ir_graph.IrStaticTensor(ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
                                             [2],
                                             block_size_data,
                                             ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE, block_size)

        # mode
        attrs.addUint32(ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_MODE, kwargs.get(ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_MODE, ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR))

        self.c_op = ir_graph.DepthToSpaceOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class DequantizeOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_DEQUANTIZE
    LEGACY_TRANSLATION_KEY = 'dequantize'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # TODO Remove once fully migrated to QNNIR
        attrs.addUint32(ir_graph.IR_OP_DEQUANTIZE_PARAM_BW, kwargs.get(ir_graph.IR_OP_DEQUANTIZE_PARAM_BW),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addFloat(ir_graph.IR_OP_DEQUANTIZE_PARAM_MIN, kwargs.get(ir_graph.IR_OP_DEQUANTIZE_PARAM_MIN, 0.0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addFloat(ir_graph.IR_OP_DEQUANTIZE_PARAM_MAX, kwargs.get(ir_graph.IR_OP_DEQUANTIZE_PARAM_MAX, 0.0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addFloat(ir_graph.IR_OP_DEQUANTIZE_PARAM_SCALE, kwargs.get(ir_graph.IR_OP_DEQUANTIZE_PARAM_SCALE, 0.0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addInt32(ir_graph.IR_OP_DEQUANTIZE_PARAM_OFFSET, kwargs.get(ir_graph.IR_OP_DEQUANTIZE_PARAM_OFFSET, 0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addBool(ir_graph.IR_OP_DEQUANTIZE_PARAM_IS_SYMMETRIC, kwargs.get(ir_graph.IR_OP_DEQUANTIZE_PARAM_IS_SYMMETRIC, False),
                      ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        self.c_op = ir_graph.DequantizeOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class DetectionOutputOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_DETECTION_OUTPUT
    LEGACY_TRANSLATION_KEY = 'detection_output'

    class PriorBoxType:
        CORNER = "PRIORBOX_TYPE_CORNER"
        CENTER_SIZE = "PRIORBOX_TYPE_CENTER_SIZE"
        CORNER_SIZE = "PRIORBOX_TYPE_CORNER_SIZE"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, num_outputs=4)

        attrs = ir_graph.IrAttributes()

        # Multiplicative scaling factors applied to each of the bounding boxes in
        # in[1] in the form of (dy, dx, dh, dw)
        attr = kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS)
        attr = np.array(attr, dtype=np.float32)
        attrs.add(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS,
                                          list(attr.shape),
                                          attr,
                                          ir_graph.QNN_DATATYPE_FLOAT_32))

        # Boxes with scores lower than this threshold are filtered prior to the application of NMS.
        confidence_threshold = max(kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_CONFIDENCE_THRESHOLD), 0)
        attrs.addFloat(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_CONFIDENCE_THRESHOLD, confidence_threshold)

        # IoU threshold for the NMS algorithm.
        attrs.addFloat(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_IOU_THRESHOLD, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_IOU_THRESHOLD))

        # REGULAR: 1 - regular multi-class NMS
        # FAST: 0 - faster variant which limits the number of classes to which NMS is applied (Default)
        if ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_TYPE in kwargs:
            attrs.addUint32(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_TYPE, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_TYPE))

        # The index in num_classes of the “background” class
        if ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX in kwargs:
            attrs.addInt32(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX))

        # Choose to include background class in computing NMS
        if ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_USE_BG_IN_NMS in kwargs:
            attrs.addBool(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_USE_BG_IN_NMS, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_USE_BG_IN_NMS))

        # True: include the background class in the output
        # False: exclude the class
        if ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_OUTPUT_BACKGROUND in kwargs:
            attrs.addBool(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_OUTPUT_BACKGROUND, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_OUTPUT_BACKGROUND))

        # True: indicate that the classes all share a common set of initial bounding boxes
        # False: indicate that they use different initial bounding boxes.
        if ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_SHARE_LOCATION in kwargs:
            attrs.addBool(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_SHARE_LOCATION, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_SHARE_LOCATION))

        # Adaptation factor for the NMS threshold.
        # This factor is applied when nms_type is set to REGULAR
        if ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_ETA in kwargs:
            attrs.addFloat(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_ETA, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_ETA))

        # Parameter that specifies:
        # (i) the maximum number of classes per detection when nms_type is set to FAST.
        # (ii) the maximum number of detections when applying NMS for each single class
        #      when nms_type is set to REGULAR.
        # Parameter is ignored if set to default value.
        # This parameter is similar to ‘nms_topK’ found in training frameworks like Caffe which set nms_type to REGULAR.
        attrs.addInt32(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_DETECTION_LIMIT, kwargs.get(ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_DETECTION_LIMIT))

        if ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_VARIANCE_ENCODED_IN_TARGET in kwargs:
            attrs.addBool(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_VARIANCE_ENCODED_IN_TARGET, kwargs.get(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_VARIANCE_ENCODED_IN_TARGET, ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL))

        if ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_PRIORBOX_DATA in kwargs:
            attrs.addFloat(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_PRIORBOX_DATA, kwargs.get(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_PRIORBOX_DATA, ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL))

        if ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_PRIORBOX_CENTER_SIZE_DATA in kwargs:
            attrs.addFloat(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_PRIORBOX_CENTER_SIZE_DATA, kwargs.get(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_PRIORBOX_CENTER_SIZE_DATA, ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL))

        if ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_CODE_TYPE in kwargs:
            attrs.addString(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_CODE_TYPE, kwargs.get(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_CODE_TYPE, ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL))

        if ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_KEEP_TOP_K in kwargs:
            attrs.addInt32(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_KEEP_TOP_K, kwargs.get(ir_graph.IR_OP_DETECTION_OUTPUT_PARAM_KEEP_TOP_K, ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL))

        self.c_op = ir_graph.DetectionOutputOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class DistributeFpnProposalsOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS
    LEGACY_TRANSLATION_KEY = 'DistributeFpnProposals'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        #roi_max_level
        roi_max_level = kwargs.get(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MAX_LEVEL, 5)
        attrs.addUint32(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MAX_LEVEL, roi_max_level)

        #roi_min_level
        roi_min_level = kwargs.get(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MIN_LEVEL, 2)
        attrs.addUint32(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MIN_LEVEL, roi_min_level)

        #roi_canonical_scale
        roi_canonical_scale = kwargs.get(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_SCALE, 244)
        attrs.addUint32(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_SCALE, roi_canonical_scale)

        #roi_canonical_level
        roi_canonical_level = kwargs.get(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_LEVEL, 4)
        attrs.addUint32(ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_LEVEL, roi_canonical_level)

        self.c_op = ir_graph.DistributeFpnProposalsOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats: list, num_outputs: int, axis_order: AxisOrder) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce input buffer axis format to NONTRIVIAL
        for input_buffer in input_buffers:
            input_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

class ElementwiseTernaryOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_ELTWISE_TERNARY

    ir_to_legacy_type = {
        ir_graph.QNN_OP_ELEMENT_WISE_SELECT: 'elementwise_select'
    }

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addString(ir_graph.IR_OP_ELTWISE_TERNARY_PARAM_TYPE,
                        kwargs.get(ir_graph.IR_OP_ELTWISE_TERNARY_PARAM_TYPE),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        # TODO Remove name placeholder_aisw once name is read-only
        self.c_op = ir_graph.ElementwiseTernaryOp(name if name is not None else "placeholder_aisw", attrs)

    @property
    def type(self):
        return self.ir_to_legacy_type[self.c_op.eltwise_type]

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        in_ranks = [len(in_buf.shape) for in_buf in input_buffers]
        max_rank = max(in_ranks)
        in_bufs_with_max_rank = list(filter(lambda buf: len(buf.shape) == max_rank, input_buffers))
        max_rank_axis_formats = [buf.axis_format for buf in in_bufs_with_max_rank]

        if AxisTracker.AxisFormat.NONTRIVIAL in max_rank_axis_formats:
            # Change axis_formats for all buffers to NONTRIVIAL
            for buf in input_buffers:
                buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        # Populate the axis format using non-Const buffers only.
        # Constant axis format does not correctly represent the format since it is based on rank alone
        non_const_buffers = [buf
                             for buf in input_buffers
                             if not isinstance(buf.producer.op, ConstantOp)]
        super().populate_axis_format(graph, buf, src_axis_order, encodings, non_const_buffers)

        # Update shape and axis format of Constant nodes to match the other input
        input_nodes = [buf.producer for buf in input_buffers]
        if len(input_nodes) == 2 and any([isinstance(node.op, ConstantOp) for node in input_nodes]):
            const_node_idx = [i for i,node in enumerate(input_nodes) if isinstance(node.op, ConstantOp)]
            if len(const_node_idx) == 1:
                const_node_idx = const_node_idx[0]
                const_buffer = input_buffers[const_node_idx]
                non_const_buffer = input_buffers[1 - const_node_idx]
                # TODO: update for 2D and 1D cases as needed
                if len(non_const_buffer.shape) == 4 and len(const_buffer.shape) == 3:
                    # Const tensor is missing Batch dimension, Extend the tensor by adding 1 at axis 0 to match Rank
                    new_shape = [1] + const_buffer.shape
                    const_node = input_nodes[const_node_idx]
                    const_node.op.tensor = np.reshape(const_node.op.tensor, new_shape)
                    const_buffer.shape = new_shape
                    const_buffer.axis_format = non_const_buffer.axis_format
                elif len(non_const_buffer.shape) == len(const_buffer.shape):
                    const_buffer.axis_format = non_const_buffer.axis_format

        elif len(input_nodes) > 2 and any([isinstance(node.op, ConstantOp) for node in input_nodes]):
            const_node_idx = [i for i, node in enumerate(input_nodes) if isinstance(node.op, ConstantOp)]
            non_const_node_idx = [i for i, node in enumerate(input_nodes) if not isinstance(node.op, ConstantOp)]
            if len(non_const_node_idx) == 1:
                non_const_node_idx = non_const_node_idx[0]
                for const_id in const_node_idx:
                    const_buffer = input_buffers[const_id]
                    non_const_buffer = input_buffers[non_const_node_idx]
                    if len(non_const_buffer.shape) == len(const_buffer.shape):
                        const_buffer.axis_format = non_const_buffer.axis_format

        else:
            pass

        # Update the data_axis_formats at the end after all adjustments are done
        self.populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class ElementwiseBinaryOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_ELTWISE_BINARY

    ir_to_legacy_type = {
        ir_graph.QNN_OP_ELEMENT_WISE_ADD: "elementwise_sum",
        ir_graph.QNN_OP_ELEMENT_WISE_AND: "elementwise_and",
        ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE: "elementwise_div",
        ir_graph.QNN_OP_ELEMENT_WISE_EQUAL: "elementwise_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV: "elementwise_floor_div",
        ir_graph.QNN_OP_ELEMENT_WISE_FMOD: "elementwise_fmod",
        ir_graph.QNN_OP_ELEMENT_WISE_GREATER: "elementwise_greater",
        ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL: "elementwise_greater_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_LESS: "elementwise_less",
        ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL: "elementwise_less_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM: "elementwise_max",
        ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM: "elementwise_min",
        ir_graph.QNN_OP_ELEMENT_WISE_MOD: "elementwise_mod",
        ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY: "elementwise_product",
        ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL: "elementwise_not_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_OR: "elementwise_or",
        ir_graph.QNN_OP_ELEMENT_WISE_POWER: "elementwise_power",
        ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT: "elementwise_sub",
        ir_graph.QNN_OP_ELEMENT_WISE_XOR: "elementwise_xor"
    }

    operation_to_legacy = {
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD: "elementwise_sum",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND: "elementwise_and",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE: "elementwise_div",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL: "elementwise_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV: "elementwise_floor_div",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD: "elementwise_fmod",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER: "elementwise_greater",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL: "elementwise_greater_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS: "elementwise_less",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL: "elementwise_less_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM: "elementwise_max",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM: "elementwise_min",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD: "elementwise_mod",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY: "elementwise_product",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL: "elementwise_not_equal",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR: "elementwise_or",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER: "elementwise_power",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT: "elementwise_sub",
        ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR: "elementwise_xor"
    }

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        operation = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION)
        attrs.addUint32(ir_graph.QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, operation)
        # TODO Remove name placeholder_aisw once name is read-only
        self.c_op = ir_graph.ElementwiseBinaryOp(name if name is not None else "placeholder_aisw", attrs)

    @property
    def type(self):
        return self.operation_to_legacy[self.c_op.operation]

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # comfirm that ElementwiseBinary Op has 2 inputs
        if len(input_buffers) != 2:
            raise ValueError("Op {} should has 2 inputs but {} are received".format(self.name, len(in_ranks)))

        # Update shape and axis format of Constant nodes to match the other input
        input_nodes = [buf.producer for buf in input_buffers]
        if len(input_nodes) == 2 and any([isinstance(node.op, ConstantOp) for node in input_nodes]):
            const_node_idx = [i for i,node in enumerate(input_nodes) if isinstance(node.op, ConstantOp)]
            if len(const_node_idx) == 1:
                const_node_idx = const_node_idx[0]
                const_buffer = input_buffers[const_node_idx]
                non_const_buffer = input_buffers[1 - const_node_idx]
                # TODO: update for 2D and 1D cases as needed
                if len(non_const_buffer.shape) == 4 and len(const_buffer.shape) == 3:
                    # Const tensor is missing Batch dimension, Extend the tensor by adding 1 at axis 0 to match Rank
                    new_shape = [1] + const_buffer.shape
                    const_node = input_nodes[const_node_idx]
                    const_node.op.tensor = np.reshape(const_node.op.tensor, new_shape)
                    const_buffer.shape = new_shape
                    const_buffer.axis_format = non_const_buffer.axis_format
                    # call populate_data_axis_format for consumer of any buffer that had the format overridden
                    for consumer in const_buffer.consumers:
                        if consumer.op != self:
                            consumer.op.populate_data_axis_formats(graph, graph.get_input_buffers(consumer))
                elif len(non_const_buffer.shape) == len(const_buffer.shape) and \
                        non_const_buffer.axis_format != const_buffer.axis_format: # avoid the infinite loop due to the shared buffer
                    const_buffer.axis_format = non_const_buffer.axis_format
                    # call populate_data_axis_format for consumer of any buffer that had the format overridden
                    for consumer in const_buffer.consumers:
                        if consumer.op != self:
                            consumer.op.populate_data_axis_formats(graph, graph.get_input_buffers(consumer))


        in_ranks = [len(in_buf.shape) for in_buf in input_buffers]
        in_axis_formats = [buf.axis_format for buf in input_buffers]
        # check whether two inputs have same ranks
        # and checker whether it's suitable to do backward-propagation
        if in_ranks[0] == in_ranks[1] and \
                AxisTracker.AxisFormat.NONTRIVIAL in in_axis_formats and \
                not all([AxisTracker.AxisFormat.NONTRIVIAL == axis_format for axis_format in in_axis_formats]) and \
                AxisTracker.AxisFormat.ANY not in in_axis_formats and AxisTracker.AxisFormat.NOT_YET_DEFINED not in in_axis_formats:
            # ElementwiseBinary has two axis_formats
            # One is NONTRIVIAL, the other is known axis-format (neither ANY nor NOT_YET_DEFINED)
            # For example, assume in_axis_formats = [NCS, NONTRIVIAL]
            # In this case, we can backward-propagate NCS to its input buffers
            # and change axis_formats for all input buffers to NCS
            NT_idx = in_axis_formats.index(AxisTracker.AxisFormat.NONTRIVIAL)
            known_axis_idx = 1 - NT_idx
            known_axis_format = in_axis_formats[known_axis_idx]
            # Dont use format from Constant Op to override NONTRIVIAL on a dynamic tensor
            # Also, call populate_data_axis_format for consumer of any buffer that had the format overridden
            if not isinstance(input_nodes[known_axis_idx].op, ConstantOp):
                input_buffers[NT_idx].axis_format = known_axis_format
                for consumer in input_buffers[NT_idx].consumers:
                    if consumer.op != self:
                        consumer.op.populate_data_axis_formats(graph, graph.get_input_buffers(consumer))

        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        # Populate the axis format using non-Const buffers only.
        # Constant axis format does not correctly represent the format since it is based on rank alone
        non_const_buffers = [buf
                             for buf in input_buffers
                             if not isinstance(buf.producer.op, ConstantOp)]
        super().populate_axis_format(graph, buf, src_axis_order, encodings, non_const_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class ElementwiseNeuronOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_ELEMENT_WISE_NEURON
    LEGACY_TRANSLATION_KEY = 'elementwiseneuron'
    neuron_to_operation = {
        ir_graph.QNN_OP_ELU: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU,
        ir_graph.QNN_OP_GELU: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU,
        ir_graph.QNN_OP_HARD_SWISH: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH,
        ir_graph.QNN_OP_RELU: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU,
        ir_graph.QNN_OP_RELU_MIN_MAX: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
        ir_graph.QNN_OP_SIGMOID: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID,
        ir_graph.QNN_OP_TANH: ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH
    }

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        # operation
        operation = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION)
        attrs.addUint32(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, operation)

        # alpha
        alpha = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA)
        if alpha is not None:
            if operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA, 1.0)))
            elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA, 0.2)))
            elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA, 1.0)))

        # beta
        beta = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA)
        if beta is not None:
            if operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA, 1.0)))
            elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA, 0.5)))
            elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA, 1.0)))

        # min_value
        min_value = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE)
        if min_value is not None:
            if operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE, 0.0)))

        # max_value
        max_value = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE)
        if max_value is not None:
            if operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE, 0.0)))

        # threshold
        threshold = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD)
        if threshold is not None:
            if operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS:
                attrs.addFloat(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD,
                               float(kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD, -1.0)))

        self.c_op = ir_graph.ElementwiseNeuronOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


OpAdapterMap.register_translation(ElementwiseNeuronOp,
                                  ElementwiseNeuronOp.TRANSLATION_KEY)


class ElementwiseUnaryOp(Op):

    TRANSLATION_KEY = ir_graph.IR_OP_ELTWISE_UNARY

    operation_to_legacy = {
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS: "elementwise_unary_abs",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN: "elementwise_unary_asin",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN: "elementwise_unary_atan",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL: "elementwise_unary_ceil",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS: "elementwise_unary_cos",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP: "elementwise_unary_exp",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR: "elementwise_unary_floor",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG: "elementwise_unary_log",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG: "elementwise_unary_neg",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT: "elementwise_unary_not",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND: "elementwise_unary_round",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT: "elementwise_unary_rsqrt",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN: "elementwise_unary_sign",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN: "elementwise_unary_sin",
        ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT: "elementwise_unary_sqrt",
    }

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        operation = kwargs.get(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION)
        attrs.addUint32(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION, operation)
        # TODO Remove name placeholder_aisw once name is read-only
        self.c_op = ir_graph.ElementwiseUnaryOp(name if name is not None else "placeholder_aisw", attrs)

    @property
    def type(self):
        return self.operation_to_legacy[self.c_op.operation]

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ErfOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_ERF
    LEGACY_TRANSLATION_KEY = 'erf'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        self.c_op = ir_graph.ErfOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ExpandOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_EXPAND
    LEGACY_TRANSLATION_KEY = 'expand'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

        if ir_graph.IR_OP_EXPAND_PARAM_SHAPE not in kwargs:
            raise KeyError("{} attribute must be specified for ExpandOp {}".format(ir_graph.IR_OP_EXPAND_PARAM_SHAPE, name))

        attrs = ir_graph.IrAttributes()
        shape_data = np.array(kwargs.get(ir_graph.IR_OP_EXPAND_PARAM_SHAPE), dtype=np.int32)
        if shape_data.ndim != 1:
            raise ValueError("{} attribute must be a 1-D tensor for ExpandOp {}".format(ir_graph.IR_OP_EXPAND_PARAM_SHAPE, name))
        attrs.add(ir_graph.IR_OP_EXPAND_PARAM_SHAPE,
                  ir_graph.IrStaticTensor(ir_graph.IR_OP_EXPAND_PARAM_SHAPE,
                                          list(shape_data.shape),
                                          shape_data,
                                          ir_graph.QNN_DATATYPE_INT_32))
        self.c_op = ir_graph.ExpandOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats:list, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ExtractGlimpseOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_EXTRACT_GLIMPSE
    LEGACY_TRANSLATION_KEY = 'extract_glimpse'

    class NoiseType:
        UNIFORM = "NOISE_UNIFORM"
        GAUSSIAN = "NOISE_GAUSSIAN"
        ZERO = "NOISE_ZERO"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        size = np.array(kwargs.get(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_SIZE))
        attrs.add(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_SIZE,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_SIZE,
                                          list(size.shape),
                                          size,
                                          ir_graph.QNN_DATATYPE_INT_32))

        if ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_CENTERED in kwargs:
            attrs.addBool(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_CENTERED, kwargs.get(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_CENTERED))

        if ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_NORMALIZED in kwargs:
            attrs.addBool(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_NORMALIZED, kwargs.get(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_NORMALIZED))

        if ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_NOISE in kwargs:
            attrs.addInt32(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_NOISE, kwargs.get(ir_graph.QNN_OP_EXTRACT_GLIMPSE_PARAM_NOISE))

        self.c_op = ir_graph.ExtractGlimpseOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ExtractPatchesOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_EXTRACT_PATCHES
    LEGACY_TRANSLATION_KEY = 'extract_patches'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # size
        size = kwargs.get(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_SIZE)
        if size is None:
            raise ValueError("size attribute must be specified for ExtractPatchesOp {}".format(name))
        size_data = np.array(size, dtype=np.uint32)
        attrs.add(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_SIZE,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_SIZE,
                                          list(size_data.shape),
                                          size_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))

        # stride
        stride = kwargs.get(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_STRIDE)
        if stride is None:
            raise ValueError("stride attribute must be specified for ExtractPatchesOp {}".format(name))
        stride_data = np.array(stride, dtype=np.uint32)
        attrs.add(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_STRIDE,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_STRIDE,
                                          list(stride_data.shape),
                                          stride_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))

        # rate
        rate = kwargs.get(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_RATE)
        if rate is None:
            raise ValueError("rate attribute must be specified for ExtractPatchesOp {}".format(name))
        rate_data = np.array(rate, dtype=np.uint32)
        attrs.add(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_RATE,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_RATE,
                                          list(rate_data.shape),
                                          rate_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))

        # padding
        padding = kwargs.get(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_PADDING)
        if padding is None:
            raise ValueError("padding attribute must be specified for ExtractPatchesOp {}".format(name))
        attrs.addUint32(ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_PADDING, padding)

        self.c_op = ir_graph.ExtractPatchesOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class FullyConnectedOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_FULLY_CONNECTED
    LEGACY_TRANSLATION_KEY = 'fully_connected'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # keep_dims
        if "keep_dims" in kwargs:
            attrs.addBool(ir_graph.QNN_OP_FULLY_CONNECTED_PARAM_KEEP_DIMS, kwargs.get("keep_dims"))

        # supplemental
        if "transpose_a" in kwargs:
            attrs.addBool(ir_graph.IR_OP_FULLY_CONNECTED_PARAM_TRANSPOSE_A, kwargs.get("transpose_a"), ir_graph.IR_ATTR_USAGE_LEGACY)
        if "transpose_b" in kwargs:
            attrs.addBool(ir_graph.IR_OP_FULLY_CONNECTED_PARAM_TRANSPOSE_B, kwargs.get("transpose_b"), ir_graph.IR_ATTR_USAGE_LEGACY)
        if "bias_op_name" in kwargs and kwargs.get("bias_op_name") is not None:
            attrs.addString(ir_graph.IR_OP_FULLY_CONNECTED_PARAM_BIAS_OP_NAME, kwargs.get("bias_op_name"), ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.FullyConnectedOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class GatherOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GATHER
    LEGACY_TRANSLATION_KEY = 'gather'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addInt32(ir_graph.QNN_OP_GATHER_PARAM_AXIS, kargs.get(ir_graph.QNN_OP_GATHER_PARAM_AXIS, 0))
        self.c_op = ir_graph.GatherOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        if input_buffers[-1].rank() == 1:
            buf.set_axis_format(input_buffers[0].axis_format)
        else:
            buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class GatherElementsOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GATHER_ELEMENTS
    LEGACY_TRANSLATION_KEY = 'gather_elements'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addInt32(ir_graph.QNN_OP_GATHER_ELEMENTS_PARAM_AXIS, kargs.get(ir_graph.QNN_OP_GATHER_ELEMENTS_PARAM_AXIS, 0))
        self.c_op = ir_graph.GatherElementsOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class GatherNDOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GATHER_ND
    LEGACY_TRANSLATION_KEY = 'gather_nd'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_GATHER_ND_PARAM_BATCH_DIMS in kwargs:
            attrs.addInt32(ir_graph.QNN_OP_GATHER_ND_PARAM_BATCH_DIMS, kwargs.get(ir_graph.QNN_OP_GATHER_ND_PARAM_BATCH_DIMS))
        self.c_op = ir_graph.GatherNDOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce indices buffer axis format to NONTRIVIAL
        input_buffers[1].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class GenerateProposalsOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GENERATE_PROPOSALS
    LEGACY_TRANSLATION_KEY = 'generate_proposals'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # img_size_ratio
        img_size_ratio_data = np.asarray(kwargs.get(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IMG_SIZE_RATIO), dtype=np.float32)
        img_size_ratio = ir_graph.IrStaticTensor(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IMG_SIZE_RATIO,
                                                 list(img_size_ratio_data.shape),
                                                 img_size_ratio_data,
                                                 ir_graph.QNN_DATATYPE_FLOAT_32)
        attrs.add(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IMG_SIZE_RATIO, img_size_ratio)

        # min_size
        min_size = kwargs.get(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_MIN_SIZE)
        if min_size is not None:
            attrs.addFloat(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_MIN_SIZE, min_size)

        # pre_nms_top_n
        pre_nms_limit = kwargs.get(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_PRE_NMS_LIMIT)
        if pre_nms_limit is not None:
            attrs.addUint32(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_PRE_NMS_LIMIT, pre_nms_limit)

        # post_nms_top_n
        post_nms_limit = kwargs.get(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_POST_NMS_LIMIT)
        if post_nms_limit is not None:
            attrs.addUint32(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_POST_NMS_LIMIT, post_nms_limit)

        # nms_thresh
        attrs.addFloat(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IOU_THRESHOLD, kwargs.get(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IOU_THRESHOLD))

        # correct_transform_coords
        correct_transform_coords = kwargs.get(ir_graph.IR_OP_GENERATE_PROPOSALS_PARAM_CORRECT_TRANSFORM_COORDS)
        if correct_transform_coords is not None:
            attrs.addBool(ir_graph.IR_OP_GENERATE_PROPOSALS_PARAM_CORRECT_TRANSFORM_COORDS, correct_transform_coords, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        # bbox_xform_clip
        bbox_xform_clip = kwargs.get(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_BBOX_XFORM_CLIP)
        if bbox_xform_clip is not None:
            attrs.addBool(ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_BBOX_XFORM_CLIP, bbox_xform_clip)

        self.c_op = ir_graph.GenerateProposalsOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class GetSparseIndicesOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GET_SPARSE_INDICES
    LEGACY_TRANSLATION_KEY = 'get_sparse_indices'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        num_specified_elements = kwargs.get('num_specified_elements')
        attrs.addUint32(ir_graph.IR_OP_GET_SPARSE_INDICES_PARAM_NUM_SPECIFIED_ELEMENTS, num_specified_elements)
        self.c_op = ir_graph.GetSparseIndicesOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class GetSparseValuesOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GET_SPARSE_VALUES
    LEGACY_TRANSLATION_KEY = 'get_sparse_values'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        self.num_specified_elements = kwargs.get('num_specified_elements')
        attrs.addUint32(ir_graph.IR_OP_GET_SPARSE_VALUES_PARAM_NUM_SPECIFIED_ELEMENTS, self.num_specified_elements)
        self.c_op = ir_graph.GetSparseValuesOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class GridSampleOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GRID_SAMPLE
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()

        align_corners = kwargs.get(ir_graph.QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS)
        if align_corners is not None:
            attrs.addBool(ir_graph.QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS, align_corners)

        mode = kwargs.get(ir_graph.QNN_OP_GRID_SAMPLE_PARAM_MODE)
        if mode is not None:
            attrs.addUint32(ir_graph.QNN_OP_GRID_SAMPLE_PARAM_MODE, mode)

        padding_mode = kwargs.get(ir_graph.QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE)
        if padding_mode is not None:
            attrs.addUint32(ir_graph.QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE, padding_mode)

        self.c_op = ir_graph.GridSampleOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].set_axis_format(graph.src_axis_order.get_axis_format(len(input_buffers[0].shape)))
        # Set grid buffer to NONTRIVIAL
        input_buffers[1].set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(input_buffers[0].axis_format)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class GroupNormOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GROUP_NORM
    # TODO: fix the lowercase constraint in unittest validator that causes LEGACY_TRANSLATION_KEY inconsistency
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY.lower()

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_GROUP_NORM_PARAM_EPSILON not in kwargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_GROUP_NORM_PARAM_EPSILON))
        attrs.addFloat(ir_graph.QNN_OP_GROUP_NORM_PARAM_EPSILON, kwargs.get(ir_graph.QNN_OP_GROUP_NORM_PARAM_EPSILON))

        if ir_graph.QNN_OP_GROUP_NORM_PARAM_GROUP not in kwargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_GROUP_NORM_PARAM_GROUP))
        attrs.addUint32(ir_graph.QNN_OP_GROUP_NORM_PARAM_GROUP, kwargs.get(ir_graph.QNN_OP_GROUP_NORM_PARAM_GROUP))

        self.c_op = ir_graph.GroupNormOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].set_axis_format(graph.src_axis_order.get_axis_format(len(input_buffers[0].shape)))
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def set_macs_params(self, input_shapes: list, output_shapes, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class GruOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_GRU
    LEGACY_TRANSLATION_KEY = 'gru'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        direction = kwargs.get(ir_graph.QNN_OP_GRU_PARAM_DIRECTION, ir_graph.QNN_OP_GRU_DIRECTION_FORWARD)
        attrs.addUint32(ir_graph.QNN_OP_GRU_PARAM_DIRECTION, direction)

        linear_before_reset = kwargs.get(ir_graph.QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET)
        if linear_before_reset is not None:
            attrs.addUint32(ir_graph.QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET, linear_before_reset)

        time_major = kwargs.get(ir_graph.QNN_OP_GRU_PARAM_TIME_MAJOR)
        if time_major is not None:
            attrs.addBool(ir_graph.QNN_OP_GRU_PARAM_TIME_MAJOR, time_major)

        # supplemental attributes
        hidden_size = kwargs.get(ir_graph.IR_OP_GRU_PARAM_HIDDEN_SIZE)
        if hidden_size is None:
            raise ValueError("hidden_size attribute must be specified for GruOp {}".format(name))
        attrs.addUint32(ir_graph.IR_OP_GRU_PARAM_HIDDEN_SIZE, hidden_size,
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        h_0_input_name = kwargs.get(ir_graph.IR_OP_GRU_PARAM_H_0_INPUT_NAME)
        if h_0_input_name is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_H_0_INPUT_NAME, h_0_input_name,
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        # legacy attributes
        activation = kwargs.get(ir_graph.IR_OP_GRU_PARAM_ACTIVATION)
        if activation is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_ACTIVATION, activation,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        gate_activation = kwargs.get(ir_graph.IR_OP_GRU_PARAM_GATE_ACTIVATION)
        if gate_activation is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_GATE_ACTIVATION, gate_activation,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        rec_gate_activation = kwargs.get(ir_graph.IR_OP_GRU_PARAM_REC_GATE_ACTIVATION)
        if rec_gate_activation is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_REC_GATE_ACTIVATION, rec_gate_activation,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        self.c_op = ir_graph.GruOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        in_data_buf = input_buffers[0]
        in_data_buf.axis_format = graph.src_axis_order.get_axis_format(len(in_data_buf.shape),
                                                                       time_series_format=True)
        # Set initial h buffer to NONTRIVIAL
        for in_buf in input_buffers[1:]:
            in_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        super().populate_data_axis_formats(graph, input_buffers)


class IdentityOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_IDENTITY
    LEGACY_TRANSLATION_KEY = 'identity'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        self.c_op = ir_graph.IdentityOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ImageProjectiveTransformOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM
    LEGACY_TRANSLATION_KEY = 'image_projective_transform'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addUint32(ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM_PARAM_INTERPOLATION_MODE,
                        kwargs.get(ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM_PARAM_INTERPOLATION_MODE,
                                   ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM_INTERPOLATION_MODE_BILINEAR))
        self.c_op = ir_graph.ImageProjectionTransformOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class InstanceNormOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_INSTANCE_NORM
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY.lower()

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        attrs = ir_graph.IrAttributes()
        attrs.addUint32(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_MODE,
                        kwargs.get(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_MODE, ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA))
        attrs.addUint32(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_REGION,
                        kwargs.get(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_REGION, ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL))
        attrs.addFloat(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_EPSILON,
                        kwargs.get(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_EPSILON, 1e-12))
        attrs.addUint8(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_NORMALIZE_VARIANCE,
                        kwargs.get(ir_graph.QNN_OP_INSTANCE_NORM_PARAM_NORMALIZE_VARIANCE, 1))

        self.c_op = ir_graph.InstancenormOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].set_axis_format(graph.src_axis_order.get_axis_format(len(input_buffers[0].shape)))
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.populate_axis_format(src_axis_order, encodings)

    def set_macs_params(self, input_shapes: list, output_shapes, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class L2NormOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_L2_NORM
    LEGACY_TRANSLATION_KEY = 'l2_norm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        axis = kwargs.get(ir_graph.QNN_OP_L2_NORM_PARAM_AXIS)
        axes = kwargs.get(ir_graph.QNN_OP_L2_NORM_PARAM_AXES)
        if axis is None and (axes is None or len(axes) == 0):
            raise ValueError("At least one of attributes axis or axes needs to be provided in L2NormOp {}".format(name))

        attrs.addInt32(ir_graph.QNN_OP_L2_NORM_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_L2_NORM_PARAM_AXIS, 0))

        # Since axes take precedence over axis, only add this attribute if provided
        if axes is not None and len(axes) > 0:
            axes_data = np.asarray(axes, dtype=np.uint32)
            axes = ir_graph.IrStaticTensor(ir_graph.QNN_OP_L2_NORM_PARAM_AXES,
                                           list(axes_data.shape),
                                           axes_data,
                                           ir_graph.QNN_DATATYPE_UINT_32)
            attrs.add(ir_graph.QNN_OP_L2_NORM_PARAM_AXES, axes)

        attrs.addFloat(ir_graph.QNN_OP_L2_NORM_PARAM_EPSILON, kwargs.get(ir_graph.QNN_OP_L2_NORM_PARAM_EPSILON, 1e-12))

        self.c_op = ir_graph.L2NormOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class LayerNormOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_LAYER_NORM
    LEGACY_TRANSLATION_KEY = 'layernorm'
    EPSILON = 1e-9

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_LAYER_NORM_PARAM_AXES not in kwargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_LAYER_NORM_PARAM_AXES))
        axes_attr = np.array(kwargs.get(ir_graph.QNN_OP_LAYER_NORM_PARAM_AXES), dtype=np.int32)
        attrs.add(ir_graph.QNN_OP_LAYER_NORM_PARAM_AXES,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_LAYER_NORM_PARAM_AXES,
                                          list(axes_attr.shape),
                                          axes_attr,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.addFloat(ir_graph.QNN_OP_LAYER_NORM_PARAM_EPSILON,
                       float(kwargs.get(ir_graph.QNN_OP_LAYER_NORM_PARAM_EPSILON, 0.001)))
        self.c_op = ir_graph.LayerNormOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class LogSoftmaxOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_LOG_SOFTMAX
    LEGACY_TRANSLATION_KEY = 'logsoftmax'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_LOG_SOFTMAX_PARAM_AXIS not in kwargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_LOG_SOFTMAX_PARAM_AXIS))
        attrs.addInt32(ir_graph.QNN_OP_LOG_SOFTMAX_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_LOG_SOFTMAX_PARAM_AXIS))
        attrs.addFloat(ir_graph.QNN_OP_LOG_SOFTMAX_PARAM_BETA,
                       float(kwargs.get(ir_graph.QNN_OP_LOG_SOFTMAX_PARAM_BETA, 1.0)))
        self.c_op = ir_graph.LogSoftmaxOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class LrnOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_LRN
    LEGACY_TRANSLATION_KEY = 'rnorm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # alpha
        alpha = kwargs.get(ir_graph.QNN_OP_LRN_PARAM_ALPHA, 1.0)
        attrs.addFloat(ir_graph.QNN_OP_LRN_PARAM_ALPHA, alpha)

        # beta
        beta = kwargs.get(ir_graph.QNN_OP_LRN_PARAM_BETA, 0.5)
        attrs.addFloat(ir_graph.QNN_OP_LRN_PARAM_BETA, beta)

        # bias
        bias = kwargs.get(ir_graph.QNN_OP_LRN_PARAM_BIAS, 1.0)
        attrs.addFloat(ir_graph.QNN_OP_LRN_PARAM_BIAS, bias)

        # radius
        radius = kwargs.get(ir_graph.QNN_OP_LRN_PARAM_RADIUS)
        if radius is None:
            raise ValueError("radius attribute must be specified for LrnOp {}".format(name))
        attrs.addInt32(ir_graph.QNN_OP_LRN_PARAM_RADIUS, radius)

        # region
        region = kwargs.get(ir_graph.QNN_OP_LRN_PARAM_REGION, ir_graph.QNN_OP_LRN_REGION_ACROSS_CHANNEL)
        attrs.addUint32(ir_graph.QNN_OP_LRN_PARAM_REGION, region)

        self.c_op = ir_graph.LrnOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class LstmOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_LSTM
    LEGACY_TRANSLATION_KEY = 'lstm'
    HIDDEN_IN_IDX, CELL_IN_IDX = 10, 11
    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        direction = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_DIRECTION, ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD)
        attrs.addUint32(ir_graph.QNN_OP_LSTM_PARAM_DIRECTION, direction)

        cell_clip_threshold = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD)
        if cell_clip_threshold is not None:
            attrs.addFloat(ir_graph.QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD, cell_clip_threshold)

        output_clip_threshold = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD)
        if output_clip_threshold is not None:
            attrs.addFloat(ir_graph.QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD, output_clip_threshold)

        time_major = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_TIME_MAJOR)
        if time_major is not None:
            attrs.addBool(ir_graph.QNN_OP_LSTM_PARAM_TIME_MAJOR, time_major)

        # supplemental attributes
        hidden_size = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_HIDDEN_SIZE)
        if hidden_size is None:
            raise ValueError("hidden_size attribute must be specified for LstmOp {}".format(name))
        attrs.addUint32(ir_graph.IR_OP_LSTM_PARAM_HIDDEN_SIZE, hidden_size,
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        reset_state_at_time_step_0 = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_RESET_STATE_AT_TIME_STEP_0)
        if reset_state_at_time_step_0 is not None:
            attrs.addBool(ir_graph.IR_OP_LSTM_PARAM_RESET_STATE_AT_TIME_STEP_0, reset_state_at_time_step_0,
                          ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        h_0_input_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_H_0_INPUT_NAME)
        if h_0_input_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_H_0_INPUT_NAME, h_0_input_name,
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        c_0_input_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_C_0_INPUT_NAME)
        if c_0_input_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_C_0_INPUT_NAME, c_0_input_name,
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        # legacy attributes
        sequence_continuation_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_SEQUENCE_CONTINUATION_NAME)
        if sequence_continuation_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_SEQUENCE_CONTINUATION_NAME, sequence_continuation_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        x_static_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_X_STATIC_NAME)
        if x_static_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_X_STATIC_NAME, x_static_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        self.c_op = ir_graph.LstmOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        in_data_buf = input_buffers[0]
        # For 3D input
        if len(in_data_buf.shape) == 3:
            in_data_buf.axis_format = graph.src_axis_order.get_axis_format(len(in_data_buf.shape),
                                                                           time_series_format=True)
        else:
            # For 2D input
            in_data_buf.set_axis_format(AxisTracker.AxisFormat.NF)

        # Set initial h/c buffers to NONTRIVIAL
        for in_buf in input_buffers[1:3]:
            in_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)
        self.params_count = self.params_count_c_op_wrapper(input_shapes, output_shapes, axis_order)


class MaskedSoftmaxOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_MASKED_SOFTMAX
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        # mode
        attrs.addUint32(ir_graph.QNN_OP_MASKED_SOFTMAX_PARAM_MODE,
                        kwargs.get(ir_graph.QNN_OP_MASKED_SOFTMAX_PARAM_MODE, ir_graph.QNN_OP_MASKED_SOFTMAX_MODE_UNCOMPRESSED))
        self.c_op = ir_graph.MaskedSoftmaxOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class MatMulOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_MAT_MUL
    LEGACY_TRANSLATION_KEY = 'matmul'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        attrs.addBool(ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
                      kwargs.get(ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, 0))
        attrs.addBool(ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                      kwargs.get(ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, 0))

        self.c_op = ir_graph.MatMulOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(input_buffers[0].axis_format)


class MergedWeightsGruOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_MERGED_WEIGHTS_GRU
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        direction = kwargs.get(ir_graph.QNN_OP_GRU_PARAM_DIRECTION, ir_graph.QNN_OP_GRU_DIRECTION_FORWARD)
        attrs.addUint32(ir_graph.QNN_OP_GRU_PARAM_DIRECTION, direction)

        linear_before_reset = kwargs.get(ir_graph.QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET)
        if linear_before_reset is not None:
            attrs.addUint32(ir_graph.QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET, linear_before_reset)

        time_major = kwargs.get(ir_graph.QNN_OP_GRU_PARAM_TIME_MAJOR)
        if time_major is not None:
            attrs.addBool(ir_graph.QNN_OP_GRU_PARAM_TIME_MAJOR, time_major)

        # supplemental attributes
        hidden_size = kwargs.get(ir_graph.IR_OP_GRU_PARAM_HIDDEN_SIZE)
        if hidden_size is None:
            raise ValueError("hidden_size attribute must be specified for MergedWeightsGruOp {}".format(name))
        attrs.addUint32(ir_graph.IR_OP_GRU_PARAM_HIDDEN_SIZE, hidden_size,
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        h_0_input_name = kwargs.get(ir_graph.IR_OP_GRU_PARAM_H_0_INPUT_NAME)
        if h_0_input_name is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_H_0_INPUT_NAME, h_0_input_name,
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        # legacy attributes
        activation = kwargs.get(ir_graph.IR_OP_GRU_PARAM_ACTIVATION)
        if activation is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_ACTIVATION, activation,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        gate_activation = kwargs.get(ir_graph.IR_OP_GRU_PARAM_GATE_ACTIVATION)
        if gate_activation is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_GATE_ACTIVATION, gate_activation,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        rec_gate_activation = kwargs.get(ir_graph.IR_OP_GRU_PARAM_REC_GATE_ACTIVATION)
        if rec_gate_activation is not None:
            attrs.addString(ir_graph.IR_OP_GRU_PARAM_REC_GATE_ACTIVATION, rec_gate_activation,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        self.c_op = ir_graph.MergedWeightsGruOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        in_data_buf = input_buffers[0]
        in_data_buf.axis_format = graph.src_axis_order.get_axis_format(len(in_data_buf.shape),
                                                                       time_series_format=True)
        # Set initial h buffer to NONTRIVIAL
        for in_buf in input_buffers[1:]:
            in_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        super().populate_data_axis_formats(graph, input_buffers)


class MomentOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_MOMENTS
    LEGACY_TRANSLATION_KEY = 'moment'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        axes = np.array(kwargs.get(ir_graph.QNN_OP_MOMENTS_PARAM_AXES), dtype=np.uint32)
        keep_dims = kwargs.get(ir_graph.QNN_OP_MOMENTS_PARAM_KEEP_DIMS)
        if axes is None:
            raise ValueError("axes attributes must be specified for Moments {}".format(name))
        if keep_dims is None:
            raise ValueError("keep_dims attributes must be specified for Moments {}".format(name))
        attrs.add(ir_graph.QNN_OP_MOMENTS_PARAM_AXES,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_MOMENTS_PARAM_AXES,
                                          list(axes.shape),
                                          axes,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.addBool(ir_graph.QNN_OP_MOMENTS_PARAM_KEEP_DIMS,keep_dims)
        self.c_op = ir_graph.MomentsOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class NonMaxSuppressionOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_NON_MAX_SUPPRESSION
    LEGACY_TRANSLATION_KEY = 'non_max_suppression'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        attrs.addFloat(ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_IOU_THRESHOLD,
                       kwargs.get(ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_IOU_THRESHOLD))
        if ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_SCORE_THRESHOLD in kwargs:
            attrs.addFloat(ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_SCORE_THRESHOLD,
                           kwargs.get(ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_SCORE_THRESHOLD))
        if ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_MAX_BOXES_SELECTED in kwargs:
            attrs.addInt32(ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_MAX_BOXES_SELECTED,
                            kwargs.get(ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_MAX_BOXES_SELECTED))

        self.c_op = ir_graph.NonMaxSuppressionOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes,input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce input axis format to NONTRIVIAL
        for input_buffer in input_buffers:
            input_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class NonZeroOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_NON_ZERO
    LEGACY_TRANSLATION_KEY = 'non_zero'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        self.c_op = ir_graph.NonZeroOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes,input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class MultiClassNmsOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_MULTI_CLASS_NMS
    LEGACY_TRANSLATION_KEY = 'multi_class_nms'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addFloat(ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_IOU_THRESHOLD,
                       kwargs.get(ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_IOU_THRESHOLD))
        if ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_SCORE_THRESHOLD in kwargs:
            attrs.addFloat(ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_SCORE_THRESHOLD,
                           kwargs.get(ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_SCORE_THRESHOLD))
        if ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_SOFT_NMS_SIGMA in kwargs:
            attrs.addFloat(ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_SOFT_NMS_SIGMA,
                           kwargs.get(ir_graph.QNN_OP_MULTI_CLASS_NMS_PARAM_SOFT_NMS_SIGMA))
        # supplemental
        attrs.addUint32(ir_graph.IR_OP_MULTI_CLASS_NMS_PARAM_MAX_TOTAL_DETECTIONS,
                        kwargs.get(ir_graph.IR_OP_MULTI_CLASS_NMS_PARAM_MAX_TOTAL_DETECTIONS),
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        if ir_graph.IR_OP_MULTI_CLASS_NMS_PARAM_MAX_DETECTIONS_PER_CLASS in kwargs:
            attrs.addUint32(ir_graph.IR_OP_MULTI_CLASS_NMS_PARAM_MAX_DETECTIONS_PER_CLASS,
                            kwargs.get(ir_graph.IR_OP_MULTI_CLASS_NMS_PARAM_MAX_DETECTIONS_PER_CLASS),
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        self.c_op = ir_graph.MultiClassNmsOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        idx = graph.get_output_buffer_idx(buf.producer, buf.name)
        # output axis_formats should equal to input for NMS feature input/output
        if idx > 3:
            buf.set_axis_format(input_buffers[idx - 2].axis_format)
        # output format for boxes, score, classes and num_valid_detections
        else:
            buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class OneHotOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_ONE_HOT
    LEGACY_TRANSLATION_KEY = 'one_hot'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        depth = kwargs.get(ir_graph.QNN_OP_ONE_HOT_PARAM_DEPTH)
        if depth is None:
            raise ValueError("depth attributes must be specified for OneHotOp {}".format(name))
        dtypeaddfunc = self.g_dtype_addfunction(np.dtype(kwargs.get(ir_graph.QNN_OP_ONE_HOT_PARAM_ON_VALUE)), attrs)

        attrs.addUint32(ir_graph.QNN_OP_ONE_HOT_PARAM_DEPTH, depth)
        attrs.addInt32(ir_graph.QNN_OP_ONE_HOT_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_ONE_HOT_PARAM_AXIS, -1))
        dtypeaddfunc(ir_graph.QNN_OP_ONE_HOT_PARAM_ON_VALUE, kwargs.get(ir_graph.QNN_OP_ONE_HOT_PARAM_ON_VALUE))
        dtypeaddfunc(ir_graph.QNN_OP_ONE_HOT_PARAM_OFF_VALUE, kwargs.get(ir_graph.QNN_OP_ONE_HOT_PARAM_OFF_VALUE))

        self.c_op = ir_graph.OneHotOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

    def __getattr__(self, key):
        if key != ir_graph.QNN_OP_ONE_HOT_PARAM_ON_VALUE and key != ir_graph.QNN_OP_ONE_HOT_PARAM_OFF_VALUE:
            return super(OneHotOp, self).__getattr__(key)
        dtype = self.c_op.attrs.get_data_type(ir_graph.QNN_OP_ONE_HOT_PARAM_ON_VALUE)
        try:
            if key in self.__dict__['attrs']:
                return self.__dict__['attrs'][key]
            elif "c_op" in self.__dict__ and self.c_op.attrs.has(key):
                # This will only work if key have add into attrs
                return self.get_attrs_keyvalue(dtype, key)
            else:
                return self.__dict__[key]
        except KeyError:
            raise AttributeError("Op %s has no attribute %s" % (self.name, key))

class PackOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_PACK
    LEGACY_TRANSLATION_KEY = 'pack'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_PACK_PARAM_AXIS not in kargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_PACK_PARAM_AXIS))
        attrs.addInt32(ir_graph.QNN_OP_PACK_PARAM_AXIS, kargs.get(ir_graph.QNN_OP_PACK_PARAM_AXIS))
        self.c_op = ir_graph.PackOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class PadOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_PAD
    LEGACY_TRANSLATION_KEY = 'pad'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        pad_attr = np.array(kwargs.get(ir_graph.QNN_OP_PAD_PARAM_PAD_AMOUNT), dtype=np.uint32)
        attrs.add(ir_graph.QNN_OP_PAD_PARAM_PAD_AMOUNT,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_PAD_PARAM_PAD_AMOUNT,
                                          list(pad_attr.shape),
                                          pad_attr,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.addUint32(ir_graph.QNN_OP_PAD_PARAM_SCHEME, kwargs.get(ir_graph.QNN_OP_PAD_PARAM_SCHEME))
        attrs.addFloat(ir_graph.QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
                       kwargs.get(ir_graph.QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE, 0.0))
        self.c_op = ir_graph.PadOp(name, attrs)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        super().populate_data_axis_formats(graph, input_buffers)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes,input_axis_formats, num_outputs, axis_order)


class TransposeOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_TRANSPOSE
    LEGACY_TRANSLATION_KEY = 'permute'

    def __init__(self, name, perm, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, **kwargs)
        perm_array = np.array(perm, dtype=np.uint32)
        attrs = ir_graph.IrAttributes()
        attrs.add(ir_graph.QNN_OP_TRANSPOSE_PARAM_PERM,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_TRANSPOSE_PARAM_PERM,
                                          list(perm_array.shape),
                                          perm_array,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        self.c_op = ir_graph.TransposeOp(name, attrs)

    def encode(self):
        return {'order' : list(self.c_op.perm)}

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        # if transpose perm is trivial then set output buffer axis format to same as input data
        # otherwise set as per the input data axis format and transpose perm
        if self.perm == [0, 1, 2, 3]:
            buf.set_axis_format(self.data_axis_formats[0])
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC and self.perm == [0, 4, 1, 2, 3]:
            buf.set_axis_format(AxisTracker.AxisFormat.NCDHW)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW and self.perm == [0, 2, 3, 4, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.NDHWC)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NSC and self.perm == [0, 3, 1, 2]:
            buf.set_axis_format(AxisTracker.AxisFormat.NCS)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NCS and self.perm == [0, 2, 3, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.NSC)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NFC and self.perm == [0, 2, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.NCF)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NCF and self.perm == [0, 2, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.NFC)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NTF and self.perm == [1, 0, 2]:
            buf.set_axis_format(AxisTracker.AxisFormat.TNF)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.TNF and self.perm == [1, 0, 2]:
            buf.set_axis_format(AxisTracker.AxisFormat.NTF)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.OIDHW and self.perm == [2, 3, 4, 1, 0]:
            buf.set_axis_format(AxisTracker.AxisFormat.DHWIO)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.IODHW and self.perm == [2, 3, 4, 0, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.DHWIO)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.DHWIO and self.perm == [3, 4, 0, 1, 2]:
            buf.set_axis_format(AxisTracker.AxisFormat.IODHW)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.DHWIO and self.perm == [4, 3, 0, 1, 2]:
            buf.set_axis_format(AxisTracker.AxisFormat.OIDHW)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.OIHW and self.perm == [2, 3, 1, 0]:
            buf.set_axis_format(AxisTracker.AxisFormat.HWIO)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.IOHW and self.perm == [2, 3, 0, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.HWIO)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.HWIO and self.perm == [2, 3, 0, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.IOHW)
        elif self.data_axis_formats[0] == AxisTracker.AxisFormat.HWIO and self.perm == [3, 2, 0, 1]:
            buf.set_axis_format(AxisTracker.AxisFormat.OIHW)
        else:
            buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class Pool1dOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_POOL1D
    LEGACY_TRANSLATION_KEY = 'pool1d'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

        attrs = ir_graph.IrAttributes()
        input_pool_type = kargs['pool_type']
        filter_data = kargs['size_x']
        stride_data = kargs['stride_x']
        pad_data = np.array([kargs.get('padx_before', 0), kargs.get('padx_after', 0)], dtype=np.uint32)
        pad_data_shape = [2]

        # Common attributes for MaxPool, AvgPool and lpPool
        attrs.addUint32(ir_graph.IR_OP_POOL_1D_PARAM_FILTER_SIZE, filter_data)
        attrs.addUint32(ir_graph.IR_OP_POOL_1D_PARAM_STRIDE, stride_data)
        attrs.add(ir_graph.IR_OP_POOL_1D_PARAM_PAD_AMOUNT,
                  ir_graph.IrStaticTensor(ir_graph.IR_OP_POOL_1D_PARAM_PAD_AMOUNT,
                                          pad_data_shape,
                                          pad_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.addUint8(ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY,
                       kargs.get(ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY,
                                 ir_graph.PADDING_SIZE_EXPLICIT),
                       ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addString(ir_graph.IR_OP_POOL_TYPE,
                        kargs.get(ir_graph.IR_OP_POOL_TYPE),
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        # Specific attributes based on pool types
        # Reuse the pool2d types
        if input_pool_type == ir_graph.QNN_OP_POOL_AVG_2D or input_pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
            attrs.addUint32(ir_graph.IR_OP_POOL_1D_PARAM_ROUNDING_MODE,
                            kargs.get(ir_graph.IR_OP_POOL_1D_PARAM_ROUNDING_MODE, ir_graph.IR_OP_POOL_ROUNDING_MODE_FLOOR))
        if input_pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
            attrs.addBool(ir_graph.IR_OP_POOL_1D_PARAM_COUNT_PAD_FOR_EDGES,
                          kargs.get(ir_graph.IR_OP_POOL_1D_PARAM_COUNT_PAD_FOR_EDGES, False))
        if input_pool_type == ir_graph.QNN_OP_L2_POOL_2D:
            attrs.addUint32(ir_graph.IR_OP_POOL_P, kargs.get(ir_graph.IR_OP_POOL_P, 2),
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.Pool1dOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class Pool2dOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_POOL2D
    LEGACY_TRANSLATION_KEY = 'pool'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        filter_data = np.array([kargs['size_y'], kargs.get('size_x', 1)], dtype=np.uint32)
        stride_data = np.array([kargs['stride_y'], kargs['stride_x']], dtype=np.uint32)
        pad_data = np.array([kargs.get('pady_before', 0), kargs.get('pady_after', 0),
                             kargs.get('padx_before', 0), kargs.get('padx_after', 0)], dtype=np.uint32)
        filter_data_shape = [2]
        stride_data_shape = [2]
        pad_data_shape = [2, 2]

        attrs = ir_graph.IrAttributes()
        input_pool_type = kargs['pool_type']
        # Common attributes for MaxPool, AvgPool and lpPool
        attrs.add(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE,
                                          filter_data_shape,
                                          filter_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.add(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_STRIDE,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_STRIDE,
                                          stride_data_shape,
                                          stride_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.add(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT,
                  ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT,
                                          pad_data_shape,
                                          pad_data,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.addUint8(ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY,
                       kargs.get(ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY,
                                 ir_graph.PADDING_SIZE_EXPLICIT),
                       ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addString(ir_graph.IR_OP_POOL_TYPE,
                        kargs.get(ir_graph.IR_OP_POOL_TYPE),
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        # Specific attributes based on pool types
        if input_pool_type == ir_graph.QNN_OP_POOL_AVG_2D or input_pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
            attrs.addUint32(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_ROUNDING_MODE,
                            kargs.get(ir_graph.QNN_OP_POOL_MAX_2D_PARAM_ROUNDING_MODE,
                                      ir_graph.IR_OP_POOL_ROUNDING_MODE_FLOOR))
        if input_pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
            attrs.addBool(ir_graph.QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES,
                          kargs.get(ir_graph.QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES, False))
        if input_pool_type == ir_graph.QNN_OP_L2_POOL_2D:
            attrs.addUint32(ir_graph.IR_OP_POOL_P, kargs.get(ir_graph.IR_OP_POOL_P, 2),
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.Pool2dOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class Pool3dOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_POOL3D
    LEGACY_TRANSLATION_KEY = 'pool3d'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        filter_data = np.array([kargs['size_z'], kargs['size_y'], kargs['size_x']], dtype=np.uint32)
        stride_data = np.array([kargs['stride_z'], kargs['stride_y'], kargs['stride_x']], dtype=np.uint32)
        pad_data = np.array([kargs.get('padz_before', 0), kargs.get('padz_after', 0),
                             kargs.get('pady_before', 0), kargs.get('pady_after', 0),
                             kargs.get('padx_before', 0), kargs.get('padx_after', 0)], dtype=np.uint32)
        filter_data_shape = [3]
        stride_data_shape = [3]
        pad_data_shape = [3, 2]

        attrs = ir_graph.IrAttributes()
        input_pool_type = kargs['pool_type']
        if input_pool_type == ir_graph.QNN_OP_POOL_AVG_3D:
            attrs.add(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE,
                    ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE,
                                            filter_data_shape,
                                            filter_data,
                                            ir_graph.QNN_DATATYPE_UINT_32))
            attrs.add(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_STRIDE,
                    ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_STRIDE,
                                            stride_data_shape,
                                            stride_data,
                                            ir_graph.QNN_DATATYPE_UINT_32))
            attrs.add(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT,
                    ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT,
                                            pad_data_shape,
                                            pad_data,
                                            ir_graph.QNN_DATATYPE_UINT_32))
            attrs.addUint32(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE,
                            kargs.get(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE, ir_graph.IR_OP_POOL_ROUNDING_MODE_FLOOR))
            attrs.addBool(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES,
                          kargs.get(ir_graph.QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES, False))
        elif input_pool_type == ir_graph.QNN_OP_POOL_MAX_3D:
            attrs.add(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_FILTER_SIZE,
                    ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_FILTER_SIZE,
                                            filter_data_shape,
                                            filter_data,
                                            ir_graph.QNN_DATATYPE_UINT_32))
            attrs.add(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_STRIDE,
                    ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_STRIDE,
                                            stride_data_shape,
                                            stride_data,
                                            ir_graph.QNN_DATATYPE_UINT_32))
            attrs.add(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_PAD_AMOUNT,
                    ir_graph.IrStaticTensor(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_PAD_AMOUNT,
                                            pad_data_shape,
                                            pad_data,
                                            ir_graph.QNN_DATATYPE_UINT_32))
            attrs.addUint32(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_ROUNDING_MODE,
                            kargs.get(ir_graph.QNN_OP_POOL_MAX_3D_PARAM_ROUNDING_MODE, ir_graph.IR_OP_POOL_ROUNDING_MODE_FLOOR))

        attrs.addUint8(ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY,
                       kargs.get(ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY,
                                 ir_graph.PADDING_SIZE_EXPLICIT),
                       ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addString(ir_graph.IR_OP_POOL_TYPE,
                        kargs.get(ir_graph.IR_OP_POOL_TYPE),
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        self.c_op = ir_graph.Pool3dOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        if self.pool_type == ir_graph.QNN_OP_POOL_AVG_3D:
            self.macs = self.get_general_macs_val(output_shapes) * self.filter_size[0] * self.filter_size[1]


class PreluOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_PRELU
    LEGACY_TRANSLATION_KEY = 'prelu'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        self.c_op = ir_graph.PreluOp(name if name is not None else "placeholder_aisw", attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class ProposalOp(Op):
    TRANSLATION_KEY = 'proposal'
    LEGACY_TRANSLATION_KEY = 'proposal'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('feat_stride', kargs)
        self.assertattr('scales', kargs)
        self.assertattr('ratios', kargs)
        self.assertattr('anchor_base_size', kargs)
        self.assertattr('min_bbox_size', kargs)
        self.assertattr('max_num_proposals', kargs)
        self.assertattr('max_num_rois', kargs)
        self.assertattr('iou_threshold_nms', kargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = [1, 1, self.max_num_rois, 5]
        return [output_shape]

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL


class QuantizeOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_QUANTIZE
    LEGACY_TRANSLATION_KEY = 'quantize'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # TODO Remove once fully migrated to QNNIR
        attrs.addUint32(ir_graph.IR_OP_QUANTIZE_PARAM_BW, kwargs.get(ir_graph.IR_OP_QUANTIZE_PARAM_BW),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addFloat(ir_graph.IR_OP_QUANTIZE_PARAM_MIN, kwargs.get(ir_graph.IR_OP_QUANTIZE_PARAM_MIN, 0.0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addFloat(ir_graph.IR_OP_QUANTIZE_PARAM_MAX, kwargs.get(ir_graph.IR_OP_QUANTIZE_PARAM_MAX, 0.0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addFloat(ir_graph.IR_OP_QUANTIZE_PARAM_SCALE, kwargs.get(ir_graph.IR_OP_QUANTIZE_PARAM_SCALE, 0.0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addInt32(ir_graph.IR_OP_QUANTIZE_PARAM_OFFSET, kwargs.get(ir_graph.IR_OP_QUANTIZE_PARAM_OFFSET, 0),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        attrs.addBool(ir_graph.IR_OP_QUANTIZE_PARAM_IS_SYMMETRIC, kwargs.get(ir_graph.IR_OP_QUANTIZE_PARAM_IS_SYMMETRIC, False),
                      ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        self.c_op = ir_graph.QuantizeOp(name, attrs)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ReduceOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_REDUCE

    ir_to_legacy_type = {
        ir_graph.QNN_OP_REDUCE_MAX: "reduce_max",
        ir_graph.QNN_OP_REDUCE_MEAN: "reduce_mean",
        ir_graph.QNN_OP_REDUCE_MIN: "reduce_min",
        ir_graph.QNN_OP_REDUCE_PROD: "reduce_prod",
        ir_graph.QNN_OP_REDUCE_SUM: "reduce_sum",
        ir_graph.IR_OP_REDUCE_L2: "reduce_l2",
    }

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        axes_attr = np.array(kwargs.get("axes"), dtype=np.int32)
        keep_dims_attr = kwargs.get("keep_dims", False)

        attrs.addString(ir_graph.IR_OP_REDUCE_PARAM_TYPE,
                        kwargs.get(ir_graph.IR_OP_REDUCE_PARAM_TYPE),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        attrs.add(ir_graph.IR_OP_REDUCE_PARAM_AXES,
                  ir_graph.IrStaticTensor(ir_graph.IR_OP_REDUCE_PARAM_AXES,
                                          list(axes_attr.shape),
                                          axes_attr,
                                          ir_graph.QNN_DATATYPE_UINT_32))
        attrs.addBool(ir_graph.IR_OP_REDUCE_PARAM_KEEP_DIMS, keep_dims_attr)

        self.c_op = ir_graph.ReduceOp(name, attrs)

    @property
    def type(self):
        return self.ir_to_legacy_type[self.c_op.reduce_type]

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        axes = self.axes.tolist()
        if not self.keep_dims:
            if self.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC and axes == [1, 2, 3]:
                buf.set_axis_format(AxisTracker.AxisFormat.NC)
            elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW and axes == [2, 3, 4]:
                buf.set_axis_format(AxisTracker.AxisFormat.NC)
            elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NSC and axes == [1, 2]:
                buf.set_axis_format(AxisTracker.AxisFormat.NC)
            elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NCS and axes == [2, 3]:
                buf.set_axis_format(AxisTracker.AxisFormat.NC)
            elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NFC and axes == [1]:
                buf.set_axis_format(AxisTracker.AxisFormat.NC)
            elif self.data_axis_formats[0] == AxisTracker.AxisFormat.NCF and axes == [2]:
                buf.set_axis_format(AxisTracker.AxisFormat.NC)
            elif len(buf.shape) == 1:
                buf.set_axis_format(AxisTracker.AxisFormat.ANY)
            else:
                buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)
        else:
            super(ReduceOp, self).populate_axis_format(graph, buf, src_axis_order, encodings, input_buffers)


class ReshapeOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_RESHAPE
    LEGACY_TRANSLATION_KEY = 'reshape'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        shape_attr = np.array(kwargs.get("shape"), dtype=np.int32)
        attrs.add(ir_graph.IR_OP_RESHAPE_PARAM_SHAPE,
                  ir_graph.IrStaticTensor(ir_graph.IR_OP_RESHAPE_PARAM_SHAPE,
                                          list(shape_attr.shape),
                                          shape_attr,
                                          ir_graph.QNN_DATATYPE_INT_32),
                  ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)
        self.c_op = ir_graph.ReshapeOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        # TODO: replace with c++ implementation once once buffer/op graph facade is integrated
        in_buf = input_buffers[0]
        if self.data_axis_formats[0] == AxisTracker.AxisFormat.NC:
            if len(buf.shape) == 5 and buf.shape[2:] == [1,1,1] and \
                    in_buf.shape[0] == buf.shape[0]:
                buf.set_axis_format(AxisTracker.AxisFormat.NCDHW)
            elif len(buf.shape) == 5 and buf.shape[1:4] == [1,1,1] and \
                    in_buf.shape[0] == buf.shape[0]:
                buf.set_axis_format(AxisTracker.AxisFormat.NDHWC)
            elif len(buf.shape) == 4 and buf.shape[2:] == [1,1] and \
                    in_buf.shape[0] == buf.shape[0]:
                buf.set_axis_format(AxisTracker.AxisFormat.NCS)
            elif len(buf.shape) == 4 and buf.shape[1:3] == [1,1] and \
                    in_buf.shape[0] == buf.shape[0]:
                buf.set_axis_format(AxisTracker.AxisFormat.NSC)
            elif len(buf.shape) == 3 and buf.shape[2:] == [1] and \
                    in_buf.shape[0] == buf.shape[0]:
                buf.set_axis_format(AxisTracker.AxisFormat.NCF)
            elif len(buf.shape) == 3 and buf.shape[1:2] == [1] and \
                    in_buf.shape[0] == buf.shape[0]:
                buf.set_axis_format(AxisTracker.AxisFormat.NFC)
            else:
                buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)
        elif self.data_axis_formats[0] in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NCDHW,
                                           AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NCS,
                                           AxisTracker.AxisFormat.NFC, AxisTracker.AxisFormat.NCF] and \
                len(buf.shape) == 2 and buf.shape[0] == in_buf.shape[0]:
            # When batch size is preserved and everything else is collapsed into 1 dimension,
            # it is NF format. e.g. input to FC layer
            buf.set_axis_format(AxisTracker.AxisFormat.NF)
        elif len(buf.shape) == 1:
            # When output buffer is 1D, set output buffer axis_format as ANY
            buf.set_axis_format(AxisTracker.AxisFormat.ANY)
        # if the ReshapeOp can be replaced with TransposeOp,
        # let ReshapeOp populate the known axis_formats (like NCS) rather than NONTRIVIAL
        elif len(in_buf.shape) == len(buf.shape) and self.data_axis_formats[0] == AxisTracker.AxisFormat.NCF and \
                AxisTracker.permute_shape(in_buf.shape, AxisTracker.AxisFormat.NCF_TO_NFC) == buf.shape and \
                in_buf.shape[1] == 1 and buf.shape[-1] == 1:
            buf.set_axis_format(AxisTracker.AxisFormat.NFC)
        elif len(in_buf.shape) == len(buf.shape) and self.data_axis_formats[0] == AxisTracker.AxisFormat.NCS and \
                AxisTracker.permute_shape(in_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC) == buf.shape and \
                in_buf.shape[1] == 1 and buf.shape[-1] == 1:
            buf.set_axis_format(AxisTracker.AxisFormat.NSC)
        elif len(in_buf.shape) == len(buf.shape) and self.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW and \
                AxisTracker.permute_shape(in_buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC) == buf.shape and \
                in_buf.shape[1] == 1 and buf.shape[-1] == 1:
            buf.set_axis_format(AxisTracker.AxisFormat.NDHWC)
        elif len(in_buf.shape) == len(buf.shape) and self.data_axis_formats[0] == AxisTracker.AxisFormat.NFC and \
                AxisTracker.permute_shape(in_buf.shape, AxisTracker.AxisFormat.NFC_TO_NCF) == buf.shape and \
                in_buf.shape[-1] == 1 and buf.shape[1] == 1:
            buf.set_axis_format(AxisTracker.AxisFormat.NCF)
        elif len(in_buf.shape) == len(buf.shape) and self.data_axis_formats[0] == AxisTracker.AxisFormat.NSC and \
                AxisTracker.permute_shape(in_buf.shape, AxisTracker.AxisFormat.NSC_TO_NCS) == buf.shape and \
                in_buf.shape[-1] == 1 and buf.shape[1] == 1:
            buf.set_axis_format(AxisTracker.AxisFormat.NCS)
        elif len(in_buf.shape) == len(buf.shape) and self.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC and \
                AxisTracker.permute_shape(in_buf.shape, AxisTracker.AxisFormat.NDHWC_TO_NCDHW) == buf.shape and \
                in_buf.shape[-1] == 1 and buf.shape[1] == 1:
            buf.set_axis_format(AxisTracker.AxisFormat.NCDHW)
        else:
            buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)


class ResizeOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_RESIZE
    LEGACY_TRANSLATION_KEY = 'resize'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # QNN parameters
        attrs.addBool(ir_graph.QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE,
                      kwargs.get(ir_graph.QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE,
                                 False),
                      ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addUint32(ir_graph.QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE,
                        kwargs.get(ir_graph.QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE,
                                   ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addUint32(ir_graph.QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE,
                        kwargs.get(ir_graph.QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE,
                                   ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addUint32(ir_graph.QNN_OP_RESIZE_PARAM_NEAREST_MODE,
                        kwargs.get(ir_graph.QNN_OP_RESIZE_PARAM_NEAREST_MODE,
                                   ir_graph.QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR),
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addFloat(ir_graph.QNN_OP_RESIZE_PARAM_CUBIC_COEFF,
                       kwargs.get(ir_graph.QNN_OP_RESIZE_PARAM_CUBIC_COEFF, -0.75),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

        # Supplemental parameters
        scale_depth = kwargs.get(ir_graph.IR_OP_RESIZE_PARAM_SCALE_DEPTH)
        if scale_depth is not None:
            attrs.addFloat(ir_graph.IR_OP_RESIZE_PARAM_SCALE_DEPTH,
                           scale_depth,
                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        scale_height = kwargs.get(ir_graph.IR_OP_RESIZE_PARAM_SCALE_HEIGHT)
        if scale_height is not None:
            attrs.addFloat(ir_graph.IR_OP_RESIZE_PARAM_SCALE_HEIGHT,
                           kwargs.get(ir_graph.IR_OP_RESIZE_PARAM_SCALE_HEIGHT),
                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        attrs.addFloat(ir_graph.IR_OP_RESIZE_PARAM_SCALE_WIDTH,
                       kwargs.get(ir_graph.IR_OP_RESIZE_PARAM_SCALE_WIDTH),
                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        self.c_op = ir_graph.ResizeOp(name, attrs)
        # TODO: Re-evaluate support for these. QNN has no support currently
        # self.addattr('pad_value', kwargs, 0.0)
        # self.addattr('maintain_aspect_ratio', kwargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].axis_format = graph.src_axis_order.get_axis_format(len(input_buffers[0].shape))
        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class RoiAlignOp(Op):
    TRANSLATION_KEY = 'roi_align'
    LEGACY_TRANSLATION_KEY = 'roi_align'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('sampling_ratio', kargs)
        self.addattr('mode', kargs, 'avg')
        # implode batch parameters
        self.addattr('tiled_batch_h', kargs, -1)
        self.addattr('tiled_batch_w', kargs, -1)
        self.addattr('batch_pad_h', kargs, -1)
        self.addattr('batch_pad_w', kargs, -1)
        self.addattr('pad_value', kargs, 0.0)
        self.addattr('aligned', kargs, False)
        self.addattr('allow_invalid_roi', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        def calc_tiled_height(in_height):
            return self.tiled_batch_h * in_height + (self.tiled_batch_h - 1) * self.batch_pad_h

        def calc_tiled_width(in_width):
            return self.tiled_batch_w * in_width + (self.tiled_batch_w - 1) * self.batch_pad_w

        input_shape = input_shapes[0][:]
        _, _, _, channel = axis_order.extract_2d_spatial_dims(input_shape)
        num_rois = input_shapes[2][0]

        if self.tiled_batch_h > 0:
            output_shape = axis_order.format_2d_spatial_output_shape(batch_size=1,
                                                                     height=calc_tiled_height(
                                                                         self.pooled_size_h),
                                                                     width=calc_tiled_width(
                                                                         self.pooled_size_w),
                                                                     channel=channel)
        else:
            output_shape = axis_order.format_2d_spatial_output_shape(batch_size=num_rois,
                                                                     height=self.pooled_size_h,
                                                                     width=self.pooled_size_w,
                                                                     channel=channel)
        return [output_shape]

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        input_buffers[0].set_axis_format(graph.src_axis_order.get_axis_format(len(input_buffers[0].shape)))
        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class RoiPoolingOp(Op):
    TRANSLATION_KEY = 'roi_pooling'
    LEGACY_TRANSLATION_KEY = 'roi_pooling'

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('spatial_scale', kargs)
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.output_shape[:]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class RolledLstmOp(Op):
    TRANSLATION_KEY = ir_graph.IR_OP_ROLLED_LSTM
    LEGACY_TRANSLATION_KEY = TRANSLATION_KEY

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        direction = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_DIRECTION, ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD)
        attrs.addUint32(ir_graph.QNN_OP_LSTM_PARAM_DIRECTION, direction)

        cell_clip_threshold = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD)
        if cell_clip_threshold is not None:
            attrs.addFloat(ir_graph.QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD, cell_clip_threshold)

        output_clip_threshold = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD)
        if output_clip_threshold is not None:
            attrs.addFloat(ir_graph.QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD, output_clip_threshold)

        time_major = kwargs.get(ir_graph.QNN_OP_LSTM_PARAM_TIME_MAJOR)
        if time_major is not None:
            attrs.addBool(ir_graph.QNN_OP_LSTM_PARAM_TIME_MAJOR, time_major)

        # supplemental attributes
        hidden_size = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_HIDDEN_SIZE)
        if hidden_size is None:
            raise ValueError("hidden_size attribute must be specified for RolledLstmOp {}".format(name))
        attrs.addUint32(ir_graph.IR_OP_LSTM_PARAM_HIDDEN_SIZE, hidden_size,
                        ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        reset_state_at_time_step_0 = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_RESET_STATE_AT_TIME_STEP_0)
        if reset_state_at_time_step_0 is not None:
            attrs.addBool(ir_graph.IR_OP_LSTM_PARAM_RESET_STATE_AT_TIME_STEP_0, reset_state_at_time_step_0,
                          ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        h_0_input_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_H_0_INPUT_NAME)
        if h_0_input_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_H_0_INPUT_NAME, h_0_input_name,
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        c_0_input_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_C_0_INPUT_NAME)
        if c_0_input_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_C_0_INPUT_NAME, c_0_input_name,
                            ir_graph.IR_ATTR_USAGE_SUPPLEMENTAL)

        # legacy attributes
        sequence_continuation_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_SEQUENCE_CONTINUATION_NAME)
        if sequence_continuation_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_SEQUENCE_CONTINUATION_NAME, sequence_continuation_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        x_static_name = kwargs.get(ir_graph.IR_OP_LSTM_PARAM_X_STATIC_NAME)
        if x_static_name is not None:
            attrs.addString(ir_graph.IR_OP_LSTM_PARAM_X_STATIC_NAME, x_static_name,
                            ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        self.c_op = ir_graph.RolledLstmOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Override input buffer axis format
        in_data_buf = input_buffers[0]
        # Allow only 3D input
        if len(in_data_buf.shape) == 3:
            in_data_buf.axis_format = graph.src_axis_order.get_axis_format(len(in_data_buf.shape),
                                                                           time_series_format=True)
        # Set initial h/c buffers to NONTRIVIAL
        for in_buf in input_buffers[1:3]:
            in_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)
        self.params_count = self.params_count_c_op_wrapper(input_shapes, output_shapes, axis_order)


class ScatterElementsOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_SCATTER_ELEMENTS
    LEGACY_TRANSLATION_KEY = 'scatter_elements'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS in kwargs:
            attrs.addInt32(ir_graph.QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS))
        if ir_graph.QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION in kwargs:
            attrs.addUint32(ir_graph.QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION, kwargs.get(ir_graph.QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION))
        self.c_op = ir_graph.ScatterElementsOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce indices and updates buffer axis format to NONTRIVIAL
        input_buffers[1].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        input_buffers[2].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class ScatterNDOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_SCATTER_ND
    LEGACY_TRANSLATION_KEY = 'scatter_nd'

    class ReductionTypes(Enum):
        REDUCTION_NONE = "none"
        REDUCTION_ADD = "add"
        REDUCTION_MUL = "mul"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addUint32(ir_graph.QNN_OP_SCATTER_ND_PARAM_REDUCTION, kwargs.get(ir_graph.QNN_OP_SCATTER_ND_PARAM_REDUCTION, ir_graph.QNN_OP_SCATTER_ND_REDUCTION_NONE))
        self.c_op = ir_graph.ScatterNDOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def populate_data_axis_formats(self, graph, input_buffers):
        # Enforce indices buffer axis format to NONTRIVIAL
        input_buffers[1].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        input_buffers[2].axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        super().populate_data_axis_formats(graph, input_buffers)

    def populate_axis_format(self, graph, buf, src_axis_order, encodings, input_buffers):
        # Output format of ScatterND depends only on input_0 format
        buf.set_axis_format(self.data_axis_formats[0])

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class SparseToDenseOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_SPARSE_TO_DENSE
    LEGACY_TRANSLATION_KEY = 'sparse_to_dense'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        self.c_op = ir_graph.SparseToDenseOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class SplitOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_SPLIT
    LEGACY_TRANSLATION_KEY = 'split'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addInt32(ir_graph.QNN_OP_SPLIT_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_SPLIT_PARAM_AXIS))

        split_index_data = np.array(kwargs.get(ir_graph.QNN_OP_SPLIT_PARAM_SPLIT_INDEX, []), dtype=np.uint32)
        split_index = ir_graph.IrStaticTensor(ir_graph.QNN_OP_SPLIT_PARAM_SPLIT_INDEX,
                                              list(split_index_data.shape),
                                              split_index_data,
                                              ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_SPLIT_PARAM_SPLIT_INDEX, split_index)

        self.c_op = ir_graph.SplitOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class StridedSliceOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_STRIDED_SLICE
    LEGACY_TRANSLATION_KEY = 'strided_slice'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        if ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES not in kwargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES))
        ranges_data = np.array(kwargs.get(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES))

        # when we convert data from np.int64 to np.int32, there will be incorrect conversion of value, so we need to clip first
        # e.g, np.array(-9223372036854775807).astype(np.int32) become np.array(1)
        ranges_data = np.clip(ranges_data, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        ranges = ir_graph.IrStaticTensor(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                         list(ranges_data.shape),
                                         ranges_data,
                                         ir_graph.QNN_DATATYPE_INT_32)
        attrs.add(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES, ranges)

        attrs.addUint32(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK, kwargs.get(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK, 0))
        attrs.addUint32(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_END_MASK, kwargs.get(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_END_MASK, 0))
        attrs.addUint32(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES, kwargs.get(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES, 0))
        attrs.addUint32(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK, kwargs.get(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK, 0))

        self.c_op = ir_graph.StridedSliceOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class SoftmaxOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_SOFTMAX
    LEGACY_TRANSLATION_KEY = 'softmax'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        attrs.addInt32(ir_graph.QNN_OP_SOFTMAX_PARAM_AXIS, kwargs.get(ir_graph.QNN_OP_SOFTMAX_PARAM_AXIS, -1))
        attrs.addFloat(ir_graph.QNN_OP_SOFTMAX_PARAM_BETA, kwargs.get(ir_graph.QNN_OP_SOFTMAX_PARAM_BETA, 1.0))
        self.c_op = ir_graph.SoftmaxOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class SpaceToBatchOp(Op):
    TRANSLATION_KEY = 'space_to_batch'
    LEGACY_TRANSLATION_KEY = 'space_to_batch'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('block_shape', kargs)
        self.addattr('paddings', kargs, [[0, 0], [0, 0]])

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        input_batch, input_height, input_width, input_channel = axis_order.extract_2d_spatial_dims(
            input_shapes[0])
        output_batch = input_batch * self.block_shape[0] * self.block_shape[1]
        output_height = (input_height + self.paddings[0][0] + self.paddings[0][1]) // self.block_shape[0]
        output_width = (input_width + self.paddings[1][0] + self.paddings[1][1]) // self.block_shape[1]
        output_shape = axis_order.format_2d_spatial_output_shape(batch_size=output_batch,
                                                                 channel=input_channel,
                                                                 height=output_height,
                                                                 width=output_width)
        return [output_shape]


class SpaceToDepthOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_SPACE_TO_DEPTH
    LEGACY_TRANSLATION_KEY = 'space_to_depth'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        # data_format
        data_format = kwargs.get(ir_graph.IR_OP_SPACE_TO_DEPTH_PARAM_DATA_FORMAT, "NHWC")
        attrs.addString(ir_graph.IR_OP_SPACE_TO_DEPTH_PARAM_DATA_FORMAT,
                        data_format,
                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)

        # block_size
        block_size_data = np.array(kwargs.get(ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE), dtype=np.uint32)
        block_size = ir_graph.IrStaticTensor(ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE,
                                             [2],
                                             block_size_data,
                                             ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE, block_size)

        # mode
        attrs.addUint32(ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, \
                        kwargs.get(ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, ir_graph.QNN_OP_SPACE_TO_DEPTH_MODE_DCR))

        self.c_op = ir_graph.SpaceToDepthOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class ThresholdedReluOp(Op):
    TRANSLATION_KEY = "ThresholdedRelu"
    LEGACY_TRANSLATION_KEY = "thresholded_relu"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('alpha', kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.macs_c_op_wrapper(input_shapes, output_shapes, axis_order)


class TileOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_TILE
    LEGACY_TRANSLATION_KEY = 'tile'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()

        multiples = np.array(kwargs.get(ir_graph.QNN_OP_TILE_PARAM_MULTIPLES), dtype=np.uint32)
        multiples_attr = ir_graph.IrStaticTensor(ir_graph.QNN_OP_TILE_PARAM_MULTIPLES,
                                      list(multiples.shape),
                                      multiples,
                                      ir_graph.QNN_DATATYPE_UINT_32)
        attrs.add(ir_graph.QNN_OP_TILE_PARAM_MULTIPLES, multiples_attr)

        self.c_op = ir_graph.TileOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class TopKOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_TOP_K
    LEGACY_TRANSLATION_KEY = 'topk'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY, num_outputs=2)
        attrs = ir_graph.IrAttributes()
        attrs.addUint32(ir_graph.QNN_OP_TOP_K_PARAM_K, (kwargs.get(ir_graph.QNN_OP_TOP_K_PARAM_K)))

        self.c_op = ir_graph.TopKOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)


class UdlOp(Op):
    TRANSLATION_KEY = 'udl'
    LEGACY_TRANSLATION_KEY = 'udl'

    def __init__(self, name, layer_type, blob, output_dims, expected_input_axis_orders,
                 expected_output_axis_orders):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.layer_type = layer_type
        self.blob = blob
        self.output_dims = output_dims
        self.expected_input_axis_orders = expected_input_axis_orders
        self.expected_output_axis_orders = expected_output_axis_orders

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.output_dims


class UnpackOp(Op):
    TRANSLATION_KEY = ir_graph.QNN_OP_UN_PACK
    LEGACY_TRANSLATION_KEY = 'unpack'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        attrs = ir_graph.IrAttributes()
        if ir_graph.QNN_OP_UN_PACK_PARAM_AXIS not in kargs:
            raise KeyError("Op %s missing required argument %s" % (name, ir_graph.QNN_OP_UN_PACK_PARAM_AXIS))
        attrs.addInt32(ir_graph.QNN_OP_UN_PACK_PARAM_AXIS, kargs.get(ir_graph.QNN_OP_UN_PACK_PARAM_AXIS))
        attrs.addInt32(ir_graph.IR_OP_UN_PACK_PARAM_NUM, kargs.get(ir_graph.IR_OP_UN_PACK_PARAM_NUM), ir_graph.IrAttrUsageType.IR_ATTR_USAGE_LEGACY)
        self.c_op = ir_graph.UnpackOp(name, attrs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.infer_shape_c_op_wrapper(input_shapes, input_axis_formats, num_outputs, axis_order)
