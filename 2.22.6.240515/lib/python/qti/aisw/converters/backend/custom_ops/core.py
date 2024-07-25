# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.custom_ops.core import *
from qti.aisw.converters.common.custom_ops.utils.config_helpers import *
import numpy
from collections import OrderedDict
from enum import Enum


# ------------------------------------------------------------------------------
#   Qnn config Core Classes
# ------------------------------------------------------------------------------
class QnnTensorLayouts(Enum):
    NHWC = 0
    NCHW = 1
    NONTRIVIAL = 2
    UNDEFINED = 3

    def describe(self):
        return self.name, self.value

    @classmethod
    def default(cls):
        return cls.UNDEFINED

    @classmethod
    def cast(cls, enum_type):
        if not isinstance(enum_type, Enum):
            raise TypeError("Enum cast failed. Expected type: {}, instead got type: {}"
                            .format(type(Enum), type(enum_type)))
        err_val = enum_type.value
        if enum_type.value == 1:
            err_val = "{}: \nOpDefs cannot have BACKEND_SPECIFIC values for offline tools. " \
                      "Specify explicit values with SupplementalOpDef.".format(err_val)
        elif enum_type.value == 0 or enum_type.value == 2:
            return cls.NHWC
        elif enum_type.value == 3:
            return cls.NCHW
        elif enum_type.value == 4:
            return cls.NONTRIVIAL
        raise TypeError('Failed to cast enum value: {}'.format(err_val))


class QnnDatatype(Enum):
    """
    Define the allowable datatypes
    """
    QNN_DATATYPE_INT_8 = 2
    QNN_DATATYPE_INT_16 = 3
    QNN_DATATYPE_INT_32 = 4
    QNN_DATATYPE_INT_64 = 5
    QNN_DATATYPE_UINT_8 = 6
    QNN_DATATYPE_UINT_16 = 7
    QNN_DATATYPE_UINT_32 = 8
    QNN_DATATYPE_UINT_64 = 9
    QNN_DATATYPE_FLOAT_16 = 10
    QNN_DATATYPE_FLOAT_32 = 11
    QNN_DATATYPE_SFIXED_POINT_8 = 12
    QNN_DATATYPE_SFIXED_POINT_16 = 13
    QNN_DATATYPE_SFIXED_POINT_32 = 14
    QNN_DATATYPE_UFIXED_POINT_8 = 15
    QNN_DATATYPE_UFIXED_POINT_16 = 16
    QNN_DATATYPE_UFIXED_POINT_32 = 17
    QNN_DATATYPE_BOOL_8 = 18

    def describe(self):
        return self.name, self.value

    @classmethod
    def default(cls):
        return cls.QNN_DATATYPE_FLOAT_32

    @classmethod
    def cast(cls, enum_type):
        if not isinstance(enum_type, Enum):
            raise TypeError("Enum cast failed. Expected type: {}, instead got type: {}"
                            .format(type(Enum), type(enum_type)))
        for member in cls.__members__.values():
            if member.value == enum_type.value:
                return member
        err_val = enum_type.value
        if enum_type.value == 0:
            err_val = "{}:" \
                      "\nOpDefs cannot have BACKEND_SPECIFIC values. " \
                      "Specify values with SupplementalOpDef.".format(err_val)
        raise TypeError('Failed to cast enum value: {}'.format(err_val))

    @classmethod
    def convert_op_def_datatypes(cls, op_def_datatypes: List) -> List:
        datatypes = []
        if len(op_def_datatypes) == 1 and op_def_datatypes[0].value == 1:
            return cls.get_types()
        for datatype in op_def_datatypes:
            if not isinstance(datatype, Enum):
                raise TypeError("Enum conversion failed. Expected type: {}, instead got type: {}"
                                .format(type(Enum), type(datatype)))
            else:
                datatypes.append(cls.cast(datatype))

        return datatypes

    @classmethod
    def get_types(cls, category='integer'):
        values = list(cls.__members__.values())
        if category == 'integer':
            return values[0:4]
        elif category == 'float':
            return [values[9]]
        elif category == 'float_fp16':
            return [values[8]]
        elif category == 'unsigned_integer':
            return values[4:8]
        elif category == "signed_quantized":
            return values[10:14]
        elif category == "unsigned_quantized":
            return values[14:len(values)]
        return values


dtype_to_qnn = {
    # int types
    numpy.dtype('int8'): QnnDatatype.QNN_DATATYPE_INT_8,
    numpy.dtype('int16'): QnnDatatype.QNN_DATATYPE_INT_16,
    numpy.dtype('int32'): QnnDatatype.QNN_DATATYPE_INT_32,
    numpy.dtype('int64'): QnnDatatype.QNN_DATATYPE_INT_64,
    numpy.dtype('uint8'): QnnDatatype.QNN_DATATYPE_UINT_8,
    numpy.dtype('uint16'): QnnDatatype.QNN_DATATYPE_UINT_16,
    numpy.dtype('uint32'): QnnDatatype.QNN_DATATYPE_UINT_32,
    numpy.dtype('uint64'): QnnDatatype.QNN_DATATYPE_UINT_64,

    # float types
    numpy.dtype('float16'): QnnDatatype.QNN_DATATYPE_FLOAT_16,
    numpy.dtype('float32'): QnnDatatype.QNN_DATATYPE_FLOAT_32,

    # bool type
    numpy.dtype('bool'): QnnDatatype.QNN_DATATYPE_BOOL_8,

    int: QnnDatatype.QNN_DATATYPE_INT_32,
    float: QnnDatatype.QNN_DATATYPE_FLOAT_32,
    str: QnnDatatype.QNN_DATATYPE_UINT_8,
    bool: QnnDatatype.QNN_DATATYPE_BOOL_8
}

def qnn_to_dtype(qnn_type):
    qnn_to_dtype_mapping = {
        # int types
        QnnDatatype.QNN_DATATYPE_INT_8: 'int8',
        QnnDatatype.QNN_DATATYPE_INT_16: 'int16',
        QnnDatatype.QNN_DATATYPE_INT_32: 'int32',
        QnnDatatype.QNN_DATATYPE_INT_64: 'int64',
        QnnDatatype.QNN_DATATYPE_UINT_8: 'uint8',
        QnnDatatype.QNN_DATATYPE_UINT_16: 'uint16',
        QnnDatatype.QNN_DATATYPE_UINT_32: 'uint32',
        QnnDatatype.QNN_DATATYPE_UINT_64: 'uint64',

        # float types
        QnnDatatype.QNN_DATATYPE_FLOAT_16: 'float16',
        QnnDatatype.QNN_DATATYPE_FLOAT_32: 'float32',

        # bool type
        QnnDatatype.QNN_DATATYPE_BOOL_8: 'bool',
    }
    return qnn_to_dtype_mapping[qnn_type]


def is_quant_type(qnn_type):
    return qnn_type in QnnDatatype.get_types('signed_quantized') \
           or qnn_type in QnnDatatype.get_types('unsigned_quantized')


def get_np_type_from_backend_type(qnn_type):
    if not isinstance(qnn_type, QnnDatatype):
        raise TypeError("Unknown QNN datatype conversion requested: {}".format(qnn_type))
    reverse_dict = {v: k for k, v in dtype_to_qnn.items()}
    return reverse_dict[qnn_type]


def get_qnn_type(data):
    dtype = type(data) if not isinstance(data, numpy.ndarray) else data.dtype
    if isinstance(data, (tuple, list)):
        dtypes = [type(data_elem) for data_elem in data]
        if check_all_equal(dtypes):
            dtype = dtypes[0]
        else:
            # extremely unlikely, but we'll check anyway
            raise TypeError("Data value is an iterator with inconsistent types: {}".format(dtypes))
    return dtype_to_qnn[dtype]


def convert_to_backend_type_from_numpy(dtype):
    if dtype not in dtype_to_qnn:
        dtype = numpy.random.randn(1).astype(dtype).dtype
    return dtype_to_qnn[dtype]


def get_internal_dtype(data, op_attr):
    try:
        candidate_type = get_qnn_type(data)
    except KeyError:
        src_type = type(data) if not isinstance(data, numpy.ndarray) else data.dtype
        raise KeyError("The provided data_type: {} is not a valid qnn_type".format(src_type))

    if op_attr.allowed_data_types and candidate_type not in op_attr.allowed_data_types:
        src_type = type(data) if not isinstance(data, numpy.ndarray) else data.dtype
        raise TypeError("The provided datatype: {} is not a valid datatype defined for: {}. "
                        "Expected one of {}"
                        .format(src_type, op_attr.name, op_attr.allowed_data_types))
    return candidate_type


class TensorInfo(CustomTensorInfo):
    allowed_data_types = aggregate_property('allowed_data_types', QnnDatatype)
    shape = property_type('shape', str)  # string for now, should be something interpretable

    # TODO: repeated, constraints, tensor_layout enum
    def __init__(self, **tensor):
        super().__init__(**tensor)
        self.layout = QnnTensorLayouts.default()

    @staticmethod
    def from_translator_tensor_element(tensor_element):
        return TensorInfo(name=tensor_element.name,
                          allowed_data_types=QnnDatatype.convert_op_def_datatypes(tensor_element
                                                                                  .datatypes),
                          shape=tensor_element.shape,
                          layout=QnnTensorLayouts.cast(tensor_element.layout)
                          if hasattr(tensor_element, "layout") else QnnTensorLayouts.default(),
                          repeated=tensor_element.repeated if hasattr(tensor_element, "repeated")
                          else False,
                          rank=tensor_element.rank,
                          default_value=tensor_element.default.value if tensor_element.default.value != ""
                          else None,
                          static=tensor_element.is_static_tensor if hasattr(tensor_element,
                                                                            "is_static_tensor")
                          else False
                          )


class ScalarParam(CustomScalarParam):
    data_type = property_type('data_type', QnnDatatype)

    # TODO: Add default value, allowed values
    def __init__(self, data, data_type=None):
        if data_type is None:
            data_type = get_qnn_type(data)  # assign datatype based on data
        super().__init__(data, data_type)


class TensorParam(CustomTensorParam):
    def __init__(self, data, tensor_info):
        super().__init__(data, tensor_info)
        self.data_type = get_qnn_type(data) if data is not None else None   # now set datatype based on data
        self.store_in_bin = tensor_info.static

    def as_dict(self):
        temp_dict = super(TensorParam, self).as_dict()
        temp_dict.update(store_in_bin=self.store_in_bin)
        return temp_dict


class StringParam(ScalarParam):
    # TODO: qnn does not accept raw strings
    def __init__(self, value):
        super().__init__(value, QnnDatatype.QNN_DATATYPE_UINT_8)


class Param(CustomParam):
    param_type = property_type('param_type', ParamTypes)
    param = union_property('param', [type(None), ScalarParam, TensorParam])

    def __init__(self, name, param_type, param=None):
        super(Param, self).__init__(name, param_type, param)


class Operator(CustomOperator):
    input = aggregate_property('input', TensorInfo)
    output = aggregate_property('output', TensorInfo)
    param = aggregate_property('param', TensorInfo)

    # setting package name statically for now, ideally it would be done from headers
    def __init__(self, type_name, package_name="qti.aisw", name="", use_default_translation=False):
        super().__init__(type_name, package_name, name, use_default_translation)

    # TODO: would ideally be done on c++ schema side, doing it here for now
    def convert_to_qnn_op_config(self):
        """
        Converts a tensor info into a Qnn Op Config
        :return: A structure which resembles its C++ counterpart as defined in QnnOpPackage.h
        """
        qnn_op_config = {'name': self.name,
                         'packageName': self.package_name,
                         'typeName': self.type_name,
                         'numOfParams': len(self.param),
                         'params': self.param,
                         'numOfInputs': len(self.input),
                         'inputs': self.input,
                         'numOfOutputs': len(self.output),
                         'outputs': self.output
                         }
        return qnn_op_config


class QnnCustomOp(CustomOp):
    __metaclass__ = ABCMeta
    methods = dict()
    inputs = aggregate_property('inputs', TensorInfo)
    outputs = aggregate_property('outputs', TensorInfo)
    param = aggregate_property('params', Param)

    def __init__(self,
                 op_type: str,
                 input_tensors: List[TensorInfo],
                 output_tensors: List[TensorInfo],
                 *,
                 params: Optional[List[Param]] = None,
                 param_info: Optional[List[TensorInfo]] = None,
                 src_op=None,
                 name: Optional[str] = ""):

        # set attributes, inputs and outputs
        super().__init__(op_type, input_tensors,
                         output_tensors, params, param_info, src_op, name)

        # set backend specific items
        self.set_axis_orders(self.inputs, tensor_layouts=QnnTensorLayouts.__members__)
        self.set_axis_orders(self.outputs, tensor_layouts=QnnTensorLayouts.__members__)
        self.set_tensor_data_types(src_op)

        # validate backend specific args
        self.validate_tensor_info_data_types()

    def as_dict(self, graph):

        inputs = OrderedDict()
        outputs = OrderedDict()
        tensor_params = {param.name: param.param.as_dict() for _, param in self.params.items()
                         if param.param_type == ParamTypes.TENSOR}
        scalar_params = {param.name: param.param.as_dict() for _, param in self.params.items()
                         if param.param_type == ParamTypes.SCALAR or param.param_type == ParamTypes.STRING}

        for input_ in self.inputs:
            inputs[input_.name] = input_.as_dict()

        for output in self.outputs:
            outputs[output.name] = output.as_dict()

        return inputs, outputs, scalar_params, tensor_params

    @classmethod
    @abstractmethod
    def extract_attrs(cls, src_op, param_info: Dict[str, TensorInfo]):
        """
        The intention of this method is to extract param_info from a framework src_op and return a
        dictionary of Param objects, such that "attr_name": "Param".
        This must be implemented, as it is called during initialization
        :param src_op: Framework src_op
        :param param_info: Parameter info
        :return: A dictionary of Params
        """

    @abstractmethod
    def infer_output_shapes(self, node, **kwargs):
        """
        This method recieves a framework node and returns the output shapes
        :param node:
        :param kwargs:
        :return: a list of lists which contain output dimensions for each output tensor
        """

    @abstractmethod
    def set_tensor_data_types(self, node):
        """
        Sets the datatype for each input and output tensor based on the operation instance
        :param node : The source framework node
        :raises An exception if data type cannot be set
        :returns
        """

    def set_static_tensor_to_param(self, tensors):
        """
        Sets a static tensor to a param. This method is called by the base class, meaning instances
        of this class are expected to have static tensors become params. This method takes a
        single tensor, and changes it to a param object. Note that a static tensor must have a
        data field defined.
        :param tensors: The tensor to be made a param.
        """
        local_tensor = []
        for tensor_info in tensors:
            if tensor_info.static:
                log_debug('Static custom input tensor: {} found for op: {} . '
                          'Note this tensor will be stored in the model output'
                          .format(tensor_info.name, self.op_type))
                self.params[tensor_info['name']] = Param(tensor_info['name'], ParamTypes.TENSOR,
                                                         TensorParam(None, tensor_info))
            else:
                local_tensor.append(tensor_info)

        return local_tensor

    def validate_tensor_info_data_types(self):
        """
        Validates the tensor_info data_type against its allowed types. Note that
        tensor info data_types are set from the tensor instance, so the set_tensor_data_types
        must have been called before this function.
        :raises: TypeError if the data_type is not in the allowed types
        """
        for tensor_info in self.outputs + self.inputs:
            if tensor_info.data_type is None or \
                    tensor_info.data_type not in tensor_info.allowed_data_types:
                raise TypeError("{} data_type: {} from op: {} does not match op package config."
                                " Expected one of {}."
                                .format(tensor_info.name,
                                        tensor_info.data_type,
                                        self.op_type,
                                        list(map(str, tensor_info.allowed_data_types))
                                        ))


BackendCustomOp = QnnCustomOp
