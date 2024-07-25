# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum, IntEnum
from copy import deepcopy
from collections import OrderedDict
from collections import defaultdict
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union, Dict, List, Any
import ast
import functools


# ------------------------------------------------------------------------------
#   Helper functions
# ------------------------------------------------------------------------------
# Define a decorator to indicate a function is deprecated
def deprecated(func):
    @functools.wraps
    def dep_print(*args, **kwargs):
        print("WARNING: Function {} is deprecated.".format(func.func_name))
        func(*args, **kwargs)
    return dep_print

def property_type(name, expected_type):
    '''
    Helper to set a property type class variable
    :param name: name of the attribute
    :param expected_type: the type to check against
    :return: Decorated getters/setters for the attribute
    '''
    attr_name = '__' + name

    @property
    def prop(self):
        return getattr(self, attr_name)

    @prop.deleter
    def prop(self):
        raise IndexError("Cannot delete this field")

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a valid object of type {}'.format(name, expected_type))
        if hasattr(self, attr_name):
            raise AttributeError('Cannot set {} field once it has been initialized'.format(attr_name))
        setattr(self, attr_name, value)

    return prop


def check_list_against_type(to_check, valid_type, list_name):
    if not isinstance(to_check, list):
        raise TypeError("Setting {} requires list not {}".format(type(to_check)))
    for elem in to_check:
        if not isinstance(elem, valid_type):
            raise TypeError("Adding to {} requires type {} not {}"
                            .format(list_name, valid_type, type(elem)))


def check_datatype_list(to_check):
    check_list_against_type(to_check, QnnDatatype, "Datatypes")


def check_constraint_list(to_check):
    check_list_against_type(to_check, Constraint, "Constraints")

def check_quant_params_list(to_check):
    check_list_against_type(to_check, QuantParam, "QuantParams")

def check_not_activation(param_tensors, op_name):
    for param in param_tensors:
        if isinstance(param, InputElement) or isinstance(param, OutputElement):
            raise TypeError("Attempted to add activation tensor {} to parameters "
                            "for OpDef {}.".format(param.name, op_name))


def add_op_def_element_common(new_tensors, old_list, element_type,
                              def_type, def_name, reset=True):
    if not isinstance(new_tensors, Iterable):
        raise TypeError("Setting list of {} for {} requires iterable not {}".format(element_type, def_type, type(new_tensors)))

    names = []
    if reset:
        del old_list[:]
    else:
        for elem in old_list:
            names.append(elem.name)

    for tensor in new_tensors:
        if not isinstance(tensor, element_type):
            raise TypeError("Attempted to add object of type {} to {} but it should be {}"
                            .format(type(tensor), def_name, element_type))
        if tensor.name in names:
            raise KeyError("Attempted to add {} to {} but it already exists".format(tensor.name, def_name))
        old_list.append(tensor)
        names.append(tensor.name)


def get_op_qnn_macro(op_name):
    # QNN_OP_<NAME>
    return "QNN_OP_" + format_qnn_string(op_name)


def get_param_qnn_macro(op_name, param_name):
    # QNN_OP_<NAME>_PARAM_<PARAM>
    return get_op_qnn_macro(op_name) + "_PARAM_" + format_qnn_string(param_name)


def get_enum_qnn_macro(op_name, param_name, enum_name):
    ## QNN_OP_<NAME>_PARAM_<PARAM>_<ENUM>
    return get_op_qnn_macro(op_name) \
           + "_" + format_qnn_string(param_name) \
           + "_" + format_qnn_string(enum_name)


def format_qnn_string(orig_string):
    if orig_string.isupper():
        return orig_string
    split_string = []
    curr_word = ""
    first = True
    for char in orig_string:
        if char.isupper() or char.isdigit():
            if curr_word.isdigit():
                if char == "D":
                    curr_word += char
                    split_string.append(curr_word)
                    curr_word = ""
                elif char.isdigit():
                    curr_word += char
                else:
                    split_string[-1] = split_string[-1] + curr_word
                    curr_word = char
                continue
            if not first:
                split_string.append(curr_word)
            curr_word = ""
        curr_word += char
        first = False

    if curr_word != "":
        if curr_word.isdigit():
            split_string[-1] = split_string[-1] + curr_word
        else:
            split_string.append(curr_word)

    return ("_".join(split_string)).upper()


def get_ctype_from_qnn_type(qnn_type):
    return {
        QnnDatatype.BACKEND_SPECIFIC     : 'float',
        QnnDatatype.QNN_DATATYPE_INT_8   : 'int8_t',
        QnnDatatype.QNN_DATATYPE_INT_16  : 'int16_t',
        QnnDatatype.QNN_DATATYPE_INT_32  : 'int32_t',
        QnnDatatype.QNN_DATATYPE_INT_64  : 'int64_t',
        QnnDatatype.QNN_DATATYPE_UINT_8  : 'uint8_t',
        QnnDatatype.QNN_DATATYPE_UINT_16 : 'uint16_t',
        QnnDatatype.QNN_DATATYPE_UINT_32 : 'uint32_t',
        QnnDatatype.QNN_DATATYPE_UINT_64 : 'uint64_t',
        QnnDatatype.QNN_DATATYPE_FLOAT_16 : 'float',
        QnnDatatype.QNN_DATATYPE_FLOAT_32 : 'float',
        QnnDatatype.QNN_DATATYPE_FLOAT_64 : 'double',
        QnnDatatype.QNN_DATATYPE_SFIXED_POINT_4  : 'int8_t',
        QnnDatatype.QNN_DATATYPE_SFIXED_POINT_8  : 'int8_t',
        QnnDatatype.QNN_DATATYPE_SFIXED_POINT_16 : 'int16_t',
        QnnDatatype.QNN_DATATYPE_SFIXED_POINT_32 : 'int32_t',
        QnnDatatype.QNN_DATATYPE_UFIXED_POINT_4  : 'uint8_t',
        QnnDatatype.QNN_DATATYPE_UFIXED_POINT_8  : 'uint8_t',
        QnnDatatype.QNN_DATATYPE_UFIXED_POINT_16 : 'uint16_t',
        QnnDatatype.QNN_DATATYPE_UFIXED_POINT_32 : 'uint32_t',
        QnnDatatype.QNN_DATATYPE_BOOL_8   : 'uint8_t',
        QnnDatatype.QNN_DATATYPE_STRING : 'const char*',
    }[qnn_type]


# ------------------------------------------------------------------------------
#   Helper Classes for Op Defs
# ------------------------------------------------------------------------------
class Layout(Enum):
    UNDEFINED = 0
    BACKEND_SPECIFIC = 1
    NHWC = 2
    NDHWC = 3
    NONTRIVIAL = 4


class ElementType(Enum):
    TENSOR = 0
    SCALAR = 1
    ENUM = 2
    STRING = 3
    BOOL = 4


class ConstraintType(Enum):
    '''
    Defines the allowable Constraints. List to be extended
    '''
    SHAPE = 0
    NUMBER = 1
    VALUE = 2
    DATATYPE = 3
    DESCRIPTION = 4


class RankType(IntEnum):
    POINT = 0
    VECTOR = 1
    MATRIX = 2
    IMAGE = 3
    BATCH_IMAGE = 4
    SCALAR = 98
    N_DIMENSIONAL = 99

class EncodingType(Enum):
    '''
    Defines the allowable Quantization Encoding schemes.
    '''
    SCALE_OFFSET = 0
    AXIS_SCALE_OFFSET = 1
    BW_SCALE_OFFSET = 2
    BW_AXIS_SCALE_OFFSET = 3

class QnnDatatype(Enum):
    '''
    Define the allowable QNN Datatypes
    '''
    BACKEND_SPECIFIC = 0
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
    QNN_DATATYPE_STRING = 19
    QNN_DATATYPE_FLOAT_64 = 20
    QNN_DATATYPE_UFIXED_POINT_4 = 21
    QNN_DATATYPE_SFIXED_POINT_4 = 22

    @classmethod
    def get_types(cls, category='integer'):
        values = list(cls.__members__.values())
        if category == 'integer':
            return values[2:6]
        elif category == 'float':
            return values[10:12] + values[20]
        elif category == 'unsigned':
            return values[6:10]
        elif category == "signed_quantized":
            return values[12:15] + values[22]
        elif category == "unsigned_quantized":
            return values[15:18] + values[21]
        elif category == "bool":
            return values[18]


class NativeDatatype(Enum):
    '''
    Define the allowable Native Datatypes
    '''

    BACKEND_SPECIFIC = QnnDatatype.BACKEND_SPECIFIC
    INT_8 = QnnDatatype.QNN_DATATYPE_INT_8
    INT_16 = QnnDatatype.QNN_DATATYPE_INT_16
    INT_32 = QnnDatatype.QNN_DATATYPE_INT_32
    INT_64 = QnnDatatype.QNN_DATATYPE_INT_64
    UINT_8 = QnnDatatype.QNN_DATATYPE_UINT_8
    UINT_16 = QnnDatatype.QNN_DATATYPE_UINT_16
    UINT_32 = QnnDatatype.QNN_DATATYPE_UINT_32
    UINT_64 = QnnDatatype.QNN_DATATYPE_UINT_64
    FLOAT_16 = QnnDatatype.QNN_DATATYPE_FLOAT_16
    FLOAT_32 = QnnDatatype.QNN_DATATYPE_FLOAT_32
    SFIXED_POINT_8 = QnnDatatype.QNN_DATATYPE_SFIXED_POINT_8
    SFIXED_POINT_16 = QnnDatatype.QNN_DATATYPE_SFIXED_POINT_16
    SFIXED_POINT_32 = QnnDatatype.QNN_DATATYPE_SFIXED_POINT_32
    UFIXED_POINT_8 = QnnDatatype.QNN_DATATYPE_UFIXED_POINT_8
    UFIXED_POINT_16 = QnnDatatype.QNN_DATATYPE_UFIXED_POINT_16
    UFIXED_POINT_32 = QnnDatatype.QNN_DATATYPE_UFIXED_POINT_32
    BOOL_8 = QnnDatatype.QNN_DATATYPE_BOOL_8
    STRING = QnnDatatype.QNN_DATATYPE_STRING
    FLOAT_64 = QnnDatatype.QNN_DATATYPE_FLOAT_64
    UFIXED_POINT_4 = QnnDatatype.QNN_DATATYPE_UFIXED_POINT_4
    SFIXED_POINT_4 = QnnDatatype.QNN_DATATYPE_SFIXED_POINT_4


class Constraint:
    def __init__(self, id, constraint, constraint_type, code="", name=""):
        self.__id = -1
        self.id = id
        self.constraint = constraint
        self.name = ""
        self.__type = None
        self.type = constraint_type
        self.code = code

    id = property_type("id", int)
    type = property_type("type", ConstraintType)

class QuantParam:
    def __init__(self, encodings, axis, symmetric, scale, offset, min, max, math_invariant):
        self.__encodings = []
        self.encodings = encodings
        self.axis = axis
        self.symmetric = symmetric
        self.scale = scale
        self.offset = offset
        self.min = min
        self.max = max
        self.math_invariant = math_invariant

    @property
    def encodings(self):
        return self.__encodings

    @encodings.setter
    def encodings(self, values):
        self.__encodings = []
        for value in values:
            if not isinstance(value, EncodingType):
                raise TypeError("Attempted to add Encoding type {} that is not valid encoding".format(value))
            self.__encodings.append(value)

class Default:
    def __init__(self, value=""):
        self.__type = None
        self.__value = None
        self.value = value

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, val):
        raise AttributeError("Cannot set Default type externally.")

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if isinstance(value, str):
            self.__value = self.__configure_from_value(value)
        elif isinstance(value, bool):
            self.__value = value
            self.__type = ElementType.BOOL
        elif isinstance(value, float) or isinstance(value, int):
            self.__value = value
            self.__type = ElementType.SCALAR
        elif isinstance(value, list):
            self.__value = value
            self.__type = ElementType.TENSOR
        else:
            raise TypeError("Attempted to set Default with unsupported type {}.".format(type(value)))

    @staticmethod
    def __is_number(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def __configure_from_value(self, raw_value):
        """
        Validates the content of the Default element
        Allowed Grammar:
          - Boolean: true, false
          - Scalar: i
          - Explicit List: {i, j, k} or  {{i, j}, {k, l}}

        :param raw_value the string representing default value
        """
        value = raw_value.replace(" ", "")

        if value.lower() == "true" or value.lower() == "false":
            self.__type = ElementType.BOOL
            return bool(value)

        if self.__is_number(value):
            self.__type = ElementType.SCALAR
            if "." not in value and "e" not in value:
                return int(value)
            return float(value)

        if any(c.isalpha() for c in value):
            self.__type = ElementType.STRING
            return raw_value

        try:
            # Account for expressions with curly braces
            converted_val = value.replace("{", "[")
            converted_val = converted_val.replace("}", "]")
            list_val = list(ast.literal_eval(converted_val))
            self.__type = ElementType.TENSOR
            return list_val
        except Exception as e:
            self.__type = ElementType.STRING
            return raw_value

        return raw_value


# ------------------------------------------------------------------------------
#   Classes for the Elements Contained in an OpDef
# ------------------------------------------------------------------------------
class SupplementalOpDefElement:
    def __init__(self, name, datatypes, quant_params, shape, constraints, layout, default_only=False):
        self.name = name
        self.__datatypes = []
        self.datatypes = datatypes
        self.__quant_params = []
        self.quant_params = quant_params
        self.__constraints = []
        self.constraints = constraints
        self.shape = shape
        self.layout = layout
        self.default_only = default_only

    name = property_type("name", str)
    shape = property_type("shape", str)
    layout = property_type("layout", Layout)

    @property
    def datatypes(self):
        return self.__datatypes

    @datatypes.setter
    def datatypes(self, dtypes):
        check_datatype_list(dtypes)
        self.__datatypes = []
        for dtype in dtypes:
            self.add_datatype(dtype)

    def add_datatype(self, dtype):
        if dtype not in self.datatypes:
            self.__datatypes.append(dtype)

    @property
    def constraints(self):
        return self.__constraints

    @constraints.setter
    def constraints(self, constraints):
        check_constraint_list(constraints)
        self.__constraints = []
        for constraint in constraints:
            self.add_constraint(constraint)

    def add_constraint(self, constraint):
        c_id = constraint.id
        if any(match for match in self.constraints if match.id == c_id):
            raise ValueError("Attempting to override existing Constraint with ID {}.".format(c_id))
        self.__constraints.append(constraint)

    @property
    def quant_params(self):
        return self.__quant_params

    @quant_params.setter
    def quant_params(self, quant_params):
        check_quant_params_list(quant_params)
        self.__quant_params = []
        for quant_param in quant_params:
            self.add_quant_param(quant_param)

    def add_quant_param(self, quant_param):
        self.__quant_params.append(quant_param)


class OpDefElement(ABC):
    '''
    The base class for an element that is part of an OpDef. All inputs, outputs, and
    params need the attributes defined here although some elements are limited in the values
    they can have
    '''

    def __init__(self, name, description, mandatory, default, datatypes, rank, shape, elem_type, constraints):
        self.markup_name = name
        self.__name = ""
        self.name = name
        self.description = description
        self.mandatory = mandatory
        self.default = default
        self.check_default()
        self.__datatypes = []
        self.datatypes = datatypes
        self.rank = rank
        self.shape = shape
        self.type = elem_type
        self.check_type_constraints()
        self.__constraints = []
        self.constraints = constraints

    # Care more about the immutability than typing
    description = property_type("description", str)

    mandatory = property_type("mandatory", bool)

    type = property_type("type", ElementType)

    shape = property_type("shape", str)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError('{} must be a valid object of type {}.'.format(new_name, str))
        if hasattr(self, "__name"):
            raise AttributeError('Cannot set {} field once it has been initialized.'.format("name"))
        if ":math:" in new_name:
            self.__name = new_name.split()[0]
            self.markup_name = new_name
        else:
            self.__name = new_name
            self.markup_name = self.name

    @property
    def rank(self):
        return self.__rank

    @rank.setter
    def rank(self, r):
        self.check_rank(r)
        self.__rank = r

    @property
    def default(self):
        return self.__default

    @default.setter
    def default(self, value):
        if hasattr(self, "__default"):
            raise AttributeError('Cannot set {} field once it has been initialized.'.format("__default"))
        self.__default = Default(value)

    @property
    def datatypes(self):
        return self.__datatypes

    @datatypes.setter
    def datatypes(self, dtypes):
        check_datatype_list(dtypes)
        self.__datatypes = []
        for dtype in dtypes:
            self.add_datatype(dtype)

    def add_datatype(self, dtype):
        if self.check_datatype(dtype) and dtype not in self.datatypes:
            self.__datatypes.append(dtype)

    @property
    def constraints(self):
        return self.__constraints

    @constraints.setter
    def constraints(self, constraints):
        self.__constraints = []
        for constraint in constraints:
            self.add_constraint(constraint)

    def add_constraint(self, constraint):
        c_id = constraint.id
        if any(match for match in self.constraints if match.id == c_id):
            raise ValueError("Attempting to override existing Constraint with ID {}.".format(c_id))
        self.__constraints.append(constraint)

    def check_datatype(self, dtype):
        '''
        Checks for Datatype restrictions. To be overridden in deriving classes when applicable
        :return: True if the Dtype is valid else false
        '''
        return True

    def check_default(self):
        if not self.mandatory and self.default.value == "":
            raise RuntimeError("OpDef Element {} is optional but has no default value.".format(self.name))

    @staticmethod
    def check_rank(r):
        '''
        Checks rank before adding. Can be overridden in deriving classes
        :param r: rank to be checked against
        :raises ValueError if the value is out of bounds
        '''
        if r not in range(0,6) and r not in range(98,100):
            raise ValueError("Rank {} is invalid. Rank must be one of RankType 0: 0-D, 1: 1-D, 2: 2-D, 3: 3-D, 4: 4-D, 5: 5-D, 98: SCALAR, or 99: N-D.".format(r))

    @abstractmethod
    def check_type_constraints(self):
        '''
        Checks type constraints
        '''
        raise NotImplementedError


# ------------------------------------------------------------------------------
#   Classes for Tensor Elements (Inputs, Outputs, and Tensor Params)
# ------------------------------------------------------------------------------
class TensorElement(OpDefElement):
    '''
    Defines a Tensor Type Element as part of the OpDef. Adds layout in addition to generic OpDef element
    attributes
    '''
    def __init__(self, name, description, mandatory, default, datatype, rank, shape, layout=Layout.UNDEFINED, constraints=[]):
        super().__init__(name=name, description=description, mandatory=mandatory, default=default, datatypes=datatype, rank=rank,
                         shape=shape, elem_type=ElementType.TENSOR, constraints=constraints)
        self.__layout = None
        self.layout = layout

    layout = property_type("layout", Layout)

    def check_type_constraints(self):
        if getattr(self, "default") is not None:
            if self.default.type == ElementType.STRING or self.default.type == ElementType.TENSOR:
                return
            elif self.rank == 0 and self.default.type == ElementType.SCALAR:
                return
            else:
                raise AttributeError("Tensor Type Element cannot have default value of type: {}.", self.default.type)

class ActivationTensorElement(TensorElement):
    '''
    Parent Class for a Input/Output Tensor elements. Allows for NULL default values on optional tensors.
    Activation tensors can be repeated, e.g. for Ops like Concat.
    '''
    def __init__(self, name, description, mandatory, default, datatype, rank, shape, layout=Layout.UNDEFINED, constraints=[], repeated=False):
        super().__init__(name=name, description=description, mandatory=mandatory, default=default, datatype=datatype, rank=rank, shape=shape, layout=layout, constraints=constraints)
        self.__repeated = None
        self.repeated = repeated

    repeated = property_type("repeated", bool)

    def check_default(self):
        # Activation tensors can have no default; it is implicitly null
        return

class InputElement(ActivationTensorElement):
    '''
    Input is a Tensor Type element, that can additionally be static tensors e.g. the case of weights and biases.
    '''

    def __init__(self, name, description, mandatory, default, datatype, rank,
                 shape, layout=Layout.UNDEFINED, constraints=[], repeated=False, is_static_tensor=False):
        super().__init__(name=name, description=description, mandatory=mandatory, default=default, datatype=datatype, rank=rank, shape=shape, layout=layout, constraints=constraints, repeated=repeated)
        self.is_static_tensor = is_static_tensor

class OutputElement(ActivationTensorElement):
    '''
    Output Elements are Activation Tensor Type Elements, with no default values allowed.
    '''

    def __init__(self, name, description, mandatory, datatype, rank, shape, layout=Layout.UNDEFINED, constraints=[], repeated=False):
        super().__init__(name=name, description=description, mandatory=mandatory, default="", datatype=datatype, rank=rank, shape=shape, layout=layout, constraints=constraints, repeated=repeated)


class TensorParam(TensorElement):
    '''
    Tensor Param is a Tensor Type element that can be created with a default value.
    '''

    def __init__(self, name, description, mandatory, default, datatype, rank,
                 shape, layout=Layout.UNDEFINED, constraints=[]):
        super().__init__(name, description, mandatory, default, datatype, rank, shape, layout, constraints)


# ------------------------------------------------------------------------------
#   Classes for Rank Zero Elements (Params)
# ------------------------------------------------------------------------------
class RankZeroElement(OpDefElement):
    '''
    RankZeroElements are a subclass of OpDef Elements whose rank is always 0. They are limited in what
    they can express for shape attributes (rank, shape text)
    '''

    def __init__(self, name, description, mandatory, default, datatype, elem_type, constraints):
        if elem_type == ElementType.TENSOR:
            super().__init__(name, description, mandatory, default, datatype, RankType.POINT, "one element", elem_type, constraints)
        else:
            super().__init__(name, description, mandatory, default, datatype, RankType.SCALAR, "scalar", elem_type, constraints)

    @staticmethod
    def check_rank(r):
        if r != RankType.POINT and r != RankType.SCALAR:
            raise ValueError("Rank {} is invalid. Rank is 0 or SCALAR for scalar-like elements.".format(r))


class ScalarElement(RankZeroElement):
    def __init__(self, name, description, mandatory, default, datatype, constraints):
        super().__init__(name, description, mandatory, default, datatype, ElementType.SCALAR, constraints)

    def check_type_constraints(self):
        if self.default.type != ElementType.STRING and self.default.type != ElementType.SCALAR:
            raise AttributeError("Scalar Type Element cannot have default value of type: {}.", self.default.type)


class EnumElement(RankZeroElement):
    '''
    Enum is a rank zero element whose fields can be expressed as either enum names
    or their underlying int values
    '''

    def __init__(self, name, description, mandatory, enum_vals, default=0, datatype=[QnnDatatype.QNN_DATATYPE_UINT_32], constraints=[]):
        self.enum = enum_vals
        super().__init__(name, description, mandatory, default, datatype, ElementType.ENUM, constraints=constraints)

    def check_datatype(self, dtype):
        if dtype == QnnDatatype.QNN_DATATYPE_FLOAT_16 or dtype == QnnDatatype.QNN_DATATYPE_FLOAT_32 or dtype == QnnDatatype.QNN_DATATYPE_FLOAT_64:
            raise ValueError("Enumeration {} received invalid datatype {}.".format(self.name, dtype))
        return True

    def check_type_constraints(self):
        if len(self.enum) == 0:
            raise ValueError("Enumeration {} has no enumeration values".format(self.name))

        if self.default.type != ElementType.STRING and self.default.type != ElementType.SCALAR:
            raise AttributeError("Enum Type Element cannot have default value of type: {}.".format(self.default.type))

        if self.default.type == ElementType.SCALAR:
            if self.default.value < 0:
                raise ValueError("Default for Enumeration {} must be greater than zero.".format(self.name))
            if self.default.value >= len(self.enum):
                raise ValueError(
                    "Default value {} exceeds enumeration range for Enumeration {}.".format(self.default.value,
                                                                                            self.name))
        else:
            if self.default.value not in self.enum:
                raise ValueError("Default value {} is not in Enumeration {}".format(self.default.value, self.name))

        if QnnDatatype.QNN_DATATYPE_BOOL_8 in self.datatypes and len(self.enum) != 2:
            raise AttributeError("Datatype is QNN_DATATYPE_BOOL_8 but there aren't two enum values.")


class BoolElement(RankZeroElement):
    '''
    Bool elements are rank zero elements whose datatype and default values are restricted
    '''

    def __init__(self, name, description, mandatory, default=False, datatype=[QnnDatatype.QNN_DATATYPE_BOOL_8], constraints=[]):
        super().__init__(name, description, mandatory, default, datatype, ElementType.BOOL, constraints=constraints)

    def check_datatype(self, dtype):
        if dtype != QnnDatatype.QNN_DATATYPE_BOOL_8:
            raise ValueError("Boolean Param {} received invalid datatype {}.".format(self.name, dtype))
        return True

    def check_type_constraints(self):
        if self.default.type != ElementType.BOOL and self.default.type != ElementType.SCALAR:
            raise AttributeError(
                "Boolean for Param {} received non-boolean default value {}.".format(self.name, self.default.value))


'''
TODO
class StringElement(OpDefElement):
    def __init__(self):
        raise NotImplementedError
'''


# ------------------------------------------------------------------------------
#   OpDef and OpDef Container Classes
# ------------------------------------------------------------------------------
class OpDefPackageInfo:
    version = property_type('version', str)
    name = property_type('name', str)
    domain = property_type('domain', str)

    def __init__(self, name, domain='aisw', version='1.0.0'):
        self.name = name
        self.domain = domain
        self.version = version

    def __repr__(self):
        return dict(package_name=self.name, version=self.version, domain=self.domain).__repr__()


class SupplementalOpDef:
    def __init__(self, name, supp_ins, supp_outs, supp_params):
        self.name = name
        self.__inputs = []
        self.inputs = supp_ins
        self.__outputs = []
        self.outputs = supp_outs
        self.__parameters = []
        self.parameters = supp_params

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, in_tensors):
        self.__add_common_supplement(in_tensors, self.__inputs)

    def add_inputs(self, *args):
        self.__add_common_supplement(args, self.__inputs, reset=False)
        return self

    @property
    def outputs(self):
        return self.__outputs

    @outputs.setter
    def outputs(self, out_tensors):
        self.__add_common_supplement(out_tensors, self.__outputs)

    def add_outputs(self, *args):
        self.__add_common_supplement(args, self.__inputs, reset=False)
        return self

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, param_tensors):
        self.__add_common_supplement(param_tensors, self.__parameters)

    def add_params(self, *args):
        self.__add_common_supplement(args, self.__inputs, reset=False)
        return self

    def __add_common_supplement(self, new_tensors, old_list, reset=True):
        add_op_def_element_common(new_tensors, old_list,
                                  SupplementalOpDefElement, SupplementalOpDef, self.name, reset)

class OpDef:
    '''
    An OpDef is generically composed of OpDef Elements. Inputs are InputElement objects, Outputs are
    OutputElement objects, and parameters are generically OpDefElements.
    '''

    def __init__(self, name, description="", references=[], ins=[], outs=[], parameters=[], use_default_translation=False):
        self.name = name
        self.description = description
        self.references = references
        self.__inputs = []
        self.inputs = ins
        self.__outputs = []
        self.outputs = outs
        self.__parameters = []
        self.parameters = parameters
        self.__package_info = None
        self.use_default_translation = use_default_translation

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, in_tensors):
        self.__add_common_full(in_tensors, self.__inputs, InputElement)

    def add_inputs(self, *args):
        self.__add_common_full(args, self.__inputs, InputElement, reset=False)
        return self

    @property
    def outputs(self):
        return self.__outputs

    @outputs.setter
    def outputs(self, out_tensors):
        self.__add_common_full(out_tensors, self.__outputs, OutputElement)

    def add_outputs(self, *args):
        self.__add_common_full(args, self.__outputs, OutputElement, reset=False)
        return self

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, param_tensors):
        check_not_activation(param_tensors, self.name)
        self.__add_common_full(param_tensors, self.__parameters, OpDefElement)

    def add_params(self, *args):
        check_not_activation(args, self.name)
        self.__add_common_full(args, self.__parameters, OpDefElement, reset=False)
        return self

    def __add_common_full(self, new_tensors, old_list, element_type, reset=True):
        add_op_def_element_common(new_tensors, old_list,
                                  element_type, OpDef, self.name, reset)

    @property
    def package_info(self):
        return self.__package_info

    @package_info.setter
    def package_info(self, package_info):
        if not isinstance(package_info, OpDefPackageInfo):
            raise TypeError('Provided package info of type: {} is not an object of type:'
                            .format(type(package_info), OpDefPackageInfo.__class__.__name__))
        self.__package_info = package_info


class OpDefCollection:
    '''
    An OpDefCollection is a dictionary-like data structure containing operation definitions,
    that cumulatively specify one or more op package. The structure consists of a core set of
    operations, and zero or more sets of supplemental operation definitions that extend or replace
    information from the core operations.
    '''

    ALL = "ALL"

    def __init__(self):
        self.op_def_dict = OrderedDict()
        self.supplemental_op_def = {}
        self.supported_ops = {}

    def add_op_def(self, op_def: OpDef):
        '''
        Adds core op definition to the collection
        :param op_def: The op def to add. Must be of type OpDef
        :return: None
        :raise: TypeError if op_def is not an OpDef
        :raise: ValueError if the op_def has already been added
        '''
        if not isinstance(op_def, OpDef):
            raise TypeError("Attempted to add non OpDef to OpDefCollection.")

        if self.is_supported(op_def.name, self.ALL):
            raise ValueError("Attempted to add OpDef {} but it already exists.".format(op_def.name))

        self.op_def_dict[op_def.name] = op_def

    def add_supplemental_op_def(self, supp_def: SupplementalOpDef, backend: str, runtime : str=None):
        '''
        Adds a supplemental op definition associated with a particular backend. If a supplemental
        definition already exists it will be updated with the newly added information.
        :param supp_def: The supplemental op def to add
        :param backend: The backend to add it to.
        :return: None
        :raise: TypeError if supp_def is not a SupplementalOpDef
        '''
        if not isinstance(supp_def, SupplementalOpDef):
            raise TypeError("Attempted to add non SupplementalOpDef to OpDefCollection.")

        if backend not in self.supplemental_op_def.keys():
            self.supplemental_op_def[backend] = {}
            self.supplemental_op_def[backend][self.ALL] = defaultdict(list)

        if runtime is not None and runtime not in self.supplemental_op_def[backend].keys():
            self.supplemental_op_def[backend][runtime] = defaultdict(list)

        elif runtime is None:
            runtime = self.ALL

        self.supplemental_op_def[backend][runtime][supp_def.name].append(supp_def)

    @deprecated
    def update_supplemental_op_def(self, supp_def: SupplementalOpDef, backend: str, runtime : str=None):
        '''
        DEPRECATED: This function is deprecated and should be replaced with add_supplemental_op_def

        Updates an existing supplemental op definition with the new content provided in supp_def.
        See __merge_tensors for information on which tensor fields are modifiable.
        :param supp_def: The new supplemental op def
        :param backend: The backend to which the new supplemental information is associated
        :return: None
        :raise: TypeError if supp_def is not of type SupplementalOpDef
        :raise: ValueError if there is no existing supplemental definition
        '''
        if not isinstance(supp_def, SupplementalOpDef):
            raise TypeError("Attempted to add non SupplementalOpDef to OpDefCollection.")

        if not (self.is_supported(supp_def.name, backend, runtime) and \
                self.is_supplemented(supp_def.name, backend, runtime)):
            raise ValueError(
                "Attempted to update SupplementalOpDef {} for Backend {} but it does not exist.".format(supp_def.name,
                                                                                                        backend))
        self.supplemental_op_def[backend][supp_def.name].append(supp_def)

    def get_op_def(self, op_def_name: str, backend: str="ALL") -> OpDef:
        '''
        Get a fully formed op definition. If backend is "ALL" the core op definition is returned. If
        a backend is specified, the core op definition will be merged with any supplemental information
        also registered with the collection. See __merge_tensors for information on what fields can be
        modified.
        :param op_def_name: The name of the op definition. This is the human readable string associated
                            with the name of the op definition e.g. "Argmax, "Conv2d", etc.
        :param backend: Optional field specifying the desired backend variant of the op def. By default
                        returns the core op definition
        :return: An OpDef object specified by the op_def_name and backend
        :raise: KeyError if the op def is not supported
        '''
        if not self.is_supported(op_def_name, backend):
            raise KeyError("OpDef {} is unsupported for Backend {}.".format(op_def_name, backend))

        op_def = None
        if backend != self.ALL and self.is_supplemented(op_def_name, backend):
            supp_op_def = self.get_supplemental_op_def(op_def_name, backend)
            orig_op_def = deepcopy(self.op_def_dict[op_def_name])
            op_def = self.__merge_supplemental_op_def(orig_op_def, supp_op_def)
        else:
            op_def = self.op_def_dict[op_def_name]
        return op_def

    def get_op_defs(self) -> Dict[str, OpDef]:
        '''
        Returns a deep copy of thedictionary containing the core op definitions.
        The keys of the dictionary are the name of the op defs, and the values are core OpDef objects.
        :return: OpDef dictionary
        '''
        return self.op_def_dict

    def get_backend_op_defs(self, backend: str) -> Dict[str, OpDef]:
        '''
        Returns a shallow copy of a dictionary of op definitions, supplemented with
        information for a particular backend. The dictionary has keys of op def names, and values
        of SupplementOpDefs. If there are no supplemental op defs for a backend, the dictionary will
        be empty.
        :param backend: The desired backend for which to get supplemental ops
        :return: SupplementalOpDef dictionary
        '''
        supported = self.get_supported_ops(backend)
        op_defs = {}
        for op_def_name in supported:
            op_defs[op_def_name] = self.get_op_def(op_def_name, backend)
        return op_defs

    def get_supplemental_op_defs(self, backend: str=None, runtime=None) -> Dict[str, SupplementalOpDef]:
        '''
        Returns a dictionary of dictionaries containing all supplemental op definitions associated
        with the collection. The dictionary is keyed by Backend, then op definition name and the values
        are SupplementalOpDefs. If a backend is provided then a deep-copy of the supplemental dictionary
        is returned.
        :param backend: The optional backend string to select a set of supplement op defs for a
                        particular backend.
        :return: A dictionary, or dictionary of dictionaries
        '''
        if backend is None:
            return self.supplemental_op_def
        if backend not in self.get_supported_backends():
            raise KeyError("No Supplemental Ops for backend {}.".format(backend))

        if runtime is None:
            return self.supplemental_op_def[backend][self.ALL]
        else:
            if runtime in self.get_supported_runtimes(backend):
                return self.supplemental_op_def[backend][runtime]
            else:
                raise KeyError("{} is not a supported runtime for backend {}.".format(runtime, backend))

    def get_supplemental_op_def(self, op_def_name, backend, runtime=None, consolidate_op_defs=True):
        '''
        Gets a supplemental op def for a particular backend, and optionally a runtime. Returns a
        list of op defs if consolidate op is false, otherwise a single op def.
        :param op_def_name: Name of the desired op def
        :param backend: string for the desired backend
        :param runtime: string for the desired runtime
        :param consolidate_op_defs: Boolean indicating whether the operation definitions should be
        consolidated
        :return: the desired supplemental op(s)
        '''
        ops = self.get_supplemental_op_defs(backend, runtime)[op_def_name]
        if consolidate_op_defs:
            op = ops[0]
            for next_op in ops[1:]:
                op = self.__merge_supplemental_op_def(op, next_op)
            return op
        else:
            return ops

    def get_supported_ops(self, backend: str, all_runtimes=True, runtime: str=None) -> List[str]:
        '''
        Get the list of supported operations for a particular backend. If "ALL" is provided, the list
        of registered core op definitions are returned. all_runtimes and runtime are mutually
        incompatible.
        :param backend: The backend string for which support is desired
        :param all_runtimes: Whether to consider supported ops across all runtimes
        :param runtime: The runtime string for which support is desired.
        :return: A list of strings containing the supported op definition names
        '''

        if all_runtimes and runtime is not None:
            raise ValueError("all_runtimes and runtime arguments are mutually exclusive")

        if backend is None:
            backend = self.ALL

        runtimes = []
        if all_runtimes:
            runtimes = self.get_supported_runtimes(backend)
            runtimes.append(self.ALL)
        else:
            if runtime is None:
                runtime = self.ALL
            runtimes.append(runtime)

        if backend == self.ALL:
            return self.op_def_dict.keys()

        if backend not in self.get_supported_backends():
            raise KeyError("No Supported Ops for backend {}.".format(backend))

        supported_ops = []
        for rt in runtimes:
            if rt not in (self.get_supported_runtimes(backend) + [self.ALL]):
                raise KeyError("Invalid Runtime option {} for BE {}".format(rt, backend))
            supported_ops += self.supported_ops[backend][rt]
        supported_ops = [*set(supported_ops)]
        return supported_ops

    def __merge_supplemental_op_def(self, original_def: Union[SupplementalOpDef, OpDef], \
                                    supp_def: SupplementalOpDef) -> Union[SupplementalOpDef, OpDef]:
        '''
        Merge an op def with supplemental information.
        :param original_def: The op definition already in the collection. Can be an OpDef or a
                             SupplementalOpDef.
        :param supp_def: The new supplement op def to merge. This must be a SupplementalOpDef
        :return: The updated original op
        '''
        self.__merge_tensors(supp_def.inputs, original_def.inputs)
        self.__merge_tensors(supp_def.outputs, original_def.outputs)
        self.__merge_tensors(supp_def.parameters, original_def.parameters)
        return original_def

    def __merge_tensors(self, supp_tensors: List[OpDefElement], orig_tensors: List[OpDefElement]):
        '''
        Datatypes are extended. Constraints can be overwritten or extended - if id matches an existing
        constraint it will be overwritten, while constraints with distinct id's will be extended.
        Shape text and layout will be overwritten if it exists.
        :param supp_tensors: The supplemental input tensors. Will be inputs, outputs or params
        :param orig_tensors: The corresponding original input tensors. Will be inputs, outputs, or params.
        '''
        for supp_tensor in supp_tensors:
            for i, orig_tensor in enumerate(orig_tensors):
                if supp_tensor.name == orig_tensor.name:
                    # Update datatypes
                    if QnnDatatype.BACKEND_SPECIFIC in orig_tensors[i].datatypes:
                        orig_tensors[i].datatypes.remove(QnnDatatype.BACKEND_SPECIFIC)

                    # Combine supplemental and backend values and remove duplicates
                    orig_tensors[i].datatypes = orig_tensors[i].datatypes + supp_tensor.datatypes
                    orig_tensors[i].datatypes = list(set(orig_tensors[i].datatypes))

                    # Overwrite shape
                    if supp_tensor.shape != "":
                        orig_tensors[i].shape = supp_tensor.shape

                    if supp_tensor.layout != Layout.UNDEFINED and orig_tensor.rank > 0:
                        orig_tensors[i].layout = supp_tensor.layout

                    # Overwrite and add constraints
                    if len(supp_tensor.constraints) > 0:
                        for new_constraint in supp_tensor.constraints:
                            added = False
                            for j, old_constraint in enumerate(orig_tensors[i].constraints):
                                if new_constraint.id == old_constraint.id:
                                    orig_tensors[i].constraints[j] = new_constraint
                                    added = True
                                    break
                            if not added:
                                orig_tensors[i].constraints.append(new_constraint)

    def get_supported_backends_per_op(self, op_def_name: str=None) -> List[str]:
        '''
        Get the backends that an op is supported by.
        :param op_def_name: The name of the op definition.
        :return: A list of strings, containing the supported backends.
        '''
        BEs = self.supported_ops.keys()
        if op_def_name is None:
            return BEs

        supported_be = []
        for be in BEs:
            if self.is_supported(op_def_name, be):
                supported_be.append(be)
        return supported_be

    def is_supported(self, op_def_name: str, backend: str=None, runtime: str=None) -> bool:
        '''
        Check if an op definition is supported by a particular backend. If "ALL" is provided as the
        backend then this function returns whether an operation is registered in the core op definitions.
        :param op_def_name: The name of the op definition of interest
        :param backend: The backend for which support is desired. By default is "ALL"
        :return: A boolean, with True indicating the op is supported for the provided backend, and False
                 indicating the provided op is not supported.
        '''
        if backend  is None:
            backend = self.ALL

        all_runtimes = True if runtime is None else False

        return True if op_def_name in self.get_supported_ops(backend, all_runtimes, runtime) else False

    def update_backend_support(self, op_def_name: str, backend: str, runtime: str=None):
        '''
        Update the support for a backend, indicating it supports an op definition of name op_def_name.
        The list of backend support can be a superset of the supplemental ops registered for that backend.
        If a backend supports the operation without also registering a supplement op def, it indicates
        the backend supports the core op def as is.
        :param op_def_name: The op definition name for which support will be updated
        :param backend: The backend for which the op_def_name is associated
        :return: None
        :raise: ValueError if there is no core op definition corresponding to op_def_name
        '''
        # Cannot update backend support if the Operation is
        # not supported generally
        if not self.is_supported(op_def_name, self.ALL):
            raise ValueError(
                "Cannot add {} backend support for OpDef {} as it is not added to Collection.".format(backend,
                                                                                                      op_def_name))
        if backend not in self.supported_ops.keys():
            self.supported_ops[backend] = OrderedDict()
            self.supported_ops[backend][self.ALL] = []

        if runtime not in self.get_supported_runtimes(backend):
            self.supported_ops[backend][runtime] = []

        if runtime is None:
            runtime = self.ALL

        if not self.is_supported(op_def_name, backend, runtime):
            self.supported_ops[backend][runtime].append(op_def_name)

    def get_op_def_attribute(self, op_def_name: str, attr: str, backend: str=None) -> Any:
        '''
        Get an attribute of an op def, without getting the op def object.
        :param op_def_name: Name of the op def to get
        :param attr: The desired attribute of the op def
        :param backend: The backend for which the op def is associated, if any
        :return: The specified attribute
        '''
        if backend is None:
            backend = self.ALL

        if not self.is_supported(op_def_name, backend):
            raise KeyError("OpDef {} is unsupported for Backend {}.".format(op_def_name, backend))
        return getattr(self.get_op(op_def_name, backend), attr)

    def is_supplemented(self, op_def_name: str, backend: str, runtime: str=None) -> bool:
        '''
        Check if there is a supplemental op definition registered with a backend.
        :param op_def_name: The name of the op definition
        :param backend: The backend to check for the op definition
        :return: True if there is a supplemental op def registered with the backend, False if there
                 is not.
        '''
        # Check runtime
        runtimes = []
        if runtime is not None:
            runtimes = self.get_supported_runtimes(backend)
            if runtime not in runtimes:
                return False
        else:
            runtime = self.ALL

        # Check backend support
        if backend in self.get_supported_backends() and \
            backend in self.supplemental_op_def.keys() and \
                op_def_name in self.supplemental_op_def[backend][runtime].keys():
            return True

        return False

    def get_supported_backends(self) -> List[str]:
        '''
        Get all the backends for which a supplemental op def is registered.
        :return: A list of strings containing the backends that the Collection contains.
        '''
        return self.supported_ops.keys()

    def get_supported_runtimes(self, backend: str) -> List[str]:
        '''
        Get all the runtimes available for a particular backend
        :param backend: the backend to query runtimes
        :return: a list of the available runtimes
        '''
        if backend not in self.get_supported_backends():
            return []
        return [key for key in self.supported_ops[backend] \
                if key != self.ALL and key is not None]

    def remove_op_def(self, op_name: str):
        '''
        Remove an op def from the collection. This removes the core definition and any registered
        any supplemental information
        :param op_name: The name of the op to remove
        :return: None
        :raise: KeyError if the op definition is not in the Collection.
        '''
        # Check if op is supported as part of base definition
        if not self.is_supported(op_name):
            raise KeyError("OpDef {} is not a supported op.".format(op_name))

        # Remove base definition
        del self.op_def_dict[op_name]

        # Remove any backend specific supplements
        for backend in self.get_supported_backends():
            if self.is_supported(op_name, backend):
                del self.supplemental_op_def[backend][op_name]
                for runtime in self.get_supported_runtimes(backend):
                    if self.is_supplemented(op_name, backend, runtime):
                        del self.supplemental_op_def[backend][runtime][op_name]