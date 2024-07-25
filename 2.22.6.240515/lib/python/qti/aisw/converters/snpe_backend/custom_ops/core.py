# ==============================================================================
#
#  Copyright (c) 2020 - 2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from collections import OrderedDict
from qti.aisw.converters.common.custom_ops.core import *
from qti.aisw.converters.snpe_backend.custom_ops.helpers.udo_module_helpers import *
from qti.aisw.converters.common.custom_ops.utils.config_helpers import *
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker


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


def qnn_to_native_dtype(qnn_type):
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


# ------------------------------------------------------------------------------
#   SNPE UDO config Core Classes
# ------------------------------------------------------------------------------
class TensorInfo(CustomTensorInfo):
    shape = property_type('shape', str)  # string for now, should be something interpretable

    def __init__(self, **tensor):
        super().__init__(**tensor)
        self.data = None
        self.quant_type = None

        # snpe needs a param type for all objects
        self.param_type = "SNPE_UDO_PARAMTYPE_TENSOR"
        self.snpe_qnn_udo_op_dict = {
            "SNPE_UDO_DATATYPE_FLOAT_32":   QnnDatatype.QNN_DATATYPE_FLOAT_32,
            "SNPE_UDO_DATATYPE_FLOAT_16":   QnnDatatype.QNN_DATATYPE_FLOAT_16,
            "SNPE_UDO_DATATYPE_INT_8":      QnnDatatype.QNN_DATATYPE_INT_8,
            "SNPE_UDO_DATATYPE_INT_16":     QnnDatatype.QNN_DATATYPE_INT_16,
            "SNPE_UDO_DATATYPE_INT_32":     QnnDatatype.QNN_DATATYPE_INT_32,
            "SNPE_UDO_DATATYPE_UINT_8":     QnnDatatype.QNN_DATATYPE_UINT_8,
            "SNPE_UDO_DATATYPE_UINT_16":    QnnDatatype.QNN_DATATYPE_UINT_16,
            "SNPE_UDO_DATATYPE_UINT_32":    QnnDatatype.QNN_DATATYPE_UINT_32,
        }

    def from_dict(self, tensor_dict, name=''):
        self.name = tensor_dict.get('name', name)
        self.dimensions = tensor_dict.get('dims', [])
        self.static = tensor_dict.get('static', False)
        self.default_value = tensor_dict.get('default_value', None)
        self.data = tensor_dict.get('data', [])
        self.layout = get_internal_tensor_layout(tensor_dict.get('tensor_layout', "NOT_YET_DEFINED"),
                                                 self.name)
        # udo only allows a single datatype to be specified per tensor
        self.data_type = get_internal_data_type(tensor_dict.get("data_type", "FLOAT_32"), self.name)
        self.allowed_data_types = [self.data_type]
        # this is not really needed, but DSP templates still use it
        # TODO: Remove once its confirmed that this value is always TF anyway
        self.quant_type = SnpeUdoConstants.SNPE_UDO_QUANT_TYPES['TF']

    @staticmethod
    def create_tensor_infos(operator_dict, dict_type):
        tensor_infos = list()
        for tensor_dict in operator_dict.get(str(dict_type), list()):
            tensor_info = TensorInfo()
            tensor_info.from_dict(tensor_dict)
            tensor_infos.append(tensor_info)

        return tensor_infos

    @staticmethod
    def create_per_core_tensor_infos(tensor_dict, type, core_types=None):
        if core_types is None:
            core_types = SnpeUdoConstants.SNPE_UDO_CORETYPES.keys()
        tensor_infos = list()

        for tensor in tensor_dict.get(str(type), list()):
            tensor_info = TensorInfo()
            tensor_info.from_dict(tensor)
            setattr(tensor_info, "repeated", tensor.get("repeated", False))
            setattr(tensor_info, "static", tensor.get("static", False))
            if 'per_core_data_types' not in tensor:
                # set per core data types based on data_type to maintain backward compatibility
                # if a user sets data_type only it will be replicated for each runtime
                data_types = [tensor.get("data_type")] * len(core_types)
                per_core_dict = get_per_core_data_types(dict(zip(core_types, data_types)))
                setattr(tensor_info, "per_core_data_types", per_core_dict)

                # In this case, it should always be assumed that static data always has an allowed
                # data type of float32 since the CPU core type was not defined
                if tensor_info.static:
                    tensor_info.allowed_data_types.append("QNN_DATATYPE_FLOAT_32")
            else:
                per_core_dict = get_per_core_data_types(tensor.get("per_core_data_types"))
                tensor_info.allowed_data_types = list(per_core_dict.values())
                if "SNPE_UDO_CORETYPE_CPU" in per_core_dict:
                    # if a core type has been defined for CPU explicitly, the default
                    # datatype should be changed
                    tensor_info.data_type = per_core_dict["SNPE_UDO_CORETYPE_CPU"]
                setattr(tensor_info, "per_core_data_types", per_core_dict)
            tensor_infos.append(tensor_info)

        return tensor_infos

    def as_dict(self):
        temp_dict = super(TensorInfo, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        temp_dict.update(per_core_data_types=getattr(self, "per_core_data_types", None))
        return temp_dict


class ScalarParam(CustomScalarParam):
    def __init__(self, data, data_type=None):
        if data_type is None:
            data_type = get_snpe_type(data)  # assign datatype based on data
        super().__init__(data, data_type)
        self.param_type = 'QNN_PARAMTYPE_SCALAR'

    def as_dict(self):
        temp_dict = super(ScalarParam, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        return temp_dict


class TensorParam(CustomTensorParam):
    def __init__(self, data, tensor_info):
        super().__init__(data, tensor_info)
        # now set datatype based on data, or use existing datatype if data is deliberately empty
        self.data_type = get_snpe_type(data) if data else tensor_info.data_type
        self.param_type = 'QNN_PARAMTYPE_TENSOR'

    def as_dict(self):
        temp_dict = super(TensorParam, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        temp_dict.update(per_core_data_types=getattr(self, "per_core_data_types", None))
        return temp_dict


class StringParam(ScalarParam):
    def __init__(self, value):
        super().__init__(value, 'QNN_DATATYPE_UINT_8')
        self.param_type = 'QNN_PARAMTYPE_STRING'

    def as_dict(self):
        temp_dict = super(StringParam, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        return temp_dict


class Operator(CustomOperator):
    """
    This object describes an operation provided in the config spec, using inputs, outputs, tensor_params and scalar_params.
    The metaclass ensures that the certain types are valid. The udo_property method ensures that those fields cannot be
    set directly, and is essentially an accessor to view the operator's members.
    """
    input = aggregate_property('input', TensorInfo)
    output = aggregate_property('output', TensorInfo)
    param = aggregate_property('param', TensorInfo)

    def __init__(self, type_name, core_types=None, *, dsp_arch_types=None):
        super().__init__(type_name)
        if core_types is not None:
            self.core_types = get_internal_core_types(core_types)
        else:
            self.core_types = [SnpeUdoConstants.SNPE_UDO_CORETYPES["CPU"]]
        self.dsp_arch_types = dsp_arch_types
        self.__param_types = dict()

    @staticmethod
    def from_dict(op_dict):
        try:
            core_types = op_dict['core_types']
            self = Operator(op_dict['type'], core_types)
            self.inputs(TensorInfo.create_per_core_tensor_infos(op_dict, 'inputs', core_types))
            self.outputs(TensorInfo.create_per_core_tensor_infos(op_dict, 'outputs', core_types))
        except KeyError as e:
            raise KeyError(
                "Required operator field: {} was not found in config".format(str(e).split(':')[-1]))

        # Create params as generic tensor info and then set param type manually
        scalar_params = TensorInfo.create_tensor_infos(op_dict, 'scalar_params')
        for param in scalar_params:
            setattr(param, "param_type", "QNN_PARAMTYPE_SCALAR")
        self.params(scalar_params)

        tensor_params = TensorInfo.create_tensor_infos(op_dict, 'tensor_params')
        self.params(tensor_params)

        self.dsp_arch_types = op_dict['dsp_arch_types'] if 'dsp_arch_types' in op_dict else []

        return self

    @property
    def scalar_param(self):
        return [param for param in self.param if param.param_type == "QNN_PARAMTYPE_SCALAR"]

    @property
    def tensor_param(self):
        return [param for param in self.param if param.param_type == "QNN_PARAMTYPE_TENSOR"]

    def __copy__(self):
        new_operator = Operator(self.name, self.core_types)
        new_operator.inputs(self.input)
        new_operator.outputs(self.output)
        new_operator.params(self.params)
        new_operator.dsp_arch_types = self.dsp_arch_types
        new_operator.__param_types = self.__param_types


class Param(CustomParam):
    param_type = property_type('param_type', ParamTypes)
    param = union_property('param', [type(None), ScalarParam, TensorParam])

    def __init__(self, name, param_type, param=None):
        super(Param, self).__init__(name, param_type, param)


class SnpeUdoCustomOp(CustomOp):
    __metaclass__ = ABCMeta
    methods = dict()
    inputs = aggregate_property('inputs', TensorInfo)
    outputs = aggregate_property('outputs', TensorInfo)
    param = aggregate_property('params', Param)

    def __init__(self,
                 op_type: str,
                 input_tensors: List[TensorInfo],
                 output_tensors: List[TensorInfo], *,
                 params: Optional[List[Param]] = None,
                 param_info: Optional[List[TensorInfo]] = None,
                 src_op=None,
                 infer_output_shapes=None,
                 name: Optional[str] = ""):
        super().__init__(op_type, input_tensors,
                         output_tensors, params, param_info, src_op, name)
        if infer_output_shapes is not None:
            self.infer_output_shapes = infer_output_shapes
        # set backend specific arguments
        self.set_axis_orders(self.inputs, tensor_layouts=SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT)
        self.set_axis_orders(self.outputs, tensor_layouts=SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT)

    def as_dict(self, graph):
        tensor_params = {param.name: param.param.as_dict() for _, param in self.params.items()
                         if param.param_type == ParamTypes.TENSOR}
        scalar_params = {param.name: param.param.as_dict() for _, param in self.params.items()
                         if param.param_type == ParamTypes.SCALAR or
                         param.param_type == ParamTypes.STRING}
        inputs = OrderedDict()
        outputs = OrderedDict()
        for input_ in self.inputs:
            inputs[input_.name] = input_.as_dict()
        for output in self.outputs:
            outputs[output.name] = output.as_dict()

        tensor_param_names = []
        for _, tensor_param in tensor_params.items():
            if tensor_param['static']:
                # if tensor is present in self.params and not in self.param_info
                # it is the input static tensor which was added to params
                if tensor_param['name'] in self.params and (tensor_param['name'] not in [param['name'] for param in self.param_info]):
                    # creates input tensor info
                    input_tensor = TensorInfo()
                    input_tensor.name = tensor_param['name']
                    input_tensor.allowed_data_types = tensor_param['allowed_data_types']
                    # input_tensor.allowed_values = tensor_param['allowed_values']
                    # input_tensor.shape = tensor_param['shape']
                    input_tensor.rank = tensor_param['rank']
                    input_tensor.default_value = tensor_param['default_value']
                    input_tensor.layout = tensor_param['layout']
                    input_tensor.repeated = tensor_param['repeated']
                    input_tensor.dimensions = tensor_param['dimensions']
                    input_tensor.static = tensor_param['static']
                    input_tensor.data = tensor_param['data']
                    input_op_name = self.name + '_' + tensor_param['name']
                    inputs[input_op_name] = input_tensor.as_dict()
                    tensor_param_names.append(tensor_param['name'])
                    # adds the constant op for static inputs and adds the buffer to graph
                    input_op = op_adapter.ConstantOp(input_op_name, tensor=input_tensor.data)
                    axis_format = AxisTracker.AxisFormat.OIHW
                    if input_tensor.rank == 1:
                        axis_format = AxisTracker.AxisFormat.ANY
                    graph.add(input_op, [], [input_op_name], axis_formats=[axis_format])

        # removes the static inputs from tensors params which were added earlier
        for param in tensor_param_names:
            tensor_params.pop(param)
        return inputs, outputs, scalar_params, tensor_params

    @classmethod
    @abstractmethod
    def extract_attrs(cls, src_op, param_info: Dict[str, TensorInfo]):
        """
        The intention of this method is to extract param_info from a framework src_op and return a dictionary of
        Param objects, such that "attr_name": "Param". This must be implemented, as it is called during
        initialization
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
        :raises An error if data_type cannot be set
        :returns
        """

    def set_static_tensor_to_param(self, tensors):
        """
        Sets a static tensor to a param. This method is called by the base class, meaning instances of this class
        are expected to have static tensors become params. This method takes a single tensor, and changes it to a
        param object. Note that a static tensor must have a data field defined.
        :param tensors: The tensor to be made a param.
        """
        local_tensor = []
        for tensor_info in tensors:
            if tensor_info.static:
                log_debug('Static custom input tensor: {} found for op: {}. '
                          'Note this tensor will be stored in the model output'
                          .format(tensor_info.name, self.op_type))
                self.params[tensor_info['name']] = Param(tensor_info['name'], ParamTypes.TENSOR,
                                                         TensorParam(None, tensor_info))
            else:
                local_tensor.append(tensor_info)

        return local_tensor


BackendCustomOp = SnpeUdoCustomOp
