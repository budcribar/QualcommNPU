# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from enum import Enum

from typing import NewType, List, Optional, Union, Any, TypeVar, Dict
from collections.abc import MutableMapping
import copy
from qti.aisw.converters.qnn_backend.custom_ops.core import *
from qti.aisw.converters.common.custom_ops.utils.config_helpers import *
from qti.aisw.op_package_generator.op_def.op_def_classes import OpDef, OpDefPackageInfo, TensorElement, OpDefElement, OpDefCollection


# ------------------------------------------------------------------------------
#   Qnn config Enum Style Classes
# ------------------------------------------------------------------------------
class QnnTemplateFileReader:
    """ Enum class that stores template file names and their corresponding types"""

    template_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')

    # GPU and CPU share implementations DSP, HTP implementation file is unique The other
    # templates are explained by the naming convention. New templates should be added here and
    # placed into corresponding file type.

    class TemplateFileTypes(Enum):
        COMMON = 0,
        AIC = 1
        CPU = 2,
        GPU = 3,
        DSP = 4,
        HTP = 5,
        HTPMCP = 6,
        HTP_FP16 = 7,
        MAKEFILE = 8,
        USER_DEFINED = 9,
        CONVERTER = 10,
        UNKNOWN = 11

        def describe(self):
            return self.name, self.value

        @classmethod
        def default(cls):
            return cls.CPU

        @classmethod
        def is_backend_type(cls, template_type):
            return template_type == cls.CPU or template_type == cls.GPU or template_type == cls.DSP or \
                   template_type == cls.HTP or template_type == cls.HTPMCP or template_type == cls.HTP_FP16 or \
                   template_type == cls.CONVERTER

    # TODO: Add backend types
    # Note: All backend types must have at least one source file, which should be listed first
    # , and one interface file which must be listed second.
    # e.x ["source.mako", "header.mako"]
    DEFAULT_TEMPLATE_FILES = {
        TemplateFileTypes.COMMON: ['common_cmakelists.mako'],
        TemplateFileTypes.CPU: ['cpu_source.mako', "cpu_interface.mako"],
        TemplateFileTypes.HTP: ['htp_source.mako', "htp_interface.mako"],
        TemplateFileTypes.HTPMCP: ['htp_source.mako', "htp_mcp_interface.mako"],
        TemplateFileTypes.HTP_FP16: ['htp_fp16_source.mako', "htp_interface.mako"],
        TemplateFileTypes.GPU: ['gpu_source.mako', 'gpu_interface.mako', 'gpu_operation.mako'],
        TemplateFileTypes.DSP: ['dsp_source.mako', 'dsp_interface.mako', 'dsp_header.mako'],
        TemplateFileTypes.MAKEFILE: ['android_makefile.mako'],
        TemplateFileTypes.CONVERTER: ['converter_op_package.mako'],
        TemplateFileTypes.AIC: [
            "aic_kernel.mako", "aic_interface.mako", "aic_functions.mako", "aic_yaml_config.mako"
        ]}

    @classmethod
    def get_template_type_by_backend(cls, backend):
        if backend == 'CPU':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.CPU]
        elif backend == 'GPU':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.GPU]
        elif backend == 'DSP':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.DSP]
        elif backend == 'HTP':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.HTP]
        elif backend == 'HTPMCP':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.HTPMCP]
        elif backend == 'HTP_FP16':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.HTP_FP16]
        elif backend == 'CONVERTER':
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.CONVERTER]
        elif backend == "AIC":
            return cls.DEFAULT_TEMPLATE_FILES[cls.TemplateFileTypes.AIC]
        raise LookupError("Cannot retrieve mako template for unknown backend: {}".format(backend))


class QnnPackageStatus(Enum):
    """
    This class contains the possible statuses of a qnn package
    """
    NOT_GENERATED = 0
    STRUCTURE_IS_SET = 1
    TEMPLATES_IMPLEMENTED = 3
    PACKAGE_CAN_COMPILE = 4  # for testing purposes only


class QnnPackageInfo:
    """
    QnnPackageInfo contains contains information gleaned from the user provided config that will constitute a package.
    It is freely editable, meaning users can add and remove information as needed. It is also the main reference point
    for constructing a package.
    """
    operators = aggregate_property('operators', Operator)
    value_info = property_type('value_info', dict)

    def __init__(self, package_name, package_root, backend=None, operators=None, version='1.0.0', domain='aisw'):
        self.name = package_name
        self.root = package_root
        self.backend = backend
        self.version = version
        self.domain = domain
        self.status = QnnPackageStatus.NOT_GENERATED
        self.cached_op_collection = OpDefCollection()
        if operators is not None:
            self.operators = operators

    def add_operator(self, operator: Operator):
        """
        Adds an operator to the package_info. Note this is the desired way to add operator in multiple steps,
        and the function can also be changed in multiple calls e.x add_operator().add_operator()
        :param operator: An object of type Operator
        :return: the package info object
        """
        if not isinstance(operator, Operator):
            raise TypeError("Expected type: {}, instead got type".format(Operator, type(operator)))
        if operator.package_name and operator.package_name != self.name:
            raise AttributeError("Expected package_name: {}"
                                 " from Operator: {} but got package_name: {} instead ",
                                 self.name, operator.type_name, operator.package_name)
        self._operators.append(operator)
        return self

    def has_operator(self, op_type: str) -> bool:
        """
         Checks for the existence of an operator in the object's operators list
        :param op_type: A string denoting the operation type
        :return: True if it exists in self.operators, else false
        """
        for operator in self.operators:
            if operator.type_name == op_type:
                return True
        return False

    def add_op_def(self, *op_defs):
        """
        Converts an op_def_api OpDef object into an Operator object, and then its to the operators list
        :param op_defs: A list of operation definition objects
        :return:
        """
        for op_def in op_defs:
            if op_def.package_info is None:
                op_def.package_info = OpDefPackageInfo(name=self.name, domain=self.domain, version=self.version)
                if op_def.name not in self.cached_op_collection.get_op_defs():
                    self.cached_op_collection.add_op_def(op_def)
                    if not self.backend in self.cached_op_collection.supported_ops:
                        self.cached_op_collection.supported_ops[self.backend] = [op_def.name]
                    else:
                        self.cached_op_collection.supported_ops[self.backend].append(op_def.name)
            else:
                self.compare_fields(op_def.package_info, ['name', 'domain', 'version'])
            self._operators.append(self.get_operator_from_translator_op_def(op_def))
        return self

    def meld(self, qnn_package_info):
        """
        Melds two package infos together by adding operations from the argument to the instance.
        :param qnn_package_info: An object of type qnn package info
        :return: None
        :raises: TypeError if the argument is not a QnnPackageInfo, and AssertionError if it does not match the instance
        name, version, backend, domain and root.
        """
        if not isinstance(qnn_package_info, type(self)):
            raise TypeError('Expected package info argument type:{}, instead got: {}'.format(type(qnn_package_info),
                                                                                             type(self)))
        if hasattr(self, "SNPE_UDO_ROOT"):
            # check package names, version, domain and root are similar
            self.compare_fields(qnn_package_info, ['name', 'version', 'domain', 'root'])
            if qnn_package_info.core_types[0] not in self.core_types:
                self.core_types += qnn_package_info.core_types
                self.supported_runtimes += qnn_package_info.supported_runtimes
            if qnn_package_info.dsp_arch_types:
                for arch_type in qnn_package_info.dsp_arch_types:
                    if arch_type not in self.dsp_arch_types:
                        self.dsp_arch_types += qnn_package_info.dsp_arch_types
            common_operators_name = set([operator.type_name for operator in self.operators]) & \
                                    set([operator.type_name for operator in qnn_package_info.operators])
            for operator in qnn_package_info.operators:
                if operator.type_name in common_operators_name:
                    for opr in self.operators:
                        if opr.type_name == operator.type_name:
                            existing_op = opr
                            break
                    existing_op.core_types += operator.core_types
                    existing_op.dsp_arch_types += operator.dsp_arch_types
                else:
                    self.operators.append(operator)
        else:
            # check package names, version, backend, domain and root are similar
            self.compare_fields(qnn_package_info, ['name', 'version', 'backend', 'domain', 'root'])
            if self.operators:
                for operator in qnn_package_info.operators:
                    if operator not in self.operators:
                        self.operators.append(operator)

    @value_info.getter
    def value_info(self):
        return {'package_name': self.root,
                'package_backend': self.backend,
                'operators': self.operators}

    def compare_fields(self, instance, arg_types: List[str]):
        for arg_type in arg_types:
            expected_arg = self[arg_type]
            instance_arg = instance[arg_type]
            if expected_arg != instance_arg:
                raise AssertionError("Expected {} to be {}, instead got {}".format(arg_type,
                                                                                   expected_arg,
                                                                                   instance_arg))

    @staticmethod
    def get_operator_from_translator_op_def(op_def: OpDef):
        """
        Transforms an op_def_classes OpDef into an instance of the Operator class.

        :param op_def: An instance of an Opdef
        :return: The created Operator if the opdef can be converted
        :raises: A KeyError if any fields are missing from the op_def, or resulting errors from the "from_translator"
        function calls.
        """
        try:
            self = Operator(op_def.name, op_def.package_info.name, "")
            self.input = list(map(TensorInfo.from_translator_tensor_element, op_def.inputs))
            self.output = list(map(TensorInfo.from_translator_tensor_element, op_def.outputs))
            self.param = list(map(TensorInfo.from_translator_tensor_element, op_def.parameters))
            self.use_default_translation = op_def.use_default_translation
        except KeyError as e:
            raise KeyError("Required operator field: {} was not found in config".format(str(e).split(':')[-1]))
        return self

    def __copy__(self):
        return QnnPackageInfo(self.package_name,
                              self.package_root,
                              self.backend,
                              copy.deepcopy(self.operators),
                              self.domain,
                              self.version)

    def __repr__(self):
        return str(dict(name=self.name, root=self.root, version=self.version,
                        backend=self.backend, domain=self.domain, operators=str(list(map(lambda x: x.__repr__(),
                                                                                         self.operators))))).replace(
            '\\', '')

    def __getattr__(self, item):
        return self.__getattribute__(item)

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, other):
        try:
            self.compare_fields(other, ['name', 'version', 'backend', 'root', 'operators', 'domain'])
        except AssertionError:
            return False
        return True


class QnnPackageCollection(MutableMapping):
    """
    Organizes package infos based on package_name, supported backend and the operations it contains.
    """

    def __init__(self, **kwargs):
        super(QnnPackageCollection, self).__init__()
        self.package_count = 0
        self.package_names = []
        if kwargs:
            for name, value in kwargs:
                setattr(self, name, value)

    def __getitem__(self, name):
        if name not in self.__dict__:
            raise KeyError("Object has not been registered with a Package Collection".format(name))
        return getattr(self, name)

    def __setitem__(self, package_name, backend_ops: Dict[str, List]):
        """
        Creates a key-value mapping of package-name ->{backend->ops}, such that:
        1. if a package name does not exist, a new entry is created.
        2. if a package name does exist, but the backend does not, a new backend entry is created
           in an existing package entry
        3. if both package name and backend exist, then the existing package info is retrieved, and the ops are
           added to the existing backend entry.
        :param package_name: The name of the package to be added.
        :param backend_ops: must be a dictionary of {backend: operators}
        :return:
        """
        if hasattr(self, package_name):
            dup_package = self.__getitem__(package_name)
            for backend, ops in backend_ops.items():
                if backend in dup_package.keys():
                    dup_package[backend].extend(ops)
                else:
                    dup_package[backend] = ops
        else:
            self.__dict__[package_name] = dict()
            for backend, ops in backend_ops.items():
                self.__dict__[package_name][backend] = ops
            self.package_count += 1
            self.package_names.append(package_name)

    def __delitem__(self, key):
        if hasattr(self, key):
            self.__dict__.pop(key)

    def __iter__(self):
        return (package_name for package_name in self.package_names)

    def __len__(self):
        return self.package_count
