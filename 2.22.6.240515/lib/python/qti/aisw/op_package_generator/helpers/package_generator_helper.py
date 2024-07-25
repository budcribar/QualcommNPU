# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from ..generator import *
import qti.aisw.op_package_generator.op_def.op_def_classes as op_def_api
import qti.aisw.op_package_generator.translator.op_def_translator as xml_package_translator
import os


# ------------------------------------------------------------------------------
#   Qnn Package Generator Helper Core Classes (IN-PROGRESS)
# ------------------------------------------------------------------------------
def compare_fields(arg_types: List[str], expected, instance):
    """
    Compares two objects by comparing the values of the specified members in arg types.
    Note the objects do not have to be the same type, but must have each of the specified argtypes.
    :param arg_types: The list of members to compare
    :param expected: The golden object for which the instance will be compared against
    :param instance: The instance object that will be validated
    :return: True, if all the fields match. False, otherwise.
    """
    for arg_type in arg_types:
        expected_arg = getattr(expected, arg_type)
        instance_arg = getattr(instance, arg_type)
        if expected_arg != instance_arg:
            return False
    return True


class QnnPackageManager:
    """
    This class controls the registration and generation of package infos using its generator member.
    """
    __generator = QnnOpPackageGenerator()
    __generator.package_infos = []
    __cached_op_collection = op_def_api.OpDefCollection()
    _generated_xml = []
    setup_logging(True)

    def __init__(self):
        pass

    @classmethod
    def create_package_info(cls,
                            package_name: str,
                            *,
                            root: str = os.getcwd(),
                            backend: str = 'CPU',
                            op_defs: Optional[List[op_def_api.OpDef]] = None,
                            version: str = '1.0.0',
                            domain: str = '',
                            register: bool = True):
        if op_defs is None:
            op_defs = []
        resolved_operators = [QnnPackageInfo.get_operator_from_translator_op_def(op_def)
                              for op_def in op_defs]
        package_info = QnnPackageInfo(package_name, root, backend, resolved_operators, version,
                                      domain)
        if register:
            cls.add_package_info(package_info, mangle=True)
        return package_info

    @classmethod
    def get_package_info(cls, name: str) -> QnnPackageInfo:
        return cls.__generator.get_package_info_by_name(name)

    @classmethod
    def add_package_info(cls, package_info: QnnPackageInfo, mangle=False):
        """
        Adds a package info to the generator.package_infos. if the package info to be registered exists, then it
        is an error unless mangle is set to True, in which case the name is mangled to ensure there are no duplicates.

        Mangling occurs in the following steps:
        if package name exists, package name becomes package_name_domain_version
        if it still exists, package_name becomes package_name_domain_version_idx until idx is 5.
        if package name still exists, then an error is raised.

        :param package_info: A package info object
        :param mangle: A boolean flag that if set to true, indicates that a duplicate package info should mangle
                       its name as much as possible to ensure registration
        :return:
        """
        backend_package_name = ''
        MAX_COUNT = 5
        package_info_names = [info.name for info in cls.__generator.package_infos]
        if package_info.name in package_info_names:
            if mangle:
                backend_package_name = package_info.name + package_info.backend
                if backend_package_name not in package_info_names:
                    log_info("Package name changed from {} to {}".format(package_info.name,
                                                                          backend_package_name))
                    package_info.name = backend_package_name
                    cls.__generator.register_package_info(package_info)
                    return
                version_package_name = '_'.join(
                    [backend_package_name + ".", package_info.domain,
                     package_info.version])
                if version_package_name not in package_info_names:
                    log_info("Package name changed from {} to {}".format(package_info.name,
                                                                          version_package_name))
                    package_info.name = version_package_name
                    cls.__generator.register_package_info(package_info)
                    return
                package_name = version_package_name
                idx = 0
                while package_name in package_info_names and idx < MAX_COUNT:
                    package_name = package_name + "_" + str(idx)
                    if package_name not in package_info_names:
                        log_info("Package name changed from {} to {}".format(package_info.name,
                                                                              package_name))
                        package_info.name = package_name
                        cls.__generator.register_package_info(package_info)
                        return
            raise AttributeError(
                "Cannot register package info with duplicate name: {}".format(package_info.name))
        else:
            cls.__generator.register_package_info(package_info)

    @classmethod
    def generate_packages(cls, force_generation=True, query=True):
        cls.__generate_xml()
        package_paths = cls.__generator.setup_file_paths(force_generation=force_generation)
        cls.__generator.implement_packages()
        if query:
            cls.__generator.generation_is_complete()

        # remove temporary generated xml since it has been moved into config dir
        for xml_file in cls._generated_xml:
            try:
                os.remove(os.path.abspath(xml_file))
            except OSError or IOError:
                pass
        return package_paths

    @classmethod
    def __generate_xml(cls):
        for package_info in cls.__generator.package_infos:
            xml_file = os.path.join("{}.xml".format(package_info.name))
            translator_instance = xml_package_translator.OpDefTranslator \
                (xml_schema=cls.__generator.SCHEMA)
            translator_instance.write_op_defs(package_info.cached_op_collection, xml_file)
            log_info("Generated XML for package: {}".format(package_info.name))
            cls._generated_xml.append(xml_file)
            cls.__generator.config_paths.update({package_info.name: [xml_file]})


# ------------------------------------------------------------------------------
#   Helper Methods
# ------------------------------------------------------------------------------
DEFAULT_TYPE = Union[int, str, bool, float, list]
CONSTRAINT_TYPE = Optional[List[op_def_api.Constraint]]


def make_scalar_param(name: str,
                      *,
                      description: str = '',
                      mandatory: bool = True,
                      default: DEFAULT_TYPE = "",
                      datatypes: List[
                          op_def_api.QnnDatatype] = op_def_api.QnnDatatype.QNN_DATATYPE_FLOAT_32,
                      constraints: CONSTRAINT_TYPE = None) -> Union[op_def_api.ScalarElement,
                                                                    op_def_api.BoolElement]:
    if constraints is None:
        constraints = []
    if datatypes[0] is QnnDatatype.QNN_DATATYPE_BOOL_8:
        return op_def_api.BoolElement(name, description, mandatory, bool(default), datatypes)
    return op_def_api.ScalarElement(name, description, mandatory, default, datatypes, constraints)


def make_enum_param(name: str,
                    enum_vals: List[Any],
                    *,
                    description: str = '',
                    mandatory: bool = True,
                    default: DEFAULT_TYPE = 0) -> op_def_api.EnumElement:
    return op_def_api.EnumElement(name, description, mandatory, enum_vals, default)


def make_op_def(name: str,
                description: str = '',
                *,
                inputs=None,
                outputs=None,
                params=None) -> op_def_api.OpDef:
    if params is None:
        params = []
    if outputs is None:
        outputs = []
    if inputs is None:
        inputs = []
    return op_def_api.OpDef(name, description, ins=inputs, outs=outputs, parameters=params)


def make_package_info(name: str,
                      *,
                      root: str = os.getcwd(),
                      backend: str = 'CPU',
                      op_defs: Optional[List[op_def_api.OpDef]] = None,
                      version: str = '1.0.0',
                      domain: str = '') -> QnnPackageInfo:
    return QnnPackageManager.create_package_info(package_name=name,
                                                 root=root,
                                                 backend=backend,
                                                 op_defs=op_defs,
                                                 version=version,
                                                 domain=domain,
                                                 register=False)


# use decorator to set defaults
def make_tensor_param(name: str,
                      *,
                      description: str = "",
                      datatypes: List[
                          op_def_api.QnnDatatype] = op_def_api.QnnDatatype.QNN_DATATYPE_FLOAT_32,
                      rank: op_def_api.RankType = op_def_api.RankType.VECTOR,
                      shape: str = "",
                      layout: op_def_api.Layout = op_def_api.Layout.NHWC,
                      constraints: CONSTRAINT_TYPE = None,
                      mandatory: bool = True,
                      default: DEFAULT_TYPE = "") -> op_def_api.TensorElement:
    if constraints is None:
        constraints = []
    return op_def_api.TensorElement(name, description, mandatory, default,
                                    datatypes, int(rank), shape, layout, constraints)


def make_input_tensor_info(name: str,
                           *,
                           description: str = "",
                           datatypes: List[
                               op_def_api.QnnDatatype] = op_def_api.QnnDatatype.QNN_DATATYPE_FLOAT_32,
                           rank: op_def_api.RankType = op_def_api.RankType.IMAGE,
                           shape: str = "",
                           layout: op_def_api.Layout = op_def_api.Layout.NHWC,
                           constraints: CONSTRAINT_TYPE = None,
                           mandatory: bool = True,
                           default: DEFAULT_TYPE = "",
                           repeated: bool = False) -> op_def_api.InputElement:
    if constraints is None:
        constraints = []
    return op_def_api.InputElement(name, description, mandatory, default,
                                   datatypes, int(rank), shape, layout, constraints, repeated)


def make_output_tensor_info(name: str,
                            *,
                            description: str = "",
                            mandatory=True,
                            datatypes: List[
                                op_def_api.QnnDatatype] = op_def_api.QnnDatatype.QNN_DATATYPE_FLOAT_32,
                            rank: op_def_api.RankType = op_def_api.RankType.IMAGE,
                            shape: str = "",
                            layout: op_def_api.Layout = op_def_api.Layout.NHWC,
                            constraints: CONSTRAINT_TYPE = None) -> op_def_api.OutputElement:
    if constraints is None:
        constraints = []
    return op_def_api.OutputElement(name, description, mandatory, datatypes, int(rank), shape,
                                    layout, constraints)


# define global variables to obscure API
Layout = op_def_api.Layout
Rank = op_def_api.RankType
QNN_DATATYPE_FLOAT_32 = op_def_api.QnnDatatype.QNN_DATATYPE_FLOAT_32
QNN_DATATYPE_FLOAT_16 = op_def_api.QnnDatatype.QNN_DATATYPE_FLOAT_16
QNN_DATATYPE_INT_32 = op_def_api.QnnDatatype.QNN_DATATYPE_INT_32
QNN_DATATYPE_INT_64 = op_def_api.QnnDatatype.QNN_DATATYPE_INT_64
QNN_DATATYPE_INT_16 = op_def_api.QnnDatatype.QNN_DATATYPE_INT_16
QNN_DATATYPE_INT_8 = op_def_api.QnnDatatype.QNN_DATATYPE_INT_8
QNN_DATATYPE_UINT_32 = op_def_api.QnnDatatype.QNN_DATATYPE_UINT_32
QNN_DATATYPE_UINT_64 = op_def_api.QnnDatatype.QNN_DATATYPE_UINT_64
QNN_DATATYPE_UINT_16 = op_def_api.QnnDatatype.QNN_DATATYPE_UINT_16
QNN_DATATYPE_UINT_8 = op_def_api.QnnDatatype.QNN_DATATYPE_UINT_8
QNN_DATATYPE_BOOL_8 = op_def_api.QnnDatatype.QNN_DATATYPE_BOOL_8
QNN_DATATYPE_UFIXED_POINT_32 = op_def_api.QnnDatatype.QNN_DATATYPE_UFIXED_POINT_32
QNN_DATATYPE_UFIXED_POINT_16 = op_def_api.QnnDatatype.QNN_DATATYPE_UFIXED_POINT_16
QNN_DATATYPE_UFIXED_POINT_8 = op_def_api.QnnDatatype.QNN_DATATYPE_UFIXED_POINT_8
QNN_DATATYPE_SFIXED_POINT_32 = op_def_api.QnnDatatype.QNN_DATATYPE_SFIXED_POINT_32
QNN_DATATYPE_SFIXED_POINT_16 = op_def_api.QnnDatatype.QNN_DATATYPE_SFIXED_POINT_16
QNN_DATATYPE_SFIXED_POINT_8 = op_def_api.QnnDatatype.QNN_DATATYPE_SFIXED_POINT_8
