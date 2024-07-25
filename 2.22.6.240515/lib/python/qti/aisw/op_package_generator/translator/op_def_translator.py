# =============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import lxml.etree as ET
import os
from typing import Union, List, Tuple

try:
    from qti.aisw.op_def.op_def_classes import *
    SCHEMA_FILE_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'OpDef.xsd'))
except ImportError:
    from qti.aisw.op_package_generator.op_def.op_def_classes import *
    SCHEMA_FILE_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpDef.xsd'))

OP_DEF_XML_PATH = os.path.abspath(os.path.join('..', '..', os.path.dirname(__file__), 'MasterOpDef.xml'))

DEFAULT_PACKAGE_NAME = "aisw"
DEFAULT_DOMAIN_NAME  = "qti"

class XmlConstants:
    '''
    Class containing the strings that are defined as part of the schema OpDef.xsd
    '''
    # OpDef Strings
    OP_DEF = "OpDef"
    REF = "Reference"
    REF_SOURCE = "Source"
    REF_URL = "Url"
    # Attr
    INPUT = "Input"
    OUTPUT = "Output"
    PARAM = "Parameter"
    # Custom Op Additions
    DEFAULT_TRANSLATION = "UseDefaultTranslation"
    SUPPORTED_BE = "SupportedBackend"
    # OpDefList
    OP_DEF_LIST = "OpDefList"

     # SupplementalOpDef
    SUPP_OP_DEF = "SupplementalOpDef"
    ONLY_DEFAULT = "OnlyDefaultSupported"
    # SupplementalOpDef List
    SUPP_OP_DEF_LIST = "SupplementalOpDefList"
    BACKEND = "Backend"
    RUNTIME = "Runtime"
    SUPPORTED_OPS = "SupportedOps"
    OP_NAME = "OpName"
    # SupplementalOpDefSpecification
    SUPP_OP_DEF_SPEC = "SupplementalOpDefSpecification"

    # OpDefCollection
    OP_DEF_COLLECTION = "OpDefCollection"
    # Package Strings
    PACKAGE_NAME = "PackageName"
    DOMAIN = "Domain"
    VERSION = "Version"

    # Tensor Strings
    # Name
    NAME = "Name"
    # Description
    DESC = "Description"
    DESC_CODE = "Code"
    DESC_CONTENT = "Content"
    # Constraint
    CONSTRAINT = "Constraint"
    CONSTRAINT_ID = "id"
    CONSTRAINT_TYPE = "Type"
    CONSTRAINT_CODE = "Code"
    # Mandatory/Default/Datatype
    MANDATORY = "Mandatory"
    DEFAULT = "Default"
    DATATYPE = "Datatype"
    # Shape
    RANK = "Rank"
    RANK_SCALAR = "SCALAR"
    RANK_0D = "0D"
    RANK_1D = "1D"
    RANK_2D = "2D"
    RANK_3D = "3D"
    RANK_4D = "4D"
    RANK_5D = "5D"
    RANK_ND = "ND"
    SHAPE = "Shape"
    SHAPE_TEXT = "Text"
    LAYOUT = "Layout"
    # ActivationTensor Adds
    REPEATED = "Repeated"
    STATIC_TENSOR = "IsStaticTensor"
    # Quantization Params
    QUANT_PARAM = "QuantParam"
    QUANT_ENCODING = "Encoding"
    QUANT_AXIS = "Axis"
    QUANT_SYMMETRIC = "Symmetric"
    QUANT_SCALE = "Scale"
    QUANT_OFFSET = "Offset"
    QUANT_MIN = "Min"
    QUANT_MAX = "Max"
    QUANT_MATH_INVARIANT = "IsMathInvariant"

    # Param specific
    ENUM = "Enumeration"
    ENUM_VAL = "Enum"


class OpDefTranslator:
    DEFAULT_SCHEMA = SCHEMA_FILE_NAME
    """
    Class to translate between XML and Python OpDef Objects.
    Parses XML and validates content and creates OpDefs.
    Parses OpDef objects and creates/writes corresponding XML.
    """

    def __init__(self, xml_file: str=None, xml_schema: str=DEFAULT_SCHEMA):
        """
        Initializes the XML Parser, XML Schema, and Element Tree
        and validates the Element Tree against the schema

        :param xml_file the XML file containing the OpDefs
        :param xml_schema the XML Schema file that validates xml_file
        """
        self.__schema = ET.XMLSchema(file=xml_schema)
        if xml_file is not None:
            self.__tree = ET.parse(xml_file)
            self.__validate_xml_against_schema()
        else:
            self.__tree = None

    def set_xml_src(self, xml_file: str):
        self.__tree = ET.parse(xml_file)
        self.__validate_xml_against_schema()

    def __validate_xml_against_schema(self, tree: ET.ElementTree=None):
        """
        Validates that the content of the Element Tree matches the Schema
        :param tree XML tree to be validated
        :raise Exception if validation fails with an error
               RuntimeError if ETree does not match Schema
        """

        if tree is None:
            if self.__tree is None:
                raise RuntimeError("Error during OpDef XML Validation. XML source not set.")
            tree = self.__tree
        try:
            self.__schema.assertValid(tree)
        except ET.DocumentInvalid as e:
            print("ERROR: OpDef XML does not match Schema")
            print(str(e))
            raise e

        if not self.__schema.validate(tree):
            raise RuntimeError("ERROR: OpDef XML does not match Schema")

    def translate_ops(self, op_def_collection: OpDefCollection=None) -> OpDefCollection:
        """
        Translates the internal Element Tree to an OpDef Collection
        :param: op_def_collection The collection of OpDefs to add to
        :return: op_def_collection The Collection containing the OpDefs with the current XML Sou
        """
        if self.__tree is None:
            raise RuntimeError("Error during OpDef Translation. XML source not set.")

        root = self.__tree.getroot()

        if op_def_collection is None:
            # initialize a new op_def_collection
            op_def_collection = OpDefCollection()
        if root.tag == XmlConstants.OP_DEF_COLLECTION:
            package_info = OpDefPackageInfo(root.get(XmlConstants.PACKAGE_NAME), \
                                            root.get(XmlConstants.DOMAIN), \
                                            root.get(XmlConstants.VERSION))
            self.__translate_op_def_list(op_def_collection, root.find(XmlConstants.OP_DEF_LIST), package_info)
            if root.findall(XmlConstants.SUPP_OP_DEF_LIST) is not None:
                for supp_elem in root.findall(XmlConstants.SUPP_OP_DEF_LIST):
                    self.__translate_supp_op_def_list(op_def_collection, supp_elem)
        elif root.tag == XmlConstants.OP_DEF_LIST:
            package_info = None
            if root.get(XmlConstants.VERSION) is not None:
                package_info = OpDefPackageInfo(DEFAULT_PACKAGE_NAME, \
                                                DEFAULT_DOMAIN_NAME, \
                                                root.get(XmlConstants.VERSION))
            self.__translate_op_def_list(op_def_collection, root, package_info)
        elif root.tag == XmlConstants.SUPP_OP_DEF_LIST:
            self.__translate_supp_op_def_list(op_def_collection, root)
        elif root.tag == XmlConstants.SUPP_OP_DEF_SPEC:
            if root.findall(XmlConstants.SUPP_OP_DEF_LIST) is not None:
                for supp_elem in root.findall(XmlConstants.SUPP_OP_DEF_LIST):
                    self.__translate_supp_op_def_list(op_def_collection, supp_elem)
        else:
            raise RuntimeError("Unsupported XML Configuration")

        return op_def_collection

    def __translate_op_def_list(self, op_def_collection: OpDefCollection,\
                                root: ET.ElementTree, \
                                package_info: OpDefPackageInfo=None):
        '''
        Translates An OpDef List XML element and adds it to an OpDefCollection
        :param op_def_collection: Collection to add OpDefs in list to
        :param root: Root XML element for the OpDefList
        :param package_info: Any package info associated with this list
        '''
        for op_def_elem in root.findall(XmlConstants.OP_DEF):
            op_def = self.__create_op_def(op_def_elem)

            # set package info if defined
            if package_info:
                op_def.package_info = package_info
            op_def_collection.add_op_def(op_def)

            if op_def_elem.findall(XmlConstants.SUPPORTED_BE) is not None:
                for supported_be in op_def_elem.findall(XmlConstants.SUPPORTED_BE):
                    op_def_collection.update_backend_support(op_def.name, supported_be.text)

    def write_op_defs(self, op_def_collection: OpDefCollection, out_file: str="OpDef.xml"):
        '''
        Writes an OpDefCollection to an XML file validated against schema
        :param op_def_collection: OpDefCollection to write
        :param out_file: output XML
        '''
        try:
            xml = self.get_op_def_xml(op_def_collection)
            self.__validate_xml_against_schema(xml)
            xml.write(out_file, pretty_print=True, xml_declaration=True,   encoding="utf-8")
        except Exception as e:
            print("ERROR during XML Generation for {}".format(out_file))
            raise

    def get_op_def_xml(self, op_def_collection: OpDefCollection) -> ET.ElementTree:
        '''
        Translates an OpDefCollection to an XML Tree
        :param op_def_collection: OpDefCollection to translate
        :return: root XML element
        '''
        root_elem = ET.Element(XmlConstants.OP_DEF_COLLECTION)
        self.__create_xml_op_def_list(root_elem, op_def_collection)
        self.__create_xml_supp_op_def_list(root_elem, op_def_collection)

        return ET.ElementTree(root_elem)

    def __create_xml_op_def_list(self, root_elem: ET.Element, op_def_collection: OpDefCollection):
        '''
        Creates an XML OpDefList from the provided op_def_collection
        :param root_elem: Root XML element for the OpDefList subelement
        :param op_def_collection: collection to read ops from
        '''
        op_def_list = ET.SubElement(root_elem, XmlConstants.OP_DEF_LIST)
        added_package_info = False
        for op_def in op_def_collection.get_op_defs().values():
            if not added_package_info:
                package_info = op_def.package_info
                if package_info is not None:
                    root_elem.set(XmlConstants.PACKAGE_NAME, package_info.name)
                    root_elem.set(XmlConstants.DOMAIN, package_info.domain)
                    root_elem.set(XmlConstants.VERSION, package_info.version)
                    added_package_info = True

            op_def_elem = ET.SubElement(op_def_list, XmlConstants.OP_DEF)
            name = ET.SubElement(op_def_elem, XmlConstants.NAME)
            name.text = op_def.name

            description = ET.SubElement(op_def_elem, XmlConstants.DESC)
            self.__get_desc_xml(description, op_def.description)

            for ref in op_def.references:
                ref_elem = ET.SubElement(op_def_elem, XmlConstants.REF)
                source, url = self.__invert_ref_string(ref)
                ref_elem.set(XmlConstants.REF_SOURCE, source)
                ref_elem.set(XmlConstants.REF_URL, url)

            for op_def_input in op_def.inputs:
                input_elem = ET.SubElement(op_def_elem, XmlConstants.INPUT)
                self.__get_xml_element_common(op_def_input, input_elem)
                if op_def_input.repeated:
                    repeated = ET.SubElement(input_elem, XmlConstants.REPEATED)
                    repeated.text = "true"
                if op_def_input.is_static_tensor:
                    is_static = ET.SubElement(input_elem, XmlConstants.STATIC_TENSOR)
                    is_static.text = "true"

            for op_def_output in op_def.outputs:
                output_elem = ET.SubElement(op_def_elem, XmlConstants.OUTPUT)
                self.__get_xml_element_common(op_def_output, output_elem)
                if op_def_output.repeated:
                    repeated = ET.SubElement(output_elem, XmlConstants.REPEATED)
                    repeated.text = "true"

            for param in op_def.parameters:
                param_elem = ET.SubElement(op_def_elem, XmlConstants.PARAM)
                self.__get_xml_element_common(param, param_elem)

                if param.type == ElementType.ENUM:
                    enumeration = ET.SubElement(param_elem, XmlConstants.ENUM)
                    for enum_val in param.enum:
                        enum = ET.SubElement(enumeration, XmlConstants.ENUM_VAL)
                        enum.text = enum_val

            if op_def.use_default_translation:
                use_default = ET.SubElement(op_def_elem, XmlConstants.DEFAULT_TRANSLATION)
                use_default.text = "true"

            for BE in op_def_collection.get_supported_backends_per_op(op_def.name):
                supported_be = ET.SubElement(op_def_elem, XmlConstants.SUPPORTED_BE)
                supported_be.text = BE

            if not added_package_info:
                raise RuntimeError("No Package Information specified in OpDef Collection")

    def __create_xml_supp_op_def_list(self, root_elem: ET.Element, op_def_collection: OpDefCollection):
        '''
        Creates an XML SupplementalOpDefList from the provided op_def_collection
        :param root_elem: Root XML element for the SupplementalOpDefList subelement
        :param op_def_collection: collection to read ops from
        '''
        all_supp_op_defs = op_def_collection.get_supplemental_op_defs()
        runtime = op_def_collection.ALL
        if XmlConstants.RUNTIME in root_elem.attrib:
            runtime = root_elem.get(XmlConstants.RUNTIME)
        for backend in all_supp_op_defs.keys():
            supp_op_defs = all_supp_op_defs[backend][runtime]
            supp_op_list = ET.SubElement(root_elem, XmlConstants.SUPP_OP_DEF_LIST)
            supp_op_list.set(XmlConstants.BACKEND, backend)
            supported_ops = op_def_collection.get_supported_ops(backend)
            supported_op_elem = ET.SubElement(supp_op_list, XmlConstants.SUPPORTED_OPS)
            for op in supported_ops:
                op_name_elem = ET.SubElement(supported_op_elem, XmlConstants.OP_NAME)
                op_name_elem.text = op

            for supp_op_def in supp_op_defs.values():
                supp_op_def = supp_op_def[0]
                supp_op_elem = ET.SubElement(supp_op_list, XmlConstants.SUPP_OP_DEF)
                name = ET.SubElement(supp_op_elem, XmlConstants.NAME)
                name.text = supp_op_def.name

                for op_def_attr in supp_op_def.inputs:
                    xml_elem = ET.SubElement(supp_op_elem, XmlConstants.INPUT)
                    self.__get_supp_xml_element_common(op_def_attr, xml_elem)
                for op_def_attr in supp_op_def.outputs:
                    xml_elem = ET.SubElement(supp_op_elem, XmlConstants.OUTPUT)
                    self.__get_supp_xml_element_common(op_def_attr, xml_elem)
                for op_def_attr in supp_op_def.parameters:
                    xml_elem = ET.SubElement(supp_op_elem, XmlConstants.PARAM)
                    self.__get_supp_xml_element_common(op_def_attr, xml_elem)

    def __get_supp_xml_element_common(self, supp_op_def_attr: SupplementalOpDefElement,\
                                      xml_elem: ET.SubElement):
        '''
        Common function to translate an attribute (input, output, parameter) of a SupplementalOpDef
        object to a corresponding XML element representation.
        :param supp_op_def_attr: The SupplementalOpDef attribute
        :param xml_elem: The root XML element that will be populated
        :return: None
        '''
        name = ET.SubElement(xml_elem, XmlConstants.NAME)
        name.text = supp_op_def_attr.name

        # Constraint
        self.__translate_constraints_to_xml(xml_elem, supp_op_def_attr)

        # Quant Params (Encoding, Axis, Symmetric, Scale, Offset, Min, Max)
        if  supp_op_def_attr.quant_params != "":
            self.__translate_quant_params_to_xml(xml_elem, supp_op_def_attr)

        # Datatype
        for dtype in supp_op_def_attr.datatypes:
            dtype_elem = ET.SubElement(xml_elem, XmlConstants.DATATYPE)
            dtype_elem.text = dtype.name

        # Shape (Layout, Text)
        if supp_op_def_attr.shape != "" or supp_op_def_attr.layout != Layout.UNDEFINED:
            shape = ET.SubElement(xml_elem, XmlConstants.SHAPE)
        if supp_op_def_attr.layout != Layout.UNDEFINED:
            layout = ET.SubElement(shape, XmlConstants.LAYOUT)
            layout.text = supp_op_def_attr.layout.name
        if supp_op_def_attr.shape != "":
            shape_text = ET.SubElement(shape, XmlConstants.SHAPE_TEXT)
            shape_text.text = supp_op_def_attr.shape

        if supp_op_def_attr.default_only:
            only_default = ET.SubElement(xml_elem, XmlConstants.ONLY_DEFAULT)
            only_default.text = "true"

    def __get_xml_element_common(self, op_def_attr: OpDefElement, xml_elem: ET.Element):
        '''
        Common function to translate an attribute (input, output, parameter, etc.) of a OpDef
        object to a corresponding XML element representation.
        :param op_def_attr: The OpDef attribute to be parsed
        :param xml_elem: The root XML element to which the new XML attribute will be added.
        :return: None
        '''

        if not isinstance(op_def_attr, OpDefElement):
            raise TypeError("Expected OpDef Element of type {} but got {}" \
                            .format(OpDefElement, type(op_def_attr)))
        # Name
        name = ET.SubElement(xml_elem, XmlConstants.NAME)
        name.text = op_def_attr.name

        # Description (Content/Code)
        if op_def_attr.description != "":
            desc = ET.SubElement(xml_elem, XmlConstants.DESC)
            self.__get_desc_xml(desc, op_def_attr.description)

        # Constraint
        self.__translate_constraints_to_xml(xml_elem, op_def_attr)

        # Mandatory
        mandatory = ET.SubElement(xml_elem, XmlConstants.MANDATORY)
        mandatory.text = "true" if op_def_attr.mandatory else "false"

        # Datatype
        for dtype in op_def_attr.datatypes:
            dtype_elem = ET.SubElement(xml_elem, XmlConstants.DATATYPE)
            dtype_elem.text = dtype.name

        # Shape (Rank, Layout, Text)
        shape = ET.SubElement(xml_elem, XmlConstants.SHAPE)
        rank = ET.SubElement(shape, XmlConstants.RANK)
        rank.text = self.__rank_to_xml(op_def_attr.rank)
        if op_def_attr.rank > 0 and op_def_attr.rank != RankType.SCALAR and op_def_attr.layout != Layout.UNDEFINED:
            layout = ET.SubElement(shape, XmlConstants.LAYOUT)
            layout.text = op_def_attr.layout.name
        if op_def_attr.shape != "":
            shape_text = ET.SubElement(shape, XmlConstants.SHAPE_TEXT)
            shape_text.text = op_def_attr.shape

        # Default
        if op_def_attr.default.value != "":
            default = ET.SubElement(xml_elem, XmlConstants.DEFAULT)
            default.text = str(op_def_attr.default.value)

    def __translate_supp_op_def_list(self, op_def_collection: OpDefCollection, root: ET.ElementTree):
        '''
        Translate a SupplementalOpDefList from XML to its python represenation.
        :param op_def_collection: The OpDefCollection python object to which SupplementalOpDef
                                  translated from XML will be added.
        :param root: The SupplementalOpDefList root XML object
        '''
        be = root.get(XmlConstants.BACKEND)
        runtime = None
        if XmlConstants.RUNTIME in root.attrib:
            runtime = root.get(XmlConstants.RUNTIME)
        for supp_op_def_elem in root.findall(XmlConstants.SUPP_OP_DEF):
            op_def_collection.add_supplemental_op_def(self.__create_supplemental_def(supp_op_def_elem), be, runtime)
        if root.find(XmlConstants.SUPPORTED_OPS) is not None:
            for supported_ops in root.find(XmlConstants.SUPPORTED_OPS).findall(XmlConstants.OP_NAME):
                op_def_collection.update_backend_support(supported_ops.text, be, runtime)

    def __create_op_def(self, op_def_elem: ET.Element) -> OpDef:
        '''
        From XML, create an OpDef object
        :param op_def_elem: The OpDef XML element
        :return: A python OpDef object parsed from XML
        '''
        name = op_def_elem.find(XmlConstants.NAME).text
        description = self.__get_desc_string(op_def_elem.find(XmlConstants.DESC))
        references = []
        ins = []
        outs = []
        params = []
        datatypes = []
        use_default_translation = False

        try:
            for ref in op_def_elem.findall(XmlConstants.REF):
                references.append(self.__get_ref_string(ref))
            for input_elem in op_def_elem.findall(XmlConstants.INPUT):
                ins.append(self.__create_input(input_elem))
                datatypes.extend([dtype.text for dtype in input_elem.findall(XmlConstants.DATATYPE)])
            for output_elem in op_def_elem.findall(XmlConstants.OUTPUT):
                outs.append(self.__create_output(output_elem))
                datatypes.extend([dtype.text for dtype in output_elem.findall(XmlConstants.DATATYPE)])
            if op_def_elem.findall(XmlConstants.PARAM) is not None:
                for param_elem in op_def_elem.findall(XmlConstants.PARAM):
                    params.append(self.__create_param(param_elem))
                    datatypes.extend([dtype.text for dtype in param_elem.findall(XmlConstants.DATATYPE)])
            if op_def_elem.find(XmlConstants.DEFAULT_TRANSLATION) is not None:
                use_default_translation = \
                    self.__text_to_bool(op_def_elem.find(XmlConstants.DEFAULT_TRANSLATION).text)

        except Exception as e:
            print("Error creating OpDef {}.".format(name))
            raise
        self.__validate_qnn_op_def_datatypes(op_def_elem, datatypes)

        return OpDef(name, description, references, ins, outs, params, use_default_translation)

    def __create_supplemental_def(self, supp_elem: ET.Element) -> SupplementalOpDef:
        '''
        From XML, create a SupplementalOpDef object
        :param supp_elem: The SupplementalOpDef XML object
        :return: A python SupplementalOpDef objects
        '''
        name = supp_elem.find(XmlConstants.NAME).text
        supp_inputs = []
        supp_outputs = []
        supp_params = []
        datatypes = []

        try:
            for input_elem in supp_elem.findall(XmlConstants.INPUT):
                supp_inputs.append(self.__create_supplemental(input_elem))
                datatypes.extend([dtype.text for dtype in input_elem.findall(XmlConstants.DATATYPE)])

            for output_elem in supp_elem.findall(XmlConstants.OUTPUT):
                supp_outputs.append(self.__create_supplemental(output_elem))
                datatypes.extend([dtype.text for dtype in output_elem.findall(XmlConstants.DATATYPE)])

            for param_elem in supp_elem.findall(XmlConstants.PARAM):
                supp_params.append(self.__create_supplemental(param_elem))
                datatypes.extend([dtype.text for dtype in param_elem.findall(XmlConstants.DATATYPE)])
        except Exception as e:
            print("Error creating Supplemental OpDef {}.".format(name))
            raise
        self.__validate_qnn_op_def_datatypes(supp_elem, datatypes)

        return SupplementalOpDef(name, supp_inputs, supp_outputs, supp_params)

    def __create_supplemental(self, elem: ET.Element) -> SupplementalOpDefElement:
        '''
        Create a SupplementalOpDef attribute, for use in a SupplementalOpDef
        :param elem: the XML element from which to parse the supplemental information
        :return: An SupplementalOpDefElement containing the supplemental information translated
                 from XML.
        '''
        name = elem.find(XmlConstants.NAME).text
        dtypes = []
        constraints = []
        quant_params = []
        shape = ""
        only_default = False
        layout = Layout.UNDEFINED

        constraint_elem = elem.findall(XmlConstants.CONSTRAINT)
        dtypes_elem = elem.findall(XmlConstants.DATATYPE)
        quant_params_elem = elem.findall(XmlConstants.QUANT_PARAM)
        shape_elem = elem.find(XmlConstants.SHAPE)
        def_elem = elem.find(XmlConstants.ONLY_DEFAULT)

        self.__translate_xml_constraints(constraint_elem, constraints)
        try:
            self.__translate_xml_quant_params(quant_params_elem, quant_params)
        except Exception as e:
            print("Error translating quantization params for op {}".format(name))
            raise

        if dtypes_elem is not None:
            for dtype in dtypes_elem:
                if dtype.text in QnnDatatype._member_names_:
                    dtypes.append(QnnDatatype[dtype.text])
                else:
                    dtypes.append(NativeDatatype[dtype.text].value)

        if shape_elem is not None:
            if shape_elem.find(XmlConstants.LAYOUT) is not None:
                for lout in shape_elem.findall(XmlConstants.LAYOUT):
                    layout_text = lout.text
                    layout = self.__get_layout(layout_text)

            if shape_elem.find(XmlConstants.SHAPE_TEXT) is not None:
                shape = self.__flatten_paragraph(shape_elem.find(XmlConstants.SHAPE_TEXT).text)

        if def_elem is not None:
            only_default = self.__text_to_bool(def_elem.text)

        return SupplementalOpDefElement(name, dtypes, quant_params, shape, constraints, layout, only_default)

    def __get_tensor_attr(self, elem: ET.Element, \
                          elem_type: XmlConstants=XmlConstants.INPUT) -> dict:
        '''
        Generically parse an XML TensorType element, and collect the attributes in a dictionary.
        :param elem: The XML element to parse
        :param elem_type: The type of the element i.e. XmlConstants.INPUT, XmlConstants.OUTPUT, etc.
                          for error messaging
        :return: A dictionary containing the python attributes of the XML element.
        '''
        name = ""
        desc = ""
        mandatory = True
        default = ""
        dtypes = []
        rank = -1
        shape = ""
        layout = Layout.UNDEFINED
        constraints = []

        name = elem.find(XmlConstants.NAME).text
        desc = self.__get_desc_string(elem.find(XmlConstants.DESC))
        mandatory = self.__text_to_bool(elem.find(XmlConstants.MANDATORY).text)

        if elem.find(XmlConstants.DEFAULT) is not None:
            default = elem.find(XmlConstants.DEFAULT).text

        for dtype in elem.findall(XmlConstants.DATATYPE):
            if dtype.text in QnnDatatype._member_names_:
                dtypes.append(QnnDatatype[dtype.text])
            else:
                dtypes.append(NativeDatatype[dtype.text].value)

        shape_elem = elem.find(XmlConstants.SHAPE)
        if shape_elem.find(XmlConstants.SHAPE_TEXT) is not None:
            shape = self.__flatten_paragraph(shape_elem.find(XmlConstants.SHAPE_TEXT).text)

        if shape_elem.find(XmlConstants.RANK) is not None:
            rank = self.__get_rank_from_xml(shape_elem.find(XmlConstants.RANK).text)

        if shape_elem.findall(XmlConstants.LAYOUT) is not None:
            for lout in shape_elem.findall(XmlConstants.LAYOUT):
                layout_text = lout.text
                if rank == 0:
                    raise ValueError("{} {} is scalar but has Tensor Layout".format(elem_type, name))
                layout = self.__get_layout(layout_text)

        self.__translate_xml_constraints(elem.findall(XmlConstants.CONSTRAINT), constraints)

        tensor_attrs = OrderedDict()
        tensor_attrs[XmlConstants.NAME] = name
        tensor_attrs[XmlConstants.DESC] = desc
        tensor_attrs[XmlConstants.MANDATORY] = mandatory
        tensor_attrs[XmlConstants.DEFAULT] = default
        tensor_attrs[XmlConstants.DATATYPE] = dtypes
        tensor_attrs[XmlConstants.RANK] = rank
        tensor_attrs[XmlConstants.SHAPE] = shape
        tensor_attrs[XmlConstants.LAYOUT] = layout
        tensor_attrs[XmlConstants.CONSTRAINT] = constraints
        return tensor_attrs

    def __get__activation_tensor_attr(self, elem: ET.Element, \
                                      tensor_type: XmlConstants=XmlConstants.INPUT) -> dict:
        '''
        Get the attributes for an activation tensor e.g. an InputElement or OutputElement
        :param elem: the XML element to parse
        :param tensor_type: The type of the tensor e.g. XmlConstants.INPUT, XmlConstants.OUTPUT
        :return: A dictionary of tensor attributes for the given element
        '''
        tensor_attrs = self.__get_tensor_attr(elem, tensor_type)
        repeated = False
        if elem.find(XmlConstants.REPEATED) is not None:
            repeated = self.__text_to_bool(elem.find(XmlConstants.REPEATED).text)
        tensor_attrs[XmlConstants.REPEATED] = repeated
        return tensor_attrs

    def __create_input(self, input_elem: ET.Element) -> InputElement:
        '''
        Create an InputElement python object translated from an Input XML element
        :param input_elem: The XML Input element to parse
        :return: A fully formed InputElement
        '''
        tensor_attrs = self.__get__activation_tensor_attr(input_elem, XmlConstants.INPUT)
        is_static_tensor = False
        if input_elem.find(XmlConstants.STATIC_TENSOR) is not None:
            is_static_tensor = self.__text_to_bool(input_elem.find(XmlConstants.STATIC_TENSOR).text)
        return InputElement(*tensor_attrs.values(),
                            is_static_tensor=is_static_tensor)

    def __create_output(self, output_elem: ET.Element) -> OutputElement:
        '''
         Create an OutputElement python object translated from an Output XML element
        :param output_elem: The XML Output element to parse
        :return: A fully formed OutputElement
        :raise: ValueError if the XML contains a default value
        '''
        tensor_attrs = self.__get__activation_tensor_attr(output_elem, XmlConstants.OUTPUT)
        default = tensor_attrs.pop(XmlConstants.DEFAULT)
        if not default == "":
            raise ValueError("Output element {} has Default value {}. Outputs cannot have Defaults.".format(tensor_attrs[XmlConstants.NAME], default))

        return OutputElement(*tensor_attrs.values())

    def __create_param(self, param_elem: ET.Element) -> Union[EnumElement, BoolElement, ScalarElement, TensorParam]:
        '''
        Create a OpDefElement object representing a parameter, translated from Parameter XML element
        :param param_elem: The XML Parameter element to parse
        :return: An OpDefElement, one of: EnumElement, BoolElement, ScalarElement, or TensorParam
        '''
        tensor_attrs = self.__get_tensor_attr(param_elem, XmlConstants.PARAM)
        enums = []
        if param_elem.find(XmlConstants.ENUM) is not None:
            for enum in param_elem.find(XmlConstants.ENUM).findall(XmlConstants.ENUM_VAL):
                enums.append(enum.text)

        default = tensor_attrs[XmlConstants.DEFAULT]
        dtypes = tensor_attrs[XmlConstants.DATATYPE]
        if tensor_attrs[XmlConstants.RANK] == 98:
            if len(enums) > 0:
                if default == "":
                    default = 0
                return EnumElement(name=tensor_attrs[XmlConstants.NAME], description=tensor_attrs[XmlConstants.DESC],\
                                   mandatory=tensor_attrs[XmlConstants.MANDATORY], enum_vals=enums, \
                                   default=default, datatype=dtypes, constraints=tensor_attrs[XmlConstants.CONSTRAINT])
            elif QnnDatatype.QNN_DATATYPE_BOOL_8 in dtypes:
                if default == "":
                    default = False
                return BoolElement(name=tensor_attrs[XmlConstants.NAME], description=tensor_attrs[XmlConstants.DESC], \
                                   mandatory=tensor_attrs[XmlConstants.MANDATORY], default=default, datatype=dtypes, \
                                   constraints=tensor_attrs[XmlConstants.CONSTRAINT])
            else:
                return ScalarElement(name=tensor_attrs[XmlConstants.NAME], description=tensor_attrs[XmlConstants.DESC], \
                                     mandatory=tensor_attrs[XmlConstants.MANDATORY], default=default,
                                     datatype=dtypes, constraints=tensor_attrs[XmlConstants.CONSTRAINT])

        return TensorParam(*tensor_attrs.values())

    def __get_layout(self, layout_text: str) -> Layout:
        '''
        Get layout enum from layout_text string
        '''
        try:
            return Layout[layout_text]
        except KeyError:
            return Layout.UNDEFINED

    def __get_rank_from_xml(self, rank_text: str) -> int:
        '''
        Get integer rank from XML enumeration
        '''
        if rank_text == XmlConstants.RANK_0D:
            return 0
        elif rank_text == XmlConstants.RANK_1D:
            return 1
        elif rank_text == XmlConstants.RANK_2D:
            return 2
        elif rank_text == XmlConstants.RANK_3D:
            return 3
        elif rank_text == XmlConstants.RANK_4D:
            return 4
        elif rank_text == XmlConstants.RANK_5D:
            return 5
        elif rank_text == XmlConstants.RANK_SCALAR:
            return 98
        elif rank_text == XmlConstants.RANK_ND:
            return 99
        else:
            raise ValueError("Rank {} is invalid.".format(rank_text))

    def __rank_to_xml(self, rank: int) -> str:
        '''
        Go from integer rank representation to XML enumeration
        '''
        if rank == 0:
            return XmlConstants.RANK_0D
        elif rank == 1:
            return XmlConstants.RANK_1D
        elif rank == 2:
            return XmlConstants.RANK_2D
        elif rank == 3:
            return XmlConstants.RANK_3D
        elif rank == 4:
            return XmlConstants.RANK_4D
        elif rank == 5:
            return XmlConstants.RANK_5D
        elif rank == 98:
            return XmlConstants.RANK_SCALAR
        elif rank == 99:
            return XmlConstants.RANK_ND
        else:
            raise ValueError("Rank {} is invalid.".format(rank))

    def __get_desc_string(self, desc_elem: ET.Element) -> str:
        '''
        Parse a Description XML element to produce a string
        :param desc_elem: The XML Description element to parse
        :return: a string representing the content and code of the description
        '''
        desc = ""
        if desc_elem is not None:
            content_set = desc_elem.findall(XmlConstants.DESC_CONTENT)
            code_set = desc_elem.findall(XmlConstants.DESC_CODE)
            num_code_blocks = len(code_set)
            num_content_blocks = len(content_set)
            idx = 0
            for content in content_set:
                if content.text is not None:
                    desc = desc + self.__split_paragraph(content.text)
                if idx < num_code_blocks:
                    desc = desc + "\n\n.. code-block:: c++\n\n"  # Specific token necessary for Doc generation in RST -> Sphinx
                    desc = desc + "   " + (self.__split_paragraph_indent(code_set[idx].text) \
                        if code_set[idx].text is not None else "") + "\n\n"
                    idx = idx + 1
        return desc

    def __invert_desc_string(self, desc: str) -> Tuple[List[str], List[str]]:
        '''
        Parse a description string, and break it into its respective Content and Code Elements to
        produce a Desciption XML element.
        :param desc: The input description string
        :return: A tuple containing the content and code parsed from desc string as lists
        '''
        content = []
        code = []
        in_code_block = False
        first_new_lines = True
        for line in desc.split("\n"):
            is_new_line = True if len(line.split()) == 0 else False
            if ".. code-block:: c++" in line:
                in_code_block = True
                code.append("")
                continue
            if in_code_block:
                if is_new_line:
                    if not first_new_lines:
                        in_code_block = False
                        first_new_lines = True
                        continue
                    else:
                        continue
                else:
                    first_new_lines = False
                    code[-1] += line.strip()+"\n"
            else:
                if len(code) >= len(content):
                    content.append("")
                content[-1] += line.strip()+"\n"

        return content, code

    def __get_desc_xml(self, desc_xml: ET.Element, desc_string: str):
        '''
        Add the description content and code to a Description XML element
        :param desc_xml: The root XML element to add Content and Code to
        :param desc_string: The string to parse
        :return: None
        '''
        content, code = self.__invert_desc_string(desc_string)
        for content_chunk, code_chunk in zip(content, code):
            desc_content = ET.SubElement(desc_xml, XmlConstants.DESC_CONTENT)
            desc_content.text = content_chunk.rstrip("\n")

            desc_code = ET.SubElement(desc_xml, XmlConstants.DESC_CODE)
            desc_code.text = code_chunk.rstrip("\n")

        if (len(content) > len(code)) and (content[-1].strip() != ""):
            desc_content = ET.SubElement(desc_xml, XmlConstants.DESC_CONTENT)
            desc_content.text = content[-1].rstrip("\n")

    def __split_paragraph(self, text: str) -> str:
        '''
        Split a paragraph, properly maintaining the formatting of bulleted lists and math blocks
        in .rst
        '''
        formatted_text = []
        in_list = False
        in_math_block = False
        first_new_line = True
        for line in text.split("\n"):
            bulleted_line = False
            math_line = False
            line_content = line.split()
            # Indicates new line
            if len(line_content) == 0:
                in_list = False
                if in_math_block and not first_new_line:
                    in_math_block = False
                    first_new_line = True
                elif in_math_block:
                    first_new_line = False
            elif line_content[0] == "*":
                in_list = True
                bulleted_line = True
            elif " ".join(line_content) == ".. math::":
                in_math_block = True
                math_line = True

            if in_list and not bulleted_line:
                line_content.insert(0, " ")
            elif in_math_block and not math_line:
                line_content.insert(0, "  ")

            formatted_text.append(" ".join(line_content))
        return "\n".join(formatted_text)

    def __split_paragraph_indent(self, text: str) -> str:
        '''
        Split a paragraph and join it using an indent
        '''
        return "\n   ".join(" ".join(line.split()) for line in text.split("\n"))

    def __text_to_bool(self, text: str) -> bool:
        '''
        Turn XML text for true/false to python bool primitive
        '''
        return True if text == "true" else False

    def __get_ref_string(self, ref_elem: ET.Element) -> str:
        '''
        Add markup to a reference string, to compose a clickable .rst link
        '''
        ref_string = ref_elem.get(XmlConstants.REF_SOURCE)
        if ref_elem.get(XmlConstants.REF_URL) is not None:
            ref_string = ref_string + ": " + "`" + ref_elem.get(XmlConstants.REF_URL) + "`_"
        return ref_string

    def __invert_ref_string(self, reference: str) -> Tuple[str, str]:
        '''
        Remove markup from a reference string
        '''
        colon_idx = reference.index(":")
        ref = reference[0:colon_idx].strip()
        url = reference[(colon_idx+1):-1]
        url = url.replace("`", "")
        url = url.strip()
        return ref, url

    def __flatten_paragraph(self, paragraph: str) -> str:
        '''
        Flatten paragraph into single line
        '''
        return " ".join(paragraph.split())

    def __translate_xml_constraints(self, constraint_elem: List[ET.Element], constraints: list):
        '''
        Translate XML constraint element to a list of constraints
        :param constraint_elem: set of XML Constraint elements
        :param constraints: the list of constraints to be appened
        '''
        if constraint_elem is not None:
            for constraint in constraint_elem:
                id_text = constraint.get(XmlConstants.CONSTRAINT_ID)
                constraint_code = constraint.find(XmlConstants.CONSTRAINT_CODE)
                if constraint_code is None:
                    constraint_code = ""
                else:
                    constraint_code = constraint_code.text
                const_type_text = constraint.get(XmlConstants.CONSTRAINT_TYPE)
                const_type = ConstraintType[const_type_text.upper()]


                constraint_text = self.__flatten_paragraph(constraint.text)
                if constraint_text is None:
                    constraint_text = ""


                constraints.append(Constraint(int(id_text), constraint_text, const_type, constraint_code))

    def __translate_constraints_to_xml(self, xml_elem: ET.Element, op_def_attr: OpDefElement):
        '''
        Translates a list of constraints to XML constraints that are appeneded to a provieded root
        element.
        :param xml_elem: The root XML element
        :param op_def_attr: The attribute from which constraints will be pulled
        '''
        for c in op_def_attr.constraints:
            constraint = ET.SubElement(xml_elem, XmlConstants.CONSTRAINT)
            constraint.set(XmlConstants.CONSTRAINT_ID, str(c.id))
            constraint.set(XmlConstants.CONSTRAINT_TYPE, c.type.name.lower().capitalize())
            constraint.text = c.constraint

    def __translate_xml_quant_params(self, quant_param_elem: List[ET.Element], quant_params: list):
        '''
        Translate XML quant_param element to a list of quant_params
        :param quant_param_elem: set of XML QuantParam elements
        :param quant_params: the list of quant_params to be appened
        '''
        if quant_param_elem is not None:
            for param in quant_param_elem:
                encodings = []
                encoding_elems = param.findall(XmlConstants.QUANT_ENCODING)
                if encoding_elems is not None:
                    for encoding in encoding_elems:
                        encodings.append(EncodingType[encoding.text])
                #Axis
                axis = None
                axis_elem = param.find(XmlConstants.QUANT_AXIS)
                if axis_elem is not None:
                    axis = int(axis_elem.text)

                #Symmetric
                symmetric = None
                symmetric_elem = param.find(XmlConstants.QUANT_SYMMETRIC)
                if symmetric_elem is not None:
                    symmetric = self.__text_to_bool(symmetric_elem.text)

                # Scale/Offset
                scale = None
                offset = None
                scale_elem = param.find(XmlConstants.QUANT_SCALE)
                offset_elem = param.find(XmlConstants.QUANT_OFFSET)
                if scale_elem is not None:
                    if offset_elem is None:
                        raise TypeError("Scale provided with no Offset")
                    scale = float(scale_elem.text)
                    offset = float(offset_elem.text)
                elif offset_elem is not None:
                    raise TypeError("Offset provided with no Scale")

                # Min/Max
                min = None
                max = None
                min_elem = param.find(XmlConstants.QUANT_MIN)
                max_elem = param.find(XmlConstants.QUANT_MAX)
                if min_elem is not None:
                    if max_elem is None:
                        raise TypeError("Min provided with no Max")
                    min = int(min_elem.text)
                    max = int(max_elem.text)
                elif max_elem is not None:
                    raise TypeError("Max provided with no Min")

                # IsMathInvariant
                math_invariant = None
                math_invariant_elem = param.find(XmlConstants.QUANT_MATH_INVARIANT)
                if math_invariant_elem is not None:
                    math_invariant = self.__text_to_bool(symmetric_elem.text)
                quant_params.append(QuantParam(encodings, axis, symmetric, scale, offset, min, max, math_invariant))

    def __translate_quant_params_to_xml(self, xml_elem: ET.Element, supp_op_def_attr: OpDefElement):
        '''
        Translates a list of quant_params to XML Quant_Params that are appended to a provided root
        element.
        :param xml_elem: The root XML element
        :param supp_op_def_attr: The attribute from which quant_params will be pulled
        '''
        for param in supp_op_def_attr.quant_params:
            #Encoding
            quant_param = ET.SubElement(xml_elem, XmlConstants.QUANT_PARAM)
            quant_param.set(XmlConstants.QUANT_ENCODING, param.encoding.name.lower().capatilize())

            #Axis
            if param.axis != "":
                quant_param.set(XmlConstants.QUANT_AXIS, str(param.axis))

            #Symmetric
            if param.symmetric != "":
                quant_param.set(XmlConstants.QUANT_SYMMETRIC, str(param.symmetric.lower()))

            #Scale/Offset
            if param.scale != "":
                if param.offset == "":
                    raise TypeError("Scale provided in QuantParam with no Offset")
                quant_param.set(XmlConstants.QUANT_SCALE, str(param.scale))
                quant_param.set(XmlConstants.QUANT_OFFSET, str(param.offset))
            elif param.offset != "":
                raise TypeError("Offset provided in QuantParam with no Scale")

            #Min/Max
            if param.min != "":
                if param.max == "":
                    raise TypeError("Min provided in QuantParam with no Max")
                quant_param.set(XmlConstants.QUANT_MIN, str(param.min))
                quant_param.set(XmlConstants.QUANT_MAX, str(param.max))
            elif param.max != "":
                raise TypeError("Max provided in QuantParam with no Min")

            #MathInvariant
            if param.math_invariant != "":
                quant_param.set(XmlConstants.QUANT_MATH_INVARIANT, str(param.math_invariant.lower()))

    def __validate_qnn_op_def_datatypes(self, op_def, datatypes):
        '''
        Validates if the QnnOpDef has Native Datatypes
        :param op_def: The OpDef XML element
        :param datatypes: Datatypes of all the elements in the OpDef XML element
        :raise Error if the QnnOpDef contains Native Datatypes
        '''
        # presence of supported backend element distinguishes QnnOpDef and CustomOpDef
        # CustomOpDef contains supported backend element whereas QnnOpDef doesn't
        is_qnn_op_def = op_def.findall(XmlConstants.SUPPORTED_BE) == None
        if is_qnn_op_def:
            for dtype in datatypes:
                if dtype != 'BACKEND_SPECIFIC' and dtype in NativeDatatype._member_names_:
                    raise TypeError("Incorrect Datatype {} specified. "
                                    "QnnOpDef does not support Native Datatypes.".format(dtype))
