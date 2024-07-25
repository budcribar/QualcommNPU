# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================

from typing import Optional, List, Type, Union, Any, ClassVar
from typing_extensions import get_args, get_origin
from pydantic import Field, FilePath, ValidationInfo, field_validator
from .common import AISWBaseModel, AISWVersion
from qti.aisw.tools.core.modules.utils.errors import *

"""
This module contains two classes: `ModuleSchemaVersion` and `ModuleSchema`.
`ModuleSchemaVersion` is used to represent and manipulate the version of a module schema.
`ModuleSchema` is a data class that defines the properties of a module.
"""


class ModuleSchemaVersion(AISWVersion):
    """
   This class represents a version of a module schema. It inherits from the `AISWVersion` class.
   """

    @staticmethod
    def get_module_schema_version_from_str(version_str: str):
        """
       This static method takes a version string and returns a `ModuleSchemaVersion` object.
       E.x. "1.3.0" -> ModuleSchemaVersion(major = 1, minor = 3, patch = 0)

       Args:
           version_str (str): The version string to be converted into a `ModuleSchemaVersion` object.

       Returns:
           ModuleSchemaVersion: A `ModuleSchemaVersion` object representing the version.

       Raises:
           SchemaFieldTypeError: If the input is not a string.
           SchemaFieldValueError: If the version string contains more than three dot separated elements.
       """
        if not isinstance(version_str, str):
            raise SchemaFieldTypeError("Schema version can only be created from str")

        version_str_split = version_str.split(".")

        if len(version_str_split) != 3:
            raise SchemaFieldValueError(f'Unknown value for version: {version_str!r}')

        major, minor, patch_plus_exp_tag = version_str_split

        patch_plus_exp_tag_split = patch_plus_exp_tag.split("-")

        if len(patch_plus_exp_tag_split) > 1:
            patch = patch_plus_exp_tag_split.pop(0)
            exp_tag = "".join(patch_plus_exp_tag_split)
            return ModuleSchemaVersion(major=major, minor=minor,
                                       patch=patch,
                                       pre_release=exp_tag)
        else:
            patch = patch_plus_exp_tag_split[0]
            return ModuleSchemaVersion(major=major, minor=minor,
                                       patch=patch)


class ModuleSchema(AISWBaseModel):
    """
    A pydantic dataclass that defines the properties of a module, and can therefore be used as an argument
    to its functions. It should be associated with a module as a class type, such that a module can provide its
    schema when queried.

    See ./interface.py for the module definition and examples/example_module.py for an example
    schema implementation.
    """

    _BASE_VERSION: ClassVar[ModuleSchemaVersion] = ModuleSchemaVersion(major=0,
                                                                       minor=1,
                                                                       patch=0)  # interface version
    _VERSION: ClassVar[ModuleSchemaVersion] = _BASE_VERSION  # derived class version

    """This class field marks the name of the module"""
    name: str
    """This class field defines a path to the module implementation (could be different from schema)"""
    path: FilePath
    """Any set of arguments required by the module"""
    arguments: Optional[AISWBaseModel] = None
    """ The outputs produced by the module """
    outputs: Optional[AISWBaseModel] = None
    """ The list of backends associated with this module if applicable e.g CPU, GPU """
    backends: Optional[List[str]] = None
    """ This class field captures the version as it relates to changes in the module schema fields."""
    version: ModuleSchemaVersion = Field(_VERSION, frozen=True)
    """ A list of serialized artifacts such as output models, jsons and other in-memory objects. Note these
        artifacts should be wrapped in a pydantic object."""
    serialized_artifacts: Optional[List[AISWBaseModel]] = None

    # TODO: Change to annotation validator with 3.10
    @classmethod
    def check_version_str(cls, version_str: str):
        """
        Checks if the version str matches the default version for this schema.

        Returns:
            True if the version matches, false otherwise

        Raises:
            Errors from get_module_schema_version_from_str if version_str cannot be
            resolved into a valid ModuleSchemaVersion object
        """
        version = ModuleSchemaVersion.get_module_schema_version_from_str(version_str)
        return cls._VERSION == version

    # Note this is a strict version check against the default or fixed version that should
    # be used with a field validator decorator.
    @classmethod
    def _check_version_with_default(cls, v: ModuleSchemaVersion, info: ValidationInfo):
        """

        This function checks the field value against the default version. It's intended
        for cases when the version field should remain fixed.

        Args:
            cls: The module schema class
            v: The version to be checked
            info: The validation info containing field information

        Returns:
            The schema version

        Raises:
           SchemaVersionError if the default value does not match the provided value
        """
        try:
            cls._check_for_non_default_value(v, cls._VERSION, info)
        except SchemaFieldValueError as e:
            raise SchemaVersionError(f'Only version: {cls._VERSION!r} is supported '
                                     f'for this schema') from e
        return v

    @staticmethod
    def _check_for_non_default_value(value: Any, default_value: Any, info: ValidationInfo) -> Any:
        """
        Internal private method that checks if a value matches a default value

        Args:
            value: The value to be checked
            default_value: A known default value
            info: The validation info for the corresponding field

        Returns:
             The value

        Raises:
           SchemaFieldValueError if the default value does not match the provided value

        """
        if default_value is None and value is not None:
            raise SchemaFieldValueError('Field value cannot be set to value other than None: {info.field_name}')
        elif value != default_value:
            raise SchemaFieldValueError('field_value_is_constant_error',
                                      f'Non default value is not allowed for field: {info.field_name}')
        return value

    @classmethod
    def get_version(cls):
        """
        Returns the class version for this schema. Note this is a class method and therefore may not match
        <instance>.version.

        Returns:
            The default version for this schema
        """
        return cls._VERSION

    @classmethod
    def get_field_type(cls, key: str) -> Type:
        """
        This class method retrieves the type annotation for a given key.

        Args:
          key (str): The key for which to retrieve the type annotation.

        Returns:
          Type: The type annotation for the given key.

        Raises:
          TypeError: If the key does not have a type annotation.
        """
        try:
            field_type = cls.model_fields.get(key).annotation
        except AttributeError as e:
            raise SchemaFieldTypeError(f'Cannot retrieve type annotation for {key}') from e

        if get_origin(field_type) == Union:
            field_type = cls._get_field_type_optional(field_type)[0]

        return field_type

    @classmethod
    def get_field_type_name(cls, key: str) -> str:
        """ Returns the name of a pydantic type given a field name"""
        return cls.get_field_type(key).__name__

    @classmethod
    def _get_field_type_optional(cls, field_type: type):
        """
        This class method retrieves the types from a Union type, excluding NoneType.

        Args:
          field_type (type): The Union type to be processed.

        Returns:
          list: A list of types included in the Union type, excluding NoneType.

        Raises:
          TypeError: If the provided field type is not a Union.
        """
        if get_origin(field_type) == Union:
            field_union_types = get_args(field_type)
        else:
            raise SchemaFieldTypeError(f'Provided field type is not a Union:{field_type!r}')

        field_union_types_no_none_types = list(
            filter(lambda union_type: union_type != type(None), field_union_types))

        return field_union_types_no_none_types

    # Note this is a strict version check, override this method if version changes
    # are desirable
    @field_validator('version')
    @classmethod
    def reject_non_default_value_version(cls, v: Any, info: ValidationInfo):
        """
        Checks if the version is equal to the default version for this schema.

        Args:
            cls: The module schema class
            v: The version to be checked
            info: The validation info containing field information

        Returns:
            The schema version

       Raises:
           SchemaVersionError if the version: v does not match the default version
           for this schema
        """
        return cls._check_version_with_default(v, info)
