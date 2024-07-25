# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================

from typing import Union, Optional, Dict, List, Any, Type, TypedDict, ClassVar, Mapping
from typing_extensions import Unpack
from abc import abstractmethod, ABC

from qti.aisw.tools.core.modules.definitions.schema import ModuleSchema, ModuleSchemaVersion

"""
This module contains a class `Module` that serves as an abstract base class for other modules. While modules
may be freely extended, it is expected that module implementations adhere to the interface. Additionally, modules should
follow compliance rules which can be found in directory <compliance>.
"""


class Module(ABC):
    """
    Abstract base class for modules. This class provides a structure that other concrete Modules should inherit from.
    It manages a schema (ModuleSchema) which describes its properties. It may also control a logger object.
    """
    _SCHEMA: ClassVar[Optional[ModuleSchema]] = None
    _LOGGER: ClassVar[Any] = None

    def __init__(self, logger=None):
        """
       Initializes a new instance of the `Module` class.

       Args:
           logger: The logger to be used by the module. If not provided, logging will be determined by the module.
       """

        self._logger = logger if logger else self._LOGGER

    @abstractmethod
    def properties(self) -> Mapping[str, Any]:
        """
        This function should describe some basic properties of this module. Ideally, it should be serializable
        and query-able. Suggested types are JSON and YAML. YAML should be used for properties that can be freely edited,
        while JSON should be used for generated properties that should remain fixed.

        Returns:
            A dictionary-like object that describes the properties of this module. The default should be pydantic's
            model_json_schema.

        """

    @classmethod
    @abstractmethod
    def get_schema(cls, version: Optional[Union[str, ModuleSchemaVersion]] = None) -> Type[ModuleSchema]:
        """
        This functions returns the module schema that matches the version specified.

        Args:
            version: A string that resolves to a valid ModuleSchemaVersion or a valid ModuleSchemaVersion or None.
        Returns:
            The module schema that matches the specified version, or the latest schema if version is None
        Raises:
            May raise a SchemaVersionError if the schema is not found
            May raise a TypeError if the version passed is not a str or ModuleSchemaVersion
        """

    @classmethod
    @abstractmethod
    def get_schemas(cls) -> List[Type[ModuleSchema]]:
        """
        This functions returns all the Module Schemas associated with this module.
        Note the return type is the Module Schema class, not an instance of a Module Schema.

        Returns:
            A list of types for all the module's schemas
        """

    @abstractmethod
    def get_logger(self) -> Any:
        """
        This should return an instance of the logger that is used. Ideally the type hint should reflect
        the actual logger type.


        Returns:
            The logger used by the module.
        """

    @abstractmethod
    def enable_debug(self, debug_level: int, **kwargs: Unpack[TypedDict]) -> Optional[bool]:
        """
         Abstract method that should be implemented by any concrete class that inherits from `Module`.
         This method should enable debugging behavior for the module.

         Args:
             debug_level (int): The level of debugging to be enabled.
             **kwargs: Arbitrary keyword arguments.

         Returns:
             True if debugging is enabled, False otherwise. If no debugging is possible at all,
             then this function may return None.
         """
