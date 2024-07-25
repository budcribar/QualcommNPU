# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================

import inspect
from typing import (Optional, Tuple, Type, Union, Callable, List,
                    get_args, get_origin, Any, TypeVar)

from qti.aisw.tools.core.modules.definitions.interface import Module
from qti.aisw.tools.core.modules.definitions.schema import ModuleSchema
from qti.aisw.tools.core.modules.definitions.common import AISWBaseModel
from qti.aisw.tools.core.modules.utils.errors import ModuleComplianceError

"""
This module contains several functions that are used to validate and verify the methods of a class.
These functions check the method signatures, filter out the public non-base methods,
and ensure the compliance of the module.
"""

ModuleSchemaType = TypeVar("ModuleSchemaType", bound=ModuleSchema)


def validate_method_signature(method: Callable, expected_param_type: Any,
                              expected_return_type: Optional[Any] = None,
                              *,
                              allow_instance_types=False,
                              ignore_keyword_params=False) -> bool:
    """
    Validates the signature of a method. Checks the type hinting for parameters and return annotations.
    Also checks if the method takes exactly one argument of the expected type and returns the expected type.

    Args:
       method (Callable): The method to be validated.
       expected_param_type: (Any): The expected type of the parameter.
       expected_return_type: (Any): The expected return type. If not provided, it is assumed to be the same as the expected parameter type.
       allow_instance_types: If allow instance types is set, then type checking allows derived types to match.
                            i.e Type[Class B] == Type[Class A] if class B inherits from class A.
       ignore_keyword_params: If set to True, then all keyword only params are ignored.


    Returns:
       bool: True if the method signature is valid, False otherwise.
    """

    def check_type_with_expected_type(type_: Any, expected_type: Any):
        if allow_instance_types:
            return expected_type in type_.__mro__
        elif type_ == expected_type:
            return True
        elif hasattr(type_, "__name__") and hasattr(expected_type, "__name__"):
            return type_.__name__ == expected_type.__name__

        return False

    def check_union_types_in_expected_type(types_: List[Any], expected_type: Any):
        if allow_instance_types:
            return all((expected_type in type_.__mro__ for type_ in types_))
        elif expected_type in types_:
            return True
        elif hasattr(expected_type, "__name__") and all(hasattr(type_, "__name__") for type_ in types_):
            return expected_type.__name__ in (type_.__name__ for type_ in types_)

        return False

    try:
        sig = inspect.signature(method)
        params_validated = True
        return_args_validated = True

        # check type hints exist
        check_signature_type_hints(sig)

        params = []
        for param in sig.parameters.values():
            if param.name in ["self", "cls"] or (
                    ignore_keyword_params and param.kind in (inspect.Parameter.KEYWORD_ONLY,
                                                             inspect.Parameter.VAR_KEYWORD)):
                continue
            params.append(param)

        expected_return_type = expected_param_type if not expected_return_type else expected_return_type

        # Check if the method takes exactly one required argument other than self or cls
        if len(params) != 1:
            return False

        # Check if the param type matches the provided schema_type
        param_type = params[0].annotation

        if get_origin(param_type) == Union:
            param_arg_types = get_args(params[0].annotation)
            params_validated = check_union_types_in_expected_type(param_arg_types, expected_param_type)
        else:
            params_validated = check_type_with_expected_type(param_type, expected_param_type)

        # check return types
        return_type = sig.return_annotation
        if get_origin(return_type) == Union:
            return_arg_types = get_args(return_type)
            return_args_validated = check_union_types_in_expected_type(return_arg_types, expected_return_type)
        else:
            return_args_validated = check_type_with_expected_type(return_type, expected_return_type)

        return params_validated and return_args_validated
    except (TypeError, ValueError) as e:
        print(e)
        return False


def check_signature_type_hints(signature: inspect.Signature):
    """
    Checks for type hints of parameters and return arguments in a signature object
    i.e. Compliance Rule 3
    Args:
        signature: A signature object
    Raises:
        TypeError if type hints are missing for parameters or return type
    """

    params = list(filter(lambda param: param.name not in ["self", "cls"], signature.parameters.values()))

    # check that type hinting is present for params
    # TODO: Use mypy for this instead
    for param in params:
        if param == param.empty:
            raise TypeError(f'Parameter with name:{param.name} is missing a type field for method: {str(signature)}')

    # check that type hinting is present for return annotations
    # TODO: Use mypy for this instead
    if signature.return_annotation == inspect.Signature.empty:
        raise TypeError(f'Return type annotation for method: {str(signature)} is missing')


def expect_module_compliance(cls: Type[Module]) -> Type:
    """
    Ensures the compliance of a module. Checks that all public, non-base class methods consume and
    return a valid AISWBaseModel.
    Also checks that at least one method consumes and returns the Module Schema or consumes
    a ModuleSchema.arguments field and returns a ModuleSchema.outputs field.

    Args:
        cls (Type): The class to be checked.

    Returns:
        Type: The class if it is compliant.

    Raises:
        TypeError if the class is not of type Module
        ModuleComplianceError if the Module class is not compliant
    """

    if not issubclass(cls, Module):
        raise TypeError(f'Class with name:{cls!r} is not of a valid class of type: {Module!r}')

    # Compliance Rule 0: filter any base class methods from Module and any other methods
    # explicitly marked internal
    public_non_base_class_methods = filter_public_non_base_methods(cls, base_class_type=Module)

    # filter "void" methods -> no param and return args or one param arg == self | cls and no return
    public_non_base_class_methods = filter_methods_without_args_and_return(public_non_base_class_methods)

    # check that module functions are all well typed
    # TODO: Should be using mypy

    # Compliance Rule 1: Check that all public, non-base class methods consume and return a valid AISWBaseModel
    for method in public_non_base_class_methods:
        verify_method_param_return_is_basemodel(method)

    # The method below checks that one of two rules is satisfied:
    # Compliance Rule 2.a: At least one method must consume and return the Module Schema
    #                                   OR
    # Compliance Rule 2.b: At least one method must consume a ModuleSchema.arguments field and return a
    # ModuleSchema.outputs field
    verify_schema_compliant_function_exists(cls, public_non_base_class_methods)

    # Store method names in the '_schema_compliant_methods' attribute
    setattr(cls, '_schema_compliant_methods', [method for method in public_non_base_class_methods])

    return cls


def filter_public_non_base_methods(cls, base_class_type: Type = type,
                                   base_class_methods: Optional[List[Tuple[str, Callable]]] = None):
    """
    Filters out the public non-base methods of a class.

    Args:
        cls: The class to be filtered.
        base_class_type: The base class whose methods should be ignored
        base_class_methods: Specific base class methods to filter, if set none then all methods are filtered out.

    Returns:
        list: A list of public non-base methods of the class.
    """

    # Get all methods of the class
    methods = inspect.getmembers(cls, predicate=lambda x: inspect.ismethod(x)
                                                          or inspect.isfunction(x)
                                                          or inspect.isgeneratorfunction(x))

    if base_class_methods is None:
        base_class_methods = inspect.getmembers(base_class_type, predicate=lambda x: inspect.ismethod(x)
                                                                                     or inspect.isfunction(x)
                                                                                     or inspect.isgeneratorfunction(x))

    base_class_method_names = list(map(lambda base_method: base_method[0], base_class_methods))

    # public non-base class methods
    public_non_base_class_methods = []

    for method_name, method in methods:
        # Filter out internal methods (those with leading underscores)
        # Filter out base class methods

        if not (method_name.startswith("_") and method_name != "_") and method_name not in base_class_method_names:
            public_non_base_class_methods.append(method)
    return public_non_base_class_methods


def filter_methods_without_args_and_return(methods: List[Callable]) -> List[Callable]:
    """
   Given a class, returns a list of methods that have at least one parameter and at least one argument,
   excluding methods with only one argument named "self" or "cls".
   """
    result = []
    for method in methods:
        signature = inspect.signature(method)
        parameters = signature.parameters
        if parameters:  # Check if the method has at least one parameter
            if signature.return_annotation != inspect.Signature.empty:  # check if there is a return argument
                # Exclude methods with only one argument named "self" or "cls"
                arg_names = [param.name for param in parameters.values()]
                if len(arg_names) > 1 or (len(arg_names) == 1 and arg_names[0] not in {"self", "cls"}):
                    result.append(method)
    return result


def verify_method_param_return_is_basemodel(method: Callable) -> bool:
    """
     Checks if the method consumes and returns a valid AISWBaseModel.

    Args:
        method (Callable): A tuple containing the method name and the method itself.

    Returns:
        bool: True if the method is valid, False otherwise.

    Raises:
        ModuleComplianceError: If the method does not consume and return a valid dataclass.
    """

    if not validate_method_signature(method, expected_param_type=AISWBaseModel, expected_return_type=AISWBaseModel,
                                     allow_instance_types=True,
                                     ignore_keyword_params=True):
        raise ModuleComplianceError(f'Public Method with name: {method.__qualname__} does not consume '
                                    f'and return a valid dataclass')
    return True


def verify_schema_compliant_function_exists(cls: Type[Module], methods: List[Callable]):
    """
    Verifies that at least one method consumes and returns the Module Schema or consumes a
    ModuleSchema.arguments field and returns a ModuleSchema.outputs field.

    Args:
       cls: The module class to be checked
       methods (List[Callable]): A list of methods to be checked.

    Raises:
       ModuleComplianceError: If no valid method exists.
    """

    # Compliance Rule 2.a: At least one method must consume and return the Module Schema
    valid_in_out_module_schema_methods = _get_valid_in_out_module_schema_methods(cls, methods)

    # OR Compliance Rule 2.b: At least one method must consume a ModuleSchema.arguments field and return a
    # ModuleSchema.outputs field
    valid_in_arg_out_arg_schema_methods = _get_valid_in_arg_out_arg_schema_methods(cls, methods)

    if not (valid_in_out_module_schema_methods or valid_in_arg_out_arg_schema_methods):
        schema = cls.get_schema()
        raise ModuleComplianceError(
            f'At least one method must have a single parameter of type: {schema!r} or '
            f'{schema.get_field_type("arguments")!r} \n and return exactly '
            f'one instance of type: {schema!r} or {schema.get_field_type("outputs")!r}')


def _get_valid_in_arg_out_arg_schema_methods(cls: Type[Module], methods: List[Callable]) -> List[Callable]:
    """
    Gets the methods that consume a ModuleSchema.arguments field and return a ModuleSchema.outputs field.

    Args:
      methods (List[Callable]): A list of methods to be checked.

    Returns:
      list: A list of valid methods.
    """
    schema = cls.get_schema()
    valid_in_arg_out_arg_schema_methods = [method for method in methods if
                                           validate_method_signature(method,
                                                                     expected_param_type=schema.get_field_type(
                                                                         "arguments"),
                                                                     expected_return_type=schema.get_field_type(
                                                                         "outputs"))]
    return valid_in_arg_out_arg_schema_methods


def _get_valid_in_out_module_schema_methods(cls: Type[Module], methods: List[Callable]) -> List[Callable]:
    """
    Gets the methods that consume and return the Module Schema.

    Args:
        methods (List[Callable]): A list of methods to be checked.

    Returns:
        list: A list of valid methods.
    """

    # Validate method signatures for at least one callable function
    def valid_in_out_module_schema_methods(schema_type_):
        return [method for method in methods if validate_method_signature(method, expected_param_type=schema_type_,
                                                                          expected_return_type=schema_type_,
                                                                          ignore_keyword_params=True)]

    if len(cls.get_schemas()) == 1:
        schema_type = cls.get_schema()
        return valid_in_out_module_schema_methods(schema_type)
    else:
        types_ = cls.get_schemas()
        unique_schema_methods = set()
        for type_ in types_:
            unique_schema_methods.update(valid_in_out_module_schema_methods(type_))
        return list(unique_schema_methods)
