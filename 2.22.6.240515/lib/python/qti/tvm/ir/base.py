# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Common base structures."""
import tvm._ffi
import tvm.error
from tvm._ffi import get_global_func, register_object
from tvm.runtime import Object, _ffi_node_api

from . import _ffi_api, json_compact


class Node(Object):
    """Base class of all IR Nodes."""


@register_object("SourceMap")
class SourceMap(Object):
    def add(self, name, content):
        return get_global_func("SourceMapAdd")(self, name, content)


@register_object("SourceName")
class SourceName(Object):
    """A identifier for a source location.

    Parameters
    ----------
    name : str
        The name of the source.
    """

    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_api.SourceName, name)  # type: ignore # pylint: disable=no-member


### [QTI Change]
@register_object("SourceQparams")
class SourceQparams(Object):
    """Stores the source quantization information.

    Parameters
    ----------
    dtype : str
        The output buffer data type.

    scale : numpy.array(float)
        The output buffer scale.

    zero_point : numpy.array(int)
        The output buffer zero_point.
    """

    def __init__(self, dtype, scale, zero_point):
        self.__init_handle_by_constructor__(
            _ffi_api.SourceQparams, dtype, scale, zero_point   # type: ignore # pylint: disable=no-member
        )

    @property
    def dtype(self):
        return str(self.dtype_)

    @property
    def scale(self):
        return self.scale_.numpy()

    @property
    def zero_point(self):
        return self.zero_point_.numpy()
### [End QTI Change]


### [QTI Change]
@register_object("Span")
class Span(Object):
    """Specifies a location in a source program.

    Parameters
    ----------
    source : SourceName
        The source name.

    lineno : int
        The line number.

    col_offset : int
        The column offset of the location.

    op_type : str
        The op type.

    output_names : list[str]
        The output buffer names.

    output_qparams : dict{str, SourceQparams}
        The output buffer quantization parameters (dtype, scales, offsets).
    """

    def __init__(self, source_name, line, end_line, column, end_column, op_type='', output_names=[], output_qparams={}):
        self.__init_handle_by_constructor__(
            _ffi_api.Span, source_name, line, end_line, column, end_column, op_type, output_names, output_qparams   # type: ignore # pylint: disable=no-member
        )

    @property
    def op_type(self):
        return self.op_type_

    @property
    def output_names(self):
        return list(self.output_names_)

    @property
    def output_qparams(self):
        return dict(self.output_qparams_)
### [End QTI Change]


### [QTI Change]
@tvm._ffi.register_object("SequentialSpan")
class SequentialSpan(Object):
    """Specifies a location in a source program.

    Parameters
    ----------
    spans : Array
        The Array of spans.
    """

    def __init__(self, spans):
        self.__init_handle_by_constructor__(_ffi_api.SequentialSpan, spans)
### [End QTI Change]


@register_object
class EnvFunc(Object):
    """Environment function.

    This is a global function object that can be serialized by its name.
    """

    def __call__(self, *args):
        return _ffi_api.EnvFuncCall(self, *args)  # type: ignore # pylint: disable=no-member

    @property
    def func(self):
        return _ffi_api.EnvFuncGetPackedFunc(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def get(name):
        """Get a static env function

        Parameters
        ----------
        name : str
            The name of the function.
        """
        return _ffi_api.EnvFuncGet(name)  # type: ignore # pylint: disable=no-member


def load_json(json_str) -> Object:
    """Load tvm object from json_str.

    Parameters
    ----------
    json_str : str
        The json string

    Returns
    -------
    node : Object
        The loaded tvm node.
    """

    try:
        return _ffi_node_api.LoadJSON(json_str)
    except tvm.error.TVMError:
        json_str = json_compact.upgrade_json(json_str)
        return _ffi_node_api.LoadJSON(json_str)


def save_json(node) -> str:
    """Save tvm object as json string.

    Parameters
    ----------
    node : Object
        A TVM object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return _ffi_node_api.SaveJSON(node)


def structural_equal(lhs, rhs, map_free_vars=False):
    """Check structural equality of lhs and rhs.

    The structural equality is recursively defined in the DAG of IRNodes.
    There are two kinds of nodes:

    - Graph node: a graph node in lhs can only be mapped as equal to
      one and only one graph node in rhs.
    - Normal node: equality is recursively defined without the restriction
      of graph nodes.

    Vars(tir::Var, TypeVar) and non-constant relay expression nodes are graph nodes.
    For example, it means that `%1 = %x + %y; %1 + %1` is not structurally equal
    to `%1 = %x + %y; %2 = %x + %y; %1 + %2` in relay.

    A var-type node(e.g. tir::Var, TypeVar) can be mapped as equal to another var
    with the same type if one of the following condition holds:

    - They appear in a same definition point(e.g. function argument).
    - They points to the same VarNode via the same_as relation.
    - They appear in a same usage point, and map_free_vars is set to be True.

    The rules for var are used to remap variables occurs in function
    arguments and let-bindings.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether free variables (i.e. variables without a definition site) should be mapped
        as equal to each other.

    Return
    ------
    result : bool
        The comparison result.

    See Also
    --------
    structural_hash
    assert_strucural_equal
    """
    lhs = tvm.runtime.convert(lhs)
    rhs = tvm.runtime.convert(rhs)
    return bool(_ffi_node_api.StructuralEqual(lhs, rhs, False, map_free_vars))  # type: ignore # pylint: disable=no-member


def get_first_structural_mismatch(lhs, rhs, map_free_vars=False):
    """Like structural_equal(), but returns the ObjectPaths of the first detected mismatch.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether free variables (i.e. variables without a definition site) should be mapped
        as equal to each other.

    Returns
    -------
    mismatch: Optional[Tuple[ObjectPath, ObjectPath]]
        `None` if `lhs` and `rhs` are structurally equal.
        Otherwise, a tuple of two ObjectPath objects that point to the first detected mismtach.
    """
    lhs = tvm.runtime.convert(lhs)
    rhs = tvm.runtime.convert(rhs)
    mismatch = _ffi_node_api.GetFirstStructuralMismatch(lhs, rhs, map_free_vars)  # type: ignore # pylint: disable=no-member
    if mismatch is None:
        return None
    else:
        return mismatch.lhs_path, mismatch.rhs_path


def assert_structural_equal(lhs, rhs, map_free_vars=False):
    """Assert lhs and rhs are structurally equal to each other.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether or not shall we map free vars that does
        not bound to any definitions as equal to each other.

    Raises
    ------
    ValueError : if assertion does not hold.

    See Also
    --------
    structural_equal
    """
    lhs = tvm.runtime.convert(lhs)
    rhs = tvm.runtime.convert(rhs)
    _ffi_node_api.StructuralEqual(lhs, rhs, True, map_free_vars)  # type: ignore # pylint: disable=no-member


def structural_hash(node, map_free_vars=False):
    """Compute structural hash of node

    The structural hash value is recursively defined in the DAG of IRNodes.
    There are two kinds of nodes:

    - Normal node: the hash value is defined by its content and type only.
    - Graph node: each graph node will be assigned a unique index ordered by the
      first occurence during the visit. The hash value of a graph node is
      combined from the hash values of its contents and the index.

    structural_hash is made to be concistent with structural_equal.
    If two nodes are structurally equal to each other,
    then their structural hash (with the same map_free_vars option)
    should be equal to each other as well.

    If the structural hash of two nodes equals to each other,
    then it is highly likely(except for rare hash value collison cases)
    that the two nodes are structurally equal to each other.

    Parameters
    ----------
    node : Object
        The input to be hashed.

    map_free_vars : bool
        If map_free_vars is set to true, we will hash free variables
        by the order of their occurrences. Otherwise, we will hash by
        their in-memory pointer address.

    Return
    ------
    result : int
        The hash result

    See Also
    --------
    structrual_equal
    """
    return _ffi_node_api.StructuralHash(node, map_free_vars)  # type: ignore # pylint: disable=no-member


def deprecated(
    method_name: str,
    new_method_name: str,
):
    """A decorator to indicate that a method is deprecated

    Parameters
    ----------
    method_name : str
        The name of the method to deprecate
    new_method_name : str
        The name of the new method to use instead
    """
    import functools  # pylint: disable=import-outside-toplevel
    import warnings  # pylint: disable=import-outside-toplevel

    def _deprecate(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            warnings.warn(
                f"{method_name} is deprecated, use {new_method_name} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return _wrapper

    return _deprecate
