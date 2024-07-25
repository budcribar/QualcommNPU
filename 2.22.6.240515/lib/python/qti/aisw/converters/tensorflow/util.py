# =============================================================================
#
#  Copyright (c) 2015-2021,2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
import logging
import os
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Dict, List, Text
from enum import Enum

import numpy as np
import tensorflow as tf
from packaging import version
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import (
    log_assert,
    log_debug1,
    log_error,
    log_warning,
)
from qti.aisw.converters.common.utils.framework_utils import (
    TensorInfo,
    determine_layout,
)
from qti.aisw.converters.tensorflow import tf_compat_v1
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.tools.optimize_for_inference_lib import ensure_graph_is_valid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

GRAPH = type(tf_compat_v1.Graph)
GRAPH_DEF = type(tf_compat_v1.GraphDef)
NODE_DEF = type(tf_compat_v1.NodeDef)

# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
DEFAULT_GENERIC_OPTIMIZATIONS = [
    "strip_unused_nodes",
    "remove_nodes(op=Identity, op=CheckNumerics)",
    "fold_constants",
    "fold_batch_norms",
    "fold_old_batch_norms",
    "sort_by_execution_order",
]

# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/rewriter_config.proto#L67
DEFAULT_GRAPPLER_OPTIMIZATION = [
    # "common_subgraph_elimination",
    # FIXME: Enabling "common_subgraph_elimination" optimization combines
    #        constants having same value. Due to this few models are failing
    #        at the sort_descriptors_in_execution_order call as the sorting
    #        logic is not written with respect to constant nodes.
    # skipping constant folding optimization from defaults as this being
    # handled externally in general tensorflow optimizations already.
    # "constfold",
    "debug_stripper",
    "dependency",
    "experimental_conditional_code_motion",
    "function",
    "implementation",
    "layout",
    "loop",
    "min_graph_nodes",
    "pruning",
    "remapping",
    "scoped_allocator",
    "shape",
    "use_plugin",
]


class ETFGrapplerOptimization(Enum):
    """
    Grappler Supported Optimizations.
    """

    CONSTFOLD = 0
    DEBUG_STRIPPER = 1
    DEPENDENCY = 2
    EXPERIMENTAL_CONDITIONAL_CODE_MOTION = 3
    FUNCTION = 4
    IMPLEMENTATION = 5
    LAYOUT = 6
    LOOP = 7
    MIN_GRAPH_NODES = 8
    PRUNING = 9
    REMAPPING = 10
    SCOPED_ALLOCATOR = 11
    SHAPE = 12
    USE_PLUGIN = 13

    @staticmethod
    def to_name(val) -> Text:
        """
        Convert ETFGrapplerOptimization to corresponding grappler
        optimization name.

        :param ETFGrapplerOptimization val: ETFGrapplerOptimization option.
        :return Text: grappler optimization in string format.
        """
        opt_mappings = {
            ETFGrapplerOptimization.CONSTFOLD: "constfold",
            ETFGrapplerOptimization.DEBUG_STRIPPER: "debug_stripper",
            ETFGrapplerOptimization.DEPENDENCY: "dependency",
            ETFGrapplerOptimization.EXPERIMENTAL_CONDITIONAL_CODE_MOTION: "experimental_conditional_code_motion",
            ETFGrapplerOptimization.FUNCTION: "function",
            ETFGrapplerOptimization.IMPLEMENTATION: "implementation",
            ETFGrapplerOptimization.LAYOUT: "layout",
            ETFGrapplerOptimization.LOOP: "loop",
            ETFGrapplerOptimization.MIN_GRAPH_NODES: "min_graph_nodes",
            ETFGrapplerOptimization.PRUNING: "pruning",
            ETFGrapplerOptimization.REMAPPING: "remapping",
            ETFGrapplerOptimization.SCOPED_ALLOCATOR: "scoped_allocator",
            ETFGrapplerOptimization.SHAPE: "shape",
            ETFGrapplerOptimization.USE_PLUGIN: "use_plugin",
        }

        if val not in opt_mappings:
            log_warning(f"{val} is invalid grappler optimization.")
            return ""

        return opt_mappings[val]

    @staticmethod
    def get_default_optimizations() -> List[Text]:
        """
        Default Grappler Optimizations.

        :return List[Text]: Returns the default grappler optimizations.
        """
        return DEFAULT_GRAPPLER_OPTIMIZATION


class GraphVisitor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_operation(self, node_def):
        """
        :type node_def: tensorflow.NodeDef
        :rtype: None
        """
        pass

    @abstractmethod
    def visit_scope(self, scope, nodes_defs):
        """
        :type scope: str
        :type nodes_defs: list[tensorflow.NodeDef]
        :return: None
        """
        pass


class VisitableGraph(object):
    def __init__(self, ops):
        """
        :type ops: list[tensorflow.Operation]
        """
        self._operations = ops

    def accept(self, visitor):
        """
        Walks the graph and calls the specified visitor for each node and scope in the graph.
        :type visitor GraphVisitor
        :rtype: None
        """
        for op in self._operations:
            visitor.visit_operation(op)

        scopes = self._scopes_for_nodes(self._operations)
        for scope, ops in list(scopes.items()):
            visitor.visit_scope(scope, ops)

    @classmethod
    def _scopes_for_nodes(cls, ops):
        """
        :type ops: list[tensorflow.Operation]
        :rtype: dict[str,tensorflow.Operation]
        """
        scope_nodes_map = OrderedDict()
        for op in ops:
            splits = op.name.split("/")
            scope = "/".join(splits[:-1]) if len(splits) > 1 else str(op.name)
            nodes = scope_nodes_map.get(op.name, [])
            if (
                len(nodes) == 0
                or len([node for node in nodes if node.type != "Const"]) == 0
            ):
                nodes = scope_nodes_map.get(scope, [])
            else:
                scope = str(op.name)
            nodes.append(op)
            scope_nodes_map[scope] = nodes
        return scope_nodes_map


class GraphPrinter(GraphVisitor, object):
    def visit_operation(self, op):
        pass

    def visit_scope(self, scope, ops):
        log_debug1(code_to_message.get_debugging_message("DEBUG_TF_SCOPE_PRINT")(scope))
        for op in ops:
            log_debug1(
                code_to_message.get_debugging_message("DEBUG_TF_OP_NAME_TYPE_PRINT")(
                    op.name, op.type
                )
            )


class GraphHelper(object):
    def __init__(self, session, model, ops):
        """
        Provides several helper methods to navigate the Tensorflow Graph.
        :type session: tensorflow.Session
        :type model: converters.tensorflow.loader.Model
        :type: ops: list[tensorflow.Operation]
        """
        self._session = session
        self._model = model
        self._graph = session.graph
        self._op_output_map = dict()
        self._tensor_shape_cache = dict()  # type: dict(str, list[int])
        self._tensor_value_cache = dict()  # type: dict(str, np.ndarray)
        self._placeholders_stubs_map = dict()
        if self._model is not None:
            input_names = [graph_input.name for graph_input in self._model.inputs]
            self._placeholders_stubs_map = self._create_placeholders_tensors(
                session, input_names
            )
        self._op_output_map = self._map_operations_outputs(ops)
        self._evaluate_tensor_shapes(ops)

    @classmethod
    def _create_placeholders_tensors(cls, session, inputs):
        placeholders_stubs_map = dict()
        # run in isolated session so that the memory gets cleared out after retrieving the output
        with tf_compat_v1.Session(graph=session.graph) as sess:
            for op in sess.graph.get_operations():
                if op.type == "Placeholder" and op.name not in inputs:
                    dtype = np.float32
                    if op.get_attr("dtype") == tf.uint8:
                        dtype = np.uint8

                    tensor = sess.graph.get_tensor_by_name(
                        GraphHelper.indexed_tensor_name(op.name)
                    )
                    shape = tensor.get_shape().as_list()
                    shape = [d if d is not None else 1 for d in shape]
                    tensor = np.zeros(shape, dtype=dtype)
                    placeholders_stubs_map[str(op.name)] = tensor
            return placeholders_stubs_map

    def dump(self):
        for n, s in list(self._tensor_shape_cache.items()):
            print(n, s)

    def get_graph_input_names(self):
        if self._model is None:
            return []
        return [graph_input.name for graph_input in self._model.inputs]

    def get_graph_input_ops(self):
        input_names = self.get_graph_input_names()
        input_ops = []
        for input_name in input_names:
            input_ops.append(self.get_tensor_by_name(input_name).op)
        return input_ops

    def get_graph_output_names(self):
        if self._model is None:
            return []
        return self._model.out_nodes_names

    def get_graph_output_tensors(self):
        output_names = self.get_graph_output_names()
        output_tensors = []
        for output_name in output_names:
            output_tensors.append(self.get_tensor_by_name(output_name))
        return output_tensors

    @classmethod
    def _map_operations_outputs(cls, operations):
        """
        :type operations: list[tensorflow.Operation]
        :rtype: dict[tensorflow.Operation, list[tensorflow.Operation]]
        """
        visitable_graph = VisitableGraph(operations)
        mapper = OutputOperationsMapper()
        visitable_graph.accept(mapper)
        return mapper.output_ops_map

    def get_tensor_by_name(self, tensor_name):
        """
        :type tensor_name: str
        :rtype: tensorflow.Tensor
        """
        return self._graph.get_tensor_by_name(self.indexed_tensor_name(tensor_name))

    @classmethod
    def indexed_tensor_name(cls, name, tensor_index=0):
        """
        :type name: str
        :type tensor_index: int
        :rtype: str
        """
        return "{}:{}".format(name, tensor_index) if ":" not in name else name

    def get_op_output_shape(self, operation, tensor_index=0):
        """
        :type operation: tensorflow.Operation
        :type tensor_index: int
        :rtype: list[int]
        """
        tensor = self._graph.get_tensor_by_name(
            GraphHelper.indexed_tensor_name(operation.name, tensor_index)
        )
        if tensor.name not in self._tensor_shape_cache:
            shape = self.get_tensor_output_shape(tensor)
            if len(shape) == 0:
                shapes = self._evaluate_tensors_output_shape([tensor])
                shape = shapes[tensor]
        else:
            shape = self._tensor_shape_cache[tensor.name]
        # return a shallow copy
        return list(shape)

    def _evaluate_tensors_output_shape(self, tensors):
        """
        :type tensor: list(tensorflow.Tensor)
        :return: dict[tensorflow.Tensor, list[int]]
        """
        shapes_map = dict()
        outputs_map = self.evaluate_tensors_output(tensors)
        for tensor, output in list(outputs_map.items()):
            shape = list(np.shape(output))
            self._tensor_shape_cache[tensor.name] = shape
            shapes_map[tensor] = shape

        return shapes_map

    def evaluate_tensor_output(self, tensor):
        """
        :type tensor: tensorflow.Tensor
        :return: np.ndarray
        """
        outputs_map = self.evaluate_tensors_output([tensor])
        return outputs_map[tensor].copy()

    def evaluate_tensors_output(self, tensors):
        """
        :type tensors: list(tensorflow.Tensor)
        :return: dict(tensorflow.Tensor, np.ndarray)
        """
        ignore_batch = True
        for t in tensors:
            if t.op.type != "Const":
                ignore_batch = False
                break

        input_tensors = dict()
        for i in self._model.inputs:
            indexed_tensor_name = GraphHelper.indexed_tensor_name(i.name)
            # batch dimension placeholder is 'None' in tf2, and '?' in tf1
            if ignore_batch and str(
                self.get_tensor_by_name(indexed_tensor_name).shape[0]
            ) in ["?", "None"]:
                input_tensors[indexed_tensor_name] = np.zeros(
                    [1] + i.shape[1:], dtype=np.float32
                )
            else:
                input_tensors[indexed_tensor_name] = np.zeros(i.shape, dtype=np.float32)

        for name, tensor in list(self._placeholders_stubs_map.items()):
            indexed_tensor_name = GraphHelper.indexed_tensor_name(name)
            if ignore_batch:
                input_tensors[indexed_tensor_name] = [tensor[0]]
            else:
                input_tensors[indexed_tensor_name] = tensor
        outputs_map = dict()
        requiring_evaluation = []
        for t in tensors:
            if t.name in self._tensor_value_cache:
                outputs_map[t] = self._tensor_value_cache[t.name]
            else:
                requiring_evaluation.append(t)

        if len(requiring_evaluation) > 0:
            try:
                # run in isolated session so that the memory gets cleared out after retrieving the output
                with tf_compat_v1.Session(graph=self._graph) as sess:
                    outputs = sess.run(
                        fetches=requiring_evaluation, feed_dict=input_tensors
                    )
                outputs = dict(list(zip(requiring_evaluation, outputs)))
                for t, o in list(outputs.items()):
                    self._tensor_value_cache[t.name] = o
                outputs_map.update(outputs)
                requiring_evaluation = []
            except InvalidArgumentError:
                pass

        for t in requiring_evaluation:
            try:
                # run in isolated session so that the memory gets cleared out after retrieving the output
                with tf_compat_v1.Session(graph=self._graph) as sess:
                    outputs = sess.run(fetches=[t], feed_dict=input_tensors)
                # outputs = self._session.run(fetches=[t], feed_dict=input_tensors)
                self._tensor_value_cache[t.name] = outputs[0]
                outputs_map[t] = outputs[0]
            except InvalidArgumentError:
                shape = (1,)
                try:
                    tensor_shape = t.get_shape().as_list()
                    if tensor_shape and None not in tensor_shape:
                        shape = tensor_shape
                except Exception:
                    pass

                outputs_map[t] = np.zeros(shape, dtype=np.float32)
        return outputs_map

    @classmethod
    def get_tensor_output_shape(cls, tensor):
        """
        :type tensor: tensorflow.Tensor
        :rtype: list[int]
        """
        shape = []
        if tensor.get_shape():
            tensor_shape = [dim if dim else -1 for dim in tensor.get_shape().as_list()]
            if len(tensor_shape) > 0 and not cls._has_unresolved_dimension(
                tensor_shape
            ):
                shape = cls._with_single_batch_dimension(tensor_shape)

        return shape

    @classmethod
    def _with_single_batch_dimension(cls, shape):
        """
        :type shape: list[int]
        :rtype: list[int]
        """
        copy = list(shape)
        if copy[0] == -1:
            copy[0] = 1
        return copy

    @classmethod
    def _has_unresolved_dimension(cls, shape):
        """
        :type shape: list[int]
        :rtype: bool
        """
        return len(shape) > 0 and -1 in shape

    @classmethod
    def filter_ops_by_type(cls, operations, operation_type):
        """
        :type operations: list[tensorflow.Operation]
        :type operation_type: str
        :rtype: list[tensorflow.Operation]
        """
        return [
            operation
            for operation in operations
            if operation.type.upper() == operation_type.upper()
        ]

    @classmethod
    def filter_op_by_type(cls, operations, operation_type):
        """
        :type operations: list[tensorflow.Operation]
        :type operation_type: str
        :rtype: tensorflow.Operation
        """
        ops = cls.filter_ops_by_type(operations, operation_type)
        if len(ops) == 0:
            raise OperationNotFoundError()
        return ops[0]

    @classmethod
    def filter_single_op_by_type(cls, operations, operation_type):
        ops = cls.filter_ops_by_type(operations, operation_type)
        if len(ops) == 0:
            operations_message = [(op.name, op.type) for op in operations]
            raise OperationNotFoundError(
                code_to_message.get_error_message("ERROR_TF_OPERATION_NOT_FOUND")(
                    operation_type, operations_message
                )
            )
        if len(ops) > 1:
            raise OperationNotFoundError(
                code_to_message.get_error_message("ERROR_TF_MULTIPLE_NODES_FOUND")(
                    operation_type
                )
            )
        return ops[0]

    def get_op_outputs(self, operation):
        return self._op_output_map.get(operation, [])

    @classmethod
    def get_op_input_tensors(cls, operations, input_types):
        """
        :type operations: tensorflow.Operation
        :type input_types:
        :return: tuple[tensorflow.Tensor]
        """
        tensors = [tensor for tensor in operations.inputs]
        types = [t.op.type for t in tensors]
        if len(types) != len(input_types):
            raise TensorNotFoundError(
                code_to_message.get_error_message(
                    "ERROR_TF_INPUT_DOES_NOT_MATCH_COUNT"
                )(operations.name, types, input_types)
            )

        input_tensors = []
        for i, t in enumerate(tensors):
            if types[i] == input_types[i] or input_types[i] == "?":
                input_tensors.append(t)
            else:
                raise TensorNotFoundError(
                    code_to_message.get_error_message(
                        "ERROR_TF_INPUT_DOES_NOT_MATCH_TYPES"
                    )(operations.name, types, input_types)
                )

        if len(input_tensors) > 1:
            return tuple(input_tensors)
        else:
            return input_tensors[0]

    @classmethod
    def get_op_sequence(cls, operation, types):
        """
        :type operation: tensorflow.Operation
        :type types: list[str]
        :rtype: list[tensorflow.Operation]
        """
        result = []
        if len(types) == 0 or operation.type != types[0]:
            raise OperationNotFoundError()

        result.append(operation)

        if len(types[1:]) > 0:
            matches = [t.op for t in operation.inputs if t.op.type == types[1]]
            if len(matches) == 1:
                result += cls.get_op_sequence(matches[0], types[1:])
            else:
                raise OperationNotFoundError()
        return result

    def _evaluate_tensor_shapes(self, ops):
        """
        :type ops: list(tensorflow.Operation)
        :rtype: None
        """
        tensors = set()
        for t in [t for op in ops for t in op.outputs]:
            tensors.add(t)

        graph_input_names = [input_obj.name for input_obj in self._model.inputs]
        for op in ops:
            if op.name not in graph_input_names:
                for t in op.inputs:
                    tensors.add(t)

        try:
            self._evaluate_tensors_output_shape(tensors)
        except Exception:
            # If we can't evaluate the graph ops in one pass
            # fallback to on-demand evaluation later
            logger = logging.getLogger()
            logger.warning(
                code_to_message.get_error_message(
                    "ERROR_TF_FALLBACK_TO_ONDEMAND_EVALUATION"
                )
            )

    @staticmethod
    def get_none_identity_input(input_tensor):
        """Traces back Op and its input to find the first op that is not an Identity op_type.
        :return the first Op tensor that is not an identity type, and a list of ops up to that tensor(inclusive)
        """
        inputs = []
        input_ = input_tensor
        while input_.op.type == "Identity":
            inputs.append(input_.op)
            input_ = input_.op.inputs[0]
        inputs.append(input_.op)
        return input_, inputs

    @staticmethod
    def get_stripped_input(input_tensor, inputs_types_to_ignore):
        """Traces back Op and its input to find the first op that is not ignore list of op_types
        :return the first Op tensor that is not in ignored op_types, and a list of ops up to that tensor(inclusive)
        """
        log_assert(
            len(inputs_types_to_ignore),
            "Requested to traceback Op input path by ignoring specified "
            "input_types, but no input_types provided.",
        )
        input_ = input_tensor
        while input_.op.type in inputs_types_to_ignore:
            input_ = input_.op.inputs[0]
        return input_

    @staticmethod
    def get_consumable_input_sequence(input_tensor):
        """Traces back an operation's sequence until it terminates in an op that can be trivially consumed.

        Note:
        -  By trivially consumed, we mean that the op contains/provides static data only, and as such can be
           consumed as part of a consequent op in some form (generally as an attribute).
        - That there is an implicit assumption that the sequence must begin and end with a trivially consumable
          op or an Identity. e.x Conv2D with the filter argument provided as a sequence of Identity->Constant.
        :return a sequence of unique consumable ops.
        """
        input_seq = set()
        consumable_types = ["Variable", "Const", "Fill"]
        if (
            input_tensor.op.type in consumable_types
            or input_tensor.op.type == "Identity"
        ):
            queue = [input_tensor]
            input_seq.add(input_tensor.op)
            while queue:
                head = queue.pop()
                for op_input_ in head.op.inputs:
                    if op_input_.op.type not in consumable_types:
                        queue.append(op_input_)
                    input_seq.add(op_input_.op)
        return input_seq

    def get_static_data_info(self, operation, input_tensor):
        """Traces back an operation's sequence until it terminates in one of
        [Shape, Constant, Identity, FakeQuant] ops
        @param operation: Operation for which input is the parameter input_tensor
        @param input_tensor: Input tensor which contains the static data

        :return a sequence of unique consumable ops.
        """
        const_value = self.evaluate_tensor_output(input_tensor)
        const_shape = self.get_op_output_shape(input_tensor)

        input_seq = set()
        consumable_types = ["Shape", "Const", "FakeQuantWithMinMaxVars", "Identity"]
        queue = [input_tensor]
        while queue:
            head = queue.pop()
            # If the Node is of one of the consumable_types or
            # has multiple outputs at least one of which is not part of the Sequence,
            # then don't add the node to the queue and sequence.
            if head.op.type not in consumable_types and all(
                [
                    ((output_op in input_seq) or output_op == operation.outputs[0].op)
                    for output_op in self.get_op_outputs(head.op)
                ]
            ):
                queue.extend(head.op.inputs)
                input_seq.add(head.op)

        consumed_nodes = list(input_seq)

        return (const_value, const_shape, consumed_nodes)

    def check_op_const_origin(self, op):
        if op.type == "Placeholder" or op in self.get_graph_input_ops():
            return False, []

        queue = [op]
        visited = []

        while queue:
            head = queue.pop()

            if head in visited:
                continue
            visited.append(head)

            for input_op in head.inputs:
                if (
                    input_op.op.type == "Placeholder"
                    or input_op.op in self.get_graph_input_ops()
                ):
                    return False, visited

                if input_op.op.type not in ["Const", "Shape"]:
                    queue.append(input_op.op)

        return True, visited

    def check_tensor_const_origin(self, tensor):
        return self.check_op_const_origin(tensor.op)


class ConverterError(Exception):
    """
    Defines a generic error for any converter errors.
    """

    pass


class OperationNotFoundError(LookupError):
    """
    Defines an error for when a required operation is not found by a method.
    """

    pass


class TensorNotFoundError(LookupError):
    """
    Defines an error for when a required operation is not found by a method.
    """

    pass


def scoped_op_name(scope_name, operation):
    """
    :type scope_name: str
    :type operation: tensorflow.Operation
    :rtype: str
    """
    op_name = str(operation.name)
    if scope_name == op_name:
        return "{}/{}".format(scope_name, op_name.split("/")[-1])
    else:
        return op_name


def get_const_op_value(op):
    if op.type != "Const":
        raise TypeError(
            "Expected Const type to resolve tensor value. Got {} for op {}".format(
                op.type, op.name
            )
        )
    tensor_type = tf.as_dtype(op.node_def.attr.get("dtype").type)
    tensor = op.node_def.attr.get("value").tensor
    if len(tensor.tensor_content):
        shape = [d.size for d in tensor.tensor_shape.dim]
        tensor_val = np.frombuffer(
            tensor.tensor_content, dtype=tensor_type.as_numpy_dtype
        )
        return np.reshape(tensor_val, newshape=shape)
    elif tensor_type.is_floating:
        return tensor.float_val[0]
    elif tensor_type.is_integer:
        return tensor.int_val[0]
    elif tensor_type.isbool:
        return tensor.bool_val[0]

    return ValueError("Unknown tensor type {} for op {}".format(tensor_type, op.name))


def expand_to_rank(shape, rank):
    """
    :type shape: list[int]
    :type rank: int
    :rtype: list[int]
    """
    result = shape[:]
    while len(result) < rank:
        result.insert(0, 1)
    return result


def is_tf2():
    return tf.__version__.startswith("2.")


class OutputOperationsMapper(GraphVisitor, object):
    def __init__(self):
        super(OutputOperationsMapper, self).__init__()
        self.output_ops_map = (
            OrderedDict()
        )  # type: dict[tf.Operation,list[tf.Operation]]

    def visit_scope(self, scope, ops):
        pass

    def visit_operation(self, op):
        """
        :type op: tensorflow.Operation
        :rtype: None
        """
        for t in op.inputs:
            if t.op not in self.output_ops_map:
                self.output_ops_map[t.op] = []
            self.output_ops_map[t.op].append(op)


class OperationExecutionSorter(object):
    def __init__(self, ops):
        self.input_ops = []
        self.output_ops = []
        self.ops_map = dict()
        for op in ops:
            op_wrapper = OperationExecutionSorter.OpWrapper(op)
            self.ops_map[op_wrapper.name] = op_wrapper
        self._connect_wrapped_ops()

    class OpWrapper:
        def __init__(self, tf_op):
            self.tf_op = tf_op
            self.name = str(tf_op.name)
            self.order = -1
            self.outputs = []

    def _connect_wrapped_ops(self):
        for op in list(self.ops_map.values()):
            for input_tensor in op.tf_op.inputs:
                if input_tensor.op.name not in self.ops_map:
                    continue
                input_op = self.ops_map[input_tensor.op.name]
                input_op.outputs.append(op)

    def sort(self, input_ops_names, output_ops_names):
        self._prepare_inputs_and_outputs(input_ops_names, output_ops_names)
        for input_op in self.input_ops:
            self._resolve_ops_in_execution_order(input_op, self.output_ops)

        self._flag_unvisited_nodes()

        sorted_in_execution_order = sorted(
            list(self.ops_map.values()), cmp=lambda a, b: a.order - b.order
        )
        return [op.tf_op for op in sorted_in_execution_order]

    def _prepare_inputs_and_outputs(self, input_ops_names, output_ops_names):
        self.input_ops = []
        self.output_ops = []
        for op_wrapper in list(self.ops_map.values()):
            if op_wrapper.name in input_ops_names:
                op_wrapper.order = 0
                self.input_ops.append(op_wrapper)
            elif op_wrapper.name in output_ops_names:
                self.output_ops.append(op_wrapper)

    @classmethod
    def _resolve_ops_in_execution_order(cls, input_op, output_ops):
        queue = [input_op]
        while len(queue) > 0:
            current_op = queue.pop(0)
            if current_op in output_ops:
                continue

            for output_op in reversed(current_op.outputs):
                output_order = max(output_op.order, current_op.order + 1)
                if output_order > output_op.order:
                    output_op.order = output_order
                    queue.insert(0, output_op)

    def _flag_unvisited_nodes(self):
        for op in list(self.ops_map.values()):
            if op.order == -1:
                op.order = sys.maxsize


def get_model_params(graph_def: GRAPH_DEF) -> int:
    """
    Calculate the tensorflow model parameters.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return int: Total number of parameters.
    """
    const_nodes = get_nodes_by_op_type(graph_def, "Const")

    total_parameters = 0
    for node in const_nodes:
        if not hasattr(node, "attr"):
            continue
        if not hasattr(node.attr["value"].tensor.tensor_shape, "dim"):
            continue

        shape = []
        for d in node.attr["value"].tensor.tensor_shape.dim:
            if not hasattr(d, "size"):
                continue
            shape.append(d.size)

        total_parameters += np.prod(shape)
    return total_parameters


def extract_inputs_from_graph(sess_graph: GRAPH) -> Dict[str, tf.Tensor]:
    """
    Get the input tensors from given session's Graph.

    :param GRAPH sess_graph: Graph object obtained from the current session.
    :return Dict[str, tf.Tensor]: Dict mapping of input tensor name to input tensor.
    """
    input_tensors = OrderedDict()

    for op in sess_graph.get_operations():
        if op.type == "Placeholder" and len(op.inputs) == 0 and len(op.outputs) == 1:
            input_tensors[op.outputs[0].name] = op.outputs[0]
        elif (
            op.type == "PlaceholderWithDefault"
            and len(op.inputs) == 1
            and len(op.outputs) == 1
        ):
            input_tensors[op.outputs[0].name] = op.outputs[0]

    return input_tensors


def extract_outputs_from_graph(sess_graph: GRAPH) -> Dict[str, tf.Tensor]:
    """
    Get the output tensors from given session's Graph.

    :param GRAPH sess_graph: Graph object obtained from the current session.
    :return Dict[str, tf.Tensor]: Dict mapping of output tensor name to output tensor.
    """
    UNLIKELY_OUTPUT_TYPES = {"Const", "Assign", "NoOp", "Placeholder"}
    output_tensors = OrderedDict()

    for op in sess_graph.get_operations():
        if op.type not in UNLIKELY_OUTPUT_TYPES and len(op.outputs) == 1:
            output_tensors[op.outputs[0].name] = op.outputs[0]

    output_tensor_names = output_tensors.keys()

    for op in sess_graph.get_operations():
        for in_t in op.inputs:
            if in_t.name in output_tensor_names:
                output_tensors.pop(in_t.name)
        for cont_op in op.control_inputs:
            for out_t in cont_op.outputs:
                if out_t.name in output_tensor_names:
                    output_tensors.pop(out_t.name)

    return output_tensors


def save_as_frozen_graph(graph_def: GRAPH_DEF, model_dir: str, model_name: str) -> None:
    """
    Save the given TF GraphDef as frozen graph at given location.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :param str model_dir: Model save directory.
    :param str model_name: Model name to be used while saving.
    """
    with tf_compat_v1.Session() as sess:
        with tf_compat_v1.Graph().as_default() as graph:
            tf_compat_v1.import_graph_def(graph_def, name="")
            if version.parse(tf.__version__) > version.parse("2.0.4"):
                tf_compat_v1.io.write_graph(
                    graph_or_graph_def=graph.as_graph_def(add_shapes=True),
                    logdir=model_dir,
                    name=model_name,
                    as_text=False,
                )
            else:
                from tensorflow.python.framework import graph_io

                graph_io.write_graph(
                    graph_or_graph_def=graph.as_graph_def(add_shapes=True),
                    logdir=model_dir,
                    name=model_name,
                    as_text=False,
                )


def remove_suffix(name: str) -> str:
    """
    Remove the suffix from the tensor name.

    :param str name: Tensor name.
    :return str: Updated Tensor name.
    """
    if name.find(":") != -1:
        return name.split(":")[0]
    return name


def update_tf_assign_op(graph_def: GRAPH_DEF) -> None:
    """
    Inplace replace the tf.assign ops with Identity ops.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    """
    for node in graph_def.node:
        if node.op == "Assign":
            node.op = "Identity"
            if "use_locking" in node.attr:
                del node.attr["use_locking"]
            if "validate_shape" in node.attr:
                del node.attr["validate_shape"]
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]


def fix_freezing_errors(ref_graph_def: GRAPH_DEF) -> GRAPH_DEF:
    """
    Fix the graph freezing errors due to Assign ops.

    :param GRAPH_DEF ref_graph_def: GraphDef reference of the model.
    :return GRAPH_DEF: Fixed GraphDef instance of the model.
    """
    assign_var_op_list = []
    for i in reversed(range(len(ref_graph_def.node))):
        if ref_graph_def.node[i].op in ["AssignVariableOp", "AssignSubVariableOp"]:
            assign_var_op_list.append(ref_graph_def.node.pop(i).name)
    names_to_be_remove = set(assign_var_op_list)
    for n in ref_graph_def.node:
        for i in reversed(range(len(n.input))):
            if n.input[i].startswith("^") and n.input[i][1:] in names_to_be_remove:
                n.input.pop(i)
    return ref_graph_def


def graphdef_to_graph(graph_def: GRAPH_DEF) -> GRAPH:
    """
    Convert the graphdef instance into graph.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return GRAPH: Converted TF Graph reference of the model.
    :raises ValueError: If the graph is incorrectly constructed.
    """
    status = graph_checker(graph_def)
    if not status:
        log_error(
            "Tensorflow graph_def to graph conversion failed because graph_def is invalid."
        )

    update_tf_assign_op(graph_def)
    try:
        with tf_compat_v1.Graph().as_default() as graph:
            tf_compat_v1.graph_util.import_graph_def(graph_def, name="")
    except:
        graph_def = fix_freezing_errors(graph_def)
        with tf_compat_v1.Graph().as_default() as graph:
            tf_compat_v1.import_graph_def(graph_def, name="")
    return graph


def get_unique_ops(graph: GRAPH) -> Dict[str, int]:
    """
    Get the count of all the unique ops.

    :param GRAPH graph: GraphDef reference of the model.
    :return Dict[str, int]: Mapping from node op type to its total occurances.
    """
    all_ops = {}
    for node in graph.get_operations():
        if node.type not in all_ops:
            all_ops[node.type] = 0
        all_ops[node.type] += 1
    return all_ops


def tensor_shape_to_list(tensor_shape: tf.TensorShape) -> List[int]:
    """
    Interprete the tf.TensorShape object into list of int indicating shape value
    of tensor.

    :param tf.TensorShape tensor_shape: TensorShape object of given tensor.
    :return List: Interpreted shape as list of ints.
    """
    if tensor_shape.ndims is None:
        return None
    return [s if s is not None else -1 for s in tensor_shape.as_list()]


def get_nodes_by_op_type(
    graph_def: GRAPH_DEF, op_type: str
) -> List[tf_compat_v1.NodeDef]:
    """
    Get the list of nodes having same op type.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :param str op_type: Name of the op for a node
    :return List[tf_compat_v1.NodeDef]: List of nodes with given op type.
    """
    filtered_nodes = []
    for node in graph_def.node:
        if node.op == op_type:
            filtered_nodes.append(node)

    return filtered_nodes


def get_nodes(graph_def: GRAPH_DEF) -> List[tf_compat_v1.NodeDef]:
    """
    Get the list of all the nodes from a given GraphDef.


    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return List[tf_compat_v1.NodeDef]: List of all the nodes in the model.
    """
    return [node for node in graph_def.node]


def get_node_mapping_dict(graph_def: GRAPH_DEF) -> Dict[str, tf_compat_v1.NodeDef]:
    """
    Get the dict with mapping from node name to node.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return Dict[str, tf_compat_v1.NodeDef]: Dict with mapping from node name to node.
    """
    return {node.name: node for node in get_nodes(graph_def)}


def get_node_by_input_name(
    graph_def: GRAPH_DEF,
) -> Dict[str, List[tf_compat_v1.NodeDef]]:
    """
    Get the dict to fetch any node by its input name.

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return Dict[str, Lst[tf_compat_v1.NodeDef]]: Dict with mapping from input name to node.
    """
    node_by_input_name = defaultdict(list)
    for node in get_nodes(graph_def):
        for inp in node.input:
            node_by_input_name[inp].append(node)
    return node_by_input_name


def get_node_by_output_name(graph_def: GRAPH_DEF) -> Dict[str, tf_compat_v1.NodeDef]:
    """
    Get the dict to fetch any node by its output name or node name (which is the
    output name).

    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return Dict[str, tf_compat_v1.NodeDef]: Dict with mapping from output name to node.
    """
    graph = graphdef_to_graph(graph_def)
    node_mapping = get_node_mapping_dict(graph_def)

    node_by_output_name = {}
    for operation in graph.get_operations():
        for output in operation.outputs:
            node_by_output_name[output.name] = node_mapping[operation.name]
    return node_by_output_name


def get_node_inputs(
    node: tf_compat_v1.NodeDef, graph_def: GRAPH_DEF
) -> List[tf.Tensor]:
    """
    Get the list of input tensors of given node.

    :param tf_compat_v1.NodeDef node: Node reference.
    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return List[tf.Tensor]: List of input tensors of given nodes.
    """
    inputs = []
    graph = graphdef_to_graph(graph_def)
    for operation in graph.get_operations():
        if operation.name == node.name:
            inputs.extend(operation.inputs)
            break
    return inputs


def get_node_outputs(
    node: tf_compat_v1.NodeDef, graph_def: GRAPH_DEF
) -> List[tf.Tensor]:
    """
    Get the list of output tensors of given node.

    :param tf_compat_v1.NodeDef node: Node reference.
    :param GRAPH_DEF graph_def: GraphDef reference of the model.
    :return List[tf.Tensor]: List of output tensors of given nodes.
    """
    outputs = []
    graph = graphdef_to_graph(graph_def)
    for operation in graph.get_operations():
        if operation.name == node.name:
            outputs.extend(operation.outputs)
            break
    return outputs


def fetch_tensor_info(tensor: tf.Tensor) -> TensorInfo:
    """
    Get the information about the tensor such as its shape, dtype and layout.

    :param tf.Tensor tensor: Tensorflow Tensor instance.
    :return TensorInfo: TensorInfo object conatining tensor related info.
    """
    tensor_shape = tensor_shape_to_list(tensor.shape)
    tensor_dtype = tensor.dtype.as_numpy_dtype
    tensor_layout = determine_layout(tensor_shape)
    tensor_name = tensor.name
    return TensorInfo(tensor_name, tensor_shape, tensor_dtype, tensor_layout)


def get_input_info(graph_def: GRAPH_DEF) -> Dict[str, TensorInfo]:
    """
    Get the input info in form of dict.

    :param GRAPH_DEF graph_def: Graphdef instance.
    :return Dict[str, TensorInfo]: Dict containing mapping of input tensor name
        to the corresponding TensorInfo object for tensor.
    """
    frozen_graph = graphdef_to_graph(graph_def)
    input_tensors_dict = extract_inputs_from_graph(frozen_graph)

    input_tensor_info_dict = {}
    for input_name, input_tensor in input_tensors_dict.items():
        input_tensor_info_dict[input_name] = fetch_tensor_info(input_tensor)
    return input_tensor_info_dict


def get_output_info(graph_def: GRAPH_DEF) -> Dict[str, TensorInfo]:
    """
    Get the output info in form of dict.

    :param GRAPH_DEF graph_def: Graphdef instance.
    :return Dict[str, TensorInfo]: Dict containing mapping of output tensor name
        to the corresponding TensorInfo object for tensor.
    """
    frozen_graph = graphdef_to_graph(graph_def)
    output_tensors_dict = extract_outputs_from_graph(frozen_graph)

    output_tensor_info_dict = {}
    for output_name, output_tensor in output_tensors_dict.items():
        output_tensor_info_dict[output_name] = fetch_tensor_info(output_tensor)
    return output_tensor_info_dict


def copy_graphdef(graph_def: GRAPH_DEF) -> GRAPH_DEF:
    """
    Make a deep copy of given graph_def instance.

    :param GRAPH_DEF graph_def: Reference graph_def.
    :return GRAPH_DEF: Copy of given graph_def.
    """
    return copy.deepcopy(graph_def)


def get_parent_nodes(graph_def: GRAPH_DEF, node_name: str) -> List[NODE_DEF]:
    """
    Get the parent nodes of a given node.

    :param GRAPH_DEF graph_def: Reference graph_def.
    :param str node_name: Name of the node.
    :return List[NODE_DEF]: List of parent nodes.
    """
    node_mapping_dict = get_node_mapping_dict(graph_def)
    node_by_op_name_dict = get_node_by_output_name(graph_def)
    node = node_mapping_dict[node_name]
    input_tensors = get_node_inputs(node, graph_def)

    par_nodes = []
    for input_tensor in input_tensors:
        par_nodes.append(node_by_op_name_dict[input_tensor.name])
    return par_nodes


def get_children_nodes(graph_def: GRAPH_DEF, node_name: str) -> List[NODE_DEF]:
    """
    Get the children nodes of a given node.

    :param GRAPH_DEF graph_def: Reference graph_def.
    :param str node_name: Name of the node.
    :return List[NODE_DEF]: List of children nodes.
    """
    node_mapping_dict = get_node_mapping_dict(graph_def)
    node_by_ip_name_dict = get_node_by_input_name(graph_def)
    node = node_mapping_dict[node_name]
    output_tensors = get_node_outputs(node, graph_def)

    children_nodes = []
    for output_tensor in output_tensors:
        children_nodes.extend(node_by_ip_name_dict[remove_suffix(output_tensor.name)])
    return children_nodes


def graph_checker(graph_def: GRAPH_DEF) -> bool:
    """
    Check the correctness of the graph.

    :param GRAPH_DEF graph_def: Reference graph_def.
    :return bool: Boolean status indicating correctness of the graph.
    """
    status = True
    try:
        ensure_graph_is_valid(graph_def)
    except ValueError:
        # In case of incorrect graph, ValueError is raised.
        log_error("TF Native Checker: The model is invalid.")
        status = False
    return status


def extract_subgraph(
    graph_def: GRAPH_DEF,
    input_tensor_info: Dict[str, Dict] = {},
    output_tensor_names: List[str] = None,
) -> GRAPH_DEF:
    """
    Clean up the graph by removing redundant nodes which don't contribute in
    computation of given output.

    Note: tensor names shall be free from ":0" suffix.

    :param GRAPH_DEF graph_def: Reference graph_def.
    :param Dict[str, Dict] input_tensor_info: Dict with mapping from tensor
        name to its shape. Defaults to {}.
    :param List[str] output_tensor_names: List of output tensor names. Defaults
        to None.
    :return GRAPH_DEF: Updated graph_def.
    """
    graph = graphdef_to_graph(graph_def)
    node_mapping = get_node_mapping_dict(graph_def)

    inferred_model_input_names = set(map(remove_suffix, get_input_info(graph_def)))

    for input_name in input_tensor_info:
        # If the user provided input is not a placeholder then convert that node
        # into a placeholder
        if input_name not in inferred_model_input_names:
            input_tensor = graph.get_tensor_by_name(input_name + ":0")

            tensor_shape = input_tensor_info[input_name]
            old_node = node_mapping[input_name]
            graph_def.node.remove(old_node)

            placeholder_node = tf_compat_v1.NodeDef(
                name=input_name,
                op="Placeholder",
                attr={
                    "dtype": attr_value_pb2.AttrValue(
                        type=input_tensor.dtype.as_datatype_enum
                    ),
                    "shape": attr_value_pb2.AttrValue(
                        shape=TensorShapeProto(
                            dim=[TensorShapeProto.Dim(size=s) for s in tensor_shape]
                        )
                    ),
                },
            )
            graph_def.node.append(placeholder_node)
            node_mapping[input_name] = placeholder_node

    if output_tensor_names is None:
        graph = graphdef_to_graph(graph_def)
        output_tensors_dict = extract_outputs_from_graph(graph)
        output_tensor_names = [
            remove_suffix(name) for name in output_tensors_dict.keys()
        ]

    # Traverse the graph from the graph output nodes.
    visited_nodes = set()
    stack = []

    for name in output_tensor_names:
        node_name = remove_suffix(name)
        node_name = node_name[1:] if node_name.startswith("^") else node_name
        if node_name in node_mapping.keys():
            node = node_mapping[node_name]
            stack.append(node)

    if len(stack) == 0:
        raise RuntimeError(
            f"Given output tensors: {output_tensor_names} are "
            "not produced by any node."
        )

    while len(stack) != 0:
        node = stack.pop()
        if node.name in visited_nodes:
            continue
        visited_nodes.add(node.name)
        for ip_name in node.input:
            node_name = remove_suffix(ip_name)
            node_name = node_name[1:] if node_name.startswith("^") else node_name
            if node_name in node_mapping:
                parent_node = node_mapping[node_name]
                if parent_node.name not in visited_nodes:
                    stack.append(parent_node)
            else:
                log_warning(f"Node {node_name} not found in graph.")

    # Till now visited_nodes is populated with nodes connected with graph outputs
    remove_nodes = [n for n in graph_def.node if n.name not in visited_nodes]

    # Remove nodes from the graph as well as from two dictionaries.
    for r_n in remove_nodes:
        graph_def.node.remove(r_n)

    status = graph_checker(graph_def)
    if not status:
        raise RuntimeError("Extracted sub-graph is invalid.")

    return graph_def


def run_tf_inference(
    graph_def: GRAPH_DEF,
    input_data: Dict[str, np.ndarray],
    output_node_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Run the inference of provided tensorflow graph using given input data.

    :param GRAPH_DEF graph_def: GraphDef representation of tf model.
    :param Dict[str, np.ndarray] input_data: Dict of input data with key as
        input tensor name and value as corresponding numpy array.
    :param List[str] output_node_names: List of output tensors for which output
        is to be computed.
    :return Dict[str, np.ndarray]: Dict of output data with key as output tensor
        name and value as corresponding numpy array.
    """
    graph = graphdef_to_graph(graph_def)
    session = tf_compat_v1.Session(graph=graph)

    input_tensor_dict = extract_inputs_from_graph(session.graph)
    output_tensor_dict = extract_outputs_from_graph(session.graph)

    filtered_output_tensors = {}
    if not output_node_names:
        filtered_output_tensors.update(output_tensor_dict)
    else:
        for output_name in output_node_names:
            try:
                tensor = session.graph.get_tensor_by_name(output_name)
                filtered_output_tensors[output_name] = tensor
            except ValueError:
                log_warning(
                    f"Given output: {output_name} is not a tensor of a "
                    "model. It refers to an operation of a model."
                )
                return None
            except KeyError:
                log_warning(
                    f"Given output: {output_name} is not identified as "
                    "model's tensor."
                )
                return None

    filtered_input_data = {}
    for input_name in input_tensor_dict.keys():
        if input_name in input_data:
            filtered_input_data[input_name] = input_data[input_name]
        else:
            log_warning(
                f"Inference input data for input: {input_name} is not found. "
                "Please provide the same for inference."
            )
            return None

    output_data = session.run(
        list(filtered_output_tensors.values()), feed_dict=filtered_input_data
    )

    result_dict = {}
    for name, data in zip(filtered_output_tensors.keys(), output_data):
        result_dict[name] = data

    return result_dict
