# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import List, Optional

from packaging import version
from qti.aisw.converters.common.framework_model_api import FrameworkModelAPI
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_error
from qti.aisw.converters.tensorflow import tf_compat_v1, util
from qti.aisw.converters.tensorflow.util import (
    DEFAULT_GENERIC_OPTIMIZATIONS,
    DEFAULT_GRAPPLER_OPTIMIZATION,
    ETFGrapplerOptimization,
    GRAPH_DEF,
)

import tensorflow as tf
from tensorflow.core.framework import graph_pb2, variable_pb2
from tensorflow.core.protobuf import config_pb2, meta_graph_pb2
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.util import compat

if version.parse(tf.__version__) <= version.parse("2.0.4"):
    from tensorflow.python.pywrap_tensorflow import TransformGraphWithStringInputs
elif version.parse("2.0.4") < version.parse(tf.__version__) <= version.parse("2.4.4"):
    from tensorflow.python._pywrap_transform_graph import TransformGraphWithStringInputs
else:
    from tensorflow.python.util._pywrap_transform_graph import (
        TransformGraphWithStringInputs,
    )

from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.tools.optimize_for_inference_lib import fold_batch_norms
from tensorflow.python.training.saver import export_meta_graph


class TFModelUtils(FrameworkModelAPI):
    def __init__(self, loader):
        self.loader = loader

    def add_inputs(self):
        raise NotImplementedError

    def add_outputs(self):
        raise NotImplementedError

    def remove_inputs(self, input_names_to_remove: List[str]):
        raise NotImplementedError

    def remove_outputs(self, output_names_to_remove: List[str]):
        raise NotImplementedError

    def clean_model(self):
        raise NotImplementedError

    def native_shape_inference(self):
        """Adds shape information to each nodes in the graph.

        :return TFModelUtils: Self instance of the class.
        """
        with tf_compat_v1.Graph().as_default() as temp_graph:
            tf_compat_v1.import_graph_def(self.loader.model.graph_def, name="")
            self.loader.model.graph_def = temp_graph.as_graph_def(add_shapes=True)
        return self

    def remove_node(self, node_name: str, input_name: str):
        """Remove node as per given node_name and connect the given inputs to
        the child nodes.

        :param str node_name: Node name of the node to be removed.
        :param str input_name: Node's input names.
        :return TFModelUtils: Returns the self instance of the class.
        """
        valid_nodes = []
        input_of_removed_node = ""
        graph_def = self.loader.model.graph_def
        for node in graph_def.node:
            if node.name == node_name:
                if (input_name not in node.input) or (len(node.input) == 0):
                    log_error(
                        f"Can't delete the given node: {node_name}. "
                        "Either the node has no inputs or provided input_name "
                        "doesn't exist in given node's inputs."
                    )
                input_of_removed_node = input_name if len(node.input) else ""
                continue
            valid_nodes.append(node)

        # modify inputs where required
        # removed name must be replaced with input of removed node
        for node in valid_nodes:
            inp_names = []
            replace = False
            for inp in node.input:
                if inp == node_name:
                    inp_names.append(input_of_removed_node)
                    replace = True
                else:
                    inp_names.append(inp)
            if replace:
                del node.input[:]
                node.input.extend(inp_names)

        mod_graph_def = tf_compat_v1.GraphDef()
        mod_graph_def.node.extend(valid_nodes)
        self.loader.update_model(mod_graph_def)
        return self

    def topological_sort(self):
        """
        Topological sort the nodes in the GraphDef.

        :return TFModelUtils: Returns the self instance of the class.
        """
        self.__optimize_by_generic_optimizer(["sort_by_execution_order"])
        return self

    def remove_nodes_by_op_type(self, op_types: List[str]):
        """
        Removes all the nodes of given op types.

        :param List[str] op_types: List of op types of nodes to be removed.
        :return TFModelUtils: Returns the self instance of the class.
        """
        for op_type in op_types:
            nodes = util.get_nodes_by_op_type(self.loader.model.graph_def, op_type)

            nodes_can_be_removed = True
            for single_node in nodes:
                operation = self.loader.session.graph.get_operation_by_name(
                    single_node.name
                )
                if len(operation.inputs) > 1 or len(operation.outputs) > 1:
                    log_debug1(
                        f"Nodes with given op_type {op_type} are not removed as they have more than one input or output."
                    )
                    nodes_can_be_removed = False
                    break

            if nodes_can_be_removed:
                self.__optimize_by_generic_optimizer([f"remove_nodes(op={op_type})"])
        return self

    def optimize(self, optimizations : Optional[List[ETFGrapplerOptimization]]  , **kwargs):
        """
        Optimize the model using generic set of optimizations and grappler
        optimizations.

        :param Optional[List[ETFGrapplerOptimization]] optimizations: Optional Grappler Optimizations.
        :return TFModelUtils: Returns the self instance of the class.
        """

        skip_optimization = kwargs.get("skip_optimization",False)

        self.__remove_unused_nodes()
        self.__remove_no_op_nodes()
        self.__remove_training_nodes()

        if skip_optimization:
            return self

        final_optimizations = []
        if not optimizations:
            final_optimizations = ETFGrapplerOptimization.get_default_optimizations()
        else:
            final_optimizations = [ETFGrapplerOptimization.to_name(opt) for opt in optimizations]


        self.__constant_folding()
        self.__fold_batch_norm()

        original_graph_def = util.copy_graphdef(self.loader.model.graph_def)
        try:
            updated_graph_def = self.__optimize_by_grappler(
                self.loader.model.graph_def, final_optimizations
            )
            self.loader.update_model(updated_graph_def)
            self.loader.native_checker()
            self.loader.utils.native_shape_inference()
        except Exception as e:
            log_error(
                f"Grappler optimization failed due to : {e}. Using previously optimized graph for further optimizations."
            )
            self.loader.update_model(original_graph_def)

        self.topological_sort()
        return self

    def __remove_training_nodes(self):
        """
        Removes the Identity and CheckNumerics nodes which are only used during
        training.

        :return TFModelUtils: Returns the self instance of the class.
        """
        log_debug1("Applying optimization: Remove Training Nodes")
        updated_graph_def = tf_compat_v1.graph_util.remove_training_nodes(
            self.loader.model.graph_def
        )
        self.loader.update_model(updated_graph_def)
        self.loader.native_checker()
        return self

    def __remove_unused_nodes(self):
        """
        Removes unused nodes from graph.

        :return TFModelUtils: Returns the self instance of the class.
        """
        log_debug1("Applying optimization: Remove Unused Nodes")
        placeholder_type_enums = []
        for _, tensor in self.loader.get_inputs().items():
            enum = tensor.dtype.as_datatype_enum
            placeholder_type_enums.append(enum)

        input_names = [
            util.remove_suffix(name) for name in self.loader.get_input_names()
        ]
        output_names = [
            util.remove_suffix(name) for name in self.loader.get_output_names()
        ]

        updated_graph_def = strip_unused_lib.strip_unused(
            self.loader.model.graph_def,
            input_names,
            output_names,
            placeholder_type_enums,
        )

        # Since the strip_unused API removes the existing input shape information,
        # we need to explictly add the shape information from original graph_def
        # to new graph_def.
        original_graph_node_mapping = util.get_node_mapping_dict(
            self.loader.model.graph_def
        )
        updated_graph_node_mapping = util.get_node_mapping_dict(updated_graph_def)
        for input_name in input_names:
            if "shape" in original_graph_node_mapping[input_name].attr.keys():
                original_shape_proto = (
                    original_graph_node_mapping[input_name].attr["shape"].shape
                )

            if "shape" not in updated_graph_node_mapping[input_name].attr.keys():
                updated_graph_node_mapping[input_name].attr["shape"].shape.CopyFrom(
                    original_shape_proto
                )

        self.loader.update_model(updated_graph_def)
        self.loader.native_checker()
        return self

    def __fold_batch_norm(self):
        """
        Remove batch norm nodes by folding them into conv nodes.

        :return TFModelUtils: Returns the self instance of the class.
        """
        log_debug1("Applying optimization: BatchNorm Folding")
        updated_graph_def = fold_batch_norms(self.loader.model.graph_def)
        self.loader.update_model(updated_graph_def)
        self.loader.native_checker()

        # Calling __remove_unused_nodes to get rid of redundant constant nodes.
        self.__remove_unused_nodes()
        return self

    def __constant_folding(self):
        """
        Looks for any sub-graphs within the model that always evaluate to
        constant expressions, and replaces them with those constants.

        :return TFModelUtils: Returns the self instance of the class.
        """
        log_debug1("Applying optimization: Constant Folding")
        self.__optimize_by_generic_optimizer(["fold_constants"])
        return self

    def __remove_identity(self):
        """
        Remove all the identity nodes except for input or output.

        :return TFModelUtils: Returns the self instance of the class.
        """
        log_debug1("Applying optimization: Remove Identity Nodes")
        self.__optimize_by_generic_optimizer(["remove_nodes(op=Identity)"])
        return self

    def __remove_no_op_nodes(self):
        """
        Remove all the NoOp nodes.

        :return TFModelUtils: Returns the self instance of the class.
        """
        log_debug1("Applying optimization: Remove NoOp Nodes")
        original_graph_def = util.copy_graphdef(self.loader.model.graph_def)
        for i in reversed(range(len(original_graph_def.node))):
            if original_graph_def.node[i].op == "NoOp":
                del original_graph_def.node[i]

        for node in original_graph_def.node:
            for i in reversed(range(len(node.input))):
                if node.input[i][0] == "^":
                    del node.input[i]

        self.loader.update_model(original_graph_def)
        self.loader.native_checker()

        # Calling __remove_unused_nodes to get rid of redundant constant nodes.
        self.__remove_unused_nodes()
        return self

    def __optimize_by_generic_optimizer(
        self, optimizations: List[str] = DEFAULT_GENERIC_OPTIMIZATIONS
    ):
        """
        Apply generic optimizations on graph_def.

        :param GRAPH_DEF graph_def: GraphDef instance of model.
        :param List[str] optimizations: List of optimizations to be applied, defaults to DEFAULT_GENERIC_OPTIMIZATIONS
        :return GRAPH_DEF: Optimized GraphDef instance.
        """
        log_debug1(
            f"Applying generic optimizer with following optimizations: {optimizations}"
        )

        input_names = self.loader.get_input_names()
        output_names = self.loader.get_output_names()

        graph_def_string = self.loader.model.graph_def.SerializeToString()
        inputs_string = compat.as_bytes(",".join(input_names))
        outputs_string = compat.as_bytes(",".join(output_names))
        optimization_string = compat.as_bytes(" ".join(optimizations))

        if version.parse(tf.__version__) <= version.parse("2.0.4"):
            from tensorflow.python.framework import errors

            with errors.raise_exception_on_not_ok_status() as status:
                optimized_graph_def_string = TransformGraphWithStringInputs(
                    graph_def_string,
                    inputs_string,
                    outputs_string,
                    optimization_string,
                    status,
                )
        else:
            optimized_graph_def_string = TransformGraphWithStringInputs(
                graph_def_string, inputs_string, outputs_string, optimization_string
            )

        optimized_graph_def = graph_pb2.GraphDef()
        optimized_graph_def.ParseFromString(optimized_graph_def_string)

        self.loader.update_model(optimized_graph_def)
        self.loader.native_checker()
        return self

    def __optimize_by_grappler(
        self,
        graph_def: GRAPH_DEF,
        optimizations: List[str] = DEFAULT_GRAPPLER_OPTIMIZATION,
    ) -> GRAPH_DEF:
        """
        Apply grappler optimizations on graph_def.

        :param GRAPH_DEF graph_def: GraphDef instance of model.
        :param List[str] optimizations: List of optimizations to be applied, defaults to DEFAULT_GRAPPLER_OPTIMIZATION
        :return GRAPH_DEF: Optimized GraphDef instance.
        """
        log_debug1(
            f"Applying Grappler optimizer with following optimizations: {optimizations}"
        )
        for function in graph_def.library.function:
            if "api_implements" in function.attr:
                del function.attr["api_implements"]

        input_names = self.loader.get_input_names()
        output_names = self.loader.get_output_names()

        graph = util.graphdef_to_graph(graph_def)
        meta_graph = export_meta_graph(graph_def=graph_def, graph=graph)
        # Clear the initializer_name for the variables collections, since they are not
        # needed after saved to saved_model.
        for name in [
            "variables",
            "model_variables",
            "trainable_variables",
            "local_variables",
        ]:
            raw_list = []
            for raw in meta_graph.collection_def["variables"].bytes_list.value:
                variable = variable_pb2.VariableDef()
                variable.ParseFromString(raw)
                variable.ClearField("initializer_name")
                raw_list.append(variable.SerializeToString())
            meta_graph.collection_def[name].bytes_list.value[:] = raw_list

        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for name in input_names + output_names:
            fetch_collection.node_list.value.append(name)
        meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

        # Initialize RewriterConfig with everything disabled except function inlining.
        config = config_pb2.ConfigProto()
        rewrite_options = config.graph_options.rewrite_options
        if version.parse(tf.__version__) >= version.parse("1.10.0"):
            rewrite_options.min_graph_nodes = -1  # do not skip small graphs
        rewrite_options.disable_model_pruning = False
        rewrite_options.optimizers.extend(optimizations)

        return tf_optimizer.OptimizeGraph(config, meta_graph)
