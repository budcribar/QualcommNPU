# =============================================================================
#
#  Copyright (c) 2016-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import copy
from typing import Any, Dict, Iterator, List, Optional, Union, Text

import qti.aisw.converters.common.utils.code_to_message as code_to_message
import tensorflow as tf
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import (
    log_debug,
    log_info,
    log_warning,
)
from qti.aisw.converters.common.utils.framework_utils import (
    FrameworkSummary,
    TensorInfo,
)
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
from qti.aisw.converters.tensorflow import tf_compat_v1, util
from qti.aisw.converters.tensorflow.tf_model_api import TFModelUtils
from qti.aisw.converters.tensorflow.util import (
    GRAPH,
    GRAPH_DEF,
    NODE_DEF,
    ConverterError,
    GraphHelper,
    GraphPrinter,
    VisitableGraph,
    is_tf2,
)
from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import Operation


class Model(object):
    """
    Model representation of a tensorflow graph coming from saved model,
    frozen model or ckpt model format.
    """

    class Input(object):
        """
        Input representation of a tensorflow model.
        """

        INPUT_TYPE_DEFAULT = "default"

        def __init__(self, name: str, shape: List[str]):
            """
            Generate Tensorflow Input using tensor's name and its shape.

            :param str name: Name of the input tensor.
            :param List[str] shape: Shape of the input tensor.
            """
            self.name = name
            self.shape = shape

    def __init__(
        self,
        model_path: str,
        input_nodes_names: List[str],
        input_nodes_shapes: List[int],
        out_node_names: List[str],
        saved_model_tag: str = "",
        saved_model_signature_key: str = "",
    ):
        """
        Generate Model object from the model path, model io information and
        saved model information.

        :param str graph_path: Path of the tensorflow model.
        :param List[str] input_nodes_names: List of input node names.
        :param List[int] input_nodes_shapes: List of input node shapes.
        :param List[str] out_node_names: List of output node names.
        :param str saved_model_tag: Saved model tag for conversion, defaults to ""
        :param str saved_model_signature_key: Saved model signature key for
            conversion, defaults to ""
        """
        # Reset the graph during init.
        tf_compat_v1.reset_default_graph()
        self.session = tf_compat_v1.Session(graph=tf_compat_v1.Graph())

        self.__load(
            model_path,
            input_nodes_names,
            input_nodes_shapes,
            out_node_names,
            saved_model_tag,
            saved_model_signature_key,
        )

    def __deepcopy__(self, memo: Dict):
        """
        Make a deep copy of Model object.
        Note: As the deep copy of session requires special workarounds, this
        function is needed.

        :param Dict memo: memorization dict to be used internally.
        :return Model: Deep copy of current Model object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "graph_def":
                temp_graph = util.graphdef_to_graph(v)
                new_session = tf_compat_v1.Session(graph=temp_graph)
                setattr(result, "session", new_session)
                setattr(result, "graph_def", temp_graph.as_graph_def())
                continue
            elif k == "session":
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __load(
        self,
        graph_path: str,
        input_nodes_names: List[str],
        input_nodes_shapes: List[str],
        out_node_names: List[str],
        saved_model_tag: str = "",
        saved_model_signature_key: str = "",
    ) -> None:
        """
        Loads the Tensorflow Graph into the class's Session's Graph and
        builds a Model instance with all the relevant information for a
        ModelConverter to use during conversion.

        :param str graph_path: Path of the tensorflow model.
        :param List[str] input_nodes_names: List of input node names.
        :param List[int] input_nodes_shapes: List of input node shapes.
        :param List[str] out_node_names: List of output node names.
        :param str saved_model_tag: Saved model tag for conversion, defaults to ""
        :param str saved_model_signature_key: Saved model signature key for
            conversion, defaults to ""
        :raises ConverterError: Number of input names and input shapes are
            not equal.
        :raises ConverterError: Rank of model's input shape and user provided
            input shape is not same.
        :raises ConverterError: User provided input shape is not correctly
            formatted.
        """
        if len(input_nodes_names) != len(input_nodes_shapes):
            raise ConverterError(
                code_to_message.get_error_message(
                    "ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH"
                )
            )

        graph_def = self.__import_graph(
            graph_path, out_node_names, saved_model_tag, saved_model_signature_key
        )
        node_mapping = util.get_node_mapping_dict(graph_def)

        inferred_model_input_names = set(
            [util.remove_suffix(name) for name in util.get_input_info(graph_def).keys()]
        )
        inferred_model_output_names = set(
            [
                util.remove_suffix(name)
                for name in util.get_output_info(graph_def).keys()
            ]
        )
        input_tensor_info = {}
        with self.session.graph.as_default():
            inputs = []
            for name, shape in zip(input_nodes_names, input_nodes_shapes):
                self.__assert_node_in_graph(graph_def, name)
                input_tensor = self.session.graph.get_tensor_by_name(
                    GraphHelper.indexed_tensor_name(name)
                )

                batched_shape = []
                try:
                    tensor_shape = input_tensor.get_shape().as_list()
                    input_shape = list(map(int, shape.split(",")))
                    if len(input_shape) != len(tensor_shape):
                        raise ConverterError(
                            code_to_message.get_error_message(
                                "ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH"
                            )
                        )
                    batched_shape = [1] * len(tensor_shape)
                    batched_shape[-len(input_shape) :] = input_shape
                except ValueError:
                    # This means the tensor has None shape.
                    pass

                if len(batched_shape) == 0:
                    try:
                        batched_shape = list(map(int, shape.split(",")))
                    except ValueError:
                        raise ConverterError(
                            code_to_message.get_error_message(
                                "ERROR_TF_INVALID_INPUT_DIMS"
                            )(shape)
                        )

                # FIXME: Changing the shape of graph input is required but due to that
                #        few models started to fail for batchsize > 1.
                # # Update the input shape with user provided shape value in graph_def.
                # if (
                #     name in inferred_model_input_names
                #     and "shape" in node_mapping[name].attr.keys()
                # ):
                #     updated_shape_proto = tf.TensorShape(dims=batched_shape).as_proto()
                #     node_mapping[name].attr["shape"].shape.CopyFrom(updated_shape_proto)
                # input_tensor_info[name] = batched_shape

                inputs.append(Model.Input(name, batched_shape))

            visitable_graph = VisitableGraph(
                self.__get_graph_operations(graph_def, self.session.graph)
            )
            visitable_graph.accept(GraphPrinter())

            model_name = os.path.splitext(os.path.basename(graph_path))[0]

        # FIXME: If the user is providing intermediate tensor as graph's input
        #        then we need to get the subgraph as per user provided inputs
        #        and outputs. But due to that few models started to fail for
        #        batchsize > 1.
        # if (set(input_nodes_names) != inferred_model_input_names) or (
        #     set(out_node_names) != inferred_model_output_names
        # ):
        #     graph_def = util.extract_subgraph(
        #         graph_def, input_tensor_info, out_node_names
        #     )
        # Since we have updated input shape in graph_def, we need to
        # update the session.graph as per the graph_def.
        temp_graph = util.graphdef_to_graph(graph_def)
        self.session = tf_compat_v1.Session(graph=temp_graph)

        self.graph_def = graph_def
        self.inputs = inputs
        self.out_nodes_names = out_node_names
        self.saved_model_tag = saved_model_tag
        self.saved_model_signature_key = saved_model_signature_key
        self.model_name = model_name

    def __get_graph_operations(
        self, graph_def: GRAPH_DEF, graph: GRAPH
    ) -> List[Operation]:
        """
        Get the list of operations in the graph.

        :param GRAPH_DEF graph_def: Reference graph_def instance.
        :param GRAPH graph: Corresponding graph instance.
        :return List[Operation]: List of operations present in the graph.
        """
        ops = [graph.get_operation_by_name(node.name) for node in graph_def.node]
        return ops

    def __prepare_savedmodel_functions(self) -> None:
        """
        Helper functions for saved model conversion into graph def.
        """
        temp_session = tf_compat_v1.Session(graph=tf_compat_v1.Graph())
        if is_tf2():
            from tensorflow.python.framework.convert_to_constants import (
                convert_variables_to_constants_v2,
            )
            from tensorflow.python.saved_model.load import load as saved_model_load

            self.savedmodel_path_check = tf.saved_model.contains_saved_model
            self.savedmodel_load = lambda path, tags=None: saved_model_load(
                path, tags=tags
            )
            self.savedmodel_get_signature = (
                lambda imported_model: imported_model.signatures
            )
            self.savedmodel_get_version = (
                lambda imported_model: imported_model.tensorflow_version
            )

            def savedmodel_get_inputs_outputs(func):
                # get the defined inputs and outputs names from model. ignore tf.resource data type ones
                inputs = [
                    tensor.name
                    for tensor in func.inputs
                    if tensor.dtype != tf.dtypes.resource
                ]
                outputs = [
                    tensor.name
                    for tensor in func.outputs
                    if tensor.dtype != tf.dtypes.resource
                ]
                return inputs, outputs

            self.savedmodel_get_inputs_outputs = savedmodel_get_inputs_outputs

            def savedmodel_convert_variables_to_constants(func, inputs, outputs):
                frozen_concrete_func = convert_variables_to_constants_v2(
                    func, lower_control_flow=False
                )
                graph_def = frozen_concrete_func.graph.as_graph_def()

                # replace outputs name by the name defined in savedmodel
                name_mapping = tf.nest.pack_sequence_as(
                    func.graph.structured_outputs,
                    frozen_concrete_func.graph.structured_outputs,
                )
                name_map = {}
                for key in name_mapping:
                    name_map[name_mapping[key].name.lstrip("^").split(":")[0]] = key
                for node in graph_def.node:
                    if node.name in name_map:
                        old_name = node.name
                        new_name = name_map.get(node.name)
                        node.name = new_name
                        log_info(
                            code_to_message.get_progress_message(
                                "INFO_TF_CHANGE_NODE_NAME"
                            )(old_name, new_name)
                        )
                        # replace related node's input
                        for _node in graph_def.node:
                            for idx, input_name in enumerate(_node.input):
                                if input_name == old_name:
                                    _node.input[idx] = new_name
                return graph_def

            self.savedmodel_convert_variables_to_constants = (
                savedmodel_convert_variables_to_constants
            )
            self.disable_eager_execution = tf_compat_v1.disable_eager_execution
            self.enable_eager_execution = tf_compat_v1.enable_eager_execution
        else:
            from tensorflow.python.framework.graph_util import (
                convert_variables_to_constants,
            )
            from tensorflow.python.saved_model.loader import load as saved_model_load

            self.savedmodel_path_check = (
                tf.saved_model.loader.maybe_saved_model_directory
            )
            self.savedmodel_load = lambda path, tags=None: saved_model_load(
                temp_session, tags, path
            )
            self.savedmodel_get_signature = (
                lambda imported_model: imported_model.signature_def
            )
            self.savedmodel_get_version = (
                lambda imported_model: imported_model.meta_info_def.tensorflow_version
            )

            def savedmodel_get_inputs_outputs(func):
                # get the defined inputs and outputs names from model. ignore tf.resource data type ones
                inputs = [
                    tensor.name.split(":")[0]
                    for _, tensor in func.inputs.items()
                    if tensor.dtype != tf.resource.as_datatype_enum
                ]
                outputs = [
                    tensor.name.split(":")[0]
                    for _, tensor in func.outputs.items()
                    if tensor.dtype != tf.resource.as_datatype_enum
                ]
                return inputs, outputs

            self.savedmodel_get_inputs_outputs = savedmodel_get_inputs_outputs

            def savedmodel_convert_variables_to_constants(func, inputs, outputs):
                graph_def = convert_variables_to_constants(
                    temp_session,
                    temp_session.graph.as_graph_def(add_shapes=True),
                    outputs,
                )
                return graph_def

            self.savedmodel_convert_variables_to_constants = (
                savedmodel_convert_variables_to_constants
            )
            self.disable_eager_execution = lambda: None
            self.enable_eager_execution = lambda: None

    def __import_graph(
        self,
        graph_path: str,
        out_nodes_names: List[str],
        saved_model_tag: str,
        saved_model_signature_key: str,
    ) -> GRAPH_DEF:
        """
        Import the tensorflow model as graph_def from saved model or
        frozen graph or ckpt-meta files.

        :param str graph_path: Path of the model file/files.
        :param List[str] out_nodes_names: List of output node names.
        :param str saved_model_tag: Saved model tag for conversion, defaults to ""
        :param str saved_model_signature_key: Saved model signature key for
        :raises ConverterError: If the model doesn't exists at the provided path.
        :raises ConverterError: If the model path for saved model is not recognized.
        :raises ConverterError: If the model path doesn't contain any tensorflow
            compatible models.
        :raises ConverterError: If the model has no nodes in the graph.
        :return GRAPH_DEF: GraphDef representation of the given tensorflow model.
        """
        if not os.path.exists(graph_path):
            raise ConverterError(
                code_to_message.get_error_message("ERROR_TF_GRAPH_FILE_DOES_NOT_EXIST")(
                    graph_path
                )
            )

        graph_path = os.path.abspath(graph_path)
        # SavedModel
        if os.path.isdir(graph_path):
            self.__prepare_savedmodel_functions()
            if self.savedmodel_path_check(graph_path):
                graph_def = self.__import_from_savedmodel(
                    graph_path, saved_model_tag, saved_model_signature_key
                )
            else:
                raise ConverterError(
                    code_to_message.get_error_message(
                        "ERROR_TF_GRAPH_PATH_CANT_BE_RECOGNIZED"
                    )(graph_path)
                )
        # Frozen Graph (pb)
        elif graph_path.endswith(".pb"):
            graph_def = self.__import_from_frozen_graph(graph_path)
        # CheckPoint + MetaGraph
        elif graph_path.endswith(".meta"):
            checkpoint = graph_path.split(".meta")[0]
            graph_def = self.__import_from_meta_graph(
                graph_path, checkpoint, out_nodes_names
            )
        else:
            raise ConverterError(
                code_to_message.get_error_message(
                    "ERROR_TF_GRAPH_PATH_CANT_BE_RECOGNIZED"
                )(graph_path)
            )

        if len(graph_def.node) == 0:
            raise ConverterError(
                code_to_message.get_error_message("ERROR_TF_NODES_NOT_FOUND_IN_GRAPH")
            )

        with self.session.graph.as_default():
            tf_compat_v1.import_graph_def(graph_def, name="")
        return graph_def

    def __import_from_savedmodel(
        self, savedmodel_path: str, tag: str, signature_key: str
    ) -> GRAPH_DEF:
        """
        Convert the saved model into a graph_def representation.

        :param str savedmodel_path: Path of the directory containing saved model.
        :param str tag: Saved model tag for conversion.
        :param str signature_key: Saved model signature key for conversion.

        :raises ConverterError: If the saved model has no signatures in it.
        :return GRAPH_DEF: GraphDef representation of the saved model.
        """
        tags = [tag]
        self.enable_eager_execution()
        try:
            imported_model = self.savedmodel_load(savedmodel_path, tags=tags)
        except:
            log_warning(
                code_to_message.get_warning_message("WARNING_TF_USE_FIRST_META_GRAPH")(
                    tags
                )
            )
            imported_model = self.savedmodel_load(savedmodel_path)
        imported_signatures = self.savedmodel_get_signature(imported_model)
        imported_signatures_keys = imported_signatures.keys()
        imported_model_version = self.savedmodel_get_version(imported_model)

        if tf.__version__.split(".")[0] != imported_model_version[0]:
            log_warning(
                code_to_message.get_warning_message(
                    "WARNING_TF_MODEL_VERSION_DOES_NOT_MATCHED"
                )(imported_model_version, tf.__version__)
            )

        if len(imported_signatures_keys) == 0:
            raise ConverterError(
                code_to_message.get_error_message(
                    "ERROR_TF_SIGNATURES_EMPTY_IN_SAVEDMODEL"
                )(savedmodel_path)
            )

        if signature_key not in imported_signatures_keys:
            input_signature_key = signature_key
            signature_key = list(imported_signatures_keys)[0]
            log_warning(
                code_to_message.get_warning_message(
                    "WARNING_TF_USE_FIRST_SIGNATURE_KEY"
                )(input_signature_key, signature_key)
            )

        func = imported_signatures[signature_key]
        inputs, outputs = self.savedmodel_get_inputs_outputs(func)
        log_debug(
            code_to_message.get_progress_message("INFO_INPUT_OUTPUT_FROM_SAVEDMODEL")(
                inputs, outputs
            )
        )

        graph_def = self.savedmodel_convert_variables_to_constants(
            func, inputs, outputs
        )
        self.disable_eager_execution()
        return graph_def

    def __import_from_frozen_graph(self, graph_path: str) -> GRAPH_DEF:
        """
        Read the frozen model from given path as graph def.

        :param str graph_path: Path of the frozen model.
        :return GRAPH_DEF: GraphDef representation of the frozen model.
        """
        graph_def = tf_compat_v1.GraphDef()
        with open(graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    def __import_from_meta_graph(
        self, meta_graph_path: str, graph_path: str, out_nodes_names: List[str]
    ) -> GRAPH_DEF:
        """
        Convert the meta graph and checkpoint file representation into a
        graph_def representation.

        :param str meta_graph_path: Path of the meta graph file.
        :param str graph_path: Path of the checkpoint file.
        :param List[str] out_nodes_names: List of output node names.
        :raises ConverterError: If the graph is not imported from meta.
        :raises ConverterError: If the meta graph is empty.
        :return GRAPH_DEF: GraphDef representation of the meta graph.
        """
        # Need to create a separate session and graph to load the model and extract
        # graphdef without altering the existing self.session.
        temp_session = tf_compat_v1.Session(graph=tf_compat_v1.Graph())
        with temp_session.graph.as_default():
            try:
                saver = tf_compat_v1.train.import_meta_graph(meta_graph_path)
            except AssertionError as e:
                raise ConverterError(
                    code_to_message.get_error_message(
                        "ERROR_TF_CANNOT_IMPORT_GRAPH_FROM_META"
                    )(e.message)
                )

            if saver is None:
                raise ConverterError(
                    code_to_message.get_error_message("ERROR_TF_GRAPH_META_EMPTY")
                )
            saver.restore(temp_session, graph_path)

        graph_def = temp_session.graph.as_graph_def(add_shapes=True)
        return self.__freeze_graph(graph_def, temp_session, out_nodes_names)

    def __freeze_graph(
        self, graph_def: GRAPH_DEF, session: Session, out_nodes_names: List[str]
    ) -> GRAPH_DEF:
        """
        Convert the variables in the given graph def into constants.

        :param GRAPH_DEF graph_def: Reference GraphDef to be freezed.
        :param Session session: Session containing the variables.
        :param List[str] out_nodes_names: List of output node names.
        :return GRAPH_DEF: Frozen graph_def representation.
        """
        for node_name in out_nodes_names:
            self.__assert_node_in_graph(graph_def, node_name)
        frozen = tf_compat_v1.graph_util.convert_variables_to_constants(
            session, graph_def, out_nodes_names
        )
        return frozen

    def __assert_node_in_graph(self, graph_def: GRAPH_DEF, node_name: str) -> None:
        """
        Checks whether given node name is present in the graph_def.

        :param GRAPH_DEF graph_def: Graph_def reference.
        :param str node_name: Name of the node to be checked.
        :raises ConverterError: If node name not found in the graph_def.
        """
        if node_name not in [node.name for node in graph_def.node]:
            raise ConverterError(
                code_to_message.get_error_message("ERROR_TF_NODE_NOT_FOUND_IN_GRAPH")(
                    node_name
                )
            )


class ModelLoader(FrameworkModelLoader):
    def __init__(self, args: Any, custom_op_factory: Optional[CustomOpFactory]):
        """
        Create Tensorflow loader instance.

        :param Any args: Args related to conversion commands.
        :param Optional[CustomOpFactory] custom_op_factory: CustomOp Factory
            object to be used for conversion. defaults to None
        """
        (in_names, in_dims) = list(zip(*args.input_dim))

        self._model = Model(
            args.input_network,
            in_names,
            in_dims,
            args.out_names,
            args.saved_model_tag,
            args.saved_model_signature_key,
        )

        super().__init__("tf", custom_op_factory, args)
        self.utils = TFModelUtils(self)

    def clone_loader(self):
        """
        Clones the ModelLoader object along with the _model instance.

        :return ModelLoader: Returns the deep copied ModelLoader instance.
        """
        loader = copy.deepcopy(self)
        return loader

    def update_model(self, graph: Union[util.GRAPH, util.GRAPH_DEF]):
        """
        Populate Model properties from given tf Graph or tf GraphDef.
        Parameters: graph : Reference tensorflow Graph or Graph Def
        """
        if isinstance(graph, tf_compat_v1.GraphDef):
            self._model.graph_def = util.copy_graphdef(graph)
            temp_graph = util.graphdef_to_graph(self._model.graph_def)
        elif isinstance(graph, tf_compat_v1.Graph):
            temp_graph = graph
            self._model.graph_def = graph.as_graph_def(add_shapes=True)
        else:
            raise RuntimeError(
                f"Can't update the model based on given graph of type {type(graph)}"
            )

        self._model.session = tf_compat_v1.Session(graph=temp_graph)
        status = self.native_checker()
        if not status:
            raise RuntimeError(
                "Failed to update the given loader's model from provided model."
            )

        updated_inputs = []
        for input_name, input_info in self.get_input_info().items():
            updated_inputs.append(
                Model.Input(util.remove_suffix(input_name), input_info.shape)
            )
        self._model.inputs = updated_inputs
        self._model.out_nodes_names = [
            util.remove_suffix(name) for name in self.get_output_names()
        ]

    def get_model(self) -> Model:
        """
        Get the Model instance from loader.

        :return Model: Tensorflow Model object.
        """
        return self._model

    @property
    def model(self) -> Model:
        """
        Get the Model instance from loader.

        :return Model: Tensorflow Model object.
        """
        return self._model

    def get_nodes_by_op_type(self, op_type: str) -> List[NODE_DEF]:
        """
        Fetch all the nodes of given type.

        :param str op_type: Node op type
        :return List[NODE_DEF]: List of all the nodes belong to given op type.
        """
        return util.get_nodes_by_op_type(self._model.graph_def, op_type)

    def get_nodes(self) -> List[NODE_DEF]:
        """
        Fetch all the nodes present in the graph.

        :return List[NODE_DEF]: List of all the nodes from graph.
        """
        return util.get_nodes(self._model.graph_def)

    def get_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Get the dict of input tensor name to corresponding tensor.

        :return Dict[str, tf.Tensor]: Dict with input tensor name as key and
            input tensor as value.
        """
        return util.extract_inputs_from_graph(self._model.session.graph)

    def get_outputs(self) -> Dict[str, tf.Tensor]:
        """
        Get the dict of output tensor name to corresponding tensor.

        :return Dict[str, tf.Tensor]: Dict with output tensor name as key and
            output tensor as value.
        """
        return util.extract_outputs_from_graph(self._model.session.graph)

    def get_input_names(self) -> List[str]:
        """
        Get the input names from the model.

        :return List[str]: List of all the input names.
        """
        return list(self.get_input_info().keys())

    def get_output_names(self) -> List[str]:
        """
        Get the output names from the model.

        :return List[str]: List of all the output names.
        """
        return list(self.get_output_info().keys())

    def get_input_info(self) -> Dict[str, TensorInfo]:
        """
        Get input tensor info from graph. E.g.Input name, shape, dtype and
        its layout.

        :return Dict[str, TensorInfo]: Dict mapping of tensor name to its
            TensorInfo object.
        """
        return util.get_input_info(self._model.graph_def)

    def get_output_info(self) -> Dict[str, TensorInfo]:
        """
        Get output tensor info from graph. E.g.Output name, shape, dtype and
        its layout.

        :return Dict[str, TensorInfo]: Dict mapping of tensor name to its
            TensorInfo object.
        """
        return util.get_output_info(self._model.graph_def)

    def get_custom_io_container(self):
        pass

    def get_parent_nodes(self, node_name: str) -> List[tf_compat_v1.NodeDef]:
        """
        Get the list of all the parent nodes of the given node name.

        :param str node_name: Name of the node.
        :return List[tf_compat_v1.NodeDef]: List of parent nodes.
        """
        return util.get_parent_nodes(self._model.graph_def, node_name)

    def get_children_nodes(self, node_name: str) -> List[tf_compat_v1.NodeDef]:
        """
        Get the list of all the child nodes of the given node name.

        :param str node_name: Name of the node.
        :return List[tf_compat_v1.NodeDef]: List of children nodes.
        """
        return util.get_children_nodes(self._model.graph_def, node_name)

    def get_node_by_name(self, node_name: str) -> Union[None, tf_compat_v1.NodeDef]:
        """
        Get the node by given name from the graph.

        :param str node_name: Node name.
        :return tf_compat_v1.NodeDef: Node reference if found else None.
        """
        for node in self.get_nodes():
            if node.name == node_name:
                return node
        return None

    def native_checker(self) -> bool:
        """
        Check the validity of the model graph.

        :return bool: Boolean value indicating the success/failure of the checker
        """
        return util.graph_checker(self._model.graph_def)

    def save_model(self, model_dir: str, model_name: str = None):
        """
        Save the model in frozen graph format at given location.

        :param str model_dir: Directory for the model
        :param str model_name: Name of the model, defaults to None
        """
        if (model_name == "") or (model_name is None):
            model_name = "frozen_model.pb"
        util.save_as_frozen_graph(self._model.graph_def, model_dir, model_name)

    def summarize_model(self) -> FrameworkSummary:
        """
        Function to get the summary about the model. e.g. Name of the model, its
        operations, total parameters etc.

        :return FrameworkSummary: Model summary instance.
        """
        self.utils.native_shape_inference()
        graph = util.graphdef_to_graph(self._model.graph_def)
        total_params = util.get_model_params(self._model.graph_def)
        ops_counter = util.get_unique_ops(graph)

        input_specs_dict = {}
        output_specs_dict = {}
        for input_name, input_tensor_info in util.get_input_info(
            self._model.graph_def
        ).items():
            input_specs_dict[input_name] = (
                input_tensor_info.shape,
                "input",
                input_name,
            )
        for output_name, output_tensor_info in util.get_output_info(
            self._model.graph_def
        ).items():
            output_specs_dict[output_name] = (
                output_tensor_info.shape,
                "output",
                output_name,
            )

        summary = FrameworkSummary(
            model_name=self._model.model_name,
            total_parameters=total_params,
            ops_counter=ops_counter,
            inp_specs=input_specs_dict,
            out_specs=output_specs_dict,
        )
        return super().summarize_model(summary)
