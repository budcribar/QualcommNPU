# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

try:
    from . import qnn_definitions
    from . import ir_graph
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that product python lib is in your PYTHONPATH")
    sys.exit(1)


from qti.aisw.converters.common.backend_base import BackendTranslationBase
from qti.aisw.converters.common.converter_ir import translation
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings, InputType
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import log_assert, log_debug3, log_info
from .qnn_mappings import *

QnnTranslations = translation.TranslationBank()


# ------------------------------------------------------------------------------
#   Translations
# ------------------------------------------------------------------------------
def register(qnn_translation):
    # Allows more than one target to be specified per class
    if isinstance(qnn_translation.TARGET, tuple) or isinstance(qnn_translation.TARGET, list):
        QnnTranslations.register_translation(qnn_translation(), *qnn_translation.TARGET)
    else:
        QnnTranslations.register_translation(qnn_translation(), qnn_translation.TARGET)
    return qnn_translation


class QnnTranslationBase(BackendTranslationBase):
    # Reasonable limit of what makes visual sense to store in model.cpp vs model.bin
    # Note: applies to static input tensors, at this time all node tensor params are required in model.cpp
    MAX_INPUT_TENSOR_BYTES_IN_CPP = 100

    def __init__(self):
        super(QnnTranslationBase, self).__init__()

    def add_op_to_backend(self, node, ir_graph, backend, **kwargs):
        return NotImplementedError("add_op_to_backend not implemented for {}".format(self.__class__.__name__))

    @staticmethod
    def squash_relu(node, graph, backend, outputs_info):
        """
        Squash Relux Op to previous node if applicable (only valid for quantized case as its for
        DSP/HTP runtime
        :return: True if layer is squashed, false otherwise
        """

        # custom ops should not be squashed
        if len(node.output_names) == 1:
            output_buf = graph.get_buffer(node.output_names[0])
            # Prune Relux Op
            if (not backend.is_online_construction and backend.c_ir_graph) and len(output_buf.consumers) == 1:
                consumer_node = list(output_buf.consumers)[0]
                if consumer_node.op.type == op_adapter.ElementwiseNeuronOp.TRANSLATION_KEY and \
                        (consumer_node.op.operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU or
                         consumer_node.op.operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX) \
                        and not backend.check_qnn_type_is_custom(consumer_node.op.type):
                    # Fold Relux
                    graph.squash(consumer_node, input_name=output_buf.name)

                    consumer_outputs_info = backend.get_outputs_info(consumer_node, graph)
                    outputs_info[0]["type"] = consumer_outputs_info[0]["type"]

                    # updated squashed output name and info
                    backend.update_tensors_info(outputs_info[0], node.output_names[0])
                    return True
        return False

    @staticmethod
    def get_pad_size_c(padding_size_strategy, total_pad_amount, pad_before, pad_after):
        """
        Get explict pad amounts since QNN Ops do not take padding strategy
        return: pad_before, pad_end after applying padding strategy
        """
        if total_pad_amount < 0:
            # if negative total pad_amount unable to traceback exact pad before and after. This is likely
            # caused because the output dim is some decimal value that is floored or ceiled and stride > filter.
            # Return IR provided pad before and after
            return pad_before, pad_after

        if padding_size_strategy == ir_graph.PADDING_SIZE_IMPLICIT_VALID:
            log_assert(total_pad_amount == 0,
                       "Padding Strategy IMPLICIT_VALID requires 0 pad value. Got {}".format(total_pad_amount))
            pad_before = pad_after = 0
        elif padding_size_strategy == ir_graph.PADDING_SIZE_IMPLICIT_SAME_BEGIN:
            pad_after = total_pad_amount // 2
            pad_before = total_pad_amount - pad_after
        elif padding_size_strategy == ir_graph.PADDING_SIZE_IMPLICIT_SAME_END:
            pad_before = total_pad_amount // 2
            pad_after = total_pad_amount - pad_before
        elif padding_size_strategy == ir_graph.PADDING_SIZE_EXPLICIT_RIGHTHANDED:
            pad_before = 0
            pad_after = total_pad_amount
        elif padding_size_strategy == ir_graph.PADDING_SIZE_EXPLICIT:
            log_assert(((pad_before + pad_after) == total_pad_amount or
                        (pad_before + pad_after + 1) == total_pad_amount),
                       "Explicit pad values ({}, {}) do not result in expected pad_value ({}) to calculate Op"
                       " output dim".format(pad_before, pad_after, total_pad_amount))
            if (pad_before + pad_after + 1) == total_pad_amount:
                # Here ceil method used when calculating output dims, which might result in pad amount being off by one.
                # To account for that we *add* the extra pad to pad_after(bottom and right).
                pad_after += 1
        elif padding_size_strategy == ir_graph.PADDING_SIZE_EXPLICIT_FLOOR:
            log_assert(((pad_before + pad_after) == total_pad_amount or
                        (pad_before + pad_after - 1) == total_pad_amount),
                       "Explicit pad values ({}, {}) do not result in expected pad_value ({}) to calculate "
                       "Op output dim".format(pad_before, pad_after, total_pad_amount))
            if (pad_before + pad_after - 1) == total_pad_amount:
                # Here floor method used when calculating output dims, which might result in pad amount being
                # off by one. To account for that we *remove* the extra pad to pad_after(bottom and right).
                pad_after -= 1
        else:
            raise ValueError("Padding Strategy ({}) not supported in QNN converter.".format(padding_size_strategy))

        return pad_before, pad_after


@register
class QnnInputTranslation(QnnTranslationBase):
    TARGET = op_adapter.InputOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        def check_consumer_is_cast_to_float(consumer):
            if (consumer.op.type == op_adapter.CastOp.TRANSLATION_KEY and
                np.dtype(consumer.op.to_type) == np.dtype('float32')):
                return True
            return False

        # check dtype
        dtype_to_np = np.dtype(node.op.input_dtype)
        if dtype_to_np not in numpy_dtype_to_qnn:
            raise TypeError("Input node: {} has type: {} which is not supported in QNN converter"
                            .format(node.op.name, dtype_to_np))

        # check encodings
        check_encodings = True
        if node.op.input_type == InputType.OPAQUE or \
                (numpy_dtype_to_qnn.get(dtype_to_np) != ir_graph.QNN_DATATYPE_FLOAT_32 and
                 # If consumer is a Cast Op to Float, then quant params of Input and Cast will match
                 # Check encoding of input in that case, so that the Cast can be optimized out as an identity
                 not all(map(check_consumer_is_cast_to_float, graph.get_op_output_nodes(node)))):
            check_encodings = False

        # retrieve source and QNN tensor axis format from output buffer
        src_axis_format = graph.get_buffer(node.output_names[0]).get_src_axis_format()
        tensor_axis_format = graph.get_buffer(node.output_names[0]).get_axis_format()

        custom_quant_params = None
        for entry in graph.user_custom_io:
            if entry['IOName'] == node.output_names[0] and 'QuantParam' in entry:
                custom_quant_params = entry['QuantParam']

        if graph.user_custom_io and custom_quant_params is not None:
            # In case when quantized inputs are passed to the model, the c_ir_graph is None. Hence,
            # this block creates a quant_params dictionary with the scale and offset information and calls the
            # add_custom_input_tensor() method instead of the add_tensor() method.
            quant_params = {"definition": qnn_definitions.QNN_DEFINITION_DEFINED, "encoding": ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                 "scale_offset": {"scale": graph.quantization_params[node.op.name]['output_encodings'][0]['scale'][0],
                                 "offset": graph.quantization_params[node.op.name]['output_encodings'][0]['offset'][0]}
                            }
            backend.add_custom_input_tensor(node.op.name, node.output_names[0], qnn_definitions.QNN_TENSOR_TYPE_APP_WRITE, node.op,
                            tensor_data_type=numpy_dtype_to_qnn.get(dtype_to_np), tensor_axis_format=tensor_axis_format,
                            quant_params = quant_params)
        else:
            backend.add_tensor(node.op.name, node.output_names[0], qnn_definitions.QNN_TENSOR_TYPE_APP_WRITE, node.op,
                               tensor_data_type=numpy_dtype_to_qnn.get(dtype_to_np), src_axis_format=src_axis_format,
                               tensor_axis_format=tensor_axis_format, check_encodings=check_encodings)


@register
class QnnArgMaxArgMinTranslation(QnnTranslationBase):
    TARGET = op_adapter.ArgOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.arg_type == ir_graph.QNN_OP_ARGMAX:
            axis_key = ir_graph.QNN_OP_ARGMAX_PARAM_AXIS
            keep_dims_key = ir_graph.QNN_OP_ARGMAX_PARAM_KEEP_DIMS
        else:
            axis_key = ir_graph.QNN_OP_ARGMIN_PARAM_AXIS
            keep_dims_key = ir_graph.QNN_OP_ARGMIN_PARAM_KEEP_DIMS
        backend.add_node(node.op.name, node.op.type, node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_INT_32,
                                                  check_encodings=False),
                         scalar_params={axis_key:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis)),
                                        keep_dims_key:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.keep_dims)),
                                        ir_graph.IR_OP_ARG_TYPE:
                                            (ir_graph.QNN_DATATYPE_UNDEFINED,
                                             str(node.op.arg_type),
                                             ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                        })


@register
class QnnAxisAlignedBboxTransformTranslation(QnnTranslationBase):
    TARGET = op_adapter.AxisAlignedBboxTransformOp.TRANSLATION_KEY

    def get_output_tensor_info(self, backend, node, graph):
        outputs_info = []
        for i, output_name in enumerate(node.output_names):
            # determine data types per spec
            # boxes
            if i == 0:
                tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            # batch_splits
            else:
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32

            check_encodings = tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32 or tensor_data_type in qnn_quantized_types

            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            outputs_info.append(output_tensor_info)
        return outputs_info

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        weights = np.asarray(node.op.weights, dtype=np.float32)
        weights_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                    ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM_PARAM_WEIGHTS)
        weights_tensor_info = backend.create_tensor_info(weights_tensor_name,
                                                         qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                         [len(weights)],
                                                         ir_graph.QNN_DATATYPE_FLOAT_32,
                                                         data=weights, quantizable=False)
        outputs_info = self.get_output_tensor_info(backend, node, graph)
        backend.add_node(node.op.name, ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM,
                         node.input_names,
                         outputs_info,
                         macs=node.op.macs,
                         tensor_params={
                             ir_graph.QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM_PARAM_WEIGHTS: weights_tensor_info})


@register
class QnnBatchnormTranslation(QnnTranslationBase):
    TARGET = op_adapter.BatchnormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):

        # Squash Relux if applicable
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)
        self.squash_relu(node, graph, backend, outputs_info)

        # add node for bn
        backend.add_node(node.op.name, node.op.type,
                         input_names=node.input_names,
                         outputs_info=outputs_info,
                         macs=node.op.macs)


@register
class QnnBatchPermutationTranslation(QnnTranslationBase):
    TARGET = op_adapter.BatchPermutationOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        outputs_info = backend.get_outputs_info(node, graph)
        backend.add_node(node.op.name, node.op.type, node.input_names, outputs_info)


@register
class QnnBatchToSpaceTranslation(QnnTranslationBase):
    TARGET = op_adapter.BatchToSpaceOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        block_size_tensor_name = backend.create_unique_qnn_tensor_name(
            node.op.name, qnn_definitions.QNN_OP_BATCH_TO_SPACE_PARAM_BLOCK_SIZE)
        block_size = np.array(node.op.block_shape, dtype=np.uint32)
        if block_size.shape != (2,):
            raise ValueError("Invalid block size shape on BatchToSpace node {}, expected: (2,), got: {}".format(
                node.op.name, block_size.shape))
        block_size_tensor_info = backend.create_tensor_info(block_size_tensor_name,
                                                            qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                            [2], ir_graph.QNN_DATATYPE_UINT_32,
                                                            data=block_size)

        crops_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                  qnn_definitions.QNN_OP_BATCH_TO_SPACE_PARAM_CROPS)
        crops = np.array(node.op.crops, dtype=np.uint32)
        if crops.shape != (2, 2):
            raise ValueError("Invalid crops shape on BatchToSpace node {}, expected: (2, 2), got: {}".format(
                node.op.name, crops.shape))
        crops_tensor_info = backend.create_tensor_info(crops_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                       [2, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                                       data=crops)

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_BATCH_TO_SPACE,
                         input_names=node.input_names,
                         outputs_info=backend.get_outputs_info(node, graph),
                         tensor_params={qnn_definitions.QNN_OP_BATCH_TO_SPACE_PARAM_BLOCK_SIZE: block_size_tensor_info,
                                        qnn_definitions.QNN_OP_BATCH_TO_SPACE_PARAM_CROPS: crops_tensor_info})


@register
class QnnBoxWithNmsLimitTranslation(QnnTranslationBase):
    TARGET = op_adapter.BoxWithNmsLimitOp.TRANSLATION_KEY

    def get_output_tensor_info(self, backend, node, graph):
        outputs_info = []
        for i, output_name in enumerate(node.output_names):
            # determine data types per spec
            # boxes
            if i == 0:
                tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            # scores
            elif i == 1:
                tensor_data_type = backend.retrieve_tensor_info(node.input_names[1])["data_type"]
            # classes, batch_indices, batch_splits, keeps, keeps_size
            else:
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32

            check_encodings = tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32 or tensor_data_type in qnn_quantized_types

            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            outputs_info.append(output_tensor_info)
        return outputs_info

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        outputs_info = self.get_output_tensor_info(backend, node, graph)
        backend.add_node(node.op.name, ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT,
                         node.input_names,
                         outputs_info,
                         scalar_params={ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_KERNEL_METHOD:
                                            (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(node.op.nms_kernel_method)),
                                        ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_SCORE_THRESHOLD:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.nms_score_threshold)),
                                        ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SCORE_THRESHOLD:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.score_threshold)),
                                        ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_PRE_NMS_LIMIT:
                                            (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(node.op.pre_nms_limit)),
                                        ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_IOU_THRESHOLD:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.iou_threshold)),
                                        ir_graph.QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SIGMA:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.sigma)),
                                        },
                         macs=node.op.macs)


@register
class QnnCastTranslation(QnnTranslationBase):
    TARGET = op_adapter.CastOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):

        input_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        output_type = numpy_dtype_to_qnn[np.dtype(node.op.to_type)]
        # First, downcast 64-bit types to 32-bit
        downcast_type = output_type

        if output_type == ir_graph.QNN_DATATYPE_INT_64:
            downcast_type = ir_graph.QNN_DATATYPE_INT_32
        elif output_type == ir_graph.QNN_DATATYPE_UINT_64:
            downcast_type = ir_graph.QNN_DATATYPE_UINT_32

        if downcast_type != output_type:
            log_debug3(code_to_message.get_debugging_message("DEBUG_DOWNCAST_TENSOR")
                       (output_type, downcast_type, node.op.name))
            output_type = downcast_type

        check_encodings = True
        if output_type != ir_graph.QNN_DATATYPE_FLOAT_32 and output_type not in qnn_quantized_types:
            check_encodings = False
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=output_type,
                                                check_encodings=check_encodings)
        output_type = outputs_info[0]["data_type"]  # update output type to final one used with quantization considered

        # TODO Remove once removing cast ops is properly handled in IR
        if input_type == output_type and graph.get_op_output_nodes(node):
            graph.squash(node, input_name=node.input_names[0], squash_into_next=True)
            return

        # In case of custom IO, we would want the (instered) cast op's output tensor to retain the scale and offset
        # obtained from the quantizer.
        setScaleOffset = True
        for entry in graph.user_custom_io:
            if entry['IOName'] in node.input_names and backend.c_ir_graph is not None:
                setScaleOffset = False
                break

        if input_type not in qnn_quantized_types and output_type in qnn_quantized_types and setScaleOffset:
            outputs_info[0]["quant_params"]["scale_offset"]["scale"] = 1.0
            outputs_info[0]["quant_params"]["scale_offset"]["offset"] = 0

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_CAST, node.input_names, outputs_info)


@register
class QnnCollectRpnProposalsTranslation(QnnTranslationBase):
    TARGET = op_adapter.CollectRpnProposalsOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MIN_LEVEL:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.rpn_min_level)),
                                        ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MAX_LEVEL:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.rpn_max_level)),
                                        ir_graph.QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_POST_NMS_TOP:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.post_nms_top))
                                        })


@register
class QnnColorTransformTranslation(QnnTranslationBase):
    TARGET = op_adapter.ColorTransformOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        scalar_params = {}
        # determine color transformation type
        if node.op.input_encoding_in == InputEncodings.NV21:
            color_transform_type = qnn_definitions.QNN_OP_NV21_TO_RGB
            output_order = qnn_definitions.QNN_OP_NV21_TO_RGB_OUTPUT_ORDER_RGB
            if node.op.input_encoding_out == InputEncodings.BGR:
                output_order = qnn_definitions.QNN_OP_NV21_TO_RGB_OUTPUT_ORDER_BGR
            scalar_params.update({qnn_definitions.QNN_OP_NV21_TO_RGB_PARAM_OUTPUT_ORDER:
                                      (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(output_order))})
        elif node.op.input_encoding_in == InputEncodings.NV12:
            color_transform_type = qnn_definitions.QNN_OP_NV12_TO_RGB
            output_order = qnn_definitions.QNN_OP_NV12_TO_RGB_OUTPUT_ORDER_RGB
            if node.op.input_encoding_out == InputEncodings.BGR:
                output_order = qnn_definitions.QNN_OP_NV12_TO_RGB_OUTPUT_ORDER_BGR
            scalar_params.update({qnn_definitions.QNN_OP_NV12_TO_RGB_PARAM_OUTPUT_ORDER:
                                      (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(output_order))})
        else:
            color_transform_type = qnn_definitions.QNN_OP_ARGB_TO_RGB
            input_order = qnn_definitions.QNN_OP_ARGB_TO_RGB_INPUT_ORDER_ARGB
            reverse_output = False
            if node.op.input_encoding_in == InputEncodings.RGBA:
                input_order = qnn_definitions.QNN_OP_ARGB_TO_RGB_INPUT_ORDER_RGBA
            if node.op.input_encoding_out == InputEncodings.BGR:
                reverse_output = True
            scalar_params.update({qnn_definitions.QNN_OP_ARGB_TO_RGB_PARAM_INPUT_ORDER:
                                      (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(input_order)),
                                  qnn_definitions.QNN_OP_ARGB_TO_RGB_PARAM_REVERSE_OUTPUT:
                                      (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(reverse_output))})

        backend.add_node(node.op.name, color_transform_type,
                         input_names=node.input_names,
                         outputs_info=backend.get_outputs_info(node, graph),
                         scalar_params=scalar_params)


@register
class QnnConstantTranslation(QnnTranslationBase):
    TARGET = op_adapter.ConstantOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        is_weight_or_bias = False
        is_bias = False
        for consumer in graph.get_buffer(node.output_names[0]).consumers:
            if consumer.op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                    op_adapter.Conv3dOp.TRANSLATION_KEY,
                                    op_adapter.TransposeConv2dOp.TRANSLATION_KEY,
                                    op_adapter.TransposeConv3dOp.TRANSLATION_KEY,
                                    op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY,
                                    op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                                    op_adapter.MatMulOp.TRANSLATION_KEY,
                                    op_adapter.LstmOp.TRANSLATION_KEY]:
                is_weight_or_bias = True
                if graph.get_input_buffer_idx(consumer, node.output_names[0]) == 2:
                    is_bias = True
            elif consumer.op.type in [op_adapter.PreluOp.TRANSLATION_KEY]:
                idx = consumer.input_names.index(node.output_names[0])
                if idx == 1 and np.all([node.op.tensor == np.ndarray.flatten(node.op.tensor)[0]]):
                    # When value was broadcasted(e.g leakyRelu case) revert back to single alpha value
                    node.op.tensor = np.array([np.ndarray.flatten(node.op.tensor)[0]])

        check_encodings = True if node.op.quantizable else False
        store_in_bin = False
        if len(node.op.tensor.tobytes()) > QnnTranslationBase.MAX_INPUT_TENSOR_BYTES_IN_CPP or is_weight_or_bias:
            store_in_bin = True

        # retrieve source and QNN tensor axis format from output buffer
        src_axis_format = graph.get_buffer(node.output_names[0]).get_src_axis_format()
        tensor_axis_format = graph.get_buffer(node.output_names[0]).get_axis_format()

        backend.add_tensor(node.op.name, node.output_names[0], qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                           node.op.tensor, check_encodings=check_encodings, tensor_data_type=numpy_dtype_to_qnn[node.op.tensor.dtype],
                           src_axis_format=src_axis_format, tensor_axis_format=tensor_axis_format,
                           orig_tensor_name="tensor", is_bias=is_bias)


@register
class QnnConvertTranslation(QnnTranslationBase):
    TARGET = op_adapter.ConvertOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_name = node.input_names[0]
        input_tensor_type = backend.retrieve_tensor_info(input_name)["type"]
        supported_input_tensor_types = [qnn_definitions.QNN_TENSOR_TYPE_APP_WRITE]
        if node.op.dynamic_input_data and input_tensor_type not in supported_input_tensor_types:
            raise ValueError("Node {}: Invalid dynamic_input_data parameter for Convert Op. Value can be true only for "
                             "tensors of type QNN_TENSOR_TYPE_APP_WRITE or QNN_TENSOR_TYPE_APP_READWRITE.".format(node.op.name))

        output_name = node.output_names[0]
        output_buf = graph.get_buffer(output_name)
        if node.op.dynamic_output_data and len(output_buf.consumers) != 0:
            raise ValueError("Node {}: Invalid dynamic_output_data parameter for Convert Op. Value can be true only for "
                             "tensors of type QNN_TENSOR_TYPE_APP_READ or QNN_TENSOR_TYPE_APP_READWRITE.".format(node.op.name))

        outputs_info = backend.get_outputs_info(node, graph)

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_CONVERT, node.input_names, outputs_info,
                         scalar_params={qnn_definitions.QNN_OP_CONVERT_PARAM_DYNAMIC_INPUT_DATA:
                                            (numpy_dtype_to_qnn[np.dtype('bool')],
                                             np.bool_(node.op.dynamic_input_data)),
                                        qnn_definitions.QNN_OP_CONVERT_PARAM_DYNAMIC_OUTPUT_DATA:
                                            (numpy_dtype_to_qnn[np.dtype('bool')],
                                             np.bool_(node.op.dynamic_output_data))})


@register
class QnnConvolutionTranslation(QnnTranslationBase):
    TARGET = [op_adapter.Conv2dOp.TRANSLATION_KEY, op_adapter.Conv3dOp.TRANSLATION_KEY, op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]

    def get_conv_params(self, backend, graph, node):
        scalar_params = {}
        num_params = 2
        if node.op.type == op_adapter.Conv2dOp.TRANSLATION_KEY:
            conv_type = ir_graph.QNN_OP_CONV_2D
            stride_param = ir_graph.QNN_OP_CONV_2D_PARAM_STRIDE
            pad_param = ir_graph.QNN_OP_CONV_2D_PARAM_PAD_AMOUNT
            dilation_param = ir_graph.QNN_OP_CONV_2D_PARAM_DILATION
            scalar_params = {ir_graph.QNN_OP_CONV_2D_PARAM_GROUP:
                                 (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.group))
                             }
            if backend.serialize_with_suppl_attr:
                scalar_params.update({ir_graph.IR_OP_CONV_2D_PARAM_PADDING_SIZE_STRATEGY:
                                          (numpy_dtype_to_qnn[np.dtype('uint8')],
                                           np.uint8(node.op.padding_size_strategy),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                      ir_graph.IR_OP_CONV_2D_BIAS_OP_NAME:
                                          (ir_graph.QNN_DATATYPE_UNDEFINED,
                                           str(node.op.bias_op_name),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                      })
        elif node.op.type == op_adapter.Conv3dOp.TRANSLATION_KEY:
            conv_type = ir_graph.QNN_OP_CONV_3D
            stride_param = ir_graph.QNN_OP_CONV_3D_PARAM_STRIDE
            pad_param = ir_graph.QNN_OP_CONV_3D_PARAM_PAD_AMOUNT
            dilation_param = ir_graph.QNN_OP_CONV_3D_PARAM_DILATION
            scalar_params = {ir_graph.QNN_OP_CONV_3D_PARAM_GROUP:
                                 (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.group)),
                             ir_graph.QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES:
                                 (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.reuse_sparse_indicies))
                             }
            num_params = 3
            if backend.serialize_with_suppl_attr:
                scalar_params.update({ir_graph.IR_OP_CONV_3D_PARAM_PADDING_SIZE_STRATEGY:
                                          (numpy_dtype_to_qnn[np.dtype('uint8')],
                                           np.uint8(node.op.padding_size_strategy),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                      ir_graph.IR_OP_CONV_3D_BIAS_OP_NAME:
                                          (ir_graph.QNN_DATATYPE_UNDEFINED,
                                           str(node.op.bias_op_name),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                      })
        else:
            conv_type = ir_graph.QNN_OP_DEPTH_WISE_CONV_2D
            stride_param = ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE
            pad_param = ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT
            dilation_param = ir_graph.QNN_OP_DEPTH_WISE_CONV_2D_PARAM_DILATION
            if backend.serialize_with_suppl_attr:
                scalar_params.update({ir_graph.IR_OP_CONV_2D_PARAM_PADDING_SIZE_STRATEGY:
                                          (numpy_dtype_to_qnn[np.dtype('uint8')],
                                           np.uint8(node.op.padding_size_strategy),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                      ir_graph.IR_OP_CONV_2D_BIAS_OP_NAME:
                                          (ir_graph.QNN_DATATYPE_UNDEFINED,
                                           str(node.op.bias_op_name),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                      })

        # Extract parameters
        strides = np.array(node.op.stride, dtype=np.uint32)
        pads = np.array(node.op.pad_amount, dtype=np.uint32)
        dilation = np.array(node.op.dilation, dtype=np.uint32)

        # get Qnn tensor-info definition for the params and add the actual data
        stride_name = backend.create_unique_qnn_tensor_name(node.op.name, stride_param)
        stride_info = backend.create_tensor_info(stride_name,
                                                 qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                 [num_params], ir_graph.QNN_DATATYPE_UINT_32,
                                                 data=strides)

        pad_name = backend.create_unique_qnn_tensor_name(node.op.name, pad_param)
        pads = np.array([[pad[0], pad[1]] for pad in pads], dtype=np.uint32)
        pad_info = backend.create_tensor_info(pad_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                              [num_params, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                              data=pads)

        dilation_name = backend.create_unique_qnn_tensor_name(node.op.name, dilation_param)
        dilation_info = backend.create_tensor_info(dilation_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                   [num_params], ir_graph.QNN_DATATYPE_UINT_32,
                                                   data=dilation)

        tensor_params = {stride_param: stride_info,
                         pad_param: pad_info,
                         dilation_param: dilation_info}

        if node.op.type == op_adapter.Conv2dOp.TRANSLATION_KEY or \
                node.op.type == op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY:
            if backend.serialize_with_suppl_attr and node.op.c_op.attrs.has(ir_graph.IR_OP_CONV_PARAM_BN_GAMMA) and node.op.c_op.attrs.has(ir_graph.IR_OP_CONV_PARAM_BN_BETA):
                gamma = np.array(node.op.gamma, dtype=np.float32)
                gamma_info = backend.create_tensor_info(ir_graph.IR_OP_CONV_PARAM_BN_GAMMA, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                        [gamma.shape], ir_graph.QNN_DATATYPE_FLOAT_32,
                                                        data=gamma)
                beta = np.array(node.op.beta, dtype=np.float32)
                beta_info = backend.create_tensor_info(ir_graph.IR_OP_CONV_PARAM_BN_BETA,
                                                       qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                       [beta.shape], ir_graph.QNN_DATATYPE_FLOAT_32,
                                                       data=beta)
                tensor_params.update({ir_graph.IR_OP_CONV_PARAM_BN_GAMMA: (gamma_info, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                      ir_graph.IR_OP_CONV_PARAM_BN_BETA: (beta_info, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)})

        return conv_type, tensor_params, scalar_params

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        conv_type, tensor_params, scalar_params = self.get_conv_params(backend, graph, node)

        outputs_info = backend.get_outputs_info(node, graph,
                                                tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)
        # Squash Relux if applicable
        self.squash_relu(node, graph, backend, outputs_info)
        backend.add_node(node.op.name, conv_type,
                         input_names=node.input_names,
                         outputs_info=outputs_info,
                         tensor_params=tensor_params,
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnConcatTranslation(QnnTranslationBase):
    TARGET = op_adapter.ConcatOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, ir_graph.QNN_OP_CONCAT,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_CONCAT_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis))
                                        })


@register
class QnnChannelShuffleTranslation(QnnTranslationBase):
    TARGET = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, ir_graph.QNN_OP_CHANNEL_SHUFFLE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.num_groups)),
                                        ir_graph.QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis))
                                        })


@register
class QnnCreateSparseTranslation(QnnTranslationBase):
    TARGET = op_adapter.CreateSparseOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        output_type = backend.retrieve_tensor_info(node.input_names[1])["data_type"]
        check_encodings = True
        if output_type != ir_graph.QNN_DATATYPE_FLOAT_32 and output_type not in qnn_quantized_types:
            check_encodings = False

        backend.add_node(node.op.name, ir_graph.QNN_OP_CREATE_SPARSE,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=output_type,
                                                  check_encodings=check_encodings))


@register
class QnnCropAndResizeTranslation(QnnTranslationBase):
    TARGET = op_adapter.CropAndResizeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        interpolation_mode = node.op.interpolation_mode

        resize_dims = np.asarray(node.op.resize_dims, dtype=np.uint32)
        resize_dims_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                        qnn_definitions.QNN_OP_CROP_AND_RESIZE_PARAM_RESIZE_DIMS)
        resize_dims_tensor_info = backend.create_tensor_info(resize_dims_tensor_name,
                                                             qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                             [len(resize_dims)],
                                                             ir_graph.QNN_DATATYPE_UINT_32,
                                                             data=resize_dims)
        output_tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_CROP_AND_RESIZE,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=output_tensor_data_type),
                         scalar_params={qnn_definitions.QNN_OP_CROP_AND_RESIZE_PARAM_EXTRAPOLATION_VALUE:
                                            (numpy_dtype_to_qnn[np.dtype('float32')],
                                             np.float32(node.op.extrapolation_value)),
                                        qnn_definitions.QNN_OP_CROP_AND_RESIZE_PARAM_INTERPOLATION_MODE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(interpolation_mode))},
                         tensor_params={
                             qnn_definitions.QNN_OP_CROP_AND_RESIZE_PARAM_RESIZE_DIMS: resize_dims_tensor_info})


@register
class QnnCumSumTranslation(QnnTranslationBase):
    TARGET = op_adapter.CumSumOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_CUMULATIVE_SUM,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={qnn_definitions.QNN_OP_CUMULATIVE_SUM_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis)), # axis needs to be positive
                                        qnn_definitions.QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], node.op.exclusive),
                                        qnn_definitions.QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], node.op.reverse)})


@register
class QnnCustomOpTranslation(QnnTranslationBase):
    TARGET = op_adapter.CustomOp.TRANSLATION_KEY

    @staticmethod
    def convert_custom_dtype_to_qnntype(custom_dtype):
        snpe_qnn_udo_op_dict = {
            "SNPE_UDO_DATATYPE_FLOAT_32":   "QNN_DATATYPE_FLOAT_32",
            "SNPE_UDO_DATATYPE_FLOAT_16":   "QNN_DATATYPE_FLOAT_16",
            "SNPE_UDO_DATATYPE_INT_8":      "QNN_DATATYPE_INT_8",
            "SNPE_UDO_DATATYPE_INT_16":     "QNN_DATATYPE_INT_16",
            "SNPE_UDO_DATATYPE_INT_32":     "QNN_DATATYPE_INT_32",
            "SNPE_UDO_DATATYPE_UINT_8":     "QNN_DATATYPE_UINT_8",
            "SNPE_UDO_DATATYPE_UINT_16":    "QNN_DATATYPE_UINT_16",
            "SNPE_UDO_DATATYPE_UINT_32":    "QNN_DATATYPE_UINT_32",
        }
        qnn_datatypes = ir_graph.Qnn_DataType_t.__members__

        for name, value in qnn_datatypes.items():
            # if block handles snpe 2.0 udo case where
            # the custom_dtype we get is a str
            if isinstance(custom_dtype, str):
                if name == snpe_qnn_udo_op_dict[custom_dtype]:
                    return value
                continue
            elif name == custom_dtype.name:
                return value
        raise TypeError("Cannot convert custom datatype: {}".format(custom_dtype.describe()))

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        outputs_info = []
        input_names = node.input_names
        # infer output datatypes from converter_op_package_lib when present
        output_datatypes = []
        if node.op.converter_op_package_lib:
            input_names, input_shapes, input_datatypes, input_axis_formats = [], [], [], []
            for input_name in node.input_names:
                input_info = backend.retrieve_tensor_info(input_name)
                input_names.append(input_info['name'])
                input_shapes.append(input_info['dims'])
                input_datatypes.append(input_info['data_type'])
                input_axis_formats.append(input_info['axis_format'])

            output_datatypes = node.op.infer_datatype(input_names, input_shapes, input_datatypes,
                                                      input_axis_formats, len(node.output_names))
        # change datatype to the actual values
        for i, name in enumerate(node.output_names):
            if name not in node.op.outputs:
                # use index instead, as name may have changed due to optimizations
                # TODO: Stop optimizations on custom ops
                # node.op.outputs is an ordered dict so this is guaranteed to match
                output = list(node.op.outputs.values())[i]
            else:
                output = node.op.outputs[name]
            if len(output_datatypes) != 0:
                output["data_type"] = output_datatypes[i]
            tensor_data_type = self.convert_custom_dtype_to_qnntype(output["data_type"])
            check_encodings = True
            # if the tensor datatype is INT32/INT16/INT8 or UINT32/UINT16/UINT8
            # we shall ignore to check for encodings
            if (tensor_data_type == ir_graph.QNN_DATATYPE_INT_32 or
                tensor_data_type == ir_graph.QNN_DATATYPE_INT_16 or
                tensor_data_type == ir_graph.QNN_DATATYPE_UINT_8 or
                tensor_data_type == ir_graph.QNN_DATATYPE_UINT_32 or
                tensor_data_type == ir_graph.QNN_DATATYPE_UINT_16 or
                tensor_data_type == ir_graph.QNN_DATATYPE_UINT_8):
                check_encodings = False

            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            # Support only default dataFormat
            output_tensor_info["dataFormat"] = qnn_definitions.QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER
            outputs_info.append(output_tensor_info)
        tensor_params = {}
        scalar_params = {}
        attr = node.op.c_op.attrs
        for name in attr.list_names():
            data_type = attr.get_data_type(name)
            attr_type = attr.get_attr_type(name)
            if 'output_dim' in name or name in ['custom_type', 'package_name']:
                continue
            if attr_type == ir_graph.Qnn_ParamType_t.QNN_PARAMTYPE_SCALAR:
                param_data = node.op.get_attrs_keyvalue(data_type, name)
                scalar_params[name] = (data_type, param_data)
            else:
                # store in bin param is not relevant to UDO. Hence, if is skipped for UDO
                # tensor_params for SnpeUdo are added in the else
                param_data = np.array(attr.get_static_tensor_data(name))
                param_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, name)
                if  not backend.serialize_with_suppl_attr and graph.has_buffer(param_tensor_name):
                    # if parameter expects its data as input it is stored as a tensor in .bin
                    # retrieve source & QNN tensor axis format from output buffer
                    src_axis_format = graph.get_buffer(param_tensor_name).get_src_axis_format()
                    tensor_axis_format = graph.get_buffer(param_tensor_name).get_axis_format()
                    backend.add_tensor(node.op.name,
                                       param_tensor_name,
                                       qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                       param_data,
                                       src_axis_format=src_axis_format,
                                       tensor_axis_format=tensor_axis_format,
                                       orig_tensor_name=name,
                                       tensor_data_type=data_type)
                    input_names.append(name)
                else:
                    param_info = backend.create_tensor_info(param_tensor_name,
                                                            qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                            list(param_data.shape),
                                                            data_type,
                                                            data=param_data)
                    # Support only default dataFormat
                    param_info["dataFormat"] = qnn_definitions.QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER
                    tensor_params[name] = param_info

        backend.add_node(node.op.name,
                         node.op.custom_type,
                         input_names=input_names,
                         outputs_info=outputs_info,
                         tensor_params=tensor_params,
                         scalar_params=scalar_params)



@register
class QnnDepthToSpaceTranslation(QnnTranslationBase):
    TARGET = op_adapter.DepthToSpaceOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        mode = node.op.mode
        block_size_tensor_param_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                             ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE)
        block_size = node.op.block_size
        block_size_info = backend.create_tensor_info(block_size_tensor_param_name,
                                                     qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                     [len(block_size)], ir_graph.QNN_DATATYPE_UINT_32,
                                                     data=block_size)

        backend.add_node(node.op.name, ir_graph.QNN_OP_DEPTH_TO_SPACE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE: block_size_info},
                         scalar_params={ir_graph.QNN_OP_DEPTH_TO_SPACE_PARAM_MODE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(mode))})


@register
class QnnDequantizeTranslation(QnnTranslationBase):
    TARGET = op_adapter.DequantizeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        output_name = node.output_names[0]
        outputs_info = []
        output_tensor_info = backend.get_output_info(graph,
                                                     tensor_name=output_name,
                                                     check_encodings=False)
        outputs_info.append(output_tensor_info)
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_DEQUANTIZE, node.input_names, outputs_info)


@register
class DetectionOutTranslation(QnnTranslationBase):
    TARGET = [
        op_adapter.DetectionOutputOp.TRANSLATION_KEY,
        op_adapter.MultiClassNmsOp.TRANSLATION_KEY
    ]

    def get_detection_output_info(self, backend, node, graph):
        # Qnn spec expects [scores, boxes, classes, valid_det] for outputs.
        # IR spec outputs [boxes, scores, classes, valid_det], so the first two output names should
        # be switched
        output_names = node.output_names[:]
        if node.op.type == op_adapter.MultiClassNmsOp.TRANSLATION_KEY:
            boxes_name = output_names[0]
            output_names[0] = output_names[1]  # scores as out [0]
            output_names[1] = boxes_name  # boxes as out [1]
        outputs_info = []
        for i, output_name in enumerate(output_names):
            check_encodings = False
            output_buf = graph.get_buffer(output_name)
            # determine data types per spec
            if i == 2:
                # class labels
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32
                # since detection_classes data type requirement is int_32, if there is a proceeding
                # eltwise sum with a const, cast the const value to int32.
                for consumer in output_buf.consumers:
                    if consumer.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]:
                        for input_name in consumer.input_names:
                            add_input_node = graph.get_producer_node(input_name)
                            if add_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                                add_input_node.op.tensor = add_input_node.op.tensor.astype(np.int32)
            elif i == 3:
                # num_valid_det
                tensor_data_type = ir_graph.QNN_DATATYPE_UINT_32
            else:
                tensor_data_type = ir_graph.QNN_DATATYPE_FLOAT_32
                if i == 0:
                    # scores
                    check_encodings = True
            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_type=qnn_definitions.QNN_TENSOR_TYPE_APP_READ,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)

            outputs_info.append(output_tensor_info)
        return outputs_info

    def add_multi_class_nms_op(self, node, graph, backend):
        def adjust_const_type(output_buf_, tensor_data_type_):
            # for classes and features, data type can be non-float, if there is a proceeding
            # eltwiseOp with a const, cast the const value accordingly.
            for consumer in output_buf_.consumers:
                if consumer.op.type in QnnElementwiseTranslation.BINARY_ELTWISE:
                    for input_name in consumer.input_names:
                        eltwise_input_node = graph.get_producer_node(input_name)
                        if eltwise_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                            eltwise_input_node.op.tensor = eltwise_input_node.op.tensor. \
                                astype(qnn_to_numpy_dtype[tensor_data_type_])
                            if not(tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32 or
                                   tensor_data_type in qnn_quantized_types):
                                eltwise_input_node.op.quantizable = False

        input_names = node.input_names
        output_names = node.output_names
        outputs_info = []
        for i, output_name in enumerate(output_names):
            check_encodings = False
            output_buf = graph.get_buffer(output_name)
            # determine data types per spec
            if i == 2:
                # class labels
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32
                adjust_const_type(output_buf, tensor_data_type)
            elif i == 3:
                # num_valid_det
                tensor_data_type = ir_graph.QNN_DATATYPE_UINT_32
            elif i > 3:
                # input must match output
                input_feature_name = input_names[i - 2]  # -2 to account for class and num_det
                tensor_data_type = backend.retrieve_tensor_info(input_feature_name)["data_type"]
                if tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32 or tensor_data_type in qnn_quantized_types:
                    check_encodings = True

                # most typically nms features require a reshape since QNN has a batch dim
                # so for adjusting possible eltwise with const input we need to look subsequent Op
                feature_consumers = list(output_buf.consumers)
                eltwise_input_buf = output_buf
                if len(feature_consumers) == 1 and \
                        feature_consumers[0].op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                    eltwise_input_buf = graph.get_buffer(feature_consumers[0].output_names[0])
                adjust_const_type(eltwise_input_buf, tensor_data_type)
            else:
                # boxes and scores
                tensor_data_type = ir_graph.QNN_DATATYPE_FLOAT_32
                if i != 0:
                    # scores
                    check_encodings = True
            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            outputs_info.append(output_tensor_info)

        scalar_params = {
            qnn_definitions.QNN_OP_MULTI_CLASS_NMS_PARAM_IOU_THRESHOLD:
                (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.iou_threshold)),
            qnn_definitions.QNN_OP_MULTI_CLASS_NMS_PARAM_SCORE_THRESHOLD:
                (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.score_threshold))
        }

        if backend.serialize_with_suppl_attr:
            scalar_params.update({ir_graph.IR_OP_MULTI_CLASS_NMS_PARAM_MAX_TOTAL_DETECTIONS:
                                  (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.max_total_detections),
                                   ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)})

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_MULTI_CLASS_NMS,
                         input_names,
                         outputs_info,
                         scalar_params=scalar_params)

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # TODO: remove handling of both nms and ssd here once new IR node
        if node.op.type == op_adapter.MultiClassNmsOp.TRANSLATION_KEY and \
                "scale_y" not in node.op.attrs:
            # Generic TF NMS case
            # delta scales indicate SSD case, where box decode is merged into NMS layer
            self.add_multi_class_nms_op(node, graph, backend)
            return

        # Handle SSD use-cases
        if node.op.type == op_adapter.DetectionOutputOp.TRANSLATION_KEY:
            delta_scales = node.op.delta_scaling_factors
        else:
            delta_scales = np.asarray([node.op.scale_y, node.op.scale_x, node.op.scale_h, node.op.scale_w],
                                      dtype=np.float32)
        delta_scales_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                  qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS)
        delta_scales_info = backend.create_tensor_info(delta_scales_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                       [4], ir_graph.QNN_DATATYPE_FLOAT_32,
                                                       data=delta_scales, quantizable=False)
        # assign params based on whether DetectionOutputOp or (decodeBox+nms)Op was used
        input_names = node.input_names
        anchor_buf = graph.get_buffer(input_names[-1])
        share_location = True
        if node.op.type == op_adapter.DetectionOutputOp.TRANSLATION_KEY:
            delta_scales = list(node.op.delta_scaling_factors)
            confidence_threshold = node.op.confidence_threshold
            iou_threshold = node.op.iou_threshold
            bg_id = getattr(node.op, ir_graph.QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX, 0)
            use_bg_in_nms = node.op.use_bg_in_nms
            share_location = node.op.share_location
            detection_limit = node.op.detection_limit
            log_assert(not node.op.hasattr('variance_encoded_in_target') or node.op.variance_encoded_in_target == False,
                       "QNN DetectionOut only supports variance encoded in boxes. Op {} has variance encoded in target."
                       .format(node.op.name))
        else:
            delta_scales = [node.op.scale_y, node.op.scale_x, node.op.scale_h, node.op.scale_w]
            confidence_threshold = node.op.score_threshold
            iou_threshold = node.op.iou_threshold
            bg_id = getattr(node.op, "background_label_id", 0)
            use_bg_in_nms = True
            share_location = True
            detection_limit = node.op.max_detections_per_class
        nms_eta = getattr(node.op, "nms_eta", 1.0)

        # reshape anchor input from [batch, num_anchors, 4] to [batch * num_anchors, 4] per QNN spec
        if anchor_buf.rank() == 3:
            # TODO: anchors are constant this shall be handled in earlier stages
            feature_reshape_node_name = input_names[-1] + "_reshape"
            feature_input_shape = [anchor_buf.shape[0] * anchor_buf.shape[1], anchor_buf.shape[2]]
            feature_reshape_info = backend.create_tensor_info(feature_reshape_node_name,
                                                              qnn_definitions.QNN_TENSOR_TYPE_NATIVE,
                                                              feature_input_shape)
            backend.add_node(feature_reshape_node_name, qnn_definitions.QNN_OP_RESHAPE,
                             [input_names[-1]],
                             [feature_reshape_info])
            input_names[-1] = feature_reshape_node_name

        outputs_info = self.get_detection_output_info(backend, node, graph)
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_DETECTION_OUTPUT,
                         input_names,
                         outputs_info,
                         tensor_params={
                             qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS: delta_scales_info},
                         scalar_params={qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_CONFIDENCE_THRESHOLD:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(confidence_threshold)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_IOU_THRESHOLD:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(iou_threshold)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_TYPE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')],
                                             np.uint32(qnn_definitions.QNN_OP_DETECTION_OUTPUT_NMS_TYPE_REGULAR)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(bg_id)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_USE_BG_IN_NMS:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(use_bg_in_nms)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_OUTPUT_BACKGROUND:
                                            (numpy_dtype_to_qnn[np.dtype('bool')],
                                             np.bool_(True)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_SHARE_LOCATION:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(share_location)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_NMS_ETA:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(nms_eta)),
                                        qnn_definitions.QNN_OP_DETECTION_OUTPUT_PARAM_DETECTION_LIMIT:
                                            (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(detection_limit))
                                        })


@register
class QnnDistributeFpnProposalsTranslation(QnnTranslationBase):
    TARGET = op_adapter.DistributeFpnProposalsOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        outputs_info = []
        input_data_types = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names]

        for idx, output_name in enumerate(node.output_names):
            if idx == 0:
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32
                check_encodings = False
            else:
                tensor_data_type = input_data_types[0]
                check_encodings = True
                if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 and tensor_data_type not in qnn_quantized_types:
                    check_encodings = False
            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            outputs_info.append(output_tensor_info)

        backend.add_node(node.op.name, ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS,
                         node.input_names,
                         outputs_info,
                         scalar_params={ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MAX_LEVEL:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.roi_max_level)),
                                        ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MIN_LEVEL:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.roi_min_level)),
                                        ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_SCALE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.roi_canonical_scale)),
                                        ir_graph.QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_LEVEL:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.roi_canonical_level))
                                        },
                                        macs=node.op.macs)


@register
class QnnElementwiseTranslation(QnnTranslationBase):
    TERNARY_ELTWISE = [op_adapter.ElementwiseTernaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SELECT]]
    BINARY_ELTWISE = [op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EQUAL],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FMOD],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_POWER],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MOD],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_OR],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT],
                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_XOR]]
    UNARY_ELTWISE = [op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN],
                     op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT]]
    TARGET = [*TERNARY_ELTWISE[:], *BINARY_ELTWISE[:], *UNARY_ELTWISE[:]]

    HAS_BOOL_OUTPUT = [
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EQUAL],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_OR],
        op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT],
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_XOR]
    ]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]:
            # Squash Relux if applicable
            outputs_info = backend.get_outputs_info(node, graph)
            if(outputs_info[0]["quantizable"]):
                self.squash_relu(node, graph, backend, outputs_info)

        # TODO This validation can be removed once QNN reference validation is implemented
        if node.op.type in self.TERNARY_ELTWISE and len(node.input_names) != 3:
            raise ValueError("QNN only support 3 inputs for ternary Eltwise operations. Got: {} for node: {}"
                             .format(node.input_names, node.op.name))
        if node.op.type in self.BINARY_ELTWISE and len(node.input_names) != 2:
            raise ValueError("QNN only support 2 inputs for binary Eltwise operations. Got: {} for node: {}"
                             .format(node.input_names, node.op.name))
        if node.op.type in self.UNARY_ELTWISE and len(node.input_names) != 1:
            raise ValueError("QNN only support 1 input for unary Eltwise operations. Got: {} for node: {}"
                             .format(node.input_names, node.op.name))

        tensor_data_type = None
        check_encodings = True

        if node.op.type in self.TERNARY_ELTWISE:
            tensor_data_type = backend.retrieve_tensor_info(node.input_names[1])["data_type"]

        if node.op.type in self.HAS_BOOL_OUTPUT:
            tensor_data_type = ir_graph.QNN_DATATYPE_BOOL_8
            check_encodings = False

        # ElementWiseFloorDiv does not support int32 input/output in backends,
        # so change them to ElementWiseDivide if all inputs are int32
        node_op_type = node.op.type
        if node_op_type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV]:
            input_data_types = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in
                                node.input_names]
            if all(input_data_type == ir_graph.QNN_DATATYPE_INT_32 for input_data_type in input_data_types):
                node_op_type = op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE]

        # Add a cast op if the input datatype is different but can be converted
        # Datatypes can be converterd: for example: <int32, float32>, <int64, float64>
        # Just support ElementwiseBinaryOp types whose output datatype is bool
        if node.op.type in self.HAS_BOOL_OUTPUT and node.op.type in self.BINARY_ELTWISE:
            first_input_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            second_input_type = backend.retrieve_tensor_info(node.input_names[1])["data_type"]
            if first_input_type != second_input_type:
                # TODO: add other datatypes into consideration
                if {first_input_type, second_input_type} == {ir_graph.QNN_DATATYPE_INT_32, ir_graph.QNN_DATATYPE_FLOAT_32}:
                    # Add cast node to graph
                    cast_input_name = node.input_names[0] if first_input_type == ir_graph.QNN_DATATYPE_INT_32 else node.input_names[1]
                    cast_op_name = cast_input_name + '_cast_to_fp32'
                    cast_op = op_adapter.CastOp(cast_op_name, from_type='int32', to_type='float32')

                    if graph.has_buffer(cast_input_name):
                        buf = graph.get_buffer(cast_input_name)
                        idx_to_insert = graph.nodes_in_order.index(buf.producer) + 1
                        cast_node = graph.add(cast_op, input_names=[cast_input_name], output_names=[cast_op_name],idx=idx_to_insert)
                        buf.consumers.remove(node)
                    else:
                        raise KeyError("Graph has no buffer %s, referred to as input for %s" % (cast_input_name, node.op.name))

                    if graph.has_buffer(cast_op_name):
                        buf = graph.get_buffer(cast_op_name)
                        buf.consumers.add(node)

                    # Add cast node to backend
                    cast_outputs_info = backend.get_outputs_info(cast_node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32, check_encodings=True)
                    backend.add_node(cast_node.op.name, qnn_definitions.QNN_OP_CAST, cast_node.input_names, cast_outputs_info)

                    # Change input names for elementwise op
                    input_idx = node.input_names.index(cast_input_name)
                    node.input_names[input_idx] = cast_op_name

        elif node.op.type == op_adapter.ElementwiseTernaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SELECT]:
            first_input_name = node.input_names[0]
            first_input_type = backend.retrieve_tensor_info(first_input_name)["data_type"]
            if first_input_type != ir_graph.QNN_DATATYPE_BOOL_8:
                cast_op_name = first_input_name + '_cast_to_bool'
                if first_input_type != ir_graph.QNN_DATATYPE_BOOL_8:

                    if not graph.has_buffer(cast_op_name):
                        cast_op = op_adapter.CastOp(cast_op_name, from_type=first_input_type, to_type='bool')
                        buf = graph.get_buffer(first_input_name)
                        idx_to_insert = graph.nodes_in_order.index(buf.producer) + 1
                        cast_node = graph.add(cast_op, input_names=[first_input_name], output_names=[cast_op_name],idx=idx_to_insert)
                        # Add cast node to backend
                        cast_outputs_info = backend.get_outputs_info(cast_node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_BOOL_8, check_encodings=True)
                        backend.add_node(cast_node.op.name, qnn_definitions.QNN_OP_CAST, cast_node.input_names, cast_outputs_info)
                        buf.consumers.remove(node)
                    else:
                        buf = graph.get_buffer(cast_op_name)
                        buf.consumers.add(node)
                        buf1 = graph.get_buffer(first_input_name)
                        buf1.consumers.remove(node)

                    # Change input names for elementwise op
                    node.input_names[0] = cast_op_name

        scalar_params = {}
        op_type = ir_consts_to_qnn[node_op_type]
        if node_op_type in self.UNARY_ELTWISE:
            op_type = ir_graph.IR_OP_ELTWISE_UNARY
            scalar_params.update({ir_graph.QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION:
                                             (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(ir_consts_to_qnn[node_op_type]))
                                  })
        elif node_op_type in self.BINARY_ELTWISE:
            op_type = ir_graph.IR_OP_ELTWISE_BINARY
            operation = node.op.operation
            scalar_params.update({ir_graph.QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION:
                                             (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(operation))
                                   })
        elif node_op_type in self.TERNARY_ELTWISE:
            op_type = ir_graph.IR_OP_ELTWISE_TERNARY
            scalar_params.update({ir_graph.IR_OP_ELTWISE_TERNARY_PARAM_TYPE:
                                      (ir_graph.QNN_DATATYPE_UNDEFINED,
                                       str(ir_consts_to_qnn[node_op_type]),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })

        backend.add_node(node.op.name, op_type,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type,
                                                  check_encodings=check_encodings),
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnExtractGlimpseTranslation(QnnTranslationBase):
    TARGET = op_adapter.ExtractGlimpseOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        extract_glimpse_param_size = node.op.size
        extract_glimpse_param_centered = node.op.centered
        extract_glimpse_param_normalized = node.op.normalized
        extract_glimpse_param_noise = node.op.noise

        extract_glimpse_size_tensor_param_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                                       qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_PARAM_SIZE)
        extract_glimpse_size_info = backend.create_tensor_info(extract_glimpse_size_tensor_param_name,
                                                               qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                               [len(extract_glimpse_param_size)],
                                                               ir_graph.QNN_DATATYPE_INT_32,
                                                               data=extract_glimpse_param_size)

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_EXTRACT_GLIMPSE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_PARAM_CENTERED:
                                            (numpy_dtype_to_qnn[np.dtype('bool')],
                                             np.bool_(extract_glimpse_param_centered)),
                                        qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_PARAM_NORMALIZED:
                                            (numpy_dtype_to_qnn[np.dtype('bool')],
                                             np.bool_(extract_glimpse_param_normalized)),
                                        qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_PARAM_NOISE:
                                            (numpy_dtype_to_qnn[np.dtype('int32')],
                                             np.int32(extract_glimpse_param_noise))},
                         tensor_params={qnn_definitions.QNN_OP_EXTRACT_GLIMPSE_PARAM_SIZE: extract_glimpse_size_info})


@register
class QnnExtractPatchesTranslation(QnnTranslationBase):
    TARGET = op_adapter.ExtractPatchesOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # size
        size_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                          ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_SIZE)
        size_info = backend.create_tensor_info(size_name,
                                               qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                               [2],
                                               ir_graph.QNN_DATATYPE_UINT_32,
                                               data=node.op.size)

        # stride
        stride_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                            ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_STRIDE)
        stride_info = backend.create_tensor_info(stride_name,
                                                 qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                 [2],
                                                 ir_graph.QNN_DATATYPE_UINT_32,
                                                 data=node.op.stride)

        # rate
        rate_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                          ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_RATE)
        rate_info = backend.create_tensor_info(rate_name,
                                               qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                               [2],
                                               ir_graph.QNN_DATATYPE_UINT_32,
                                               data=node.op.rate)

        backend.add_node(node.op.name, ir_graph.QNN_OP_EXTRACT_PATCHES,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_PADDING:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')],
                                             np.uint32(node.op.padding))},
                         tensor_params={ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_SIZE: size_info,
                                        ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_STRIDE: stride_info,
                                        ir_graph.QNN_OP_EXTRACT_PATCHES_PARAM_RATE: rate_info})


@register
class QnnExpandTranslation(QnnTranslationBase):
    TARGET = op_adapter.ExpandOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_names = node.input_names
        output_names = node.output_names
        output_buf = graph.get_output_buffers(node)[0]

        # To avoid data type mismatch, we assign the shape tensor with input data type in qnn_translation instead of IR translation
        input_tensor_data_type = backend.retrieve_tensor_info(input_names[0])["data_type"]

        const_op_name = node.op.name + "_coeff"
        const_op = op_adapter.ConstantOp(const_op_name, np.ones(output_buf.shape, qnn_to_numpy_dtype[input_tensor_data_type]))

        check_encodings = True if const_op.quantizable else False
        store_in_bin = len(const_op.tensor.tobytes()) > QnnTranslationBase.MAX_INPUT_TENSOR_BYTES_IN_CPP

        src_axis_format = graph.get_buffer(node.output_names[0]).get_src_axis_format()
        tensor_axis_format = graph.get_buffer(node.output_names[0]).get_axis_format()

        backend.add_tensor(const_op_name, const_op_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                           const_op.tensor, check_encodings=check_encodings, tensor_data_type=input_tensor_data_type,
                           src_axis_format=src_axis_format, tensor_axis_format=tensor_axis_format,
                           orig_tensor_name="tensor")

        # QNN_OP_ELEMENT_WISE_MULTIPLY can not support bool_8 dtype, but QNN_OP_ELEMENT_WISE_AND can support it.
        # Here use QNN_OP_ELEMENT_WISE_AND to handle bool_8 dtype.
        if input_tensor_data_type == ir_graph.QNN_DATATYPE_BOOL_8:
            backend.add_node(node.op.name, ir_consts_to_qnn[op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND]],
                             [input_names[0], const_op_name],
                             backend.get_outputs_info(node, graph, tensor_data_type=input_tensor_data_type, check_encodings=check_encodings),
                             macs=node.op.macs)
        else:
            backend.add_node(node.op.name, ir_consts_to_qnn[op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY]],
                             [input_names[0], const_op_name],
                             backend.get_outputs_info(node, graph, tensor_data_type=input_tensor_data_type, check_encodings=check_encodings),
                             macs=node.op.macs)


@register
class QnnFullyConnectedTranslation(QnnTranslationBase):
    TARGET = op_adapter.FullyConnectedOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # Squash Relux if applicable
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)
        self.squash_relu(node, graph, backend, outputs_info)
        scalar_params = {}
        if backend.serialize_with_suppl_attr:
            scalar_params.update({ir_graph.IR_OP_FULLY_CONNECTED_PARAM_BIAS_OP_NAME:
                                      (ir_graph.QNN_DATATYPE_UNDEFINED,
                                       str(node.op.bias_op_name),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_FULLY_CONNECTED,
                         input_names=node.input_names,
                         outputs_info=outputs_info,
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnGatherTranslation(QnnTranslationBase):
    TARGET = op_adapter.GatherOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axis = node.op.axis
        input_rank = graph.get_buffer(node.input_names[0]).rank()
        if axis < 0:
            axis += input_rank

        # gather has 2 inputs with the second one being indices which is always int, hence use the first one
        tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        check_encodings = True
        if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 and tensor_data_type not in qnn_quantized_types:
            check_encodings = False
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_GATHER,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type,
                                                  check_encodings=check_encodings),
                         scalar_params={qnn_definitions.QNN_OP_GATHER_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(axis))
                                        })


@register
class QnnGatherElementsTranslation(QnnTranslationBase):
    TARGET = op_adapter.GatherElementsOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # make axis positive
        axis = node.op.axis
        input_rank = graph.get_buffer(node.input_names[0]).rank()
        if axis < 0:
            axis += input_rank

        # gather has 2 inputs with the second one being indices which is always int, hence use the first one
        tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        check_encodings = True
        if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 and tensor_data_type not in qnn_quantized_types:
            check_encodings = False

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_GATHER_ELEMENTS,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type,
                                                  check_encodings=check_encodings),
                         scalar_params={qnn_definitions.QNN_OP_GATHER_ELEMENTS_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.int32(axis))
                                        })


@register
class QnnGatherNDTranslation(QnnTranslationBase):
    TARGET = op_adapter.GatherNDOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        batch_dims = qnn_definitions.QNN_OP_GATHER_ND_PARAM_BATCH_DIMS

        if batch_dims is None:
            raise ValueError("Expected param batch_dims must be 0 or a positive integer.")

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_GATHER_ND,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={qnn_definitions.QNN_OP_GATHER_ND_PARAM_BATCH_DIMS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.batch_dims))})


@register
class QnnGenerateProposalsTranslation(QnnTranslationBase):
    TARGET = op_adapter.GenerateProposalsOp.TRANSLATION_KEY

    def get_output_tensor_info(self, backend, node, graph):
        # Qnn spec expects [score, position, batch_index] for outputs
        outputs_info = []
        for i, output_name in enumerate(node.output_names):
            check_encodings = False
            # determine data types per spec
            # scores
            if i == 0:
                tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
            # boxes
            elif i == 1:
                tensor_data_type = backend.retrieve_tensor_info(node.input_names[3])["data_type"]
            else:
                # batch index of each bounding box
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32

            check_encodings = tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32 or tensor_data_type in qnn_quantized_types

            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            outputs_info.append(output_tensor_info)
        return outputs_info

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        img_size_ratio_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                           ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IMG_SIZE_RATIO)
        img_size_ratio_data = node.op.img_size_ratio
        img_size_ratio_info = backend.create_tensor_info(img_size_ratio_tensor_name,
                                                         qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                         [len(img_size_ratio_data)], ir_graph.QNN_DATATYPE_FLOAT_32,
                                                         data=np.asarray(img_size_ratio_data, dtype=np.float32), quantizable=False)

        outputs_info = self.get_output_tensor_info(backend, node, graph)
        backend.add_node(node.op.name, ir_graph.QNN_OP_GENERATE_PROPOSALS,
                         node.input_names,
                         outputs_info,
                         tensor_params={ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IMG_SIZE_RATIO: img_size_ratio_info},
                         scalar_params={ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_MIN_SIZE:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.min_size)),
                                        ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_PRE_NMS_LIMIT:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.pre_nms_limit)),
                                        ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_POST_NMS_LIMIT:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.post_nms_limit)),
                                        ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_IOU_THRESHOLD:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.iou_threshold)),
                                        ir_graph.QNN_OP_GENERATE_PROPOSALS_PARAM_BBOX_XFORM_CLIP:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.bbox_xform_clip))})


@register
class QnnGetSparseIndicesTranslation(QnnTranslationBase):
    TARGET = op_adapter.GetSparseIndicesOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        output_tensor_data_type = ir_graph.QNN_DATATYPE_INT_32
        check_encodings = output_tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32
        backend.add_node(node.op.name, ir_graph.QNN_OP_GET_SPARSE_INDICES,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=output_tensor_data_type,
                                                  check_encodings=check_encodings))


@register
class QnnGetSparseValuesTranslation(QnnTranslationBase):
    TARGET = op_adapter.GetSparseValuesOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        output_tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        check_encodings = output_tensor_data_type == ir_graph.QNN_DATATYPE_FLOAT_32
        backend.add_node(node.op.name, ir_graph.QNN_OP_GET_SPARSE_VALUES,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=output_tensor_data_type,
                                                  check_encodings=check_encodings))


@register
class QnnGridSampleTranslation(QnnTranslationBase):
    TARGET = op_adapter.GridSampleOp.TRANSLATION_KEY
    SUPPORTED_MODES = [ir_graph.QNN_OP_GRID_SAMPLE_MODE_BILINEAR,
                       ir_graph.QNN_OP_GRID_SAMPLE_MODE_NEAREST]
    SUPPORTED_PADDING_MODES = [ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS,
                               ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER,
                               ir_graph.QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.mode not in self.SUPPORTED_MODES:
            raise ValueError("Node {}: Invalid parameter value for mode, got {}".format(node.op.name, node.op.mode))
        if node.op.padding_mode not in self.SUPPORTED_PADDING_MODES:
            raise ValueError("Node {}: Invalid parameter value for padding_mode, got {}".format(node.op.name, node.op.padding_mode))

        backend.add_node(node.op.name, ir_graph.QNN_OP_GRID_SAMPLE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.align_corners)),
                                        ir_graph.QNN_OP_GRID_SAMPLE_PARAM_MODE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.mode)),
                                        ir_graph.QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.padding_mode))},
                         macs=node.op.macs)


@register
class QnnGroupNormTranslation(QnnTranslationBase):
    TARGET = op_adapter.GroupNormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        num_groups = np.uint32(node.op.group)
        num_channels = graph.get_buffer(node.input_names[0]).shape[-1]
        if num_channels % num_groups != 0:
            raise ValueError("Node {}: number of groups must be a divisor of the number of channels {}. Got {}".format(
                node.op.name, num_channels, num_groups))

        if graph.get_input_shapes(node)[1][0] != num_channels:
            raise ValueError("Node {}: Weight input shape must be equal to number of channels. Expected {} but got {} ".format(
                node.op.name, num_channels, graph.get_input_shapes(node)[1][0]))

        if graph.get_input_shapes(node)[2][0] != num_channels:
            raise ValueError("Node {}: Bias input shape must be equal to number of channels. Expected {} but got {} ".format(
                node.op.name, num_channels, graph.get_input_shapes(node)[2][0]))

        scalar_params = {}
        scalar_params.update({ir_graph.QNN_OP_GROUP_NORM_PARAM_EPSILON:
                                  (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.epsilon))})
        scalar_params.update({ir_graph.QNN_OP_GROUP_NORM_PARAM_GROUP:
                                  (numpy_dtype_to_qnn[np.dtype('uint32')], num_groups)})

        backend.add_node(node.op.name, node.op.type,
                         input_names=node.input_names,
                         outputs_info=backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32),
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnGruTranslation(QnnTranslationBase):
    TARGET = op_adapter.GruOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # If the input name is null (doesn't exist) or we're lowering and the input is unprovided add null tensors
        for i, name in enumerate(node.input_names):
            if not name or (not backend.is_online_construction and "_unprovided_input_" in name):
                node.input_names[i] = node.op.name + "_unprovided_input_" + str(i)
                backend.add_tensor(node.op.name, node.input_names[i], qnn_definitions.QNN_TENSOR_TYPE_NULL,
                                   np.ndarray([0]), check_encodings=False)

        # Several params are passed their default value, support does not exist in both IR and QNN backends
        scalar_params = {ir_graph.QNN_OP_GRU_PARAM_DIRECTION:
                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.direction)),
                         ir_graph.QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET:
                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.linear_before_reset)),
                         ir_graph.QNN_OP_GRU_PARAM_TIME_MAJOR:
                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.time_major))
                        }

        if backend.serialize_with_suppl_attr:
            scalar_params.update({ir_graph.IR_OP_GRU_PARAM_HIDDEN_SIZE:
                                       (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.hidden_size),
                                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  ir_graph.IR_OP_GRU_PARAM_H_0_INPUT_NAME:
                                       (ir_graph.QNN_DATATYPE_UNDEFINED, str(node.op.h_0_input_name),
                                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  })

        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)

        backend.add_node(node.op.name, ir_graph.QNN_OP_GRU, node.input_names,
                         outputs_info,
                         scalar_params=scalar_params
                        )


@register
class QnnImageProjectionTransform(QnnTranslationBase):
    TARGET = op_adapter.ImageProjectiveTransformOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM,
                         node.input_names, backend.get_outputs_info(node, graph, check_encodings=False),
                         scalar_params={ir_graph.QNN_OP_IMAGE_PROJECTION_TRANSFORM_PARAM_INTERPOLATION_MODE:
                                        (numpy_dtype_to_qnn[np.dtype('uint32')],
                                         np.uint32(node.op.interpolation_mode))})


@register
class QnnInstanceNormTranslation(QnnTranslationBase):
    TARGET = op_adapter.InstanceNormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):

        scalar_params = {}
        # Converter IR frontend supports MVN and InstanceNorm as one op so differentiate here
        if node.op.mode == ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA:
            # verify no unsupported MVN attribute values
            if node.op.region != ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL or not node.op.normalize_variance:
                # Source frameworks such as Caffe, have MVN layers with options to disable normalizing variance and
                # perform across_channel mvn.
                raise ValueError("No full support for Mean-Variance Normalization(MVN) in QNN. "
                                 "normalize_variance and across_spatial attributes must be true.")
            scalar_params.update({ir_graph.QNN_OP_INSTANCE_NORM_PARAM_EPSILON:
                                      (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.epsilon))})
        else:
            # Currently QNN only supports normalization done using mean/variance. With use_mu_sigma set to False
            # that will imply normalization using RMS(Root Mean Square) which is not supported.
            raise ValueError("QNN only supports normalization using mean/variance. Requested for RMS (Root Mean "
                             "Square) Method")

        # Squash Relux if applicable
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)
        self.squash_relu(node, graph, backend, outputs_info)

        # add node for bn
        backend.add_node(node.op.name, node.op.type,
                         input_names=node.input_names,
                         outputs_info=outputs_info,
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnL2NormTranslation(QnnTranslationBase):
    TARGET = op_adapter.L2NormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axis = node.op.axis
        if isinstance(node.op.axis, np.ndarray):
            if len(node.op.axis) != 1:
                raise ValueError("axis attribute for L2Norm should be scalar, got array results {} for op {}"
                                 .format(node.op.axis, node.op.name))
            axis = int(node.op.axis[0])

        # qnn requires axis to be positive valued
        if axis < 0:
            axis += graph.get_buffer(node.input_names[0]).rank()

        tensor_params = {}
        # currently axes attribute is only supported in HTP backend
        if node.op.hasattr("axes") and len(node.op.axes) > 0:
            axes_tensor_param_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                           ir_graph.QNN_OP_L2_NORM_PARAM_AXES)
            axes_data = node.op.axes
            axes_info = backend.create_tensor_info(axes_tensor_param_name,
                                                   qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                   [len(axes_data)], ir_graph.QNN_DATATYPE_UINT_32,
                                                   data=np.asarray(axes_data, dtype=np.uint32))
            tensor_params[ir_graph.QNN_OP_L2_NORM_PARAM_AXES] = axes_info

        backend.add_node(node.op.name, ir_graph.QNN_OP_L2_NORM,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32),
                         tensor_params=tensor_params,
                         scalar_params={ir_graph.QNN_OP_L2_NORM_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(axis)),
                                        ir_graph.QNN_OP_L2_NORM_PARAM_EPSILON:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.epsilon))},
                         macs=node.op.macs)


@register
class QnnLayernormTranslation(QnnTranslationBase):
    TARGET = op_adapter.LayerNormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axes_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, qnn_definitions.QNN_OP_LAYER_NORM_PARAM_AXES)
        axes = node.op.axes
        axes_tensor_info = backend.create_tensor_info(axes_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                      [len(axes)], ir_graph.QNN_DATATYPE_UINT_32,
                                                      data=np.asarray(axes, dtype=np.uint32))

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_LAYER_NORM,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={qnn_definitions.QNN_OP_LAYER_NORM_PARAM_AXES: axes_tensor_info},
                         scalar_params={qnn_definitions.QNN_OP_LAYER_NORM_PARAM_EPSILON:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.epsilon))},
                         macs=node.op.macs)


@register
class QnnLogSoftmaxTranslation(QnnTranslationBase):
    TARGET = op_adapter.LogSoftmaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axis = node.op.axis
        input_rank = graph.get_buffer(node.input_names[0]).rank()
        if axis < 0:
            axis += input_rank

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_LOG_SOFTMAX,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32),
                         scalar_params={qnn_definitions.QNN_OP_LOG_SOFTMAX_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(axis)),
                                        qnn_definitions.QNN_OP_LOG_SOFTMAX_PARAM_BETA:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta))
                                        })


@register
class QnnLrnTranslation(QnnTranslationBase):
    TARGET = op_adapter.LrnOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_LRN,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32),
                         scalar_params={qnn_definitions.QNN_OP_LRN_PARAM_ALPHA:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.alpha)),
                                        qnn_definitions.QNN_OP_LRN_PARAM_BETA:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta)),
                                        qnn_definitions.QNN_OP_LRN_PARAM_BIAS:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.bias)),
                                        qnn_definitions.QNN_OP_LRN_PARAM_RADIUS:
                                            (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(node.op.radius)),
                                        qnn_definitions.QNN_OP_LRN_PARAM_REGION:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.region))
                                        },
                         macs=node.op.macs)


@register
class QnnLstmTranslation(QnnTranslationBase):
    TARGET = op_adapter.LstmOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):

        # If the input name is null (doesn't exist) or we're lowering and the input is unprovided add null tensors
        for i, name in enumerate(node.input_names):
            if not name or (not backend.is_online_construction and "_unprovided_input_" in name):
                node.input_names[i] = node.op.name + "_unprovided_input_" + str(i)
                backend.add_tensor(node.op.name, node.input_names[i], qnn_definitions.QNN_TENSOR_TYPE_NULL,
                                   np.ndarray([0]), check_encodings=False)

        # Several params are passed their default value, support does not exist in both IR and QNN backends
        scalar_params = {ir_graph.QNN_OP_LSTM_PARAM_DIRECTION:
                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.direction)),
                         ir_graph.QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.cell_clip_threshold)),
                         ir_graph.QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.output_clip_threshold)),
                         ir_graph.QNN_OP_LSTM_PARAM_TIME_MAJOR:
                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.time_major)),
                         ir_graph.QNN_OP_LSTM_PARAM_INPUT_GATE_QSCALE:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(0.0)),
                         ir_graph.QNN_OP_LSTM_PARAM_FORGET_GATE_QSCALE:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(0.0)),
                         ir_graph.QNN_OP_LSTM_PARAM_CELL_GATE_QSCALE:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(0.0)),
                         ir_graph.QNN_OP_LSTM_PARAM_OUTPUT_GATE_QSCALE:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(0.0)),
                         ir_graph.QNN_OP_LSTM_PARAM_HIDDEN_STATE_OFFSET:
                            (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(0)),
                         ir_graph.QNN_OP_LSTM_PARAM_HIDDEN_STATE_QSCALE:
                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(0.0))
                        }
        if backend.serialize_with_suppl_attr:
            scalar_params.update({ir_graph.IR_OP_LSTM_PARAM_HIDDEN_SIZE:
                                       (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.hidden_size),
                                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  ir_graph.IR_OP_LSTM_PARAM_RESET_STATE_AT_TIME_STEP_0:
                                       (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.reset_state_at_time_step_0),
                                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  ir_graph.IR_OP_LSTM_PARAM_H_0_INPUT_NAME:
                                       (ir_graph.QNN_DATATYPE_UNDEFINED, str(node.op.h_0_input_name),
                                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  ir_graph.IR_OP_LSTM_PARAM_C_0_INPUT_NAME:
                                       (ir_graph.QNN_DATATYPE_UNDEFINED, str(node.op.c_0_input_name),
                                        ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  })

        node.output_names = node.output_names[:3]
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)

        backend.add_node(node.op.name, ir_graph.QNN_OP_LSTM, node.input_names,
                         outputs_info,
                         scalar_params=scalar_params,
                         macs=node.op.macs
                         )


@register
class QnnMaskedSoftmaxTranslation(QnnTranslationBase):
    TARGET = op_adapter.MaskedSoftmaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        mode = node.op.mode
        outputs_info = backend.get_outputs_info(node, graph)
        backend.add_node(node.op.name, ir_graph.QNN_OP_MASKED_SOFTMAX,
                         node.input_names, outputs_info,
                         scalar_params={ir_graph.QNN_OP_MASKED_SOFTMAX_PARAM_MODE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(mode))})


@register
class QnnMatMulTranslation(QnnTranslationBase):
    TARGET = op_adapter.MatMulOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # If the bitwidth of second input of the matmul is not 8 then insert a convertOp.
        # HTP only support second input to be of 8 bitwidth.
        if backend.c_ir_graph is not None and backend.c_ir_graph.get_output_tensor(node.input_names[1]) is not None:
            input_encoding = backend.c_ir_graph.get_output_tensor(node.input_names[1]).get_encoding()
            input_encoding_enc_info = input_encoding.encInfo
            input_bw = input_encoding_enc_info.bw
            if input_bw != 8:
                input_name = node.input_names[1]
                convert_name = node.input_names[1] + "_converted_QNN_DATATYPE_UFIXED_POINT_8"
                convert_op = op_adapter.ConvertOp(convert_name)
                if graph.has_buffer(convert_name):
                    convert_buffer = graph.buffers[convert_name]
                    consumer = graph.nodes_by_name[node.op.name]
                    convert_buffer.consumers.add(consumer)
                    node.input_names[1] = convert_name
                    input_buffer = graph.buffers[input_name]
                    input_buffer.consumers.remove(consumer)
                else:
                    consumers = list()
                    for consumer in graph.buffers[input_name].consumers:
                        if consumer.op.type == node.op.type and consumer.input_names[1] == input_name:
                            consumers.append(consumer.op.name)
                    graph.inject(convert_op, input_name, convert_name, consumer_names=consumers)
                    convert_node = graph.nodes_by_name[convert_name]
                    producer_encoding = backend.get_producer_encoding(convert_node, graph)
                    quant_params, producer_tensor_encoding = backend.get_qnn_quant_params(producer_encoding)

                    # set scale offset params using a performance optimized scale factor
                    quant_params['scale_offset']['scale'] = quant_params['scale_offset']['scale'] * 256.0
                    quant_params['scale_offset']['offset'] = round(quant_params['scale_offset']['offset'] / 256.0)

                    output_matmul_name = node.output_names[0]
                    outputs_info = backend.get_outputs_info(convert_node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_UFIXED_POINT_8, original_output=output_matmul_name)
                    outputs_info[0]['data_type'] = ir_graph.QNN_DATATYPE_UFIXED_POINT_8
                    outputs_info[0]['quant_params'] = quant_params

                    backend.add_node(convert_node.op.name, qnn_definitions.QNN_OP_CONVERT, convert_node.input_names, outputs_info,
                                     scalar_params={qnn_definitions.QNN_OP_CONVERT_PARAM_DYNAMIC_INPUT_DATA:
                                                        (numpy_dtype_to_qnn[np.dtype('bool')],
                                                         np.bool_(convert_node.op.dynamic_input_data)),
                                                    qnn_definitions.QNN_OP_CONVERT_PARAM_DYNAMIC_OUTPUT_DATA:
                                                        (numpy_dtype_to_qnn[np.dtype('bool')],
                                                         np.bool_(convert_node.op.dynamic_output_data))})

        backend.add_node(node.op.name, ir_graph.QNN_OP_MAT_MUL,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], node.op.transpose_in0),
                                        ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], node.op.transpose_in1)
                                        },
                         macs=node.op.macs)


@register
class QnnMomentsTranslation(QnnTranslationBase):
    TARGET = op_adapter.MomentOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axes_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, ir_graph.QNN_OP_MOMENTS_PARAM_AXES)
        axes = node.op.axes
        axes_tensor_info = backend.create_tensor_info(axes_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                      [len(axes)], ir_graph.QNN_DATATYPE_INT_32,
                                                      data=np.asarray(axes, dtype=np.int32))

        backend.add_node(node.op.name, ir_graph.QNN_OP_MOMENTS,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={ir_graph.QNN_OP_MOMENTS_PARAM_AXES: axes_tensor_info},
                         scalar_params={ir_graph.QNN_OP_MOMENTS_PARAM_KEEP_DIMS:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], node.op.keep_dims)
                                        })


@register
class QnnElementwiseNeuronTranslation(QnnTranslationBase):
    TARGET = op_adapter.ElementwiseNeuronOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        neuron_scalar_params = {}
        operation = node.op.operation
        if operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX:
            neuron_scalar_params = {ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.min_value)),
                                    ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.max_value))
                                    }
        elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU:
            neuron_scalar_params = {ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.alpha))
                                    }
        elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID:
            neuron_scalar_params = {ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.alpha)),
                                    ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta))
                                    }
        elif operation == ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS:
            neuron_scalar_params = {ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta)),
                                    ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD:
                                        (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.threshold))
                                    }
        neuron_scalar_params.update({ir_graph.QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION:
                                         (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(operation))})

        backend.add_node(node.op.name, node.op.type,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params=neuron_scalar_params)


@register
class QnnNonMaxSuppressionTranslation(QnnTranslationBase):
    TARGET = op_adapter.NonMaxSuppressionOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        output_names = node.output_names
        outputs_info = []
        for i, output_name in enumerate(output_names):
            # determine data types per spec
            if i == 0:
                # selected_indices
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32
            else:
                # valid number of selected indices
                tensor_data_type = ir_graph.QNN_DATATYPE_UINT_32
            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=False)
            outputs_info.append(output_tensor_info)

        scalar_params = {
            ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_IOU_THRESHOLD:
                (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.iou_threshold)),
            ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_SCORE_THRESHOLD:
                (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.score_threshold)),
            ir_graph.QNN_OP_NON_MAX_SUPPRESSION_PARAM_MAX_BOXES_SELECTED:
                (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.max_boxes_selected))
        }

        backend.add_node(node.op.name, ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
                         node.input_names,
                         outputs_info,
                         scalar_params=scalar_params)


@register
class QnnNonZeroTranslation(QnnTranslationBase):
    TARGET = op_adapter.NonZeroOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, ir_graph.QNN_OP_NON_ZERO,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_INT_32,
                                                  check_encodings=False))


@register
class QnnOneHotTranslation(QnnTranslationBase):
    TARGET = op_adapter.OneHotOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_rank = graph.get_buffer(node.input_names[0]).rank()

        # derive output tensor data type from on_value param data type
        tensor_data_type = node.op.get_default_output_dtypes_c_op_wrapper(node.op.num_outputs)[0]

        backend.add_node(node.op.name, ir_graph.QNN_OP_ONE_HOT,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type),
                         scalar_params={qnn_definitions.QNN_OP_ONE_HOT_PARAM_DEPTH:
                                            (ir_graph.QNN_DATATYPE_UINT_32, np.uint32(node.op.depth)),
                                        qnn_definitions.QNN_OP_ONE_HOT_PARAM_AXIS:
                                            (ir_graph.QNN_DATATYPE_UINT_32, np.uint32(node.op.axis)),
                                        qnn_definitions.QNN_OP_ONE_HOT_PARAM_ON_VALUE:
                                            (tensor_data_type, node.op.on_value),
                                        qnn_definitions.QNN_OP_ONE_HOT_PARAM_OFF_VALUE:
                                            (tensor_data_type, node.op.off_value)
                                        })


@register
class QnnPackTranslation(QnnTranslationBase):
    TARGET = op_adapter.PackOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_PACK,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={qnn_definitions.QNN_OP_PACK_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis))
                                        })

@register
class QnnPadTranslation(QnnTranslationBase):
    TARGET = op_adapter.PadOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        supported_modes = [ir_graph.QNN_OP_PAD_SCHEME_CONSTANT,
                           ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT,
                           ir_graph.QNN_OP_PAD_SCHEME_MIRROR_SYMMETRIC,
                           ir_graph.QNN_OP_PAD_SCHEME_EDGE]

        # Validate that specified pad mode is supported
        if node.op.scheme not in supported_modes:
            raise ValueError(code_to_message.get_error_message("ERROR_PAD_UNSUPPORTED_MODE")(node.op.scheme))

        # Add scalar parameters pad scheme and constant value, if specified
        pad_scalar_params = {qnn_definitions.QNN_OP_PAD_PARAM_SCHEME:
                                 (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.scheme))
                             }

        if node.op.scheme == ir_graph.QNN_OP_PAD_SCHEME_CONSTANT:
            pad_scalar_params.update({qnn_definitions.QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE:
                                          (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.pad_constant_value))
                                      })

        # Add tensor parameter pad amount with shape (input_rank, 2)
        pad_tensor_name = backend.create_unique_qnn_tensor_name(
            node.op.name, ir_graph.QNN_OP_PAD_PARAM_PAD_AMOUNT)
        input_rank = graph.get_buffer(node.input_names[0]).rank()
        pad_tensor_info = backend.create_tensor_info(pad_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                     [input_rank, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                                     data=np.asarray(node.op.pad_amount, dtype=np.uint32))

        # Add node for pad
        backend.add_node(node.op.name, ir_graph.QNN_OP_PAD,
                         input_names=node.input_names,
                         outputs_info=backend.get_outputs_info(node, graph),
                         tensor_params={ir_graph.QNN_OP_PAD_PARAM_PAD_AMOUNT: pad_tensor_info},
                         scalar_params=pad_scalar_params)


@register
class QnnPoolTranslation(QnnTranslationBase):
    TARGET = [op_adapter.Pool2dOp.TRANSLATION_KEY,
              op_adapter.Pool3dOp.TRANSLATION_KEY]

    def get_pool_pad_size(self, input_size, output_size, stride_size, filter_size, padding_size_strategy,
                          pad_before, pad_after):
        if padding_size_strategy not in [ir_graph.PADDING_SIZE_IMPLICIT_SAME_BEGIN,
                                         ir_graph.PADDING_SIZE_IMPLICIT_SAME_END]:
            pad_value = (output_size - 1) * stride_size + filter_size - input_size
            return self.get_pad_size_c(padding_size_strategy, pad_value, pad_before, pad_after)
        return pad_before, pad_after

    def get_pool_params(self, backend, graph, node):
        scalar_params = {}
        num_params = 2
        if node.op.pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
            stride_tensor_param = ir_graph.QNN_OP_POOL_AVG_2D_PARAM_STRIDE
            pad_tensor_param = ir_graph.QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT
            filter_tensor_param = ir_graph.QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE
            scalar_params.update({ir_graph.QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES:
                                      (numpy_dtype_to_qnn[np.dtype('bool')],
                                       np.bool_(node.op.count_pad_for_edges))
                                  })
        elif node.op.pool_type == ir_graph.QNN_OP_POOL_AVG_3D:
            stride_tensor_param = ir_graph.QNN_OP_POOL_AVG_3D_PARAM_STRIDE
            pad_tensor_param = ir_graph.QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT
            filter_tensor_param = ir_graph.QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE
            scalar_params.update({ir_graph.QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES:
                                      (numpy_dtype_to_qnn[np.dtype('bool')],
                                       np.bool_(node.op.count_pad_for_edges))
                                  })
            num_params = 3
        elif node.op.pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
            stride_tensor_param = ir_graph.QNN_OP_POOL_MAX_2D_PARAM_STRIDE
            pad_tensor_param = ir_graph.QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT
            filter_tensor_param = ir_graph.QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE
        elif node.op.pool_type == ir_graph.QNN_OP_POOL_MAX_3D:
            stride_tensor_param = ir_graph.QNN_OP_POOL_MAX_3D_PARAM_STRIDE
            pad_tensor_param = ir_graph.QNN_OP_POOL_MAX_3D_PARAM_PAD_AMOUNT
            filter_tensor_param = ir_graph.QNN_OP_POOL_MAX_3D_PARAM_FILTER_SIZE
            num_params = 3
        elif node.op.pool_type == ir_graph.QNN_OP_L2_POOL_2D:
            if node.op.p != 2:
                raise ValueError("Attribute p value {} for LpPool is not supported, expected 2".format(node.op.p))
            stride_tensor_param = qnn_definitions.QNN_OP_L2_POOL_2D_PARAM_STRIDE
            pad_tensor_param = qnn_definitions.QNN_OP_L2_POOL_2D_PARAM_PAD_AMOUNT
            filter_tensor_param = qnn_definitions.QNN_OP_L2_POOL_2D_PARAM_FILTER_SIZE
        else:
            raise ValueError("Unsupported Pooling type: {}".format(node.op.pool_type))

        # get Qnn tensor-info definition for the params and add the actual data to the binary file
        stride_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, stride_tensor_param)
        strides = np.array(node.op.stride, dtype=np.uint32)
        stride_tensor_info = backend.create_tensor_info(stride_tensor_name,
                                                        qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                        [num_params], ir_graph.QNN_DATATYPE_UINT_32,
                                                        data=strides)

        pad_amount = np.array(node.op.pad_amount, dtype=np.uint32)
        pad_pairs = []
        for dim in range(num_params):
            in_size, out_size = graph.get_buffer(node.input_names[0]).shape[1+dim], graph.get_buffer(node.output_names[0]).shape[1+dim]
            pads = self.get_pool_pad_size(in_size, out_size, strides[dim], node.op.filter_size[dim],
                                                     node.op.padding_size_strategy, pad_amount[dim][0], pad_amount[dim][1])
            pad_pairs.append([pads[0], pads[1]])

        pad_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, pad_tensor_param)
        pads = np.array(pad_pairs, dtype=np.uint32)
        pad_tensor_info = backend.create_tensor_info(pad_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                     [num_params, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                                     data=pads)

        filter_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, filter_tensor_param)
        filters = np.array([sz for sz in node.op.filter_size], dtype=np.uint32)
        filter_tensor_info = backend.create_tensor_info(filter_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                        [num_params], ir_graph.QNN_DATATYPE_UINT_32,
                                                        data=filters)

        tensor_params = {stride_tensor_param: stride_tensor_info,
                         pad_tensor_param: pad_tensor_info,
                         filter_tensor_param: filter_tensor_info}

        scalar_params.update({ir_graph.IR_OP_POOL_TYPE:
                                  (ir_graph.QNN_DATATYPE_UNDEFINED,
                                   str(node.op.pool_type),
                                   ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)})

        if backend.serialize_with_suppl_attr:
            scalar_params.update({ir_graph.IR_OP_POOL_PADDING_SIZE_STRATEGY:
                                      (numpy_dtype_to_qnn[np.dtype('uint8')],
                                       np.uint8(node.op.padding_size_strategy),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })

        return tensor_params, scalar_params

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        tensor_params, scalar_params = self.get_pool_params(backend, graph, node)
        backend.add_node(node.op.name, node.op.type,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32),
                                                  tensor_params=tensor_params,
                                                  scalar_params=scalar_params,
                                                  macs=node.op.macs)


@register
class QnnPreluTranslation(QnnTranslationBase):
    TARGET = [op_adapter.PreluOp.TRANSLATION_KEY]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_buf = graph.get_buffer(node.input_names[0])
        input_coeff_buf = graph.get_buffer(node.input_names[1])
        output_buf = graph.get_buffer(node.output_names[0])

        # QNN only supports uni-directional broadcasting of coeff to input, hence first input shape must be
        # same as output shape
        if input_buf.shape != output_buf.shape:
            raise RuntimeError("QNN only supports unidirectional broadcasting. Got input {} and coeff {} "
                               .format(input_buf.shape, input_coeff_buf.shape))

        backend.add_node(node.op.name, ir_graph.QNN_OP_PRELU,
                         node.input_names,
                         outputs_info=backend.get_outputs_info(node, graph,
                                                               tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32),
                         macs=node.op.macs)


@register
class QnnQuantizeTranslation(QnnTranslationBase):
    TARGET = op_adapter.QuantizeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        outputs_info = backend.get_outputs_info(node, graph)
        backend.update_quant_param_info(node, graph, backend, outputs_info[0])
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_QUANTIZE, node.input_names, outputs_info)


@register
class QnnReductionTranslation(QnnTranslationBase):
    TARGET = [op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX],
              op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN],
              op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN],
              op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD],
              op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM],]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axes_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                 ir_consts_to_qnn[node.op.type]["axes"])
        axes_tensor_info = backend.create_tensor_info(axes_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                      [len(node.op.axes)], ir_graph.QNN_DATATYPE_UINT_32,
                                                      data=np.asarray(node.op.axes, dtype=np.uint32))
        backend.add_node(node.op.name, ir_graph.IR_OP_REDUCE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={ir_consts_to_qnn[node.op.type]["axes"]: axes_tensor_info},
                         scalar_params={ir_consts_to_qnn[node.op.type]["keep_dims"]:
                                            (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.keep_dims)),
                                        ir_graph.IR_OP_REDUCE_PARAM_TYPE:
                                            (ir_graph.QNN_DATATYPE_UNDEFINED,
                                             str(ir_consts_to_qnn[node.op.type]["qnn_type"]),
                                             ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                        })


@register
class QnnReshapeTranslation(QnnTranslationBase):
    TARGET = op_adapter.ReshapeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        tensor_params = {}
        if backend.serialize_with_suppl_attr:
            shape_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name, ir_graph.IR_OP_RESHAPE_PARAM_SHAPE)
            shape_tensor_info = backend.create_tensor_info(shape_tensor_name,
                                                           qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                           [len(node.op.shape)],
                                                           ir_graph.QNN_DATATYPE_INT_32,
                                                           data=node.op.shape,
                                                           )
            tensor_params.update({ir_graph.IR_OP_RESHAPE_PARAM_SHAPE:
                                      (shape_tensor_info, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })
        backend.add_node(node.op.name, ir_graph.QNN_OP_RESHAPE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params=tensor_params)


@register
class QnnResizeTranslation(QnnTranslationBase):
    TARGET = op_adapter.ResizeOp.TRANSLATION_KEY
    ir_consts_to_qnn = {
        ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR:
            {"qnn_type": qnn_definitions.QNN_OP_RESIZE_BILINEAR,
             "align_corners": qnn_definitions.QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS,
             "half_pixel_centers": qnn_definitions.QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS},
        ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST:
            {"qnn_type": qnn_definitions.QNN_OP_RESIZE_NEAREST_NEIGHBOR,
             "align_corners": qnn_definitions.QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS,
             "half_pixel_centers": qnn_definitions.QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS}
    }

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        op_type = ir_graph.QNN_OP_RESIZE
        scalar_params = {}
        output_shape = graph.get_output_buffers(node)[0].shape

        # Interpolation mode cubic is supported in QNN Resize op with the constraint of input rank 4
        if len(output_shape) in [3,5] or node.op.interpolation_mode == ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC:
            scalar_params.update({ir_graph.QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE:
                                      (numpy_dtype_to_qnn[np.dtype('bool')], node.op.exclude_outside),
                                  ir_graph.QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE:
                                      (numpy_dtype_to_qnn[np.dtype('uint32')], node.op.transformation_mode),
                                  ir_graph.QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE:
                                      (numpy_dtype_to_qnn[np.dtype('uint32')], node.op.interpolation_mode)
                                  })
            # According to QNN OpDef validation, nearest_mode can only be used when interpolation_mode is nearest
            if node.op.interpolation_mode == ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST:
                scalar_params.update({ir_graph.QNN_OP_RESIZE_PARAM_NEAREST_MODE:
                                          (numpy_dtype_to_qnn[np.dtype('uint32')], node.op.nearest_mode)})
            # According to QNN OpDef, interpolation_mode must be set to cubic for param cubic_coeff to be valid
            elif node.op.interpolation_mode == ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC:
                scalar_params.update({ir_graph.QNN_OP_RESIZE_PARAM_CUBIC_COEFF:
                                          (numpy_dtype_to_qnn[np.dtype('float32')], node.op.cubic_coeff)})
        elif len(output_shape) == 4:
            # TODO: backends dont support the new resize yet so we need to map to old style till then
            qnn_resize_op = self.ir_consts_to_qnn[node.op.interpolation_mode]
            op_type = qnn_resize_op["qnn_type"]
            align_corners = node.op.transformation_mode == ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS
            half_pixel_centers = node.op.transformation_mode == ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL
            _, output_height, output_width, _ = output_shape
            # for backward compatibility where py_torch_half_pixel was mapped for half_pixel
            if node.op.transformation_mode == ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_PYTORCH_HALF_PIXEL and \
                    (output_height > 1 and output_width > 1):
                half_pixel_centers = True

            scalar_params.update({qnn_resize_op["align_corners"]:
                                      (numpy_dtype_to_qnn[np.dtype('bool')], align_corners),
                                  qnn_resize_op["half_pixel_centers"]:
                                      (numpy_dtype_to_qnn[np.dtype('bool')], half_pixel_centers),
                                  })
        else:
            raise ValueError("Node {}: Expected ResizeOp with output rank 3/4/5, but got {} ".format(node.op.name, len(output_shape)))

        if backend.serialize_with_suppl_attr:
            scalar_params.update({ir_graph.IR_OP_RESIZE_PARAM_SCALE_WIDTH:
                                      (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.scale_width),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })
            if len(output_shape) >= 4:
                scalar_params.update({ir_graph.IR_OP_RESIZE_PARAM_SCALE_HEIGHT:
                                          (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.scale_height),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                      })
            if len(output_shape) == 5:
                scalar_params.update({ir_graph.IR_OP_RESIZE_PARAM_SCALE_DEPTH:
                                          (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.scale_depth),
                                           ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                      })

        backend.add_node(node.op.name, op_type,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnRoiAlignTranslation(QnnTranslationBase):
    TARGET = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.mode != 'avg':
            raise ValueError(
                "Node {}: Invalid parameter value for mode. Expected {}, instead got {}".format(node.op.name, 'avg',
                                                                                                node.op.mode))
        image_size_ratio_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                             qnn_definitions.QNN_OP_ROI_ALIGN_PARAM_IMG_SIZE_RATIO)
        image_size_ratio = np.array([node.op.spatial_scale, node.op.spatial_scale], dtype=np.float32)
        image_size_ratio_info = backend.create_tensor_info(image_size_ratio_tensor_name,
                                                           qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                           [2], ir_graph.QNN_DATATYPE_FLOAT_32,
                                                           data=image_size_ratio, quantizable=False)

        scalar_params = {qnn_definitions.QNN_OP_ROI_ALIGN_PARAM_NUM_SAMPLES_X:
                             (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(node.op.sampling_ratio)),
                         qnn_definitions.QNN_OP_ROI_ALIGN_PARAM_NUM_SAMPLES_Y:
                             (numpy_dtype_to_qnn[np.dtype('int32')], np.int32(node.op.sampling_ratio)),
                         'aligned':
                             (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.aligned)),
                         'allow_invalid_roi':
                             (numpy_dtype_to_qnn[np.dtype('bool')], np.bool_(node.op.allow_invalid_roi))
                         }
        if backend.serialize_with_suppl_attr:
            pass  # TODO: populate when migration is implemented

        tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_ROI_ALIGN,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type),
                         tensor_params={qnn_definitions.QNN_OP_ROI_ALIGN_PARAM_IMG_SIZE_RATIO: image_size_ratio_info},
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnRoiPoolingTranslation(QnnTranslationBase):
    TARGET = op_adapter.RoiPoolingOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        img_size_ratio_tensor_param_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                                 qnn_definitions.
                                                                                 QNN_OP_ROI_POOLING_PARAM_IMG_SIZE_RATIO)

        img_size_ratio = np.array([node.op.spatial_scale, node.op.spatial_scale], dtype=np.float32)
        img_size_ratio_info = backend.create_tensor_info(img_size_ratio_tensor_param_name,
                                                         qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                         [2], ir_graph.QNN_DATATYPE_FLOAT_32,
                                                         data=img_size_ratio, quantizable=False)
        tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_ROI_POOLING,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type),
                         tensor_params={qnn_definitions.QNN_OP_ROI_POOLING_PARAM_IMG_SIZE_RATIO: img_size_ratio_info},
                         macs=node.op.macs)


@register
class QnnScatterElementsTranslation(QnnTranslationBase):
    TARGET = op_adapter.ScatterElementsOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        supported_reductions = {ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE: qnn_definitions.QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE,
                                ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD: qnn_definitions.QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD,
                                ir_graph.QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL: qnn_definitions.QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL}

        reduction_param_value = supported_reductions.get(node.op.reduction, None)

        if reduction_param_value is None:
            raise ValueError("Expected reduction param to be one of {}, instead got {}"
                             .format(list(supported_reductions.keys()), node.op.reduction))

        # input(data/updates) dtype should be equal with output's one
        tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_SCATTER_ELEMENTS,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type),
                         scalar_params={qnn_definitions.QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis)),
                                        qnn_definitions.QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(reduction_param_value))})


@register
class QnnScatterNDTranslation(QnnTranslationBase):
    TARGET = op_adapter.ScatterNDOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        supported_reductions = {ir_graph.QNN_OP_SCATTER_ND_REDUCTION_NONE: qnn_definitions.QNN_OP_SCATTER_ND_REDUCTION_NONE,
                                ir_graph.QNN_OP_SCATTER_ND_REDUCTION_ADD: qnn_definitions.QNN_OP_SCATTER_ND_REDUCTION_ADD,
                                ir_graph.QNN_OP_SCATTER_ND_REDUCTION_MUL: qnn_definitions.QNN_OP_SCATTER_ND_REDUCTION_MUL}

        reduction_param_value = supported_reductions.get(node.op.reduction, None)

        if reduction_param_value is None:
            raise ValueError("Expected reduction param to be one of {}, instead got {}"
                             .format(list(supported_reductions.keys()), node.op.reduction))

        # input(updates) dtype should be equal with output
        tensor_data_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        update_data_type = backend.retrieve_tensor_info(node.input_names[2])["data_type"]
        if tensor_data_type != update_data_type:
            raise ValueError("Expected input Data dtype same with Updata dtype, but got Data dtype: {}, Updata dtype: {}"
                             .format(tensor_data_type, update_data_type))
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_SCATTER_ND,
                         node.input_names,
                         backend.get_outputs_info(node, graph, tensor_data_type=tensor_data_type),
                         scalar_params={qnn_definitions.QNN_OP_SCATTER_ND_PARAM_REDUCTION:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(reduction_param_value))})


@register
class QnnSoftmaxTranslation(QnnTranslationBase):
    TARGET = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, ir_graph.QNN_OP_SOFTMAX,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_SOFTMAX_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis)),
                                        ir_graph.QNN_OP_SOFTMAX_PARAM_BETA:
                                            (numpy_dtype_to_qnn[np.dtype('float32')], np.float32(node.op.beta))
                                        })


@register
class QnnSpaceToBatchTranslation(QnnTranslationBase):
    TARGET = op_adapter.SpaceToBatchOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        block_size_tensor_name = backend.create_unique_qnn_tensor_name(
            node.op.name, qnn_definitions.QNN_OP_SPACE_TO_BATCH_PARAM_BLOCK_SIZE)
        block_size = np.array(node.op.block_shape, dtype=np.uint32)
        if block_size.shape != (2,):
            raise ValueError("Invalid block size shape on SpaceToBatch node {}, expected: (2,), got: {}".format(
                node.op.name, block_size.shape))
        block_size_tensor_info = backend.create_tensor_info(block_size_tensor_name,
                                                            qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                            [2], ir_graph.QNN_DATATYPE_UINT_32,
                                                            data=block_size)

        paddings_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                     qnn_definitions.QNN_OP_SPACE_TO_BATCH_PARAM_PAD_AMOUNT)
        paddings = np.array(node.op.paddings, dtype=np.uint32)
        if paddings.shape != (2, 2):
            raise ValueError("Invalid paddings shape on SpaceToBatch node {}, expected: (2, 2), got: {}".format(
                node.op.name, paddings.shape))
        paddings_tensor_info = backend.create_tensor_info(paddings_tensor_name,
                                                          qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                          [2, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                                          data=paddings)

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_SPACE_TO_BATCH,
                         input_names=node.input_names,
                         outputs_info=backend.get_outputs_info(node, graph),
                         tensor_params={qnn_definitions.QNN_OP_SPACE_TO_BATCH_PARAM_BLOCK_SIZE: block_size_tensor_info,
                                        qnn_definitions.QNN_OP_SPACE_TO_BATCH_PARAM_PAD_AMOUNT: paddings_tensor_info})


@register
class QnnSpaceToDepthTranslation(QnnTranslationBase):
    TARGET = op_adapter.SpaceToDepthOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        mode = node.op.mode
        block_size_tensor_param_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                             ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE)
        block_size = node.op.block_size
        block_size_info = backend.create_tensor_info(block_size_tensor_param_name,
                                                     qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                     [len(block_size)], ir_graph.QNN_DATATYPE_UINT_32,
                                                     data=block_size)
        backend.add_node(node.op.name, ir_graph.QNN_OP_SPACE_TO_DEPTH,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE: block_size_info},
                         scalar_params={ir_graph.QNN_OP_SPACE_TO_DEPTH_PARAM_MODE:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(mode))})


@register
class QnnSparseToDenseTranslation(QnnTranslationBase):
    TARGET = op_adapter.SparseToDenseOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        output_type = backend.retrieve_tensor_info(node.input_names[0])["data_type"]
        check_encodings = True
        if output_type != ir_graph.QNN_DATATYPE_FLOAT_32 and output_type not in qnn_quantized_types:
            check_encodings = False

        backend.add_node(node.op.name, ir_graph.QNN_OP_SPARSE_TO_DENSE,
                         node.input_names,
                         backend.get_outputs_info(node, graph,
                                                  tensor_data_type=output_type,
                                                  check_encodings=check_encodings))

@register
class QnnSplitTranslation(QnnTranslationBase):
    TARGET = op_adapter.SplitOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        split_index_tensor_param_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                              ir_graph.QNN_OP_SPLIT_PARAM_SPLIT_INDEX)
        split_index = np.array(node.op.split_index, dtype=np.uint32)
        split_index_info = backend.create_tensor_info(split_index_tensor_param_name,
                                                      qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                      [len(split_index)], ir_graph.QNN_DATATYPE_UINT_32,
                                                      data=split_index)

        backend.add_node(node.op.name, ir_graph.QNN_OP_SPLIT,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_SPLIT_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis))},
                         tensor_params={ir_graph.QNN_OP_SPLIT_PARAM_SPLIT_INDEX: split_index_info})


@register
class QnnStridedSliceTranslation(QnnTranslationBase):
    TARGET = op_adapter.StridedSliceOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        ranges_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                            ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES)
        ranges_info = backend.create_tensor_info(ranges_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                 list(node.op.ranges.shape), ir_graph.QNN_DATATYPE_INT_32,
                                                 data=node.op.ranges)

        backend.add_node(node.op.name, ir_graph.QNN_OP_STRIDED_SLICE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={ir_graph.QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.begin_mask)),
                                        ir_graph.QNN_OP_STRIDED_SLICE_PARAM_END_MASK:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.end_mask)),
                                        ir_graph.QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')],
                                             np.uint32(node.op.shrink_axes)),
                                        ir_graph.QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')],
                                             np.uint32(node.op.new_axes_mask))},
                         tensor_params={ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES: ranges_info})


@register
class QnnTileTranslation(QnnTranslationBase):
    TARGET = op_adapter.TileOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        multiples = np.asarray(node.op.multiples, dtype=np.uint32)

        multiples_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                               qnn_definitions.QNN_OP_TILE_PARAM_MULTIPLES)
        multiples_info = backend.create_tensor_info(multiples_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                    [len(multiples)], ir_graph.QNN_DATATYPE_UINT_32,
                                                    data=multiples)

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_TILE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={qnn_definitions.QNN_OP_TILE_PARAM_MULTIPLES: multiples_info})


@register
class QnnTopKTranslation(QnnTranslationBase):
    TARGET = op_adapter.TopKOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        outputs_info = []
        input_data_types = [backend.retrieve_tensor_info(input_name)["data_type"] for input_name in node.input_names]
        for idx, output_name in enumerate(node.output_names):
            if idx == 0:
                tensor_data_type = input_data_types[0]
                check_encodings = True
                if tensor_data_type != ir_graph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32 and tensor_data_type not in qnn_quantized_types:
                    check_encodings = False
            else:
                tensor_data_type = ir_graph.QNN_DATATYPE_INT_32
                check_encodings = False
            output_tensor_info = backend.get_output_info(graph,
                                                         tensor_name=output_name,
                                                         tensor_data_type=tensor_data_type,
                                                         check_encodings=check_encodings)
            outputs_info.append(output_tensor_info)

        backend.add_node(node.op.name, qnn_definitions.QNN_OP_TOP_K,
                         node.input_names,
                         outputs_info,
                         scalar_params={qnn_definitions.QNN_OP_TOP_K_PARAM_K:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.k))})



@register
class QnnTransposeTranslation(QnnTranslationBase):
    TARGET = op_adapter.TransposeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        perm_order = np.asarray(node.op.perm, dtype=np.uint32)

        perm_order_name = backend.create_unique_qnn_tensor_name(node.op.name, ir_graph.QNN_OP_TRANSPOSE_PARAM_PERM)
        perm_order_info = backend.create_tensor_info(perm_order_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                     [len(perm_order)], ir_graph.QNN_DATATYPE_UINT_32,
                                                     data=perm_order)

        backend.add_node(node.op.name, ir_graph.QNN_OP_TRANSPOSE,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         tensor_params={ir_graph.QNN_OP_TRANSPOSE_PARAM_PERM: perm_order_info})


@register
class QnnTransposeConv2dTranslation(QnnTranslationBase):
    TARGET = op_adapter.TransposeConv2dOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # Extract parameters
        strides = np.array(node.op.stride, dtype=np.uint32)
        pads = np.array(node.op.pad_amount, dtype=np.uint32)
        output_padding = np.array(node.op.output_padding, dtype=np.uint32)

        # get Qnn tensor-info definition for the params and add the actual data to the binary file
        stride_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                   ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE)
        stride_tensor_info = backend.create_tensor_info(stride_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                        [2], ir_graph.QNN_DATATYPE_UINT_32,
                                                        data=strides)

        pad_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                         ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT)
        pad_info = backend.create_tensor_info(pad_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                              [2, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                              data=pads)

        # add output padding information
        output_padding_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                    ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING)
        output_padding_info = backend.create_tensor_info(output_padding_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                         [2], ir_graph.QNN_DATATYPE_UINT_32,
                                                         data=output_padding)
        # Squash Relux if applicable
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)
        self.squash_relu(node, graph, backend, outputs_info)

        tensor_params = {ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE: stride_tensor_info,
                         ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT: pad_info,
                         ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING: output_padding_info}
        scalar_params = {ir_graph.QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP:
                             (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.group))
                         }
        if backend.serialize_with_suppl_attr:
            output_size_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                            ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_SIZE)
            output_size_tensor_info = backend.create_tensor_info(output_size_tensor_name,
                                                                 qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                                 [len(node.op.output_size)],
                                                                 ir_graph.QNN_DATATYPE_UINT_32,
                                                                 data=node.op.output_size,
                                                                 )
            tensor_params.update({ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_SIZE:
                                      (output_size_tensor_info, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })
            scalar_params.update({ir_graph.IR_OP_TRANSPOSE_CONV_2D_PARAM_PADDING_SIZE_STRATEGY:
                                      (numpy_dtype_to_qnn[np.dtype('uint8')],
                                       np.uint8(node.op.padding_size_strategy),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  ir_graph.IR_OP_TRANSPOSE_CONV_2D_BIAS_OP_NAME:
                                      (ir_graph.QNN_DATATYPE_UNDEFINED,
                                       str(node.op.bias_op_name),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })

        # add node for transpose conv
        backend.add_node(node.op.name, ir_graph.QNN_OP_TRANSPOSE_CONV_2D,
                         input_names=node.input_names,
                         outputs_info=outputs_info,
                         tensor_params=tensor_params,
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnTransposeConv3dTranslation(QnnTranslationBase):
    TARGET = op_adapter.TransposeConv3dOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # Extract parameters
        strides = np.array(node.op.stride, dtype=np.uint32)
        pads = np.array(node.op.pad_amount, dtype=np.uint32)
        dilations = np.array(node.op.dilation, dtype=np.uint32)
        output_padding = np.array(node.op.output_padding, dtype=np.uint32)

        # get Qnn tensor-info definition for the params and add the actual data to the binary file
        stride_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                   ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE)
        stride_tensor_info = backend.create_tensor_info(stride_tensor_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                        [3], ir_graph.QNN_DATATYPE_UINT_32,
                                                        data=strides)

        pad_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                         ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT)
        pad_info = backend.create_tensor_info(pad_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                              [3, 2], ir_graph.QNN_DATATYPE_UINT_32,
                                              data=pads)

        dilation_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                              ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION)
        dilation_info = backend.create_tensor_info(dilation_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                   [3], ir_graph.QNN_DATATYPE_UINT_32,
                                                   data=dilations)

        # add output padding information
        output_padding_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                    ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_PADDING)
        output_padding_info = backend.create_tensor_info(output_padding_name, qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                         [3], ir_graph.QNN_DATATYPE_UINT_32,
                                                         data=output_padding)
        # Squash Relux if applicable
        outputs_info = backend.get_outputs_info(node, graph, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32)
        self.squash_relu(node, graph, backend, outputs_info)

        tensor_params = {ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE: stride_tensor_info,
                         ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT: pad_info,
                         ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION: dilation_info,
                         ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_PADDING: output_padding_info}
        scalar_params = {ir_graph.QNN_OP_TRANSPOSE_CONV_3D_PARAM_GROUP:
                             (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.group))
                         }
        if backend.serialize_with_suppl_attr:
            output_size_tensor_name = backend.create_unique_qnn_tensor_name(node.op.name,
                                                                            ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_SIZE)
            output_size_tensor_info = backend.create_tensor_info(output_size_tensor_name,
                                                                 qnn_definitions.QNN_TENSOR_TYPE_STATIC,
                                                                 [len(node.op.output_size)],
                                                                 ir_graph.QNN_DATATYPE_UINT_32,
                                                                 data=node.op.output_size,
                                                                 )
            tensor_params.update({ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_SIZE:
                                      (output_size_tensor_info, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })
            scalar_params.update({ir_graph.IR_OP_TRANSPOSE_CONV_3D_PARAM_PADDING_SIZE_STRATEGY:
                                      (numpy_dtype_to_qnn[np.dtype('uint8')],
                                       np.uint8(node.op.padding_size_strategy),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
                                  ir_graph.IR_OP_TRANSPOSE_CONV_3D_BIAS_OP_NAME:
                                      (ir_graph.QNN_DATATYPE_UNDEFINED,
                                       str(node.op.bias_op_name),
                                       ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                                  })

        backend.add_node(node.op.name, ir_graph.QNN_OP_TRANSPOSE_CONV_3D,
                         input_names=node.input_names,
                         outputs_info=outputs_info,
                         tensor_params=tensor_params,
                         scalar_params=scalar_params,
                         macs=node.op.macs)


@register
class QnnUnPackTranslation(QnnTranslationBase):
    TARGET = op_adapter.UnpackOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.add_node(node.op.name, qnn_definitions.QNN_OP_UN_PACK,
                         node.input_names,
                         backend.get_outputs_info(node, graph),
                         scalar_params={qnn_definitions.QNN_OP_UN_PACK_PARAM_AXIS:
                                            (numpy_dtype_to_qnn[np.dtype('uint32')], np.uint32(node.op.axis))
                                        })
