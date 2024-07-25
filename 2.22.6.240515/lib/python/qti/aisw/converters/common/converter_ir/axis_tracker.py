# ==============================================================================
#
#  Copyright (c) 2018-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *


class AxisTracker(object):
    """
    This class holds all enums, modules and subclasses needed to handle axis tracking between
    source framework to our desired format.
    """
    # Class (an another way of enum class)
    # TBD: Once minimum python version is upgraded for converter from 2.7 to 3.0
    #      replace with enum class
    class AxisAnnotations(object):
        """
        This class contains axis annotations required for axis tracking.
        """
        DEPTH = 0
        HEIGHT = 1
        WIDTH = 2
        CHANNEL = 3
        BATCH = 4
        TIME = 5
        FEATURE = 6
        ANY = 7
        # NONTRIVIAL indicates none of axis annotation is valid and not trivial to be derived
        # Layers such as reshape/flatten specify this axis annotation.
        NONTRIVIAL = 8
        # Weights annotations
        INPUT_CHANNELS = 9
        OUTPUT_CHANNELS = 10

    class AxisFormat(object):
        """
        Contains axis commonly used axis orders along with permute order to go to/from this well-defined formats
        """
        # Batch,Channel,Depth,Height,Width. With one batch and three spatial dimensions.
        # Used for 5D ops.
        NCDHW = 'NCDHW'
        # Batch,Depth,Height,Width,Channel. With one batch and three spatial dimensions.
        # Used for 5D ops. This is the native data order for SNPE ops.
        NDHWC = 'NDHWC'
        # Batch,Channel,Spatial. With one batch and two spatial dimensions,
        # equivalent to NCHW
        NCS = 'NCHW'
        # Batch,Spatial,Channel. With one batch and two spatial dimensions,
        # equivalent to NHWC. This is the native data order for SNPE ops with
        # output feature maps.
        NSC = 'NHWC'
        # Batch,Channel,Feature. With one batch and one spatial dimension,
        # Used for Conv1D, BatchNorm1D etc.
        NCF = 'NCF'
        # Batch,Feature,Channel. With one batch and one spatial dimension,
        # Used for Conv1D, BatchNorm1D etc. This is the native data order for SNPE ops.
        NFC = 'NFC'
        # Time,Batch,Feature.
        TNF = 'TNF'
        # Batch,Time,Feature. This is the native data order for SNPE RNN ops.
        NTF = 'NTF'
        # Batch,Feature.
        NF = 'NF'
        # Batch,Channels. Used with Reduce Ops to identify that reduce happened across Spatial dimensions
        NC = 'NC'
        # used by Constant Op and 1D tensor
        ANY = 'ANY'
        # Op specific data format.
        NONTRIVIAL = 'NONTRIVIAL'
        # Enum value used by buffers which have not yet undergone axis tracking.
        NOT_YET_DEFINED = 'NOT_YET_DEFINED'
        # Enum value used by null buffer
        NULL = 'NULL'

        # well-known permute orders
        NCDHW_TO_NDHWC = [0, 2, 3, 4, 1]
        NDHWC_TO_NCDHW = [0, 4, 1, 2, 3]
        NCS_TO_NSC = [0, 2, 3, 1]
        NSC_TO_NCS = [0, 3, 1, 2]
        NCF_TO_NFC = NFC_TO_NCF = [0, 2, 1]
        TNF_TO_NTF = NTF_TO_TNF = [1, 0, 2]

        # Weight axis formats
        DHWIO = "DHWIO"
        DHWOI = "DHWOI"
        IODHW = "IODHW"
        OIDHW = "OIDHW"
        HWIO = "HWIO"
        HWOI = "HWOI"
        IOHW = "IOHW"
        OIHW = "OIHW"
        FIO = "FIO"
        FOI = "FOI"
        IOF = "IOF"
        OIF = "OIF"

        # Weight permute orders
        IODHW_TO_DHWIO = OIDHW_TO_DHWOI = [2, 3, 4, 0, 1]
        DHWIO_TO_IODHW = [3, 4, 0, 1, 2]
        OIDHW_TO_DHWIO = [2, 3, 4, 1, 0]
        DHWIO_TO_OIDHW = [4, 3, 0, 1, 2]
        DHWOI_TO_DHWIO = [0, 1, 2, 4, 3]
        IOHW_TO_HWIO = HWIO_TO_IOHW = OIHW_TO_HWOI = [2, 3, 0, 1]
        OIHW_TO_HWIO = [2, 3, 1, 0]
        HWIO_TO_OIHW = [3, 2, 0, 1]
        HWOI_TO_HWIO = [0, 1, 3, 2]
        IOF_TO_FIO = OIF_TO_FOI = [2, 0, 1]
        FIO_TO_IOF = [1, 2, 0]
        OIF_TO_FIO = FIO_TO_OIF = [2, 1, 0]
        FOI_TO_FIO = [0, 2, 1]

        @staticmethod
        def get_valid_formats():
            return [AxisTracker.AxisFormat.NCDHW,
                    AxisTracker.AxisFormat.NDHWC,
                    AxisTracker.AxisFormat.NCS,
                    AxisTracker.AxisFormat.NSC,
                    AxisTracker.AxisFormat.NCF,
                    AxisTracker.AxisFormat.NFC,
                    AxisTracker.AxisFormat.TNF,
                    AxisTracker.AxisFormat.NTF,
                    AxisTracker.AxisFormat.NF,
                    AxisTracker.AxisFormat.NC,
                    AxisTracker.AxisFormat.ANY,
                    AxisTracker.AxisFormat.NONTRIVIAL]

        ir_to_c_axis_format = {
            NCDHW: ir_graph.AxisFormat.NCDHW,
            NDHWC: ir_graph.AxisFormat.NDHWC,
            NCS: ir_graph.AxisFormat.NCHW,
            NSC: ir_graph.AxisFormat.NHWC,
            NCF: ir_graph.AxisFormat.NCF,
            NFC: ir_graph.AxisFormat.NFC,
            TNF: ir_graph.AxisFormat.TNF,
            NTF: ir_graph.AxisFormat.NTF,
            NF: ir_graph.AxisFormat.NF,
            NC: ir_graph.AxisFormat.NC,
            ANY: ir_graph.AxisFormat.ANY,
            NONTRIVIAL: ir_graph.AxisFormat.NONTRIVIAL,
            DHWIO: ir_graph.AxisFormat.DHWIO,
            DHWOI: ir_graph.AxisFormat.DHWOI,
            IODHW: ir_graph.AxisFormat.IODHW,
            OIDHW: ir_graph.AxisFormat.OIDHW,
            HWIO: ir_graph.AxisFormat.HWIO,
            HWOI: ir_graph.AxisFormat.HWOI,
            IOHW: ir_graph.AxisFormat.IOHW,
            OIHW: ir_graph.AxisFormat.OIHW,
            FIO: ir_graph.AxisFormat.FIO,
            FOI: ir_graph.AxisFormat.FOI,
            IOF: ir_graph.AxisFormat.IOF,
            OIF: ir_graph.AxisFormat.OIF,
            NOT_YET_DEFINED: ir_graph.AxisFormat.NOT_YET_DEFINED
        }

    @classmethod
    def input_axis_formats_intact(cls, graph, node, input_nontrivial_as_changed=False):
        input_axis_formats = graph.get_input_axis_formats(node)

        if input_axis_formats != node.op.data_axis_formats or \
                (input_nontrivial_as_changed and AxisTracker.AxisFormat.NONTRIVIAL in input_axis_formats):
            return False
        return True

    @classmethod
    def get_axis_annotation_from_format(cls, axis_format):
        if axis_format == cls.AxisFormat.NCDHW:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.DEPTH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.NDHWC:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.DEPTH,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH,
                    AxisTracker.AxisAnnotations.CHANNEL]
        elif axis_format == cls.AxisFormat.NSC:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif axis_format == cls.AxisFormat.NCS:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.NCF:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.NFC:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE,
                    AxisTracker.AxisAnnotations.CHANNEL]
        elif axis_format == cls.AxisFormat.NF:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.NC:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL]
        elif axis_format == cls.AxisFormat.NTF:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.TNF:
            return [AxisTracker.AxisAnnotations.TIME, AxisTracker.AxisAnnotations.BATCH,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.NONTRIVIAL:
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]
        elif axis_format == cls.AxisFormat.DHWIO:
            return [AxisTracker.AxisAnnotations.DEPTH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.INPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.OUTPUT_CHANNELS]
        elif axis_format == cls.AxisFormat.DHWOI:
            return [AxisTracker.AxisAnnotations.DEPTH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.INPUT_CHANNELS]
        elif axis_format == cls.AxisFormat.IODHW:
            return [AxisTracker.AxisAnnotations.INPUT_CHANNELS, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.DEPTH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.OIDHW:
            return [AxisTracker.AxisAnnotations.OUTPUT_CHANNELS, AxisTracker.AxisAnnotations.INPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.DEPTH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.HWIO:
            return [AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH,
                    AxisTracker.AxisAnnotations.INPUT_CHANNELS, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS]
        elif axis_format == cls.AxisFormat.HWOI:
            return [AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH,
                    AxisTracker.AxisAnnotations.OUTPUT_CHANNELS, AxisTracker.AxisAnnotations.INPUT_CHANNELS]
        elif axis_format == cls.AxisFormat.IOHW:
            return [AxisTracker.AxisAnnotations.INPUT_CHANNELS, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.OIHW:
            return [AxisTracker.AxisAnnotations.OUTPUT_CHANNELS, AxisTracker.AxisAnnotations.INPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.FIO:
            return [AxisTracker.AxisAnnotations.FEATURE, AxisTracker.AxisAnnotations.INPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.OUTPUT_CHANNELS]
        elif axis_format == cls.AxisFormat.FOI:
            return [AxisTracker.AxisAnnotations.FEATURE, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.INPUT_CHANNELS]
        elif axis_format == cls.AxisFormat.IOF:
            return [AxisTracker.AxisAnnotations.INPUT_CHANNELS, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.OIF:
            return [AxisTracker.AxisAnnotations.OUTPUT_CHANNELS, AxisTracker.AxisAnnotations.INPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.FEATURE]
        else:
            raise ValueError("Uknown axis format for get_axis_annotations: {}".format(axis_format))

    @classmethod
    def get_axis_format_from_annotation(cls, axis_annotation):
        if axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.CHANNEL,
                               cls.AxisAnnotations.DEPTH, cls.AxisAnnotations.HEIGHT,
                               cls.AxisAnnotations.WIDTH]:
            return cls.AxisFormat.NCDHW
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.DEPTH,
                                 cls.AxisAnnotations.HEIGHT, cls.AxisAnnotations.WIDTH,
                                 cls.AxisAnnotations.CHANNEL]:
            return cls.AxisFormat.NDHWC
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.CHANNEL,
                               cls.AxisAnnotations.HEIGHT, cls.AxisAnnotations.WIDTH]:
            return cls.AxisFormat.NCS
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.HEIGHT,
                                 cls.AxisAnnotations.WIDTH, cls.AxisAnnotations.CHANNEL]:
            return cls.AxisFormat.NSC
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.CHANNEL,
                               cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.NCF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.FEATURE,
                                 cls.AxisAnnotations.CHANNEL]:
            return cls.AxisFormat.NFC
        elif axis_annotation == [cls.AxisAnnotations.TIME, cls.AxisAnnotations.BATCH,
                                 cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.TNF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.TIME,
                                 cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.NTF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.NF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.CHANNEL]:
            return cls.AxisFormat.NC
        else:
            return cls.AxisFormat.NONTRIVIAL

    @staticmethod
    def compute_permute_order(current_order, expected_order):
        log_debug("Current Axes=" + str(current_order) + " Expected Axes=" + str(expected_order))
        log_assert(set(current_order) == set(expected_order),
                   "Error: computing permute order for current and expected axes orders: values do not match;"
                   " Current order " + str(current_order) + " Expected order:" + str(expected_order) +
                   ". Make sure you are using correct Axis Annotations for orders.")
        permute_order = []
        for axis in expected_order:
            permute_order.append(current_order.index(axis))
        return permute_order

    @staticmethod
    def permute_shape(shape, order):
        return [shape[i] for i in order]

    @classmethod
    def enforce_input_axis_format(cls, graph, input_name, target_format, permute_order,
                                  valid_input_axis_formats=None, consumers=None):
        """

        :param graph: IrOpGraph
        :param input_name: Input buffer name
        :param target_format: Target input axis format to enforce
        :param permute_order: Permute order to get to Target if the input format is different
        :param valid_input_axis_formats: These are input_formats for which the permute_order
                                         is valid to get to target_format
        :return:
        """
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == target_format:
            # Nothing to do, its already in target format
            return False

        if input_buf.axis_format in [cls.AxisFormat.ANY, cls.AxisFormat.NF]:
            return False
        elif valid_input_axis_formats:
            if input_buf.axis_format in valid_input_axis_formats:
                graph.inject_implicit_permute(input_name, target_format, permute_order, consumers=consumers)
                return True
            else:
                return False
        elif input_buf.axis_format in [cls.AxisFormat.NONTRIVIAL]:
            if input_buf.rank() == len(permute_order):
                graph.inject_implicit_permute(input_name, target_format, permute_order, consumers=consumers)
                return True
            else:
                log_debug2("inject_implicit_permute ignored for NONTRIVIAL axis format due to rank"
                           "({}) and permute_order({}) mismatch for input name: {}",
                           input_buf.rank(), len(permute_order), input_name)
                return False
        else:
            raise ValueError(code_to_message.get_error_message('ERROR_INPUT_DATA_ORDER_UNEXPECTED')
                             (input_name, set(valid_input_axis_formats + [target_format]), input_buf.axis_format))

    @classmethod
    def enforce_channel_last_input(cls, input_buf, node, graph):
        if (input_buf.axis_format in [cls.AxisFormat.NCDHW]) or \
           (len(input_buf.shape) == 5 and input_buf.axis_format == cls.AxisFormat.NONTRIVIAL):
            return cls.enforce_input_axis_format(graph, input_buf.name, cls.AxisFormat.NDHWC, cls.AxisFormat.NCDHW_TO_NDHWC,
                                          valid_input_axis_formats=[cls.AxisFormat.NCDHW,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])
        elif (input_buf.axis_format in [cls.AxisFormat.NCS]) or \
             (len(input_buf.shape) == 4 and input_buf.axis_format == cls.AxisFormat.NONTRIVIAL):
            return cls.enforce_input_axis_format(graph, input_buf.name, cls.AxisFormat.NSC, cls.AxisFormat.NCS_TO_NSC,
                                          valid_input_axis_formats=[cls.AxisFormat.NCS,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])
        elif input_buf.axis_format == cls.AxisFormat.NCF:
            return cls.enforce_input_axis_format(graph, input_buf.name, cls.AxisFormat.NFC, cls.AxisFormat.NCF_TO_NFC,
                                          valid_input_axis_formats=[cls.AxisFormat.NCF,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])

    @classmethod
    def enforce_spatial_last_input(cls, input_buf, node, graph):
        if (input_buf.axis_format in [cls.AxisFormat.NDHWC]) or \
           (len(input_buf.shape) == 5 and input_buf.axis_format == cls.AxisFormat.NONTRIVIAL):
            return cls.enforce_input_axis_format(graph, input_buf.name, cls.AxisFormat.NCDHW, cls.AxisFormat.NDHWC_TO_NCDHW,
                                          valid_input_axis_formats=[cls.AxisFormat.NDHWC,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])
        elif (input_buf.axis_format in [cls.AxisFormat.NSC]) or \
             (len(input_buf.shape) == 4 and input_buf.axis_format == cls.AxisFormat.NONTRIVIAL):
            return cls.enforce_input_axis_format(graph, input_buf.name, cls.AxisFormat.NCS, cls.AxisFormat.NSC_TO_NCS,
                                          valid_input_axis_formats=[cls.AxisFormat.NSC,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])

    @classmethod
    def enforce_feature_last_input(cls, input_buf, node, graph):
        if input_buf.axis_format == cls.AxisFormat.NFC:
            return cls.enforce_input_axis_format(graph, input_buf.name, cls.AxisFormat.NCF, cls.AxisFormat.NFC_TO_NCF,
                                          valid_input_axis_formats=[cls.AxisFormat.NFC,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])

    @classmethod
    def image_to_channel_last_order(cls, node, graph, enforce_input_channel_last=True):
        """Axis transformation for layers which take in and emit only image-valued data"""
        cls.log_axes_transformation(node, graph)

        # If any of our inputs are NONTRIVIAL, put a permute in front of them.
        # This will be shared with everyone who consumes that buffer, so don't specify consumers
        if enforce_input_channel_last:
            for name in node.input_names:
                # fetch input buffers one by one to avoid degenerate case where
                # an op uses the same input more than once and needs to permute it.
                input_buf =graph.get_buffer(name)
                cls.enforce_channel_last_input(input_buf, node, graph)

        # Fetch the input_axis_formats after enforcing input format above so that the input buffers are updated
        input_axis_formats = graph.get_input_axis_formats(node)

        # Fetch indices with max rank
        # Use those axis formats to update output buffer
        input_shapes = graph.get_input_shapes(node)
        input_ranks = [len(shape) for shape in input_shapes]
        max_rank = max(input_ranks)
        max_rank_indices = [i for i, rank in enumerate(input_ranks) if rank == max_rank]

        # Update the output buffers with the new input buffers
        # Use for loop to skip the undesired axis_format in input buffers like ANY,
        # For the input rank mismatch cases, the new input_axis_formats may be [ANY, NHWC]
        # In this case, we shouldn't only use 1st input_axis_formats to update the output axis_format
        for i in max_rank_indices:
            if cls.AxisFormat.NONTRIVIAL == input_axis_formats[i]:
                # Update output buffers to NONTRIVIAL when enforce_input_axis_format fails to
                # inject implicit permute for NONTRIVIAL input buffer
                for buf in graph.get_output_buffers(node):
                    buf.axis_format = cls.AxisFormat.NONTRIVIAL
                break
            elif (cls.AxisFormat.NDHWC == input_axis_formats[i] and
                    # Check that the input axis format was changed after op creation time
                    input_axis_formats[i] != node.op.data_axis_formats[i]):
                # Update all of our output buffers to be in NDHWC order. Output buffer is not
                # explicitly checked, it is assumed to be in NCDHW order.
                for buf in graph.get_output_buffers(node):
                    buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.NCDHW_TO_NDHWC)
                    buf.axis_format = cls.AxisFormat.NDHWC
                    # TODO: Remove setting value for node.op.output_shape once RoiPooling op is implemented in C++
                    node.op.output_shape = buf.shape
                break
            elif (cls.AxisFormat.NSC == input_axis_formats[i] and
                    # Check that the input axis format was changed after op creation time
                    input_axis_formats[i] != node.op.data_axis_formats[i]):
                # Update all of our output buffers to be in NSC order. Output buffer is not
                # explicitly checked, it is assumed to be in NCS order.
                for buf in graph.get_output_buffers(node):
                    if buf.rank() == 4:
                        buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.NCS_TO_NSC)
                        buf.axis_format = cls.AxisFormat.NSC
                        node.op.output_shape = buf.shape
                break

        log_debug2("{}\n" +
                   "OpNode {} Got Input formats {} original input formats {}",
                        cls.image_to_channel_last_order.__name__,
                        node,
                        input_axis_formats,
                        node.op.data_axis_formats)

    @classmethod
    def feature_to_channel_last_order(cls, node, graph):
        """Axis transformation for layers which take in and emit only image-valued data"""
        cls.log_axes_transformation(node, graph)

        input_axis_formats = graph.get_input_axis_formats(node)

        # If any of our inputs are NONTRIVIAL, put a permute in front of them.
        # This will be shared with everyone who consumes that buffer, so don't specify consumers
        for name in node.input_names:
            # fetch input buffers one by one to avoid degenerate case where
            # an op uses the same input more than once and needs to permute it.
            input_buf =graph.get_buffer(name)
            if input_buf.axis_format in [cls.AxisFormat.NCF]:
                ret = cls.enforce_input_axis_format(graph, name, cls.AxisFormat.NFC, cls.AxisFormat.NCF_TO_NFC,
                                                    valid_input_axis_formats=[cls.AxisFormat.NCF,
                                                                              cls.AxisFormat.NONTRIVIAL],
                                                    consumers=[node.op.name])
            elif (input_buf.rank() == 3 and input_buf.axis_format == cls.AxisFormat.NONTRIVIAL and
                  # The following condition determines the need to add permute when input is NONTRIVIAL
                  cls.AxisFormat.NCF in [*node.op.data_axis_formats, *input_axis_formats]):
                ret = cls.enforce_input_axis_format(graph, name, cls.AxisFormat.NFC, cls.AxisFormat.NCF_TO_NFC,
                                                    valid_input_axis_formats=[cls.AxisFormat.NCF,
                                                                              cls.AxisFormat.NONTRIVIAL],
                                                    consumers=[node.op.name])

        # Fetch the input_axis_formats after enforcing input format above so that the input buffers are updated
        input_axis_formats = graph.get_input_axis_formats(node)

        # Fetch indices with max rank
        # Use those axis formats to update output buffer
        input_shapes = graph.get_input_shapes(node)
        input_ranks = [len(shape) for shape in input_shapes]
        max_rank = max(input_ranks)
        max_rank_indices = [i for i, rank in enumerate(input_ranks) if rank == max_rank]

        # Update the output buffers with the new input buffers
        # Use for loop to skip the undesired axis_format in input buffers like ANY,
        # For the input rank mismatch cases, the new input_axis_formats may be [ANY, NFC]
        # In this case, we shouldn't only use 1st input_axis_formats to update the output axis_format
        for i in max_rank_indices:
            if cls.AxisFormat.NONTRIVIAL == input_axis_formats[i]:
                # Update output buffers to NONTRIVIAL when enforce_input_axis_format fails to
                # inject implicit permute for NONTRIVIAL input buffer
                for buf in graph.get_output_buffers(node):
                    buf.axis_format = cls.AxisFormat.NONTRIVIAL
                break
            elif (cls.AxisFormat.NFC == input_axis_formats[i] and
                    # Check that the input axis format was changed after op creation time
                    input_axis_formats[i] != node.op.data_axis_formats[i]):
                # Update all of our output buffers to be in NFC order. Output buffer is not
                # explicitly checked, it is assumed to be in NCF order.
                for buf in graph.get_output_buffers(node):
                    if buf.rank() == 3:
                        buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.NCF_TO_NFC)
                        buf.axis_format = cls.AxisFormat.NFC
                        node.op.output_shape = buf.shape
                break

        log_debug2("{}\n" +
                   "OpNode {} Got Input formats {} original input formats {}",
                        cls.image_to_channel_last_order.__name__,
                        node,
                        input_axis_formats,
                        node.op.data_axis_formats)

    @classmethod
    def time_series_to_batch_first_order(cls, node, graph):
        for name in node.input_names:
            cls.enforce_input_axis_format(graph, name, cls.AxisFormat.NTF, cls.AxisFormat.TNF_TO_NTF,
                                          valid_input_axis_formats=[cls.AxisFormat.TNF,
                                                                    cls.AxisFormat.NONTRIVIAL],
                                          consumers=[node.op.name])

        for buf in graph.get_output_buffers(node):
            if buf.rank() == 3:
                buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.TNF_TO_NTF)
                buf.axis_format = cls.AxisFormat.NTF
            elif buf.rank() == 4:
                buf.axis_format = cls.AxisFormat.NSC
            elif buf.rank() == 5:
                buf.axis_format = cls.AxisFormat.NDHWC

    @classmethod
    def alter_axis_format_to_ir_order(cls, node, graph):
        input_buffers = graph.get_input_buffers(node)
        input_buf_formats = [buf.axis_format for buf in input_buffers]
        output_buffers = graph.get_output_buffers(node)
        output_buf_formats = [buf.axis_format for buf in output_buffers]
        if cls.AxisFormat.NCDHW in node.op.data_axis_formats or \
                cls.AxisFormat.NDHWC in input_buf_formats:
            cls.image_to_channel_last_order(node, graph)
        elif cls.AxisFormat.NCS in node.op.data_axis_formats or \
                cls.AxisFormat.NSC in input_buf_formats:
            cls.image_to_channel_last_order(node, graph)
        elif cls.AxisFormat.NCF in node.op.data_axis_formats or \
                cls.AxisFormat.NFC in input_buf_formats or \
                (cls.AxisFormat.NONTRIVIAL in input_buf_formats and len(input_buffers[0].shape) == 3 and
                    not graph.is_time_series_input()):
            cls.feature_to_channel_last_order(node, graph)
        elif cls.AxisFormat.TNF in node.op.data_axis_formats or \
                cls.AxisFormat.NTF in input_buf_formats or \
                (cls.AxisFormat.NONTRIVIAL in input_buf_formats and len(input_buffers[0].shape) == 3 and
                    graph.is_time_series_input()):
            cls.time_series_to_batch_first_order(node, graph)
        elif any(filter(lambda fmt: fmt in AxisOrder().axis_formats, output_buf_formats)):
            # No change needed
            pass
        else:
            log_verbose("{}: Op {} Type {} input_formats {} output_formats {}".format(
                cls.alter_axis_format_to_ir_order.__name__,
                node,
                node.op.type,
                node.op.data_axis_formats,
                output_buf_formats))

    @staticmethod
    def log_axes_transformation(node, graph):
        log_debug(code_to_message.get_debugging_message("DEBUG_AXES_TRANSFORMATION_ENTRY")
                  (node.op.name))
        for input_name in node.input_names:
            log_debug(
                      code_to_message.get_debugging_message("DEBUG_AXES_TRANSFORMATION_INPUT_SIZE")
                      (input_name, str(graph.get_buffer(input_name).shape)))


class AxisOrder(object):
    def __init__(self):
        # Default to SNPE order
        self.axis_formats = [
            AxisTracker.AxisFormat.ANY,
            AxisTracker.AxisFormat.NF,
            AxisTracker.AxisFormat.NFC,
            AxisTracker.AxisFormat.NSC,
            AxisTracker.AxisFormat.NDHWC,
            AxisTracker.AxisFormat.NTF,
            AxisTracker.AxisFormat.FIO,
            AxisTracker.AxisFormat.HWIO,
            AxisTracker.AxisFormat.DHWIO,
            AxisTracker.AxisFormat.NC
        ]
        self.conv1d_weights_format = AxisTracker.AxisFormat.FIO
        self.conv2d_weights_format = AxisTracker.AxisFormat.HWIO
        self.conv3d_weights_format = AxisTracker.AxisFormat.DHWIO
        # Currently TF uses these formats but the TF Deconv weights are actually HWOI. Since the weights are static,
        # there is a transpose in the frontend. Once dynamic weights are supported, TF deconv weights format needs to
        # be changed to HWOI.
        self.deconv2d_weights_format = AxisTracker.AxisFormat.HWIO
        self.deconv3d_weights_format = AxisTracker.AxisFormat.DHWIO
        self.time_series_format = AxisTracker.AxisFormat.NTF
        # permute_sequence_from_ir - Contains the permute sequence from IR order to source framework order
        # permute_sequence_to_ir - Contains the permute sequence from source framework order to IR order
        self.permute_sequence_from_ir = self.permute_sequence_to_ir = [
            [0], # ANY
            [0, 1], # NF
            [0, 1, 2], # NFC_TO_NFC
            [0, 1, 2, 3], # NSC_TO_NSC
            [0, 1, 2, 3, 4] # NDHWC_TO_NDHWC
        ]
        self.permute_time_series_from_ir = self.permute_time_series_to_ir = list(range(len(self.time_series_format)))
        self.permute_conv1d_weights_to_ir = self.permute_conv1d_weights_from_ir = list(range(len(self.conv1d_weights_format)))
        self.permute_conv2d_weights_to_ir = self.permute_conv2d_weights_from_ir = list(range(len(self.conv2d_weights_format)))
        self.permute_conv3d_weights_to_ir = self.permute_conv3d_weights_from_ir = list(range(len(self.conv3d_weights_format)))
        self.permute_deconv2d_weights_to_ir = self.permute_deconv2d_weights_from_ir = list(range(len(self.deconv2d_weights_format)))
        self.permute_deconv3d_weights_to_ir = self.permute_deconv3d_weights_from_ir = list(range(len(self.deconv3d_weights_format)))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        self_vars = dict(self.__dict__)
        other_vars = dict(other.__dict__)
        for var in list(self_vars.keys()):
            if self_vars[var] != other_vars[var]:
                return False
        return True

    def permute_shape_to_ir(self, shape: list, format=None, conv_weights=False, deconv_weights=False) -> list:
        if format and format in [AxisTracker.AxisFormat.NTF, AxisTracker.AxisFormat.TNF]:
            order = self.permute_time_series_to_ir
            return AxisTracker.permute_shape(shape, order)
        if conv_weights and deconv_weights:
            raise ValueError("conv_weights and deconv_weights cannot be True at the same time")
        if conv_weights:
            if format and format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NCDHW]:
                order = self.permute_conv3d_weights_to_ir
                return AxisTracker.permute_shape(shape, order)
            elif format and format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NCS]:
                order = self.permute_conv2d_weights_to_ir
                return AxisTracker.permute_shape(shape, order)
        if deconv_weights:
            if format and format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NCDHW]:
                order = self.permute_deconv3d_weights_to_ir
                return AxisTracker.permute_shape(shape, order)
            elif format and format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NCS]:
                order = self.permute_deconv2d_weights_to_ir
                return AxisTracker.permute_shape(shape, order)

        try:
            order = self.permute_sequence_to_ir[len(shape)-1]
            return AxisTracker.permute_shape(shape, order)
        except IndexError:
            raise ValueError("Unable to permute shape {} to IR ordering".format(shape))

    def permute_shape_from_ir(self, shape: list, format=None, conv_weights=False, deconv_weights=False) -> list:
        if format and format in [AxisTracker.AxisFormat.NTF, AxisTracker.AxisFormat.TNF]:
            order = self.permute_time_series_from_ir
            return AxisTracker.permute_shape(shape, order)
        if conv_weights and deconv_weights:
            raise ValueError("conv_weights and deconv_weights cannot be True at the same time")
        if conv_weights:
            if format and format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NCDHW]:
                order = self.permute_conv3d_weights_from_ir
                return AxisTracker.permute_shape(shape, order)
            elif format and format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NCS]:
                order = self.permute_conv2d_weights_from_ir
                return AxisTracker.permute_shape(shape, order)
        if deconv_weights:
            if format and format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NCDHW]:
                order = self.permute_deconv3d_weights_from_ir
                return AxisTracker.permute_shape(shape, order)
            elif format and format in [AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NCS]:
                order = self.permute_deconv2d_weights_from_ir
                return AxisTracker.permute_shape(shape, order)

        try:
            order = self.permute_sequence_from_ir[len(shape) - 1]
            return AxisTracker.permute_shape(shape, order)
        except IndexError:
            raise ValueError("Unable to permute shape {} from IR ordering".format(shape))

    def get_axis_format(self, rank, time_series_format=False):
        if rank == 3 and time_series_format:
            return self.time_series_format

        if 6 > rank > 0:
            return self.axis_formats[rank-1]
        else:
            return AxisTracker.AxisFormat.NONTRIVIAL

    @classmethod
    def extract_time_series_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        return shape[:]

    @classmethod
    def format_time_series_output_shape(cls, batch_size, time_steps, feature):
        return [batch_size, time_steps, feature]

    @classmethod
    def extract_1d_spatial_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        return shape[:]

    @classmethod
    def extract_2d_spatial_dims(cls, shape):
        if len(shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(shape))
        return shape[:]

    @classmethod
    def extract_3d_spatial_dims(cls, shape):
        if len(shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(shape))
        return shape[:]

    @classmethod
    def format_1d_spatial_output_shape(cls, batch_size, feature, channel):
        return [batch_size, feature, channel]

    @classmethod
    def format_2d_spatial_output_shape(cls, batch_size, height, width, channel):
        return [batch_size, height, width, channel]

    @classmethod
    def format_3d_spatial_output_shape(cls, batch_size, depth, height, width, channel):
        return [batch_size, depth, height, width, channel]

    @classmethod
    def extract_conv1d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(weights_shape))
        # weights are expected to be in shape [FIO]
        return weights_shape[:]

    @classmethod
    def extract_conv2d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        # weights are expected to be in shape [HWIO]
        return weights_shape[:]

    @classmethod
    def extract_conv3d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(weights_shape))
        # weights are expected to be in shape [DHWIO]
        return weights_shape[:]

    @classmethod
    def format_conv1d_weights_output_shape(cls, feature, in_channels, out_channels):
        return [feature, in_channels, out_channels]

    @classmethod
    def format_conv2d_weights_output_shape(cls, height, width, in_channels, out_channels):
        return [height, width, in_channels, out_channels]

    @classmethod
    def format_conv3d_weights_output_shape(cls, depth, height, width, in_channels, out_channels):
        return [depth, height, width, in_channels, out_channels]

    @classmethod
    def extract_deconv2d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        # Note: This function will need to be updated once handling of HWOI_TO_HWIO transpose is not handled by the
        #       Tensorflow front-end. This is because deconv weights are HWOI in Tensorflow and expected output format
        #       from this function is HWIO.
        return weights_shape[:]

    @classmethod
    def extract_deconv3d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(weights_shape))
        # Note: This function will need to be updated once handling of DHWOI_TO_DHWIO transpose is not handled by the
        #       Tensorflow front-end. This is because deconv weights are DHWOI in Tensorflow and expected output format
        #       from this function is DHWIO.
        return weights_shape[:]

    @classmethod
    def extract_fc_weights_dims(cls, weights_shape):
        if len(weights_shape) != 2:
            raise ValueError("Shape needs to be of length 2, passed {}".format(weights_shape))
        # weights are expected to be in shape [out_channels, in_channels]
        return weights_shape[:]


class SpatialFirstAxisOrder(AxisOrder):
    def __init__(self):
        # SpatialFirstAxisOrder is same as SNPE order, Do Nothing
        super(SpatialFirstAxisOrder, self).__init__()


class SpatialLastAxisOrder(AxisOrder):
    def __init__(self):
        self.axis_formats = [
            AxisTracker.AxisFormat.ANY,
            AxisTracker.AxisFormat.NF,
            AxisTracker.AxisFormat.NCF,
            AxisTracker.AxisFormat.NCS,
            AxisTracker.AxisFormat.NCDHW,
            AxisTracker.AxisFormat.TNF,
            AxisTracker.AxisFormat.IOF,
            AxisTracker.AxisFormat.OIF,
            AxisTracker.AxisFormat.IOHW,
            AxisTracker.AxisFormat.OIHW,
            AxisTracker.AxisFormat.IODHW,
            AxisTracker.AxisFormat.OIDHW,
            AxisTracker.AxisFormat.NC
        ]
        self.time_series_format = AxisTracker.AxisFormat.TNF
        self.conv1d_weights_format = AxisTracker.AxisFormat.OIF
        self.conv2d_weights_format = AxisTracker.AxisFormat.OIHW
        self.conv3d_weights_format = AxisTracker.AxisFormat.OIDHW
        self.deconv2d_weights_format = AxisTracker.AxisFormat.IOHW
        self.deconv3d_weights_format = AxisTracker.AxisFormat.IODHW
        # Contains the permute sequence from IR order to source framework order
        self.permute_sequence_from_ir = [
            [0], # ANY
            [0, 1], # NF
            AxisTracker.AxisFormat.NFC_TO_NCF,
            AxisTracker.AxisFormat.NSC_TO_NCS,
            AxisTracker.AxisFormat.NDHWC_TO_NCDHW
        ]
        # Contains the permute sequence from source framework order to IR order
        self.permute_sequence_to_ir = [
            [0], # ANY
            [0, 1], # NF
            AxisTracker.AxisFormat.NCF_TO_NFC,
            AxisTracker.AxisFormat.NCS_TO_NSC,
            AxisTracker.AxisFormat.NCDHW_TO_NDHWC
        ]
        self.permute_time_series_from_ir = self.permute_time_series_to_ir =\
            AxisTracker.AxisFormat.NTF_TO_TNF # Same as TNF_TO_NTF
        self.permute_conv1d_weights_to_ir = AxisTracker.AxisFormat.OIF_TO_FIO
        self.permute_conv2d_weights_to_ir = AxisTracker.AxisFormat.OIHW_TO_HWIO
        self.permute_conv3d_weights_to_ir = AxisTracker.AxisFormat.OIDHW_TO_DHWIO
        self.permute_conv1d_weights_from_ir = AxisTracker.AxisFormat.FIO_TO_OIF
        self.permute_conv2d_weights_from_ir = AxisTracker.AxisFormat.HWIO_TO_OIHW
        self.permute_conv3d_weights_from_ir = AxisTracker.AxisFormat.DHWIO_TO_OIDHW
        self.permute_deconv2d_weights_to_ir = AxisTracker.AxisFormat.IOHW_TO_HWIO
        self.permute_deconv3d_weights_to_ir = AxisTracker.AxisFormat.IODHW_TO_DHWIO
        self.permute_deconv2d_weights_from_ir = AxisTracker.AxisFormat.HWIO_TO_IOHW
        self.permute_deconv3d_weights_from_ir = AxisTracker.AxisFormat.DHWIO_TO_IODHW

    @classmethod
    def extract_time_series_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        time_steps, batch_size, feature = shape[:]
        return [batch_size, time_steps, feature]

    @classmethod
    def format_time_series_output_shape(cls, batch_size, time_steps, feature):
        return [time_steps, batch_size, feature]

    @classmethod
    def extract_1d_spatial_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        batch_size, channel, feature = shape[:]
        return [batch_size, feature, channel]

    @classmethod
    def extract_2d_spatial_dims(cls, shape):
        if len(shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(shape))
        batch_size, channel, height, width = shape[:]
        return [batch_size, height, width, channel]

    @classmethod
    def extract_3d_spatial_dims(cls, shape):
        if len(shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(shape))
        batch_size, channel, depth, height, width = shape[:]
        return [batch_size, depth, height, width, channel]

    @classmethod
    def format_1d_spatial_output_shape(cls, batch_size, feature, channel):
        return [batch_size, channel, feature]

    @classmethod
    def format_2d_spatial_output_shape(cls, batch_size, height, width, channel):
        return [batch_size, channel, height, width]

    @classmethod
    def format_3d_spatial_output_shape(cls, batch_size, depth, height, width, channel):
        return [batch_size, channel, depth, height, width]

    @classmethod
    def extract_conv1d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(weights_shape))
        out_channels, in_channels, feature = weights_shape[:]
        return [feature, in_channels, out_channels]

    @classmethod
    def extract_conv2d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        out_channels, in_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_conv3d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(weights_shape))
        out_channels, in_channels, depth, height, width = weights_shape[:]
        return [depth, height, width, in_channels, out_channels]

    @classmethod
    def format_conv1d_weights_output_shape(cls, feature, in_channels, out_channels):
        return [out_channels, in_channels, feature]

    @classmethod
    def format_conv2d_weights_output_shape(cls, height, width, in_channels, out_channels):
        return [out_channels, in_channels, height, width]

    @classmethod
    def format_conv3d_weights_output_shape(cls, depth, height, width, in_channels, out_channels):
        return [out_channels, in_channels, depth, height, width]

    @classmethod
    def extract_deconv1d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(weights_shape))
        in_channels, out_channels, feature = weights_shape[:]
        return [feature, in_channels, out_channels]

    @classmethod
    def extract_deconv2d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        in_channels, out_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_deconv3d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(weights_shape))
        in_channels, out_channels, depth, height, width = weights_shape[:]
        return [depth, height, width, in_channels, out_channels]

    @classmethod
    def extract_fc_weights_dims(cls, weights_shape):
        if len(weights_shape) != 2:
            raise ValueError("Shape needs to be of length 2, passed {}".format(weights_shape))
        in_channels, out_channels = weights_shape[:]
        return [out_channels, in_channels]


class CaffeAxisOrder(AxisOrder):
    def __init__(self):
        self.axis_formats = [
            AxisTracker.AxisFormat.ANY,
            AxisTracker.AxisFormat.NF,
            AxisTracker.AxisFormat.NCF,
            AxisTracker.AxisFormat.NCS,
            AxisTracker.AxisFormat.NCDHW,
            AxisTracker.AxisFormat.TNF,
            AxisTracker.AxisFormat.IOF,
            AxisTracker.AxisFormat.OIF,
            AxisTracker.AxisFormat.IOHW,
            AxisTracker.AxisFormat.OIHW,
            AxisTracker.AxisFormat.IODHW,
            AxisTracker.AxisFormat.OIDHW,
            AxisTracker.AxisFormat.NC
        ]
        self.time_series_format = AxisTracker.AxisFormat.TNF
        self.conv1d_weights_format = AxisTracker.AxisFormat.OIF
        self.conv2d_weights_format = AxisTracker.AxisFormat.OIHW
        self.conv3d_weights_format = AxisTracker.AxisFormat.OIDHW
        self.deconv2d_weights_format = AxisTracker.AxisFormat.IOHW
        self.deconv3d_weights_format = AxisTracker.AxisFormat.IODHW
        # Contains the permute sequence from IR order to source framework order
        self.permute_sequence_from_ir = [
            [0],  # ANY
            [0, 1],  # NF
            AxisTracker.AxisFormat.NFC_TO_NCF,
            AxisTracker.AxisFormat.NSC_TO_NCS,
            AxisTracker.AxisFormat.NDHWC_TO_NCDHW
        ]
        # Contains the permute sequence from source framework order to IR order
        self.permute_sequence_to_ir = [
            [0],  # ANY
            [0, 1],  # NF
            AxisTracker.AxisFormat.NCF_TO_NFC,
            AxisTracker.AxisFormat.NCS_TO_NSC,
            AxisTracker.AxisFormat.NCDHW_TO_NDHWC
        ]
        self.permute_time_series_from_ir = self.permute_time_series_to_ir = \
            AxisTracker.AxisFormat.NTF_TO_TNF  # Same as TNF_TO_NTF
        self.permute_conv1d_weights_to_ir = AxisTracker.AxisFormat.OIF_TO_FIO
        self.permute_conv2d_weights_to_ir = AxisTracker.AxisFormat.OIHW_TO_HWIO
        self.permute_conv3d_weights_to_ir = AxisTracker.AxisFormat.OIDHW_TO_DHWIO
        self.permute_conv1d_weights_from_ir = AxisTracker.AxisFormat.FIO_TO_OIF
        self.permute_conv2d_weights_from_ir = AxisTracker.AxisFormat.HWIO_TO_OIHW
        self.permute_conv3d_weights_from_ir = AxisTracker.AxisFormat.DHWIO_TO_OIDHW
        self.permute_deconv2d_weights_to_ir = AxisTracker.AxisFormat.IOHW_TO_HWIO
        self.permute_deconv3d_weights_to_ir = AxisTracker.AxisFormat.IODHW_TO_DHWIO
        self.permute_deconv2d_weights_from_ir = AxisTracker.AxisFormat.HWIO_TO_IOHW
        self.permute_deconv3d_weights_from_ir = AxisTracker.AxisFormat.DHWIO_TO_IODHW

    @classmethod
    def extract_time_series_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        time_steps, batch_size, feature = shape[:]
        return [batch_size, time_steps, feature]

    @classmethod
    def format_time_series_output_shape(cls, batch_size, time_steps, feature):
        return [time_steps, batch_size, feature]

    @classmethod
    def extract_1d_spatial_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        batch_size, channel, feature = shape[:]
        return [batch_size, feature, channel]

    @classmethod
    def extract_2d_spatial_dims(cls, shape):
        if len(shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(shape))
        batch_size, channel, height, width = shape[:]
        return [batch_size, height, width, channel]

    @classmethod
    def extract_3d_spatial_dims(cls, shape):
        if len(shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(shape))
        batch_size, channel, depth, height, width = shape[:]
        return [batch_size, depth, height, width, channel]

    @classmethod
    def format_1d_spatial_output_shape(cls, batch_size, feature, channel):
        return [batch_size, feature, channel]

    @classmethod
    def format_2d_spatial_output_shape(cls, batch_size, height, width, channel):
        return [batch_size, channel, height, width]

    @classmethod
    def format_3d_spatial_output_shape(cls, batch_size, depth, height, width, channel):
        return [batch_size, channel, depth, height, width]

    @classmethod
    def extract_conv1d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(weights_shape))
        out_channels, in_channels, feature = weights_shape[:]
        return [feature, in_channels, out_channels]

    @classmethod
    def extract_conv2d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        out_channels, in_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_conv3d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(weights_shape))
        out_channels, in_channels, depth, height, width = weights_shape[:]
        return [depth, height, width, in_channels, out_channels]

    @classmethod
    def format_conv1d_weights_output_shape(cls, feature, in_channels, out_channels):
        return [out_channels, in_channels, feature]

    @classmethod
    def format_conv2d_weights_output_shape(cls, height, width, in_channels, out_channels):
        return [out_channels, in_channels, height, width]

    @classmethod
    def format_conv3d_weights_output_shape(cls, depth, height, width, in_channels, out_channels):
        return [out_channels, in_channels, depth, height, width]

    @classmethod
    def extract_deconv1d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(weights_shape))
        in_channels, out_channels, feature = weights_shape[:]
        return [feature, in_channels, out_channels]

    @classmethod
    def extract_deconv2d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        in_channels, out_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_deconv3d_weights_dims(cls, weights_shape):
        if len(weights_shape) != 5:
            raise ValueError("Shape needs to be of length 5, passed {}".format(weights_shape))
        in_channels, out_channels, depth, height, width = weights_shape[:]
        return [depth, height, width, in_channels, out_channels]

    @classmethod
    def extract_fc_weights_dims(cls, weights_shape):
        if len(weights_shape) != 2:
            raise ValueError("Shape needs to be of length 2, passed {}".format(weights_shape))
        in_channels, out_channels = weights_shape[:]
        return [out_channels, in_channels]


class RelayAxisOrder(AxisOrder):
    def __init__(self):
        # Relay has changed part of ops into channel-last
        # so set it as SNPE order
        super(RelayAxisOrder, self).__init__()

    def get_default_input_axis_format(self, rank, time_series_format=False):
        return self.get_axis_format(rank,
                                    time_series_format=time_series_format)


class PytorchAxisOrder(RelayAxisOrder):
    def __init__(self):
        super(PytorchAxisOrder, self).__init__()

    def get_default_input_axis_format(self, rank, time_series_format=False):
        # default input_layout for Pytorch model is spatial-last, so override this function for Pytorch
        return SpatialLastAxisOrder().get_axis_format(rank,
                                                      time_series_format=time_series_format)


class AxisOrders(object):
    TF = SpatialFirstAxisOrder()
    ONNX = SpatialLastAxisOrder()
    # In TVM ConvertLayout Pass, heavily layout-sensitive ops will be transformed into SpatialFirst format
    ONNX_RELAY = SpatialFirstAxisOrder()
    # TODO: Abandon CaffeAxisOrder here because Caffe converter has been deprecated
    CAFFE = CaffeAxisOrder()
    RELAY = RelayAxisOrder()
    PYTORCH = PytorchAxisOrder()

    # TODO: Abandon CaffeAxisOrder in ir_graph because Caffe converter has been deprecated
    # TODO: Check whether TFAxisOrder/OnnxAxisOrder can be renamed as SpatialFirst/SpatialLast in ir_graph
    TF_C = ir_graph.TfAxisOrder()
    ONNX_C = ir_graph.OnnxAxisOrder()
    CAFFE_C = ir_graph.CaffeAxisOrder()
    RELAY_C = ir_graph.TfAxisOrder()

    @classmethod
    def python_to_c_axis_orders(cls, axis_order):
        # TODO: Remove check for Caffe once CaffeAxisOrder is abandoned
        if type(axis_order) == CaffeAxisOrder:
            return cls.CAFFE_C
        elif type(axis_order) == SpatialLastAxisOrder:
            return cls.ONNX_C
        return cls.TF_C
