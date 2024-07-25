# ==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.relay.relay_to_ir import RelayConverterFrontend
from qti.aisw.converters.relay.importers.tflite_importer import TFLiteImporter
from qti.aisw.converters.relay.custom_ops.utils.tflite_helpers import TfliteCustomOpFactory
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
import numpy as np
import sys

class TFLiteConverterFrontend(RelayConverterFrontend):
    class ArgParser(RelayConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(TFLiteConverterFrontend.ArgParser, self).__init__(conflict_handler='resolve',
                                                                    parents=[TFLiteImporter.ArgParser()],
                                                                    **kwargs)

    def __init__(self, args, custom_op_factory=None, **kwargs):
        super(TFLiteConverterFrontend, self).__init__(args,
                                                      importer=TFLiteImporter(args, custom_op_factory=TfliteCustomOpFactory()),
                                                      axis_order=AxisOrders.TF,
                                                      **kwargs)
        if self.dump_io_config_template:
            self.dump_io_config_yaml_template()
            sys.exit(0)
        # for pre-quantized model, the input dtypes may change after dequantize pass
        # we need to rewrite them
        for input_name, dtype in self.importer.dtype_dict.items():
            self.graph.input_dtypes_dict[input_name] = np.dtype(dtype)
