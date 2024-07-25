# ==============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import argparse

from qti.aisw.converters.common.custom_ops.utils.custom_op_helpers import populate_custom_op_collection
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.relay.passes.pattern_match import tflite_detection_postprocess, tflite_dequantize
from qti.aisw.converters.relay.utils import get_key_from_expr
from .relay_importer import RelayImporter, RelaySpanParser
import tvm
from tvm.relay.frontend import tflite as tflite_to_relay
from tvm.relay.frontend.tflite import get_tensor_name
from tvm.relay.build_module import bind_params_by_name

# TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
try:
    import tflite
except TypeError:
    import tflite.Model as tflite


class TFLiteImporter(RelayImporter):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(TFLiteImporter.ArgParser, self).__init__(conflict_handler='resolve', **kwargs)
            self.add_optional_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The names and dimensions of the network input layers specified "
                                            "in the format [input_name comma-separated-dimensions], "
                                            "for example: \n"
                                            "    'data' 1,224,224,3\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dim on the command "
                                            "line like: \n"
                                            "    --input_dim 'data1' 1,224,224,3 --input_dim 'data2' 1,50,100,3")
            self.add_optional_argument('--input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'),
                                       help="The names and datatype of the network input layers specified "
                                            "in the format [input_name datatype], "
                                            "for example: \n"
                                            "    'data' 'float32'\n"
                                            "Default is float32 if not specified\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dtype on the command "
                                            "line like: \n"
                                            "    --input_dtype 'data1' 'float32' --input_dtype 'data2' 'float32'")
            self.add_optional_argument('--signature_name', '-sn', type=str, default="",
                                       help='Use this option to specify a specific Subgraph signature to convert')
            self.add_optional_argument('--partial_graph_input_name', action='append',
                                       help=argparse.SUPPRESS)

    def __init__(self, args, **kwargs):
        super(TFLiteImporter, self).__init__(args, custom_op_factory=kwargs.get('custom_op_factory', None))

        self.signature_name = args.signature_name
        self.custom_op_config_paths = args.custom_op_config_paths
        self.out_names = args.out_names

        if args.partial_graph_input_name:
            for in_name in args.partial_graph_input_name:
                if in_name not in self.shape_dict:
                    self.shape_dict[in_name] = None

    def convert_to_relay(self, input_model_path, **kwargs):
        if isinstance(input_model_path, str):
            tflite_model_buf = open(input_model_path, "rb").read()
        elif isinstance(input_model_path, bytes):
            tflite_model_buf = input_model_path
        else:
            raise TypeError("Unsupported type {} for {}".format(type(input_model_path), input_model_path))
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

        populate_custom_op_collection(tflite_model, 'tflite',
                                      custom_op_config_paths=self.custom_op_config_paths,
                                      custom_op_factory=self.custom_op_factory)

        # In the description of argument --out_node, it takes output tensor name as input.
        # However, it should represent output node name according to the literal meaning.
        # In tflite converter, both node name and tensor name are supported for user convenience.
        # We add a check here to change tensor name to its corresponding node name.
        # TODO: Reconsider the scope of argument --out_node
        if self.out_names:
            subgraph = tflite_model.Subgraphs(0)
            for subgraph_index in range(1, tflite_model.SubgraphsLength()):
                if tflite_model.Subgraphs(subgraph_index).Name().decode() == self.signature_name:
                    subgraph = tflite_model.Subgraphs(subgraph_index)
                    break

            out_node_names = set()
            out_tensor_names = set()
            op_idx_str_list = [str(op_idx) for op_idx in range(subgraph.OperatorsLength())]
            for name in self.out_names:
                out_node_names.add(name) if name in op_idx_str_list else out_tensor_names.add(name)

            if out_tensor_names:
                for op_idx in range(subgraph.OperatorsLength()):
                    if not out_tensor_names:
                        break
                    op = subgraph.Operators(op_idx)
                    for tensor_idx in op.OutputsAsNumpy():
                        tensor_name = get_tensor_name(subgraph, tensor_idx)
                        if tensor_name in out_tensor_names:
                            out_node_names.add(str(op_idx))
                            out_tensor_names.remove(tensor_name)
                self.out_names = list(sorted(out_node_names))

        self.mod, self.params = tflite_to_relay.from_tflite(tflite_model,
                                                            subgraph_name=self.signature_name,
                                                            shape_dict=self.shape_dict,
                                                            dtype_dict=self.dtype_dict,
                                                            output_node_names=self.out_names,
                                                            custom_op_factory=self.custom_op_factory)

        self._post_process()
        return self.mod, self.params, self.expr_to_source_info_dict, self.custom_op_factory

    def _post_process(self):
        """post-process Relay module, including necessary fixes and optimizations"""

        def _populated_encodings_dict(span_to_encodings):
            def visit_module(expr):
                if hasattr(expr, 'span'):
                    span = expr.span
                    if isinstance(expr, tvm.relay.Constant) and isinstance(expr.span, tvm.relay.SequentialSpan):
                        span = expr.span.spans[0]

                    key = get_key_from_expr(expr)
                    if span in span_to_encodings:
                        self.expr_to_source_info_dict[key].set_encodings(span_to_encodings[span])
            tvm.relay.analysis.post_order_visit(self.mod["main"], visit_module)

        # bind TVM params variance to const
        self.mod["main"] = bind_params_by_name(self.mod["main"], self.params)

        # use span as key to record encodings and populate them after other transform pass since span will not change
        # e.g.,
        # %1 = Call(nn.conv2d(...), span=spanA, checked_type=undefined), => hash_value1, span=spanA
        # => after InferType transform pass
        # %1 = Call(nn.conv2d(...), span=spanA, checked_type=defined), => hash_value2 (changed), span=spanA (not change)
        span_to_encodings = {}

        # Prepare for Relay Passes
        seq = tvm.transform.Sequential([
            tflite_dequantize.DequantizePass(self.dtype_dict, span_to_encodings),
            # compress detection_postprocess expression back to one ir
            tflite_detection_postprocess.IdentifyTFLiteDetectionPostProcess(),
            tvm.relay.transform.FoldConstant(),
            tvm.relay.transform.InferType(),
        ])

        # need opt_level=3 to trigger ConvertLayout
        with tvm.transform.PassContext(opt_level=3):
            self.mod = seq(self.mod)
        RelaySpanParser(self.expr_to_source_info_dict).visit(self.mod['main'])
        _populated_encodings_dict(span_to_encodings)
