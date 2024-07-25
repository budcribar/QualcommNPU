# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.converter_base import ConverterFrontend
import argparse
import yaml
import ast

class QairtConverterFrontendArgParser(ConverterFrontend.ArgParserv2):
    def __init__(self, **kwargs):
        super(QairtConverterFrontendArgParser, self).__init__(conflict_handler='resolve', **kwargs)

        onnx_group = self.add_argument_group(title='Onnx Converter Options')
        onnx_group.add_argument('--onnx_no_simplification', dest='no_simplification', action='store_true', default=False,
                                         help="Do not attempt to simplify the model automatically. This may prevent some models from properly converting \n"
                                              "when sequences of unsupported static operations are present.")
        onnx_group.add_argument('--onnx_batch', dest='batch', type=int, default=None,
                                         help="The batch dimension override. This will take the first dimension of all "
                                              "inputs and treat it as a batch dim, overriding it with the value provided "
                                              "here. For example:\n"
                                              "--batch 6\n"
                                              "will result in a shape change from [1,3,224,224] to [6,3,224,224].\n"
                                              "If there are inputs without batch dim this should not be used and each input "
                                              "should be overridden independently using -d option for input dimension overrides.")
        onnx_group.add_argument('--onnx_define_symbol', dest='define_symbol', nargs=2, action='append',
                                         metavar=('SYMBOL_NAME', 'VALUE'),
                                         help="This option allows overriding specific input dimension symbols. For instance you "
                                              "might see input shapes specified with variables such as :\n"
                                              "data: [1,3,height,width]\n"
                                              "To override these simply pass the option as:\n"
                                              "--define_symbol height 224 --define_symbol width 448\n"
                                              "which results in dimensions that look like:\n"
                                              "data: [1,3,224,448]")
        onnx_group.add_argument('--dump_inferred_model', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        onnx_group.add_argument('--dump_value_info', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        # hidden flag for onnx relay converter
        onnx_group.add_argument('--use_onnx_relay', action='store_true', default=False,
                                         help=argparse.SUPPRESS)
        onnx_group.add_argument('--dump_relay', type=str, default=None,
                                         help=argparse.SUPPRESS)

        tf_group = self.add_argument_group(title='TensorFlow Converter Options')
        # add command-line options custom to tensorflow converter
        tf_group.add_argument('--tf_no_optimization', dest='no_optimization', action='store_true', default=False,
                                       help="Do not attempt to optimize the model automatically.")
        tf_group.add_argument("--tf_show_unconsumed_nodes", dest='show_unconsumed_nodes', action="store_true",
                                       help="Displays a list of unconsumed nodes, if there any are found. Nodes"
                                            "which are unconsumed do not violate the structural fidelity of the"
                                            "generated graph.",
                                       default=False)
        tf_group.add_argument("--tf_saved_model_tag", dest='saved_model_tag', type=str, action='store',
                                       help="Specify the tag to seletet a MetaGraph from savedmodel. ex: "
                                            "--saved_model_tag serve. Default value will be 'serve' when it "
                                            "is not assigned.",
                                       default="serve")
        tf_group.add_argument("--tf_saved_model_signature_key", dest='saved_model_signature_key', type=str, action='store',
                                       help="Specify signature key to select input and output of the model. "
                                            "ex: --saved_model_signature_key serving_default. Default value "
                                            "will be 'serving_default' when it is not assigned",
                                       default="serving_default")
        tf_group.add_argument("--tf_validate_models", dest='validate_models', action="store_true",
                                       help="Validate the original TF model against optimized TF model.\n"
                                            "Constant inputs with all value 1s will be generated and will be used \n"
                                            "by both models and their outputs are checked against each other.\n"
                                            "The %% average error and 90th percentile of output differences will be calculated for this.\n"
                                            "Note: Usage of this flag will incur extra time due to inference of the models.")
        # TODO: remove once QNN supports known LSTM variants completely (such as multiple time-steps)
        # Added as a workaround to match lstm nodes as low-level ops
        tf_group.add_argument("--disable_match_lstms", action='store_true', default=False,
                                       help=argparse.SUPPRESS)

        tflite_group = self.add_argument_group(title='Tflite Converter Options')
        tflite_group.add_argument('--tflite_signature_name', dest='signature_name', type=str, default="",
                                   help='Use this option to specify a specific Subgraph signature to convert')
        tflite_group.add_argument('--partial_graph_input_name', action='append',
                                   help=argparse.SUPPRESS)
        tflite_group.add_argument('--dump_relay', type=str, default=None,
                                   help=argparse.SUPPRESS)

        pytorch_group = self.add_argument_group(title='PyTorch Converter Options')
        pytorch_group.add_argument('--pytorch_custom_op_lib', type=str, default="",
                                    help=argparse.SUPPRESS)
        pytorch_group.add_argument('--dump_relay', type=str, default=None,
                                   help=argparse.SUPPRESS)

    # Convert argsv2 (from quart_converter which accepts i/o yaml) to argsv1 (used by SNPE/QNN)

def convert_args_v2_to_v1(args):
    args_dict = vars(args)

    # input_dims is parsed as [['ip1', 'a,b,c,d'], ['ip1', 'd,e,f,g']]
    input_dims = None
    input_encoding = []
    input_layout = []
    input_dtype = []
    output_names = []
    user_custom_io = []
    # in case user provides multiple dimensions for an input, network specialization will be enabled (supported only
    # in onnx) and input_dims will be populated as [['ip1', ((a,b,c), (d,e,f))], ['ip2', ((a',b',c'), (d',e',f'))]]
    network_specialization = False

    if args.io_config:
        f = open(args.io_config)
        io_config_dict = yaml.safe_load(f)

        input_layout_dict = {}
        output_layout_dict = {}


        if 'Input Tensor Configuration' in io_config_dict:
            for i in range(len(io_config_dict['Input Tensor Configuration'])):
                for key, val in io_config_dict['Input Tensor Configuration'][i].items():
                    if key == 'Name':
                        if val:
                            name = str(val)
                    elif key == 'Src Model Parameters':
                        if 'DataType' in val and val['DataType']:
                            input_dtype.append([name, val['DataType']])
                        if 'Layout' in val and val['Layout']:
                            input_layout.append([name, val['Layout']])
                            input_layout_dict[name] = val['Layout']
                    elif key == 'Desired Model Parameters':
                        if 'Shape' in val and val['Shape']:
                            if input_dims is None:
                                input_dims = []

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: 1
                            if isinstance(val["Shape"], int):
                                val["Shape"] = "(" + str(val['Shape']) + ",)"

                            dim = ast.literal_eval(val['Shape'])

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: (1)
                            if isinstance(dim, int):
                                dim = (dim,)

                            if type(dim[0]) is tuple:
                                network_specialization = True
                            input_dims.append([name, dim])

                        custom_io_options = dict()
                        custom_io_options['IOName'] = name
                        if 'DataType' in val and val['DataType']:
                            custom_io_options['Datatype'] = val['DataType']
                        if 'Layout' in val and val['Layout']:
                            custom_io_options['Layout'] = dict()
                            custom_io_options['Layout']['Custom'] = val['Layout']
                            # Get the model layout corresponding to the custom layout for current input
                            if name in input_layout_dict:
                                custom_io_options['Layout']['Model'] = input_layout_dict[name]
                        # if any of the quant params are provided
                        if 'QuantParams' in val and (val['QuantParams']['Scale'] or val['QuantParams']['Offset']):
                            custom_io_options['QuantParam'] = val['QuantParams']
                            custom_io_options['QuantParam']['Type'] = 'QNN_DEFINITION_DEFINED'
                        if len(custom_io_options) > 1:
                            user_custom_io.append(custom_io_options)
                        if 'Color Conversion' in val and val['Color Conversion']:
                            input_encoding.append([name, val['Color Conversion']])

        if 'Output Tensor Configuration' in io_config_dict:
            for i in range(len(io_config_dict['Output Tensor Configuration'])):
                for key, val in io_config_dict['Output Tensor Configuration'][i].items():
                    if key == 'Name':
                        if val:
                            output_names.append(str(val))
                            name = str(val)
                    elif key == 'Src Model Parameters':
                        if 'Layout' in val and val['Layout']:
                            output_layout_dict[name] = val['Layout']
                    elif key == 'Desired Model Parameters':
                        custom_io_options = dict()
                        custom_io_options['IOName'] = name
                        if 'Layout' in val and val['Layout']:
                            custom_io_options['Layout'] = dict()
                            custom_io_options['Layout']['Custom'] = val['Layout']
                            # Get the model layout corresponding to the custom layout for current output
                            if name in output_layout_dict:
                                custom_io_options['Layout']['Model'] = output_layout_dict[name]
                        if 'DataType' in val and val['DataType']:
                            custom_io_options['Datatype'] = val['DataType']
                        # if any of the quant params are provided
                        if 'QuantParams' in val and (val['QuantParams']['Scale'] or val['QuantParams']['Offset']):
                            custom_io_options['QuantParam'] = val['QuantParams']
                            custom_io_options['QuantParam']['Type'] = 'QNN_DEFINITION_DEFINED'
                        if len(custom_io_options) > 1:
                            user_custom_io.append(custom_io_options)


    # update following args only if they were not provided on the commandline
    if not args_dict['input_dim']:
        # convert name:str, dim:tuple to name:str, dim:str if network specialization is disabled
        if input_dims and not network_specialization:
            for i in range(len(input_dims)):
                # convert tuple of dimension to comma separated string
                if type(input_dims[i][1]) is tuple:
                    input_dims[i][1] = ','.join(map(str, input_dims[i][1]))
                # remove whitespaces if any from string of dimension
                elif isinstance(input_dims[i][1], str):
                    input_dims[i][1] = input_dims[i][1].replace(" ", "")

        args_dict["input_dim"] = input_dims

    if not args_dict['input_layout']:
        args_dict['input_layout'] = input_layout
    if not args_dict['input_dtype']:
        args_dict['input_dtype'] = input_dtype
    if not args_dict['input_encoding']:
        args_dict['input_encoding'] = input_encoding

    # following arguments will be unused
    args_dict['input_type'] = []
    args_dict['dump_custom_io_config_template'] = ""

    if not args_dict['out_names']:
        args_dict['out_names'] = output_names

    args_dict['user_custom_io'] = user_custom_io

    # populate preserve_io_arg with [['layout']] to apply it to all inputs/outputs
    args_dict['preserve_io'] = [['layout']]
    if args.disable_preserve_io:
        args_dict['preserve_io'] = []

    return argparse.Namespace(**args_dict)
