# ==============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import argparse
import re
import sys

# tvm, relay
try:
    import tvm
    from tvm import relay
except ModuleNotFoundError as e:
    print("Error while importing Relay...\n")
    raise e
except ImportError as e:
    print("TVM not found in PYTHONPATH. Ensure PYTHONPATH includes <path/to/tvm>/python.\n"
          "You can download and install TVM from https://tvm.apache.org/docs/install/from_source.html\n")
    sys.exit(1)
except Exception as e:
    print("Error while importing TVM...\n")
    raise e

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ConstantOp,
    Op,
)
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
from qti.aisw.converters.common.utils.converter_utils import (
    converter_type,
    log_assert,
    log_debug1,
    log_debug2,
    log_debug3,
    log_error,
    log_info,
    log_verbose,
    log_warning,
)

from qti.aisw.converters.relay.importers.relay_importer import RelayImporter
from qti.aisw.converters.relay.utils import get_key_from_expr
from .translations import RelayTranslations


def get_op_type(op_type):
    # Some op type names are repeated since they are stored under a class hierarchy. In order
    # to ensure the correct translations are used these op classes leverage the full op names
    SPECIAL_CLASSES = ['qnn']
    if not op_type.split(('.'))[0] in SPECIAL_CLASSES:
        op_type = str(op_type).split(('.'))[-1]
    log_debug2("op_type in get_translation {}", op_type)
    return converter_type(op_type, 'relay')


def get_translation(expr, custom_op_factory: CustomOpFactory = None):
    span = expr.span
    if isinstance(expr.span, tvm.relay.SequentialSpan):
        # for activation we use last one since the order of spans in SequentialSpan is
        # topological order
        span = expr.span.spans[-1]
    # For Pytorch,
    #   The type name of Pytorch source op contains "::".
    #   Discard it along with the namespace because "::" has a different meaning in C++.
    # Other frontends are not affected because "::" is not present in the op type.
    span_op_type = span.op_type.split("::")[-1] if span else None
    span_domain_name = (
        span.op_type.split("::")[-2]
        if span and len(span.op_type.split("::")) > 1
        else None
    )
    if custom_op_factory and span_op_type in custom_op_factory.op_collection \
            and span_op_type == str(expr.op.name):
        op_type = get_op_type("custom")
    elif span_domain_name == "qti_aisw":
        op_type = get_op_type("qti_aisw")
    else:
        op_type = get_op_type(str(expr.op.name))
    if op_type in RelayTranslations.translations:
        return RelayTranslations.translations[op_type]
    else:
        raise TypeError("Unsupported Op type {}".format(expr.op.name))


class RelayConverterContext:
    """
    Class that contains all data structures and methods for Op Conversion
    """
    def __init__(self, quir_graph: IROpGraph, relay_params: dict, expr_to_source_info_dict: dict=None):
        self.relay_params = relay_params
        self.expr_to_source_info_dict = expr_to_source_info_dict
        if self.expr_to_source_info_dict:
            log_verbose("Output Names in expr_to_source_info Dict:")
            for expr, source_info in self.expr_to_source_info_dict.items():
                op_name = source_info.get_op_name()
                if op_name:
                    log_verbose("\t {}: {}", op_name, source_info.get_output_names())
                else:
                    log_verbose("\t{}", source_info.get_output_names())
            log_verbose("\n")
        self.type_count = {}
        self.quir_graph = quir_graph

    def get_op_name(self, expr: relay.expr, op_type: str, legacy_translation_key: str = None):
        """
        Generates Op name that is unique using ref count per Op Type
        :param expr: Relay Expr
        :param op_type: QuIR Op Type
        :param legacy_translation_key: Legacy Python IR op type
        :return: Str
        """
        key = get_key_from_expr(expr)
        op_name = self.expr_to_source_info_dict[key].get_op_name()
        if not op_name:
            count = self.type_count.get(op_type, 0)
            self.type_count[op_type] = count + 1
            if legacy_translation_key:
                name_prefix_str = str(legacy_translation_key)
            else:
                name_prefix_str = str(op_type)
            op_name = "%s_%d" % (name_prefix_str, count)
        self.expr_to_source_info_dict[key].set_op_name(op_name)
        log_verbose("op_name {}", op_name)
        return op_name

    def get_input_names(self, expr: relay.expr):
        """
        Get Input Names for input Relay Expr. It uses recursive tree traversal to get output names of
        inputs to the Input Expr
        :param expr: Relay Expr
        :return: List of input names
        """
        inputs = []
        if isinstance(expr, relay.Call):
            for arg in expr.args:
                outputs = self.get_output_names(arg)
                log_verbose("Call outputs {}", outputs)
                inputs.extend(outputs)
        elif isinstance(expr, relay.Var):
            k = get_key_from_expr(expr)
            output_names = self.expr_to_source_info_dict[expr].get_output_names()
            if output_names:
                log_verbose("Var name {} outputs {}", expr.name_hint, output_names)
                inputs.append(output_names)
            else:
                raise KeyError("Span or Expr for {} not found in dictionary expr_to_source_info_dict".format(expr))
        elif isinstance(expr, relay.TupleGetItem):
            log_verbose("tuple item input index {}", expr.index)
            tuple_inputs = self.get_output_names(expr.tuple_value)[expr.index]
            log_verbose("Appending tuple item input {}", tuple_inputs)
            inputs.extend(tuple_inputs)
        elif isinstance(expr, relay.Tuple):
            for elem in expr.fields:
                log_verbose("inputs before Tuple {}", inputs)
                inputs.extend(self.get_output_names(elem))
                log_verbose("inputs after Tuple {}", inputs)
        else:
            raise TypeError("Unsupported Expr type {} for get_input_names".format(type(expr)))

        return inputs

    def get_input_shapes(self, expr: relay.expr):
        """
        Get Buffer Shapes from QuIR Graph for inputs to the Relay expr
        :param expr: Relay Expr
        :return: List of input shapes
        """
        inputs = self.get_input_names(expr)
        input_shapes = []
        for input_name in inputs:
            if self.quir_graph.has_buffer(input_name):
                input_shape = self.quir_graph.get_buffer(input_name).shape
            elif input_name in self.relay_params:
                input_shape = self.relay_params[input_name].shape
                input_shape = list(map(int, input_shape))
            else:
                raise KeyError("input_name {} is not found in graph buffers, nor relay_params".format(input_name))
            input_shapes.append(input_shape)
        log_verbose("input_shapes {}", *zip(inputs, input_shapes))
        return input_shapes

    def get_output_names(self, expr: relay.expr, num_outputs: int=None):
        """
        Get output names of given Relay Expr
        :param expr: Relay Expr
        :param num_outputs:
        :return:
        """

        key = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[key].get_output_names()
        if not output_names:
            if isinstance(expr, relay.Var):
                log_verbose("Var name {}", expr.name_hint)
                output_names = [expr.name_hint]
                self.expr_to_source_info_dict[key].set_output_names(output_names)
            elif isinstance(expr, relay.Tuple):
                output_names = []
                for elem in expr.fields:
                    log_verbose("tuple outputs before {}", output_names)
                    output_names.extend(self.get_output_names(elem))
                    log_verbose("tuple outputs after {}", output_names)
            elif isinstance(expr, relay.TupleGetItem):
                output_names = [self.get_output_names(expr.tuple_value)[expr.index]]
                log_verbose("Appending tuple item output {}", output_names)
            else:
                # expr is not in self.expr_to_source_info_dict
                if num_outputs:
                    output_names = self.generate_output_names(expr, num_outputs)
                    self.expr_to_source_info_dict[key].set_output_names(output_names)
                else:
                    log_error("Unknown expr:\n{}\ntype {}\n", expr, type(expr))
                    raise KeyError("Unknown Expr found while getting output names")
        else:
            if num_outputs is not None:
                log_assert(len(output_names)==num_outputs, "output_names not match num_outputs for expr:\n{}", expr)

        return output_names

    def get_output_datatype(self, expr: relay.expr, output_name: str):
        key = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[key].get_output_names()

        if output_names:
            if output_name in output_names and isinstance(expr, relay.Call):
                return expr.checked_type.dtype
        else:
            if isinstance(expr, relay.Tuple):
                for elem in expr.fields:
                    elem_dtype = self.get_output_datatype(elem, output_name)
                    if elem_dtype:
                        return elem_dtype
            else:
                return None

    def generate_output_names(self, expr: relay.expr, num_outputs: int):
        """
        Generate output tensor names for given Relay Expr since they were not already provided
        :param expr: Relay Expr
        :param num_outputs:
        :return:
        """
        k = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[k].get_output_names()
        if not output_names:
            output_names = [self.expr_to_source_info_dict[k].get_op_name() + '_' +
                            str(i) for i in range(num_outputs)]
            log_verbose("generated output names {}", output_names)
        return output_names

    def add_op_to_graph(self,
                        expr: relay.expr,
                        op: Op,
                        input_names: list,
                        output_names: list,
                        axis_formats: list=None,
                        idx: int=-1):
        """
        Add QuIR Op to QuIR OpGraph and update the dictionary of expr to output names
        :param expr: Relay Expr
        :param op: QuIR Op
        :param input_names: List of input names
        :param output_names: List of output names
        :param axis_formats:
        :param idx: Index in graph to insert the Node
        :return: QuIR OpNode
        """
        key = get_key_from_expr(expr)

        # add Constant Op for input_name in relay_param but not in the graph.
        for input_name in input_names:
            if not self.quir_graph.has_buffer(input_name):
                log_assert(input_name in self.relay_params,
                           "Input {} not found in Graph or Params", input_name)
                log_debug3("Adding ConstantOp for {} due to op {}", input_name, op.name)

                const_input_tensor = self.relay_params[input_name]
                if isinstance(const_input_tensor, (tvm.runtime.ndarray.NDArray, tvm.runtime.NDArray)):
                    const_input_tensor = const_input_tensor.asnumpy()
                self.quir_graph.add(ConstantOp(input_name, const_input_tensor),
                               input_names=[],
                               output_names=[input_name])

        return self.quir_graph.add(op, input_names, output_names, axis_formats, idx)


class RelayConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(RelayConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--dump_relay', type=str, default=None,
                                       help="Dump Relay ASM and Params at the path provided with the argument\n"
                                            "Usage: --dump_relay <path_to_dump>")
            # dry-run options
            self.add_optional_argument('--dry_run', default=False, action="store_true",
                                       help="Evaluates the model without actually converting any ops, and\n"
                                            " returns unsupported ops if any.")
            self.add_optional_argument('--dump_out_names', default=False, action="store_true",
                                       help="Dump output names mapped from QNN CPP stored names to converter used\n"
                                            "names and save to file 'model_output_names.json'.")

    def __init__(self, args, importer: RelayImporter=None, mod: tvm.IRModule=None, params: dict=None, **kwargs):
        super(RelayConverterFrontend, self).__init__(args,
                                                     **kwargs)
        self.dry_run = args.dry_run
        if self.dry_run:
            self.module_op_types = set()
        self.dump_out_names = False
        if hasattr(args, "dump_out_names"):
            self.dump_out_names = args.dump_out_names
        self.dump_out_path = args.output_path if args.output_path else args.input_network
        self.c_utils = ir_graph.IrUtils()

        self.importer = importer
        if self.importer and isinstance(self.importer, RelayImporter):
            self.relay_mod, self.relay_params, self.expr_to_source_info_dict, self.custom_op_factory =\
                self.importer.convert_to_relay(self.input_model_path,
                                               quantization_overrides=self.graph.user_quantization_overrides,
                                               dry_run = self.dry_run)
        else:
            mod = mod
            params = params
            if not mod or not params:
                raise SyntaxError("{} should be initialized with either an importer or with (mod, params). "
                                  "None of these provided.".format(self.__class__.__name__))
            self.expr_to_source_info_dict = {}
            self.relay_mod = mod
            self.relay_params = params

        if args.dump_relay:
            self.dump_relay_data(args)
        self.converter_context = RelayConverterContext(self.graph,
                                                       relay_params=self.relay_params,
                                                       expr_to_source_info_dict=self.expr_to_source_info_dict)

    def dump_relay_data(self, args):
        ########## debugging ###########
        import os
        if not args.dump_relay:
            path = '/'.join(os.path.realpath(args.input_network).split('/')[:-1])
        else:
            path = args.dump_relay

        log_verbose("Dumping Relay data at {}", path)

        full_mod_txt_path = os.path.join(path, "mod.txt")
        full_mod_json_path = os.path.join(path, "mod.json")
        self.dump_mod(full_mod_txt_path, full_mod_json_path)

        full_params_path = os.path.join(path, "params.txt")
        self.dump_params(full_params_path)
        ########## end debugging ###########

    def dump_params(self, file_name):
        with open(file_name, "w") as f:
            for k, v in self.relay_params.items():
                f.write(k)
                f.write(':')
                f.write(str(v))
                f.write('\n')

    def dump_mod(self, mod_txt_path, mod_json_path):
        with open(mod_txt_path, "w") as f:
            f.write(self.relay_mod.astext(show_meta_data=False))

        with open(mod_json_path, "w") as f:
            f.write(tvm.ir.save_json(self.relay_mod))

    def dump_out_tensor_names(self):
        from pathlib import Path
        import json
        dump_out_path = Path(self.dump_out_path)
        if dump_out_path.is_file():
            dump_out_path = dump_out_path.parent.joinpath("model_output_names.json")
        else:
            dump_out_path = dump_out_path.joinpath("model_output_names.json")
        output_names = {}
        for out in self.graph.output_names:
            output_names[self.c_utils.sanitize_name(out)] = out
        with open(dump_out_path, 'w') as f:
            json.dump(output_names, f, indent=2)

    def add_input(self, expr: relay.expr):
        if not isinstance(expr, relay.Var):
            return

        var_name = str(expr).split("\n")[1].lstrip("%v")

        k = get_key_from_expr(expr)

        if var_name in self.relay_params:
            output_names = self.expr_to_source_info_dict[k].get_output_names()
            if not output_names:
                self.expr_to_source_info_dict[k].set_output_names([var_name])
            param = self.relay_params[var_name]
            log_verbose("param {}", var_name)
            log_verbose("shape {}", param.shape)
        else:
            log_verbose("input {}", var_name)
            log_verbose("type {}", type(expr))
            log_verbose('shape {}', list(expr.type_annotation.shape))
            input_shape = [int(val) for val in expr.type_annotation.shape]
            input_dtype = expr.type_annotation.dtype
            input_node = self.graph.add_input(var_name, input_shape, input_dtype=input_dtype)
            output_names = self.expr_to_source_info_dict[k].get_output_names()
            if not output_names:
                self.expr_to_source_info_dict[k].set_output_names([var_name])

            # populate quantization info for input var
            key = get_key_from_expr(expr)
            encodings = self.expr_to_source_info_dict[key].get_encodings()
            if encodings:
                self.graph.set_overridden_encoding(input_node.op.name, encodings, is_param=False)

    def add_constant(self, expr: relay.expr):
        if not isinstance(expr, relay.Constant):
            return

        key = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[key].get_output_names()
        if not output_names:
            constant_name = self.converter_context.get_op_name(expr, 'relay_constant', "")
            self.expr_to_source_info_dict[key].set_output_names([constant_name])
        else:
            constant_name = output_names[0]
        # update relay_params
        constant_array = expr.data
        self.relay_params[constant_name] = constant_array

    def add_op(self, expr: relay.expr):
        if isinstance(expr, relay.Call):
            # op_name = str(expr.op.name).replace("nn.", "")
            # log_verbose("name {}", expr.op.name)
            log_debug1("")
            log_debug1("Relay Op name {}", expr.op)

            ##### DEBUG PRINTS #####
            attributes = {}
            if expr.attrs:
                for attr in expr.attrs.list_field_info():
                    attributes[attr.name] = {}
                    attributes[attr.name]['value'] = getattr(expr.attrs, attr.name)
            log_verbose("attributes:")
            for k, v in attributes.items():
                log_verbose("\t{}:{}", k, v)
            ##### END DEBUG #####

            translation = get_translation(expr, self.custom_op_factory)
            translation.add_op(expr,
                               self.graph,
                               converter_context=self.converter_context,
                               relay_params=self.relay_params)
        else:
            pass

    def visit_module(self, expr: relay.expr):
        log_debug2("")
        log_debug2("##### NEW OP Translation #####")
        if isinstance(expr, relay.Var):
            self.add_input(expr)
        elif isinstance(expr, relay.Constant):
            self.add_constant(expr)
        elif isinstance(expr, relay.Call):
            self.add_op(expr)
        else:
            log_verbose("{}", type(expr))

        log_debug2("\n")

    def get_module_op_types(self, expr: relay.expr):
        if isinstance(expr, relay.Call):
            self.module_op_types.add(get_op_type(str(expr.op.name)))
        else:
            pass

    def enable_preserve_io(self):
        # --custom_io has higher precedence than --preserve_io. Skip the tensors for which dtype is
        # supplied using the --custom_io option.
        tensors_having_custom_dtypes = []
        if self.graph.user_custom_io:
            for entry in self.graph.user_custom_io:
                if "Datatype" in entry:
                    tensors_having_custom_dtypes.append(str(entry['IOName']))

        for arg in self.graph.preserve_io:
            if self.graph.preserve_io_datatype_passed == 1 and arg[0] == 'datatype':
                for buffer_name in arg[1:]:
                    if buffer_name not in tensors_having_custom_dtypes:
                        self.graph.preserve_datatype_tensors[buffer_name] = None

        # self.graph.preserve_io_datatype_passed = 1 indicates that user intends to preserve datatype only for the specified tensors
        # self.graph.preserve_io_datatype_passed = 2 indicates that user intends to preserve datatype for all the input and output tensors

        # relay class reference: https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1relay_1_1Var.html
        # inputs
        for param in self.relay_mod["main"].params:
            if ((self.graph.preserve_io_datatype_passed == 1 and param.name_hint in self.graph.preserve_datatype_tensors) or \
                    self.graph.preserve_io_datatype_passed == 2) and param.name_hint not in tensors_having_custom_dtypes:
                self.graph.preserve_datatype_tensors[param.name_hint] = param.type_annotation.dtype

        # outputs
        for output_name in self.graph.output_names:
            if ((self.graph.preserve_io_datatype_passed == 1 and output_name in self.graph.preserve_datatype_tensors) or \
                    self.graph.preserve_io_datatype_passed == 2) and output_name not in tensors_having_custom_dtypes:
                self.graph.preserve_datatype_tensors[output_name] = self.converter_context.get_output_datatype(self.relay_mod["main"].body, output_name)
        for key in self.graph.preserve_datatype_tensors:
            if not self.graph.preserve_datatype_tensors[key]:
                log_error("Cannot fetch the datatype for the tensor \"{}\" from relay expression.".format(key))

        # Throw an error if there is a conflict between the dtype passed using the --input_dtype option and the original dtype
        for k in self.graph.input_dtypes_dict:
            if k in self.graph.preserve_datatype_tensors and self.graph.input_dtypes_dict[k] != self.graph.preserve_datatype_tensors[k]:
                log_error("Datatype mismatch for tensor %s. %s datatype set with --input_dtype and %s datatype set with --preserve_io!" \
                        % (k, str(self.graph.input_dtypes_dict[k]), self.graph.preserve_datatype_tensors[k]))
                sys.exit(-1)

        for k in self.graph.preserve_datatype_tensors:
            if self.graph.preserve_datatype_tensors[k] == None:
                log_error("Graph does not have the tensor %s" % (k))
                sys.exit(-1)

    def convert_to_ir(self):
        def visit_module_wrapper(expr: relay.expr):
            """
            The second parameter of relay.analysis.post_order_visit requires a function that only
            takes relay.expr as its argument. As for self.visit_module, it is defined as a method
            and must set `self` as its first parameter, so we can not directly pass visit_module
            to relay.analysis.post_order_visit.
            Here we create the wrapper function to fulfill the above requirement.
            """
            self.visit_module(expr)

        def get_module_op_types_wrapper(expr: relay.expr):
            """
            The wrapper function to fulfill the argument requirement in relay.analysis.post_order_visit
            """
            self.get_module_op_types(expr)

        if self.dry_run:
            relay.analysis.post_order_visit(self.relay_mod["main"], get_module_op_types_wrapper)
            missing = [op_type for op_type in self.module_op_types if op_type not in RelayTranslations.translations]
            if missing:
                msg = "[QNNIR Dryrun] The following operators are not implemented: {}".format(missing)
                raise NotImplementedError(msg)
            else:
                log_info("[QNNIR Dryrun] PASS\n"
                         "| All operators in current TVM relay module are supported in QNNIR.")
                sys.exit(0)
        relay.analysis.post_order_visit(self.relay_mod["main"], visit_module_wrapper)
        self.graph.eval_macs_params()

        # If --out_node gives non-existing output, remove them and print warnings
        relay_output_names = self.converter_context.get_output_names(self.relay_mod["main"].body)
        if self.graph.output_names:
            updated_output_names = []
            for output_name in self.graph.output_names:
                if output_name in relay_output_names:
                    updated_output_names.append(output_name)
                else:
                    log_warning("The out node '{}' does not exist. Hence remove it.".format(output_name))
            if updated_output_names:
                self.graph.output_names = updated_output_names
            else:
                self.graph.output_names = relay_output_names
        else:
            self.graph.output_names.extend(relay_output_names)

        if self.dump_out_names:
            self.dump_out_tensor_names()

        if self.graph.preserve_io_datatype_passed:
            self.enable_preserve_io()

        return self.graph

    def convert(self):
        # Wrapper for combination of convert_to_relay and convert_to_ir
        return self.convert_to_ir()
