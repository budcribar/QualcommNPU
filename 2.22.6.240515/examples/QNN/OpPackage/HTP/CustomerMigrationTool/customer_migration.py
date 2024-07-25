#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

#!/usr/bin/env python3
"""Customer Migration Tool
A tool to assist customers with Custom OpPackages to
maximize compatibility with current and future changes to
the hexagon-nn-v3 (HTP Core) backend.

The first application of this tool is to assist with new features
that try to land tensors in the proper layout (Crouton vs Non-Crouton/Flat)
and in the proper memory location (TCM or Main Memory/DDR)
"""
import argparse
from collections import Counter
import dataclasses
from enum import Enum
from multiprocessing import Value
import os
import re
import shutil
import subprocess
from typing import Dict, Iterable, List, Set, Tuple, Union


def resolve_to_integer(expr_str: str) -> int:
    """This replaces unsafe `eval` calls and turns
    simple integer expressions into their true values"""
    while "(" in expr_str:
        start = expr_str.index("(")
        end = expr_str.index(")")
        # remove parens and attempt to resolve
        result = resolve_to_integer(expr_str[start + 1 : end])
        # update with our result from symbol resolution
        expr_str = str(expr_str[0:start] + str(result) + expr_str[end + 1 :])

    # then, insert spaces into our expr string for splitting up the work
    for operator in ("-", "+"):
        if operator in expr_str:
            expr_list = expr_str.split(operator)
            expr_str = f" {operator} ".join(expr_list)

    # split it up to get our individual symbols
    symbols = expr_str.split()

    # pylint: disable=consider-using-enumerate
    for index in range(len(symbols)):
        if symbols[index].startswith("0x"):
            return str(int(symbols[index], base=16))
        if symbols[index].startswith("0b"):
            return str(int(symbols[index], base=2))

    if len(symbols) == 1:
        return int(symbols[0])
    if symbols[1] == "+":
        return int(symbols[0]) + int(symbols[2])
    if symbols[1] == "-":
        return int(symbols[0]) - int(symbols[2])
    assert False, f"Could not resolve expression: {expr_str}"


def resolve_to_arguments(expr_str: str) -> List[str]:
    """Split up a string of expressions separated by commas
    into separate arguments"""
    args = []
    lparen = 0
    start_index = 0
    curr_index = 0
    for _curr_index, char_found in enumerate(expr_str):
        curr_index = _curr_index
        if char_found == "(":
            lparen += 1
        elif char_found == ")":
            assert lparen != 0, f"parens malformed in expr {expr_str} at {curr_index}"
            lparen -= 1
        elif char_found == "," and lparen == 0:
            args.append(expr_str[start_index:curr_index].strip())
            start_index = curr_index + 1
    args.append(expr_str[start_index:].strip())
    assert lparen == 0, f"parens malformed in expr {expr_str} at {curr_index}"
    return args


class Op:
    """Class to support easy parsing of an operator's key details
    - opstr (str): An Op's "name" used to reference a given operator
      within our optimizer BEFORE a concrete implementation is chosen
    - inputs (Union["Op",str]): The child Ops (or strs that are bound
      to unseen Ops) within a given input pattern.
    - n_outputs (int): Typically Ops will default to having a single
      unnamed output, but occasionally we discover that an Op has
      several outputs based upon info we find in the rules or
      registration information (OpMultiOut in rules, or
      non-const parameters in function headers for registrations)."""

    _process_safe_counter = Value("i", 0)

    def __init__(self, opstr: str, inputs: Union["Op", str], n_outputs: int = 1):
        with self.__class__._process_safe_counter.get_lock():
            self._unique_id = self.__class__._process_safe_counter.value
            self.__class__._process_safe_counter.value += 1
        self.opstr = opstr
        for param in inputs:
            assert isinstance(param, (Op, str)), "inputs may only be strs or Ops"
        self.inputs = inputs
        self.n_outputs = n_outputs

    def __str__(self) -> str:
        return f"{self.opstr}({','.join((str(p) for p in self.inputs))})"

    def __eq__(self, other: "Op") -> bool:
        return isinstance(other, Op) and self.unique_id == other.unique_id

    def __hash__(self) -> int:
        return self.unique_id

    @property
    def unique_id(self) -> int:
        """Get process-safe unique id for an Op"""
        return int(self._unique_id)

    @property
    def subgraph_ops(self) -> List["Op"]:
        """Return all subgraph_ops and the node itself"""
        return [self] + [succ for p in self.inputs if isinstance(p, Op) for succ in p.subgraph_ops]

    def contains_opstrs(self, opstrs: Iterable[str]) -> bool:
        """Check if this Op or any of its children contain any of the
        enumerated opstrs"""
        return self.opstr in opstrs or any(
            child.contains_opstrs(opstrs) for child in self.inputs if isinstance(child, Op)
        )

    def structural_match(self, other) -> Dict["Op", "Op"]:
        """For 2 Ops, attempt a structural mapping of 2 Op trees,
        so that we can approximatenwhat Ops will transform into what
        other Ops

        An example of this could be:
            Op("A", Op("B"), "some_str")
             ->  Op("C", Op("D"), Op("E"))
            resulting in the mapping
            {Op("A"):Op("C"), Op("B"):Op("D")}
        Return None if such a mapping is impossible to complete
        """
        if not isinstance(other, Op) or len(self.inputs) != len(other.inputs):
            return None
        op_map = {self: other}
        min_arity = min(len(self.inputs), len(other.inputs))
        for index in range(min_arity):
            s_param = self.inputs[index]
            o_param = other.inputs[index]
            if isinstance(s_param, Op) and isinstance(o_param, Op):
                ret = s_param.structural_match(o_param)
                if not ret:
                    return None
                op_map.update(ret)
        return op_map


@dataclasses.dataclass
class SimpleRule:
    """A subset of the features of a typical optimization rule
    that makes it easier to process common requests for customer
    migration purposes. Ignores constraints, flags, and many
    intermediate expressions that don't pertain to Op structure"""

    filename: str
    lineno: int
    priority: int
    pattern: Op
    result: Tuple[Op, str]

    @property
    def all_ops(self) -> List[Op]:
        """return all Ops inside of this rule's input/outputs"""
        if isinstance(self.result, Op):
            return self.pattern.subgraph_ops + self.result.subgraph_ops
        return []


Diagnostic = Enum("Diagnostic", ["ERROR", "WARNING", "INFO"])
Constraints = Enum("Constraints", ["Flat", "Crouton", "Tcm", "MainMemory"])


def _extract_constraints(argtype_set, is_output) -> Set[Constraints]:
    """Extract all DEF_TENSOR_PROPERTIES constraints for a given argument"""
    constraints = set()
    # Case 0) check if we're dealing with `Tensor``, which is treated as
    # Flat + MM for outputs (absent other tensor definitions) and as
    # unconstrained for inputs (can assume any layout/mem location)
    if "Tensor" in argtype_set:
        if not is_output:
            return constraints  # unconstrained on inputs
        # we are an output, thus Tensor becomes Flat+MM, so need to see if
        # other impl options are available
        constraints = {Constraints.Flat, Constraints.MainMemory}
        if any("TCM" in argtype for argtype in argtype_set):
            constraints.remove(Constraints.MainMemory)
        if any("Crouton" in argtype for argtype in argtype_set):
            constraints.remove(Constraints.Flat)
        return constraints

    # Case 1) arg has only TCM tensor types
    # (allow generic Tensor type for inputs to be unconstrained)
    if all("TCM" in argtype for argtype in argtype_set):
        constraints.add(Constraints.Tcm)
    # Case 2) arg has only MainMemory tensor types
    if all("TCM" not in argtype for argtype in argtype_set):
        constraints.add(Constraints.MainMemory)
    # Case 3) arg has only Crouton tensor types
    if all("Crouton" in argtype for argtype in argtype_set):
        constraints.add(Constraints.Crouton)
    # Case 4) arg has only Flat (AKA non-Crouton) tensor types
    if all("Crouton" not in argtype for argtype in argtype_set):
        constraints.add(Constraints.Flat)
    return constraints


def _aggregate_opstrs_and_argtypes(registrations: List[Dict]) -> Dict[str, List[Set[str]]]:
    """Aggregate all of our opstrs and their argument types across registrations.
    This gives us a better picture of potential types for each tensor, BUT is
    incomplete, since it doesn't account for the relationships between arguments.
    For the majority of cases, this is fine."""
    opstr2argtype_sets = {}
    for registration in registrations:
        opstr, argtypes = registration["opstr"], registration["arg_tensor_types"]
        if opstr not in opstr2argtype_sets:
            opstr2argtype_sets[opstr] = []
        for index, argtype in enumerate(argtypes):
            if len(opstr2argtype_sets[opstr]) <= index:
                opstr2argtype_sets[opstr].append(set())
            opstr2argtype_sets[opstr][index].add(argtype)
    return opstr2argtype_sets


def get_def_tensor_properties(registrations: List[Dict], opstr2n_outputs: Dict[str, int]) -> List[str]:
    """Function for generating suggested DEF_TENSOR_PROPERTIES based on
    existing OpPackage-defined functions and tensor info"""
    # Step 1) Aggregate argtypes available across all registrations
    opstr2argtype_sets = _aggregate_opstrs_and_argtypes(registrations)
    # Step #2) Convert aggregated potential Tensor types for an opstr to constraints
    opstr2arg_constraints = {}
    for opstr, argtype_sets in opstr2argtype_sets.items():
        opstr2arg_constraints[opstr] = [
            _extract_constraints(argtype_set, is_output=index < opstr2n_outputs[opstr])
            for index, argtype_set in enumerate(argtype_sets)
        ]
    # Step #3) Create a DEF_TENSOR_PROPERTIES for every opstr
    # based on the constraints on each argument
    dtps = []
    for opstr, arg_constraints in opstr2arg_constraints.items():
        n_outputs = opstr2n_outputs[opstr]
        # 3A) Create preamble with correct number of args
        opstr_args = [f'"in{i-n_outputs}"' for i in range(n_outputs, len(arg_constraints))]
        dtp = f"DEF_TENSOR_PROPERTIES(Op({opstr},{','.join(opstr_args)})"
        # 3B) Check to see if we need an Outputs(...) clause (for multi-output case)
        if n_outputs > 1:
            dtp += ",Outputs(" + ",".join(f'"out{i}"' for i in range(n_outputs)) + ")"
        # 3C) Check for all our constraint modes
        member_args = {
            "Flat": [i for i, c in enumerate(arg_constraints) if Constraints.Flat in c],
            "Crouton": [i for i, c in enumerate(arg_constraints) if Constraints.Crouton in c],
            "MainMemory": [i for i, c in enumerate(arg_constraints) if Constraints.MainMemory in c],
            "Tcm": [i for i, c in enumerate(arg_constraints) if Constraints.Tcm in c],
        }
        # 3D) Construct constraints
        for wrapper, arg_ids in member_args.items():
            if arg_ids:
                if n_outputs == 1:
                    relevant_args = [f'"in{i-n_outputs}"' if i != 0 else '"*"' for i in arg_ids]
                else:
                    relevant_args = [f'"in{i-n_outputs}"' if i >= n_outputs else f'"out{i}"' for i in arg_ids]
                dtp += f",{wrapper}({','.join(relevant_args)})"
        dtps.append(dtp + ")")  # add closing paren
    return dtps


def get_rule_recommendations(
    args: argparse.Namespace, simple_rule: SimpleRule
) -> List[Tuple[Diagnostic, SimpleRule, str, str]]:
    """Run our simple rule through various checks to see how compliant
    they are with our current internal features."""
    recommendations = []
    # Check #1: Any migration-related ops referenced outside of expected stage
    migration_ops = {
        "flat_to_vtcm",
        "flat_from_vtcm",
        "crouton_to_vtcm",
        "crouton_from_vtcm",
        "ForceFormat_Flat",
        "ForceFormat_Crouton",
        "ConvLayer.opt.activations_to_vtcm",
        "ConvLayer.opt.activations_from_vtcm",
    }
    # Capture as many ways the the preprocessor might emit migraiton ops as possible
    migration_ops = (
        {f"{args.package_name}::{opstr}" for opstr in migration_ops}
        | {f"{args.package_name}{opstr}" for opstr in migration_ops}
        | migration_ops
    )
    if simple_rule.pattern.contains_opstrs(migration_ops) or (
        isinstance(simple_rule.result, Op) and simple_rule.result.contains_opstrs(migration_ops)
    ):
        # Case 1a) ERROR -> We have a migration op AFTER central migration zone
        if simple_rule.priority > args.migration_phase:
            recommendations.append(
                (
                    Diagnostic.ERROR,
                    simple_rule,
                    "References layout/placement-related Ops after LAYOUT_AND_PLACEMENT phase.",
                    "Suggest removing rule or moving components related to format\n"
                    + "and data location to LAYOUT_AND_PLACEMENT pass.",
                )
            )
        # Case 1b) WARNING -> We have a migration op BEFORE central migration zone
        elif simple_rule.priority < args.migration_phase:
            recommendations.append(
                (
                    Diagnostic.WARNING,
                    simple_rule,
                    "References layout/placement-related Ops before LAYOUT_AND_PLACEMENT phase.",
                    "Suggest removing rule or moving components related to format\n"
                    + "and data location to LAYOUT_AND_PLACEMENT pass.",
                )
            )

        # Case 1c) INFO -> We have a migration op within central migration pass
        else:  # (sr.priority == args.migration_phase
            recommendations.append(
                (
                    Diagnostic.INFO,
                    simple_rule,
                    "References layout/placement-related Ops within LAYOUT_AND_PLACEMENT phase.",
                    "HTP Core will attempt to respect these constraints on format/"
                    + "data location,\nbut consider if you can represent your "
                    + "personally-defined operator tensor attributes via DEF_TENSOR_PROPERTIES",
                )
            )

    # Check #2: Did we introduce a migration op?
    if not simple_rule.pattern.contains_opstrs(migration_ops) and (
        isinstance(simple_rule.result, Op) and simple_rule.result.contains_opstrs(migration_ops)
    ):
        # Case 2a) WARNING -> We didn't have any migration ops before-hand, but now we do.
        recommendations.append(
            (
                Diagnostic.WARNING,
                simple_rule,
                "Introduces a layout/placement op where there wasn't one before.",
                "Suggest removing rule and writing a DEF_TENSOR_PROPERTIES.",
            )
        )

    # Check #3: Did we propagate a migration op?
    if simple_rule.pattern.contains_opstrs(migration_ops) and (
        isinstance(simple_rule.result, Op) and simple_rule.result.contains_opstrs(migration_ops)
    ):
        migration_pattern_opstrs = Counter(
            op.opstr for op in simple_rule.pattern.subgraph_ops if op.opstr in migration_ops
        )
        migration_result_opstrs = Counter(
            op.opstr for op in simple_rule.result.subgraph_ops if op.opstr in migration_ops
        )
        difference_counter = migration_result_opstrs - migration_pattern_opstrs
        # Case 3a) Any migration-related ops in result came from pattern (no propagation)
        if sum(difference_counter.values()) == 0:
            recommendations.append(
                (
                    Diagnostic.WARNING,
                    simple_rule,
                    "This rule appears to contain equal numbers of"
                    + " LAYOUT_AND_PLACEMNT-related ops on input and output.",
                    "Suggest removing components of this rule containing migration ops,\n"
                    + "as it may interfere with the work of Central Migration.",
                )
            )

        # Case 3b) There are new migration-related ops in the result
        elif sum(difference_counter.values()) > 0:
            recommendations.append(
                (
                    Diagnostic.ERROR,
                    simple_rule,
                    "This rule appears to introduce new LAYOUT_AND_PLACEMENT-related ops.",
                    "Suggest removing rule entirely, as it may interfere with " + "centralized migration on HTP Core.",
                )
            )
        else:  # difference_counter.total() < 0
            recommendations.append(
                (
                    Diagnostic.ERROR,
                    simple_rule,
                    "This rule appears to remove LAYOUT_AND_PLACEMENT-related ops.",
                    "Suggest removing rule entirely, as it may interfere with " + "centralized migration on HTP Core.",
                )
            )

    return recommendations


def _resolve_to_simplified_pattern(
    args: argparse.Namespace, pattern: str, let_binding_map: Dict[str, Op]
) -> Union[Op, str]:
    """basic depth-first traversal to resolve pattern and grab let bindings"""
    # Check if we are dealing with a subexpression that we expect to be functional
    if "(" in pattern and ")" in pattern:
        first_lparen_index = pattern.index("(")
        final_rparen_index = pattern.rindex(")")
        assert first_lparen_index < final_rparen_index, f"Final ) precedes first ( in {pattern}"
        assert pattern.count("(") == pattern.count(")"), f"Unbalanced parens in {pattern}"
        inputs = resolve_to_arguments(pattern[first_lparen_index + 1 : final_rparen_index])
        inputs = [_resolve_to_simplified_pattern(args, p, let_binding_map) for p in inputs]
        if pattern.startswith("Op"):
            return Op(inputs[0].strip().strip('"'), inputs[1:])
        if pattern.startswith("LET"):
            assert len(inputs) == 2, f"LET binding statement is malformed: {pattern}"
            assert isinstance(inputs[1], Op), "LET binding expects Op-type as second parameter"
            let_binding_map[inputs[0].strip()] = inputs[1]
            return inputs[1]
        # unhandled case we missed --> must account for
        assert False, f"Unknown Start of pattern: {pattern}"
    # We are now looking at a non-functional expression in our pattern
    # AKA --> likely a string or number
    assert not pattern.startswith("Op") and not pattern.startswith("LET")
    return pattern.strip().strip('"')  # strip any quotes from strs used as reference


def _resolve_to_simplified_result(
    args: argparse.Namespace, result: str, let_binding_map: Dict[str, Op]
) -> Union[Op, str]:
    # pylint: disable=too-many-return-statements
    # Check if we are dealing with a subexpression that we expect to be functional
    if "(" in result and ")" in result:
        first_lparen_index = result.index("(")
        final_rparen_index = result.rindex(")")
        assert first_lparen_index < final_rparen_index, f"Final ) precedes first ( in {result}"
        assert result.count("(") == result.count(")"), f"Unbalanced parens in {result}"
        inputs = resolve_to_arguments(result[first_lparen_index + 1 : final_rparen_index])
        # WITH_* statements grab final param in simplified form
        if result.startswith("WITH_") or result.startswith("AUTOSPLIT("):
            return _resolve_to_simplified_result(args, inputs[-1], let_binding_map)
        if (
            # TYPICAL_SLICE will be simplified to just the first param
            result.startswith("TYPICAL_SLICE")
            or result.startswith("AUTOSPLIT_SLICE")
            or result.startswith("OP_ITER")
        ):
            return _resolve_to_simplified_result(args, inputs[0], let_binding_map)
        if (
            # SHAPEFN_APPLY will generate a whole new value, so treat as str
            result.startswith("SHAPEFN_APPLY")
            or result.startswith("AUTOSPLIT_SHAPEFN_APPLY")
            or result.startswith("EXTERNAL_REPLACE")
            or result.startswith("gen_Const")
            or result.startswith("gen_Shape")
        ):
            # NOTE: the below is not precisely correct, but we should consider it
            # a fair solution for now since we can sort of treat the above functions
            # as black boxes
            return Op(result, [])
        # WrapOp does a bit more than this, but we can treat it
        if result.startswith("WrapOp"):
            # like surrounding a node with the lhs, so swap WrapOp for Op
            inputs = [_resolve_to_simplified_result(args, p, let_binding_map) for p in inputs]
            return Op(inputs[0].strip().strip('"'), inputs[1:])
        # OpMultiOut is used for multi-output ops in the result
        if result.startswith("OpMultiOut"):
            # and will contain info on the number of outputs as the first parameter
            inputs = [_resolve_to_simplified_result(args, p, let_binding_map) for p in inputs]
            n_outputs = int(inputs[0])
            return Op(inputs[2].strip().strip('"'), inputs[3:], n_outputs=n_outputs)
        # NOTE: we are definitely missing cases here. We tested on OpPackage symbols we saw
        # on internal examples. If you encounter the assert toward the end that it's an unknown
        # symbol, you'll want to see which of these symbols it's most like and add it.
        # (worst case: either ignore it (pass) or treat it as an op: `return Op(result, [])`)
        assert result.startswith("Op"), f"Unknown Start of result: {result}"
        inputs = [_resolve_to_simplified_result(args, p, let_binding_map) for p in inputs]
        return Op(inputs[0].strip().strip('"'), inputs[1:])
    # for all other cases (likely without inputs) -> AKA strs that could be in our
    # let binding map, other strs, or ints
    if result.strip() in let_binding_map:
        return let_binding_map[result.strip()]
    return result.strip().strip('"')


def resolve_to_simplified_rule(args: argparse.Namespace, raw_rule: Dict[str, str]) -> SimpleRule:
    """Convert a rule containing various raw text attributes into a
    SimpleRule by simplifying the input pattern and output patterns
    by removing unneeded intermediate nodes that aren't Ops or
    str references to Ops"""
    let_binding_map = {}
    simplified_pattern = _resolve_to_simplified_pattern(args, raw_rule["pattern"], let_binding_map)
    # '*' tracks our output node at the root of input pattern
    let_binding_map['"*"'] = simplified_pattern
    simplified_result = _resolve_to_simplified_result(args, raw_rule["result"], let_binding_map)
    return SimpleRule(
        raw_rule["filename"], raw_rule["lineno"], raw_rule["priority"], simplified_pattern, simplified_result
    )


def _enhance_registration(
    registration: Dict[str, str], impl_name2arguments: Dict[str, str], impl_name2typenames: Dict[str, str]
) -> Dict[str, str]:
    """Add additional details to our discovered registrations based on information discovered
    about the function headers within the file."""
    # First, get the impl_name and template arguments for the registration
    function_re = re.compile(
        r"\(?(?P<impl_name>[a-zA-Z0-9_]+[a-zA-Z_^\(]*)" + r"(\<(?P<arguments>[a-zA-Z0-9_,\s:<>]+\>)?\)?)?"
    )
    function_match = function_re.search(registration["function"])
    argtypes = [ent["type"] for ent in impl_name2arguments[function_match["impl_name"].strip()]]
    if function_match["arguments"]:
        template_re = re.compile(
            r"(?P<template_var_name>[a-zA-Z_]+[a-zA-Z0-9_]*\s*\=\s*)?"
            + r"(?P<tensor_type>[a-zA-Z_]+[a-zA-Z0-9_]*\s*\,?\s*)"
        )
        template_matches = template_re.findall(function_match["arguments"])
        # Map template names to their types
        template_varnames = [ent["varname"] for ent in impl_name2typenames[function_match["impl_name"].strip()]]
        template_var_type = [tensor_type for _, tensor_type in template_matches]
        template_name2assigned_type = dict(zip(template_varnames, template_var_type))

        # Now fixup argtypes (TensorType -> QuantUint16Tensor)
        fixed_argtypes = [
            template_name2assigned_type[at] if at in template_name2assigned_type else at for at in argtypes
        ]
    else:
        fixed_argtypes = argtypes  # no change, likely since we don't have templated typenames

    # Finally, create an entry in our registration that contains the tensors at each argument
    registration["arg_tensor_types"] = fixed_argtypes

    registration["arg_is_output"] = [
        arg["is_output"] for arg in impl_name2arguments[function_match["impl_name"].strip()]
    ]
    return registration


DEF_OPT_RE = re.compile(r"__def_opt__\((.*)\)<<<\"(.*)\",\s*([0-9]+)>>>")


def _extract_rule(line: str) -> Dict[str, Union[str, int]]:
    def_opt_match = DEF_OPT_RE.search(line)
    if def_opt_match:
        priority, flags, rest = def_opt_match.group(1).split(",", 2)
        priority = resolve_to_integer(priority)  # preferred to python's eval()
        flags = flags.strip()
        filename = os.path.split(def_opt_match.group(2))[-1].strip()

        lineno = int(def_opt_match.group(3))
        rest_args = resolve_to_arguments(rest)
        assert len(rest_args) == 3, "Incorrect parse of pattern/constraint/result in rule"
        return {
            "filename": filename.strip(),
            "lineno": lineno,
            "priority": priority,
            "flag": flags.strip(),
            "pattern": rest_args[0].strip(),
            "constraint": rest_args[1].strip(),
            "result": rest_args[2].strip(),
        }
    # not def_opt_match:
    assert "__def_opt__" not in line, f"{filename}: Missed Optimization: {line}"
    assert not line.startswith("DEF_PACKAGE_OPTIMIZATION"), f"{filename}: Missed Optimization: {line}"
    return None


REG_OP_RE = re.compile(r"__reg_op__\((.*)\)<<<\"(.*)\",\s*([0-9]+)>>>")


def _extract_registration(line: str) -> Dict[str, Union[str, int]]:
    reg_op_match = REG_OP_RE.search(line.strip())
    if reg_op_match:
        function, opstr = reg_op_match.group(1).rsplit(",", 1)
        filename = os.path.split(reg_op_match.group(2))[-1].strip()
        lineno = int(reg_op_match.group(3))
        return {"filename": filename.strip(), "lineno": lineno, "function": function.strip(), "opstr": opstr.strip()}
    # not reg_op_match
    assert "__reg_op__" not in line, f"{filename}: Missed Op registration: {line}"
    assert not line.startswith("DEF_PACKAGE_OP"), f"{filename}: Missed Op registration: {line}"
    assert not line.startswith("REGISTER_OP"), f"{filename}: Missed Op registration: {line}"
    return None


ARGUMENT_RE = re.compile(
    r"(?P<is_input>const\s+)?\s*(?P<type>[a-zA-Z_]+[a-zA-Z0-9_<>]*)\s*&?\s*"
    + r"(?P<varname>[a-zA-Z_]+[a-zA-Z0-9_<>:]*)\s*,?\s*"
)


def _process_impl_args(impl_args: str) -> Dict[str, Union[bool, str]]:
    ret = []
    for is_input, tensor_type, varname in ARGUMENT_RE.findall(impl_args):
        # we know that something is an output based on if it has "const" in its var name
        ret.append({"is_output": len(is_input.strip()) == 0, "type": tensor_type, "varname": varname})
    return ret


TEMPLATE_OP_RE = re.compile(
    r"(template\s*\<(?P<typenames>\s*typename\s*[a-zA-Z0-9_\s,]+)\>)?\s*"
    + r"(GraphStatus|int)\s+(?P<impl_name>[a-zA-Z_]+[a-zA-Z0-9_^]*)\("
    + r"(?P<arguments>[\sa-zA-Z0-9_&,^<>:]+)\)"
)


def _extract_impl_name_args_and_typenames(lines) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    impl_name2arguments = {}
    impl_name2typenames = {}

    whole_file = "".join(line.strip() for line in lines)
    for _template, typenames, _graph_status, impl_name, impl_args in TEMPLATE_OP_RE.findall(whole_file):
        impl_name2arguments[impl_name] = _process_impl_args(impl_args)
        impl_name2typenames[impl_name] = _process_impl_args(typenames)

    return impl_name2arguments, impl_name2typenames


DTP_RE = re.compile(r"__dtp__\(([^<]*)\s*\)\s*<<<\s*\"[^>]*\"\s*,\s*[0-9]+>>>")


def _extract_dtp_opstr(file_contents, filename) -> List["DefTensorProperties"]:
    discovered_dtp_opstrs = []
    for opstr_contents in DTP_RE.findall(file_contents):
        discovered_dtp_opstrs.append(re.search(r'"([^"]*)"', opstr_contents).group(1))
    dtp_count = file_contents.count("__dtp__")
    discovered_count = len(discovered_dtp_opstrs)
    assert discovered_count == dtp_count, f"Missed {dtp_count - discovered_count} DEF_TENSOR_PROPERTIES in {filename}"
    return discovered_dtp_opstrs


def get_rules_and_registrations_from_file(
    args: argparse.Namespace, filename: str
) -> Tuple[List[SimpleRule], List[Dict[str, str]]]:
    """Run our scripts through clang preprocessor with relevant internal
    symbols on HTP Core/hexagon-nn-v3 included.
    Example 1: Resolving symbolic passes
        -> LATE+N might map to the integer 20000+N
    Example 2: Unraveling nested macro symbols
        -> IS_FLOAT('X')  == EQ(DTYPE_OF('X'), DType::Float32)"""
    # pylint: disable=too-many-locals
    compile_command = [
        f"{args.clang_path}",
        "-std=c++17",
        f"-I{args.htp_include_dir}",
        f"-I{args.libnative_include_dir}",
        f"-I{args.qnn_op_pkg_include_dir}",
    ]
    # add any additional paths
    if args.additional_include_paths:
        for additional_include_path in args.additional_include_paths:
            compile_command.append(f"-I{additional_include_path}")
    compile_command.extend(
        [
            f'-DQNN_HTP_VERSION="{args.qnn_htp_version}"',
            "-DSETUP_OS_VOTE",
            "-DHEXNN_ARGMMAX_RESIZE_OPT",
            f"-DVTCM_MB={args.vtcm_mb}",
            f"-DTHIS_PKG_NAME={args.package_name}",
            "-D__HVXDBL__",
            "-DUSE_OS_LINUX",
            "-DDEF_OPT_COMPILE=1",
            "-DOP_REG_COMPILE=1",
            "-DDTP_COMPILE=1",
            "-DWITH_OPT_DEBUG=1",
            f"-DHEX_ARCH={args.hex_arch}",
            "-E",
            "-o",
            "-",
            filename,
        ]
    )

    rules = []
    registrations = []

    lines = []
    for line in output(compile_command):
        lines.append(line)
        # NOTE: This is some tidying for the output of a common function to get
        # internal packages from HTP Core.
        line = line.replace(").c_str()", "")
        line = line.replace('hnnx::get_opname_with_default_pkg_prefix("', f'"{args.package_name}')
        # Optimization Registration
        def_opt_match = _extract_rule(line.strip())
        if def_opt_match:
            rules.append(def_opt_match)

        # Implementation Registration
        reg_op_match = _extract_registration(line.strip())
        if reg_op_match:
            registrations.append(reg_op_match)

    # Grab arguments and typename information from our registrations
    impl_name2arguments, impl_name2typenames = _extract_impl_name_args_and_typenames(lines)

    # enhance registrations with type info for all args
    registrations = [_enhance_registration(r, impl_name2arguments, impl_name2typenames) for r in registrations]

    # extract any opstrs w/ DTPs that might exist while skipping all the irrelevant stuff
    first_index_with_dtp = None
    for index, line in enumerate(lines):
        if "__dtp__" in line:
            first_index_with_dtp = index
            break
    existing_dtp_opstrs = []
    if first_index_with_dtp:
        existing_dtp_opstrs = _extract_dtp_opstr(" ".join(lines[first_index_with_dtp:]), filename)
    return rules, registrations, existing_dtp_opstrs


def output(cmd: List[str]) -> List[str]:
    """Execute cmd and return output as list of lines"""
    return subprocess.check_output(cmd, encoding="utf-8").split("\n")[:-1]


def _extract_transients_and_replacements(
    args: argparse.Namespace,
    simple_rules: List[SimpleRule],
    all_opstrs_pre_check: Set[str],
    registered_opstrs: Set[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    replacement2transients = {}
    transient2transients = {}
    simple_rules.sort(key=lambda x: x.priority)
    for simple_rule in simple_rules:
        structural_match = simple_rule.pattern.structural_match(simple_rule.result)
        if not structural_match:  # no match
            continue
        for transient, replacement in structural_match.items():
            # don't say an op is a replacement of itself
            if transient.opstr == replacement.opstr:
                continue
            # check to see if op existed prior to this stage through
            # our DEF_PACKAGE_OPTIMIZATION rules
            if transient.opstr not in all_opstrs_pre_check:
                continue
            # likewise, check to see if our transient opstr
            if transient.opstr in registered_opstrs:
                continue
            # confirm that the replacement opstr (for this pass)
            # is truly a registered osptr and not just a transient one
            if replacement.opstr not in registered_opstrs:
                if transient.opstr not in transient2transients:
                    transient2transients[transient.opstr] = set()
                transient2transients[transient.opstr].add(replacement.opstr)
                continue
            if simple_rule.priority <= args.migration_phase:
                continue
            # finally, we have an opstr we know won't exist at
            # the end of optimization, and we are noting at least
            # one of the replacement opstrs that are relevant for it
            if replacement.opstr not in replacement2transients:
                replacement2transients[replacement.opstr] = set()
            replacement2transients[replacement.opstr].add(transient.opstr)
    return transient2transients, replacement2transients


def infer_transient_opstrs(
    args: argparse.Namespace, simple_rules: List[SimpleRule], registered_opstrs: Set[str]
) -> Dict[str, Set[str]]:
    """There's a chance that we may have an opstr that is entirely
    replaced by the end of optimization, but is still in the graph when we
    do LAYOUT_AND_PLACEMENT. This requires us to have a DEF_TENSOR_PROPERTIES
    entry for the opstr, even though it doesn't have a registration assigned!

    We call these "transient" since they disappear after a short while
    in our optimizer.

    Example:
    Phase LAYOUT_AND_PLACEMENT    : Creation of "Example" node
    // Our centralized layout and placement steps run
    Phase LAYOUT_AND_PLACEMENT + 1: "Example" -> "OtherOpStr"

    We only have DEF_PACKAGE_OP registrations for OtherOpStr, BUT
    when we ran our centralized layout/placement steps we looked
    for properties related to "Example". If we don't have property info
    then we assume Flat + MainMemory, which may be incorrect!

    The solution for now is to infer the properties of
    "Example" based on what we know about "OtherOpStr"
    """

    # First, collect all opstrs that precede our central migration step
    all_opstrs_pre_check = {op.opstr for sr in simple_rules if sr.priority <= args.migration_phase for op in sr.all_ops}
    # Next, go through all rules after this phase and see which ones transform to something else
    transient2transients, replacement2transients = _extract_transients_and_replacements(
        args, simple_rules, all_opstrs_pre_check, registered_opstrs
    )

    def _get_descendants(transient_opstr):
        if transient_opstr not in transient2transients:
            return set()
        desc = set(transient2transients[transient_opstr])
        for child_transient_opstr in transient2transients[transient_opstr]:
            desc |= _get_descendants(child_transient_opstr)
        return desc

    # Fixup for transient2transients cases, where we have chains of
    # rewrite rules covering opstrs that won't have registrations
    # pylint: disable=consider-using-dict-items
    for replacement_opstr in replacement2transients:
        prior_transients = set(replacement2transients[replacement_opstr])
        new_additions = set()
        for parent_transient_opstr in prior_transients:
            new_additions |= _get_descendants(parent_transient_opstr)
        replacement2transients[replacement_opstr] = new_additions | prior_transients

    transient_opstr2replacements = {}
    for replacement_opstr, transients in replacement2transients.items():
        for transient_opstr in transients:
            if transient_opstr not in transient_opstr2replacements:
                transient_opstr2replacements[transient_opstr] = set()
            transient_opstr2replacements[transient_opstr].add(replacement_opstr)

    return transient_opstr2replacements


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    Logging level is set in this function based on parsed options.
    Returns: Parsed command line options."""
    description = (
        "This tool assists customers with Custom OpPackages to"
        + " maximize compatibility with current and future changes to"
        + " the hexagon-nn-v3 (HTP Core) backend."
    )
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--op-pkg-src", help="Regex path(s) containing op src files", required=True, nargs="+")
    parser.add_argument(
        "--htp-include-dir", help="Path to directory containing essential HTP Core definitions", required=True
    )
    parser.add_argument(
        "--libnative-include-dir", help="Path to directory containing essential libnative definitions", required=True
    )
    parser.add_argument(
        "--qnn-op-pkg-include-dir", help="Path to root directory containing headers to support OpPackage", required=True
    )
    parser.add_argument("--clang-path", help="Path to clang in local environment", default=shutil.which("clang++"))
    parser.add_argument(
        "--qnn-htp-version",
        help="Commit hash reflecting version of QNN-HTP (defaults to older version)",
        default="404971e27",
    )
    parser.add_argument(
        "--hex-arch", help="Hexagon architecture version number we are generating targets for.", type=int, default=75
    )
    parser.add_argument("--vtcm-mb", help="Maximum amount of VTCM Available in MB.", type=int, default=8)
    parser.add_argument("--package-name", help="Prefix of default package name.", default="q")
    parser.add_argument(
        "--migration-phase",
        help=(
            "Phase in our optimizer at the end of which we'll do centralized"
            " decision-making about memory location (TCM vs Main Memory)\n"
            "and layout (Crouton vs. non-Crouton) of our tensors "
            "(Default is 21100, which maps to LAYOUT_AND_PLACEMENT)"
        ),
        default=21100,
        type=int,
    )
    parser.add_argument(
        "--additional-include-paths",
        help="More paths you'd like to provide to the preprocessor to resolve the macros in your OpPackage",
        default=[],
        nargs="+",
    )

    args = parser.parse_args()
    # Input Validation
    if not os.path.exists(args.clang_path):
        parser.error("Couldn't find local clang++")
    return args


def pretty_print_recommendations(
    rule_recommendations: List[Tuple[Diagnostic, SimpleRule, str, str]],
    missing_dtp_opstrs: List[str],
    dtps: List[str],
    transient_opstr2replacements: Dict[str, Set[str]],
):
    """Provide a clean output that display results inferred from the inputs provided"""
    if rule_recommendations:
        # pretty print rule recommendations
        print("*****************************************")
        print("*** OPTIMIZATION RULE RECOMMENDATIONS ***")
        print("*****************************************")
        print("Per-file + line number recommendations of fixes you need to (ERROR), ought to (WARNING)")
        print("and might still (INFO) implement to maximize compatibility of your OpPackage with our")
        print("latest developments for efficient data layout/placement on the HTP Core Backend.\n")
        for level in Diagnostic:
            for diagnostic_level, simple_rule, identifier, solution in rule_recommendations:
                if diagnostic_level == level:
                    print(f"{diagnostic_level} ({simple_rule.filename}:{simple_rule.lineno})")
                    print(f"Cause: {identifier}\nSolution: {solution}\n")

    # Pretty print missing DTP opstrs
    if missing_dtp_opstrs:
        print("*****************************************")
        print("*** OPS MISSING DEF_TENSOR_PROPERTIES ***")
        print("*****************************************")
        for missing_dtp_opstr in missing_dtp_opstrs:
            print(f"{Diagnostic.ERROR}")
            print(f"Cause: Missing DEF_TENSOR_PROPERTIES for {missing_dtp_opstr}")
            print("Solution: Either use the synthesized DTPs below or write your own.\n")

    # pretty print DTPs
    if dtps:
        print("*****************************************")
        print("*** SYNTHESIZED DEF_TENSOR_PROPERTIES ***")
        print("*****************************************")
        print("Recommended DEF_TENSOR_PROPERTIES to replace current optimization rules pertaining")
        print("to memory location + data format. DTPs are best-effort and derive from any")
        print("implementation definitions within the file where you registered them.\n")
        for dtp in dtps:
            print("\n" + "),\n\t".join(dtp.split("),")) + "\n")

    if transient_opstr2replacements:
        print("*****************************************")
        print("*** TRANSIENT OPS  (+ suggested DTPs) ***")
        print("*****************************************")
        print("Some Ops exist at the LAYOUT_AND_PLACEMENT phase, but don't have corresponding")
        print("registrations. This means that we expect that (by the end of optimization) all")
        print("of these transient ops will have been transformed into another op w/ a DTP.\n")
        for transient_opstr, replacements in transient_opstr2replacements.items():
            print(f"Transient Op: {transient_opstr}")
            print("Should have a DEF_TENSOR_PROPERTIES created that mimics:")
            for replacement in replacements:
                print(f"\t{replacement}")
            print()
    if not (rule_recommendations or dtps or transient_opstr2replacements or missing_dtp_opstrs):
        print("NO RECOMMENDATIONS TO MAKE -- DOUBLE CHECK THAT INCLUDE PATHS ARE VALID")


def main():
    """main method for parsing key customer OpPackage data structures"""
    # pylint: disable=too-many-locals
    args = parse_args()
    # Grab files we'll want to look at from OpPackage
    relevant_files = []
    for src_regex in args.op_pkg_src:
        cmd = (
            rf'grep -lr --include=*.cpp --include=*.cc "DEF_PACKAGE_OP\|REGISTER_OP\|DEF_TENSOR_PROPERTIES" {src_regex}'
        )
        try:
            relevant_files.extend(subprocess.check_output(["bash"], input=cmd, encoding="utf-8").strip().split("\n"))
        except subprocess.CalledProcessError as check_ret_code:
            assert check_ret_code.returncode == 1, f"Unexpected return code {check_ret_code.returncode} for cmd: {cmd}"

    # Run files through preprocessor and get optimization rules + registrations
    raw_rules = []
    registrations = []
    existing_dtp_opstrs = []
    for relevant_file in relevant_files:
        rrs, regs, existing_opstrs = get_rules_and_registrations_from_file(args, relevant_file)
        raw_rules.extend(rrs)
        registrations.extend(regs)
        existing_dtp_opstrs.extend(existing_opstrs)

    # Construct simplified version of optimizations (only op relationships, no constraints/flags)
    simplified_rules = [resolve_to_simplified_rule(args, raw_rule) for raw_rule in raw_rules]

    # Generate a set of recommendations based on the rules and on the registrations
    rule_recommendations = [
        rec for simplified_rule in simplified_rules for rec in get_rule_recommendations(args, simplified_rule)
    ]

    # Note any opstrs with multiple outputs
    # (bookkeeping for DEF_TENSOR_PROPERTIES generation down the line)
    opstr2n_outputs = {}
    for reg in registrations:
        opstr2n_outputs[reg["opstr"]] = len([is_output for is_output in reg["arg_is_output"] if is_output])

    # Generate any potential DEF_TENSOR_PROPERTIES that could replace existing optimization rules
    synthesized_dtps = get_def_tensor_properties(registrations, opstr2n_outputs)

    # Infer potentially-transient opstrs
    registered_opstrs = {reg["opstr"].strip('"').strip() for reg in registrations}
    transient_opstr2replacements = infer_transient_opstrs(args, simplified_rules, registered_opstrs)

    # Generate a set of ERRORs if we are missing DEF_TENSOR_PROPERTIES for opstrs
    # which have affiliated registrations
    missing_dtp_opstrs = []
    if len(existing_dtp_opstrs) < len(synthesized_dtps) + len(transient_opstr2replacements):
        for synthesized_dtp in synthesized_dtps:
            synthesized_opstr = re.search(r'"([^"]*)"', synthesized_dtp).group(1)
            if synthesized_opstr not in existing_dtp_opstrs:
                missing_dtp_opstrs.append(synthesized_opstr)
        for transient_opstr in transient_opstr2replacements:
            if transient_opstr not in existing_dtp_opstrs:
                missing_dtp_opstrs.append(transient_opstr)

    pretty_print_recommendations(
        rule_recommendations, missing_dtp_opstrs, synthesized_dtps, transient_opstr2replacements
    )


if __name__ == "__main__":
    main()