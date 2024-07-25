# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
from abc import abstractmethod
import numpy as np
import qti.aisw.arch_checker.constants as const
from qti.aisw.converters.common import ir_graph

class ModelModifier:
    def __init__(self, c_ir_graph, constraints_json, logger, query, df):
        self.c_ir_graph = c_ir_graph
        self.constraints_json = constraints_json
        self.logger = logger
        self.query = query
        self.modifiable_rules = []
        self.df = df

    @abstractmethod
    def save_modified_graph(self):
        pass

    def do_modification(self):
        mrule_ids = self.get_modifiable_rules()
        if len(mrule_ids) == 0:
            return False
        if self.query == "all":
            # Modifications for all applicable rules
            for i in range(len(self.df[const.INTERNAL_RULEID])):
                rule_info = self.get_rule_info_from_id(self.df[const.INTERNAL_RULEID][i])
                if eval(rule_info[const.MODIFIABLE]):
                    eval(rule_info[const.MOD_CONDITION])
        elif "apply" in self.query:
            # Modifications as per user supplied rule names
            rules_supplied = ''.join(self.query.split('=')[1:])
            need_mod_list = rules_supplied.split(',')
            rules_to_modify = self.input_sanity_check(need_mod_list, mrule_ids)
            for i in range(len(self.df[const.INTERNAL_RULEID])):
                rule_id = self.df[const.INTERNAL_RULEID][i]
                if rule_id in rules_to_modify:
                    rule_info = self.get_rule_info_from_id(self.df[const.INTERNAL_RULEID][i])
                    if eval(rule_info[const.MODIFIABLE]):
                        eval(rule_info[const.MOD_CONDITION])
        elif self.query == "" or self.query == 'show':
            # Show modifications that can be applied
            self.logger.info("====== Here is the list of possible modifications.======")
            self.logger.info("Re-run the checker with either 'all' or provide comma separated rule names to apply the modifications eg: apply=elwisediv,prelu")
            for rule_name, rule_id in mrule_ids.items():
                rule_info = self.get_rule_info_from_id(rule_id)
                self.logger.info("\t Rule name : " + rule_name + " | Modification: " + rule_info[const.MOD_STRING])
        else:
            self.logger.error("Invalid modify argument. Please provide valid --modify argument.")
            return False
        return True

    def get_modifiable_rules(self):
        # get rules that can be modified in a dict with key as rule name
        candidates = set(self.df[const.INTERNAL_RULEID].tolist())
        modifiable_rules = {}
        for cand in candidates:
            rule_cat = self.find_rule_category(cand)
            if eval(self.constraints_json[rule_cat][cand][const.MODIFIABLE]):
                rule_name = self.constraints_json[rule_cat][cand][const.RULE_NAME]
                # Check sn-4 rule is modifiable
                if cand == "sn-4":
                    check_modifiablility = self.check_sn4_modifiability()
                    if not check_modifiablility:
                        continue
                modifiable_rules[rule_name] = cand
        return modifiable_rules

    def get_rule_info_from_id(self, rule_id):
        rule_cat = self.find_rule_category(rule_id)
        return self.constraints_json[rule_cat][rule_id]

    def find_rule_category(self, rule_id):
        if rule_id.startswith("g-"):
            return const.RULE_CTGY_G
        elif rule_id.startswith("sn-"):
            return const.RULE_CTGY_SN
        elif rule_id.startswith("p-"):
            return const.RULE_CTGY_P
        else:
            self.logger.error("Unknown rule category")
            exit(-1)

    def input_sanity_check(self, need_mod_list, mrule_ids):
        # check if user specified rules are valid and return the rules ids for the same
        if len(set(need_mod_list)) != len(need_mod_list):
            self.logger.error("Remove duplicated ids.")
            sys.exit(-1)
        rules_to_modify = []
        for x in range(0, len(need_mod_list)):
            if need_mod_list[x] not in mrule_ids:
                if need_mod_list[x]:
                    temp_rule_name = need_mod_list[x]
                else:
                    temp_rule_name = "Provided rule"
                self.logger.error(str(temp_rule_name) + " is not modifiable. Double check the modifiable list and input the correct rule name as apply=rule_name1,rule_name2 without spaces.")
                sys.exit(-1)
            else:
                rules_to_modify.append(mrule_ids[need_mod_list[x]])
        return rules_to_modify

    @abstractmethod
    def get_header_name(self):
        pass

    def get_old_op_info(self, i):
        node_name = self.get_header_name()
        old_op_name = self.df[node_name][i]
        old_op = self.c_ir_graph.get_op(old_op_name)
        old_in_tensors = old_op.inputs()
        old_out_tensors = old_op.outputs()
        return old_op_name, old_op, old_in_tensors, old_out_tensors

    def modify_prelu(self, i):
        # Update prelu op with relu
        old_op_name, old_op, old_in_tensors, old_out_tensors = self.get_old_op_info(i)

        # Prelu takes 2 inputs whereas Relu takes single input, update op with in[0]
        attrs = ir_graph.IrAttributes()
        attrs.addString(ir_graph.IR_OP_NEURON_TYPE, ir_graph.QNN_OP_RELU, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        new_op = ir_graph.NeuronOp(old_op_name, attrs)
        old_op.set_input_tensors([old_in_tensors[0]])
        new_op.set_input_tensors([old_in_tensors[0]])
        new_op.set_output_tensors(old_out_tensors)
        self.c_ir_graph.replace(old_op, new_op)
        self.logger.info("Found prelu possible modification for " + str(old_op_name) + ". Successfully applied prelu modification!")
        rule_info = self.get_rule_info_from_id(self.df[const.INTERNAL_RULEID][i])
        self.df[const.O_C_MODIFICATION_INFO][i] = rule_info[const.MOD_STRING]
        self.df[const.O_C_MODIFICATION][i] = const.MODIFICATION_STATUS

    def extract_encoding_info(self, tensor):
        # get encoding information like scale and offset
        encoding = tensor.get_encoding()
        encoding_info = encoding.encInfo
        if encoding_info.offset > 0:
            offset = -(encoding_info.offset)
        else:
            offset = encoding_info.offset
        return encoding_info.scale, offset

    def quantized_datatype(self, tensor_dtype):
        if tensor_dtype == ir_graph.QNN_DATATYPE_UFIXED_POINT_8:
            return np.dtype('uint8')
        elif tensor_dtype == ir_graph.QNN_DATATYPE_SFIXED_POINT_8:
            return np.dtype('int8')
        elif tensor_dtype == ir_graph.QNN_DATATYPE_UFIXED_POINT_16:
            return np.dtype('uint16')
        elif tensor_dtype == ir_graph.QNN_DATATYPE_SFIXED_POINT_16:
            return np.dtype('int16')
        elif tensor_dtype == ir_graph.QNN_DATATYPE_UFIXED_POINT_32:
            return np.dtype('uint32')
        elif tensor_dtype == ir_graph.QNN_DATATYPE_SFIXED_POINT_32:
            return np.dtype('int32')

    def check_sn4_modifiability(self):
        # Check for elwisediv if the tensor type is static then only modify
        total_elwise_div_rules = (self.df[const.INTERNAL_RULEID].loc[lambda x: x=="sn-4"].index).tolist()
        node_name = self.get_header_name()
        for row_id in total_elwise_div_rules:
            op_name = self.df[node_name][row_id]
            op = self.c_ir_graph.get_op(op_name)
            op_const_tensor = op.inputs()[1]
            if not op_const_tensor.is_static_tensor():
                return False
        return True

    def modify_div(self, i):
        # If elwisediv found, replace the op with elwise multipy and reciprocal value
        old_op_name, old_op, old_in_tensors, old_out_tensors = self.get_old_op_info(i)
        old_const_tensor = old_in_tensors[1]
        # If its a static tensor
        if old_const_tensor.is_static_tensor():
            old_static_tensor_data = old_const_tensor.get_data()
            if old_const_tensor.is_quantized():
                scale, offset = self.extract_encoding_info(old_const_tensor)
                data_type = self.quantized_datatype(old_const_tensor.data_type())
                dequantized_data = scale * (old_static_tensor_data + offset).astype(np.float32)
                recip_data = np.reciprocal(dequantized_data, where=dequantized_data!=0)
                requantized_data = (np.clip(np.rint(recip_data/scale) - offset, np.iinfo(data_type).min, np.iinfo(data_type).max)).astype(data_type)
                new_const_tensor = ir_graph.IrStaticTensor(old_const_tensor.name(), old_const_tensor.dims(), requantized_data, old_const_tensor.data_type())
            else:
                recip_data = np.reciprocal(old_static_tensor_data, where=old_static_tensor_data!=0)
                new_const_tensor = ir_graph.IrStaticTensor(old_const_tensor.name(), old_const_tensor.dims(), recip_data, old_const_tensor.data_type())

            new_in_tensors = [old_in_tensors[0], new_const_tensor]

            attrs = ir_graph.IrAttributes()
            attrs.addString(ir_graph.IR_OP_ELTWISE_BINARY_PARAM_TYPE, ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
            new_op = ir_graph.ElementwiseBinaryOp(old_op_name, attrs)
            new_op.set_input_tensors(new_in_tensors)
            new_op.set_output_tensors(old_out_tensors)
            self.c_ir_graph.replace(old_op, new_op)
            self.logger.info("Found elwisediv possible modification for " + str(old_op_name) + ". Successfully applied elwisediv modification!")
            rule_info = self.get_rule_info_from_id(self.df[const.INTERNAL_RULEID][i])
            self.df[const.O_C_MODIFICATION_INFO][i] = rule_info[const.MOD_STRING]
            self.df[const.O_C_MODIFICATION][i] = const.MODIFICATION_STATUS
