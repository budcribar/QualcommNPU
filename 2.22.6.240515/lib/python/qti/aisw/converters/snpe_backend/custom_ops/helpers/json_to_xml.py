# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import argparse
from qti.aisw.converters.common.utils.converter_utils import *
import json
import os
import qti.aisw.op_package_generator.translator.op_def_translator as xml_package_translator
from qti.aisw.converters.common.utils.io_utils import *
from qti.aisw.converters.snpe_backend.custom_ops.snpe_udo_config import *
from collections import defaultdict


def json_to_xml(config_path, xml_path):

    # check file and xml path
    io_utils.check_validity(config_path, extensions=[".json"])
    io_utils.check_validity(xml_path, is_directory=True)

    # Import config
    with open(config_path, 'r') as json_config:
        config_vars = json.load(json_config)

    xml_config_paths = []
    for udo_package_name, udo_package_dict in config_vars.items():
        op_def_collection = OpDefCollection()
        new_udo_package = UdoPackage(udo_package_dict['UDO_PACKAGE_NAME'])
        udo_package_info = UdoPackageInfo.from_dict(udo_package_dict)
        package_info = OpDefPackageInfo(new_udo_package.name)
        new_udo_package.add_package_info(udo_package_info)

        for operator in udo_package_info.operators:
            for snpe_core_type in operator.core_types:
                core = SnpeUdoConstants.snpe_udo_coretypes[snpe_core_type]
                if core == 'DSP' and len(operator.dsp_arch_types):
                    for arch_type in operator.dsp_arch_types:
                        core = "DSP_" + arch_type.upper()
                        if core not in op_def_collection.supported_ops:
                            op_def_collection.supported_ops[core] = defaultdict(list)
                        if core in op_def_collection.supported_ops.keys():
                            op_def_collection.supported_ops[core]['ALL'].append(operator.type_name)
                        else:
                            op_def_collection.supported_ops[core]['ALL'] = [operator.type_name]
                else:
                    if core not in op_def_collection.supported_ops:
                        op_def_collection.supported_ops[core] = defaultdict(list)
                    if core in op_def_collection.supported_ops.keys():
                        op_def_collection.supported_ops[core]['ALL'].append(operator.type_name)
                    else:
                        op_def_collection.supported_ops[core]['ALL'] = [operator.type_name]

        from qti.aisw.op_package_generator.op_def.op_def_classes import QnnDatatype
        for operator in udo_package_info.operators:
            ins = []
            for input in operator.input:
                mandatory = False
                if input.default_value is None:
                    input.default_value = ""
                    mandatory = True
                inp = InputElement(name=input.name, description="", mandatory=mandatory, default=input.default_value,
                                   datatype=[QnnDatatype.BACKEND_SPECIFIC], rank=input.rank,shape=input.shape,
                                   layout=snpe_to_qnn_layout[input.layout], constraints=[], repeated=input.repeated,
                                   is_static_tensor=input.static)
                ins.append(inp)
            outs = []
            for output in operator.output:
                out = OutputElement(name=output.name, description="", mandatory=True, datatype=[QnnDatatype.BACKEND_SPECIFIC],
                                    rank=output.rank, shape=output.shape, layout=snpe_to_qnn_layout[output.layout],
                                    constraints=[], repeated=output.repeated)
                outs.append(out)
            params = []
            for param in operator.param:
                if param.param_type == 'QNN_PARAMTYPE_SCALAR':
                    dtype = snpe_udo_to_qnn[param.data_type]
                    mandatory = False
                    if isinstance(param.default_value, bool):
                        param.default_value = int(param.default_value)
                    if param.default_value is None:
                        param.default_value = ""
                        mandatory = True
                    scalar_param = ScalarElement(name=param.name, description="", mandatory=mandatory,
                                                 default=param.default_value, datatype=[dtype], constraints=[])
                    params.append(scalar_param)
                elif param.param_type == 'SNPE_UDO_PARAMTYPE_TENSOR':
                    dtype = snpe_udo_to_qnn[param.data_type]
                    mandatory = False
                    if param.default_value is None:
                        param.default_value = ""
                        mandatory = True
                    tensor_param = TensorElement(name=param.name, description="", mandatory=mandatory,
                                                 default=param.default_value, datatype=[dtype], rank=param.rank,
                                                 shape=param.shape, layout=snpe_to_qnn_layout[param.layout], constraints=[])
                    params.append(tensor_param)

            op_def = OpDef(name=operator.type_name, description="", references=[], ins=ins, outs=outs,
                           parameters=params, use_default_translation=False)
            op_def.package_info = package_info
            op_def_collection.add_op_def(op_def)

            for snpe_core_type in operator.core_types:
                core_type = SnpeUdoConstants.snpe_udo_coretypes[snpe_core_type]
                supp_ins = []
                for input in operator.input:
                    dtype = snpe_udo_to_qnn[input.per_core_data_types[snpe_core_type]]
                    inp = SupplementalOpDefElement(name=input.name, datatypes=[dtype], quant_params=[], shape=input.shape,
                                                   layout=snpe_to_qnn_layout[input.layout], constraints=[])
                    supp_ins.append(inp)
                supp_outs = []
                for output in operator.output:
                    dtype = snpe_udo_to_qnn[output.per_core_data_types[snpe_core_type]]
                    out = SupplementalOpDefElement(name=output.name, datatypes=[dtype], quant_params=[], shape=output.shape,
                                                   layout=snpe_to_qnn_layout[output.layout], constraints=[])
                    supp_outs.append(out)
                supp_params = []
                supp_op_def = SupplementalOpDef(name=operator.type_name, supp_ins=supp_ins, supp_outs=supp_outs,
                                                supp_params=supp_params)

                if core_type == 'DSP' and len(operator.dsp_arch_types):
                    for arch_type in operator.dsp_arch_types:
                        core_type = "DSP_" + arch_type.upper()
                        op_def_collection.add_supplemental_op_def(supp_op_def, backend=core_type)
                else:
                    op_def_collection.add_supplemental_op_def(supp_op_def, backend=core_type)

        SCHEMA_PATH = os.path.abspath(os.path.dirname(xml_package_translator.__file__))
        SCHEMA = os.path.join(SCHEMA_PATH, xml_package_translator.OpDefTranslator.DEFAULT_SCHEMA)
        translator_instance = xml_package_translator.OpDefTranslator(xml_schema=SCHEMA)
        path = os.path.join(os.path.realpath(xml_path), udo_package_info.name + '.xml')
        translator_instance.write_op_defs(op_def_collection, path)
        xml_config_paths.append(path)
    return xml_config_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_json', required=True, help='path to Json config')
    parser.add_argument("-o", '--output_xml', required=True, help='path to save converted XML Configs')
    args = parser.parse_args()
    json_to_xml(args.input_json, args.output_xml)
    log_debug_msg_as_status("Conversion Completed")
