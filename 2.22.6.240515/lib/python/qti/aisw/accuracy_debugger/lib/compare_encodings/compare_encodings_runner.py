# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import xlsxwriter
import argparse
import sys
import numpy as np
import json
from collections import OrderedDict
import os
import re

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError


class CompareEncodingsRunner(object):

    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self.args = args
        self._logger = logger
        self.encoding_diff_path = os.path.join(args.output_dir, 'encodings_diff.xlsx')
        self.extracted_encodings_path = os.path.join(args.output_dir, 'extracted_encodings.json')
        self.engine_type = None

    def run(self, engine_type):
        self.engine_type = engine_type
        self._logger.info(f'Arguments received to encodings comparison tool: {self.args}')

        if self.engine_type == Engine.QNN.value:
            self.compare_encodings_qnn()
        elif self.engine_type == Engine.SNPE.value:
            self.compare_encodings_snpe()
        else:
            raise ParameterError(
                f'Engine type should be either {Engine.QNN.value} or {Engine.SNPE.value} but given {self.engine_type}'
            )

    def check_missing_encodings(self, extracted_encodings=None, aimet_encodings=None):
        """
        Helper function to find encodings present in AIMET but not in Target(QNN/SNPE) and vice-versa
        """
        self._logger.info(
            f'Finding encodings present only in {self.engine_type} encodings but not in AIMET encodings:'
        )
        for enc_type in extracted_encodings:
            if enc_type in aimet_encodings:
                self._logger.info(f'Checking {enc_type}...')
                for layer in extracted_encodings[enc_type]:
                    if layer not in aimet_encodings[enc_type]:
                        self._logger.warning(
                            f'{layer} present only in {self.engine_type} encodings')
            else:
                self._logger.warning(f'{enc_type} present only in {self.engine_type} encodings')

        self._logger.info(
            f'Encodings present only in AIMET encodings but not in {self.engine_type} encodings:')
        for enc_type in aimet_encodings:
            if enc_type in extracted_encodings:
                self._logger.info(f'Checking {enc_type}...')
                for layer in aimet_encodings[enc_type]:
                    if layer not in extracted_encodings[enc_type]:
                        self._logger.warning(f'{layer} present only in AIMET encodings')
            else:
                self._logger.warning(f'{enc_type} present only in AIMET encodings')

    def compare_encodings_qnn(self):
        output_dict = {}
        output_dict['activation_encodings'] = {}
        output_dict['param_encodings'] = {}

        def difference_dict(Dict_A, Dict_B, a, b, pre):
            output_dict = {}
            aj = 0
            for key in Dict_A.keys():
                if key in Dict_B.keys():
                    output_dict[key] = {}
                    for j in Dict_A[key].keys():
                        if j in Dict_B[key].keys():
                            l = []
                            c = 0
                            while (c < len(Dict_A[key][j])):
                                t = 0
                                dt = {}
                                # x will be used to indicate that bitwidth has changed from 8 to 16 or vice-versa. Thus, x=1 means scale and offset will be checked for valid conversion
                                x = 0
                                if ('bitwidth' in Dict_B[key][j][c].keys()
                                        and 'bitwidth' in Dict_A[key][j][c].keys()):
                                    if (Dict_B[key][j][c]['bitwidth'] == Dict_A[key][j][c]
                                        ['bitwidth']):
                                        dt.update({"bitwidth": str(Dict_B[key][j][c]['bitwidth'])})
                                    # checking for consistent scale and offset conversion in regards to bitwidth conversion from 8 to 16 bits and vice-versa by using QnnOptimization formulas
                                    elif (Dict_B[key][j][c]['bitwidth'] == 8
                                          and Dict_A[key][j][c]['bitwidth'] == 16):
                                        t = t + 1
                                        x = 1
                                        dt.update({
                                            "bitwidth":
                                            "| Input JSON encoding=" +
                                            str(Dict_B[key][j][c]['bitwidth']) + "  " +
                                            " QNN encoding=" + str(Dict_A[key][j][c]['bitwidth'])
                                        })
                                        if ('offset' in Dict_B[key][j][c].keys()
                                                and 'offset' in Dict_A[key][j][c].keys()):
                                            if (round(Dict_A[key][j][c]['offset'], pre) == round(
                                                (Dict_B[key][j][c]['offset'] * 256.0), pre)
                                                    or round(Dict_A[key][j][c]['offset'], pre)
                                                    == round((Dict_B[key][j][c]['offset'] * 257.0),
                                                             pre)):
                                                dt.update({
                                                    "offset":
                                                    "| " + str(Dict_A[key][j][c]['offset']) + " |"
                                                })
                                            else:
                                                t = t + 1
                                                dt.update({
                                                    "offset":
                                                    "* Offset not consistent according to bitwidth conversion Input JSON encoding="
                                                    + str(Dict_B[key][j][c]['offset']) + "  " +
                                                    " QNN encoding=" +
                                                    str(Dict_A[key][j][c]['offset'])
                                                })
                                        if ('scale' in Dict_B[key][j][c].keys()
                                                and 'scale' in Dict_A[key][j][c].keys()):
                                            if (round(Dict_A[key][j][c]['scale'], pre) == round(
                                                    Dict_B[key][j][c]['scale'] / 256.0, pre) or
                                                    round(Dict_A[key][j][c]['scale'], pre) == round(
                                                        Dict_B[key][j][c]['scale'] / 257.0, pre)):
                                                dt.update({
                                                    "scale":
                                                    "| " + str(Dict_A[key][j][c]['scale']) + " |"
                                                })
                                            else:
                                                t = t + 1
                                                dt.update({
                                                    "scale":
                                                    "* Scale not consistent according to bitwidth conversion Input JSON encoding="
                                                    + str(Dict_B[key][j][c]['scale']) + "  " +
                                                    " QNN encoding=" +
                                                    str(Dict_A[key][j][c]['scale'])
                                                })
                                    elif (Dict_B[key][j][c]['bitwidth'] == 16
                                          and Dict_A[key][j][c]['bitwidth'] == 8):
                                        t = t + 1
                                        x = 1
                                        dt.update({
                                            "bitwidth":
                                            "| Input JSON encoding=" +
                                            str(Dict_B[key][j][c]['bitwidth']) + "  " +
                                            " QNN encoding=" + str(Dict_A[key][j][c]['bitwidth'])
                                        })
                                        if ('offset' in Dict_B[key][j][c].keys()
                                                and 'offset' in Dict_A[key][j][c].keys()):
                                            if (round(Dict_A[key][j][c]['offset'], pre) == round(
                                                    Dict_B[key][j][c]['offset'] / 256.0, pre)
                                                    or round(Dict_A[key][j][c]['offset'], pre)
                                                    == round(Dict_B[key][j][c]['offset'] / 257.0,
                                                             pre)):
                                                dt.update({
                                                    "offset":
                                                    "| " + str(Dict_A[key][j][c]['offset']) + " |"
                                                })
                                            else:
                                                t = t + 1
                                                dt.update({
                                                    "offset":
                                                    "* Offset not consistent according to bitwidth conversion Input JSON encoding="
                                                    + str(Dict_B[key][j][c]['offset']) + "  " +
                                                    " QNN encoding=" +
                                                    str(Dict_A[key][j][c]['offset'])
                                                })
                                        if ('scale' in Dict_B[key][j][c].keys()
                                                and 'scale' in Dict_A[key][j][c].keys()):
                                            if (round(Dict_A[key][j][c]['scale'], pre) == round(
                                                    Dict_B[key][j][c]['scale'] * 256.0, pre) or
                                                    round(Dict_A[key][j][c]['scale'], pre) == round(
                                                        Dict_B[key][j][c]['scale'] * 257.0, pre)):
                                                dt.update({
                                                    "scale":
                                                    "| " + str(Dict_A[key][j][c]['scale']) + " |"
                                                })
                                            else:
                                                t = t + 1
                                                dt.update({
                                                    "scale":
                                                    "* Scale not consistent according to bitwidth conversion Input JSON encoding="
                                                    + str(Dict_B[key][j][c]['scale']) + "  " +
                                                    " QNN encoding=" +
                                                    str(Dict_A[key][j][c]['scale'])
                                                })

                                    else:
                                        t = t + 1
                                        dt.update({
                                            "bitwidth":
                                            "* Activation bitwidth conversions from Input JSON encoding="
                                            + str(Dict_B[key][j][c]['bitwidth']) + " to " +
                                            " QNN encoding=" + str(Dict_A[key][j][c]['bitwidth']) +
                                            "not supported"
                                        })
                                if ('dtype' in Dict_B[key][j][c].keys()
                                        and 'dtype' in Dict_A[key][j][c].keys()):
                                    if (Dict_B[key][j][c]['dtype'] == Dict_A[key][j][c]['dtype']):
                                        dt.update({"dtype": str(Dict_B[key][j][c]['dtype'])})
                                    else:
                                        t = t + 1
                                        dt.update({
                                            "dtype":
                                            "* Input JSON encoding=" +
                                            str(Dict_B[key][j][c]['dtype']) + "  " +
                                            " QNN encoding=" + str(Dict_A[key][j][c]['dtype'])
                                        })
                                if ('is_symmetric' in Dict_A[key][j][c].keys()
                                        and 'is_symmetric' in Dict_B[key][j][c].keys()):
                                    if (Dict_B[key][j][c]['is_symmetric'] == Dict_A[key][j][c]
                                        ['is_symmetric']):
                                        dt.update({
                                            "is_symmetric":
                                            str(Dict_B[key][j][c]['is_symmetric'])
                                        })
                                    else:
                                        t = t + 1
                                        dt.update({
                                            "is_symmetric":
                                            "* Input JSON encoding=" +
                                            str(Dict_B[key][j][c]['is_symmetric']) + "  " +
                                            " QNN encoding=" +
                                            str(Dict_A[key][j][c]['is_symmetric'])
                                        })
                                if ('max' in Dict_B[key][j][c].keys()
                                        and 'max' in Dict_A[key][j][c].keys()):
                                    if (round(Dict_B[key][j][c]['max'],
                                              pre) == round(Dict_A[key][j][c]['max'], pre)):
                                        dt.update({"max": str(Dict_B[key][j][c]['max'])})
                                    else:
                                        t = t + 1
                                        dt.update({
                                            "max":
                                            "* Input JSON encoding=" +
                                            str(Dict_B[key][j][c]['max']) + "  " +
                                            " QNN encoding=" + str(Dict_A[key][j][c]['max'])
                                        })
                                if ('min' in Dict_B[key][j][c].keys()
                                        and 'min' in Dict_A[key][j][c].keys()):
                                    if (round(Dict_B[key][j][c]['min'],
                                              pre) == round(Dict_A[key][j][c]['min'], pre)):
                                        dt.update({"min": str(Dict_B[key][j][c]['min'])})
                                    else:
                                        t = t + 1
                                        dt.update({
                                            "min":
                                            "* Input JSON encoding=" +
                                            str(Dict_B[key][j][c]['min']) + "  " +
                                            " QNN encoding=" + str(Dict_A[key][j][c]['min'])
                                        })
                                if (x == 0):
                                    if ('offset' in Dict_B[key][j][c].keys()
                                            and 'offset' in Dict_A[key][j][c].keys()):
                                        if (round(Dict_B[key][j][c]['offset'],
                                                  pre) == round(Dict_A[key][j][c]['offset'], pre)):
                                            dt.update({"offset": str(Dict_B[key][j][c]['offset'])})
                                        else:
                                            t = t + 1
                                            dt.update({
                                                "offset":
                                                "* Input JSON encoding=" +
                                                str(Dict_B[key][j][c]['offset']) + "  " +
                                                " QNN encoding=" + str(Dict_A[key][j][c]['offset'])
                                            })
                                    if ('scale' in Dict_B[key][j][c].keys()
                                            and 'scale' in Dict_A[key][j][c].keys()):
                                        if (round(Dict_B[key][j][c]['scale'],
                                                  pre) == round(Dict_A[key][j][c]['scale'], pre)):
                                            dt.update({"scale": str(Dict_B[key][j][c]['scale'])})
                                        else:
                                            t = t + 1
                                            dt.update({
                                                "scale":
                                                "* Input JSON encoding=" +
                                                str(Dict_B[key][j][c]['scale']) + "  " +
                                                " QNN encoding=" + str(Dict_A[key][j][c]['scale'])
                                            })
                                c = c + 1
                                l.append(dt)
                                output_dict[key].update({j: l})
                                if (t > 0):
                                    if (key == "activation_encodings"):
                                        a = a + t
                                    else:
                                        b = b + t
            self._logger.info(f'Number of activation encoding differences observed: {a}')
            self._logger.info(f'Number of param encoding differences observed: {b}')
            self._logger.info(f'Total number of encoding differences observed: {a+b}')
            return output_dict

        dta = {}
        with open(self.args.input) as json_file:
            data = json.load(json_file)
            #for k in data['graph']['nodes']:
            #print(k)
            dta = data['graph']['tensors']
            dtanodes = data['graph']['nodes']

        ac_inp = {}
        ac_inp['activation_encodings'] = {}
        ac_inp['param_encodings'] = {}
        with open(self.args.aimet_encodings_json) as json_file:
            ac = json.load(json_file)
            for k in ac.keys():
                if k == "activation_encodings" or k == "param_encodings":
                    for kr in ac[k].keys():
                        ac_inp[k][santize_node_name(kr)] = ac[k][kr]

        c = 0
        out_dict = {}
        out_dict['activation_encodings'] = {}
        out_dict['param_encodings'] = {}
        #list containing the Bias Ops and Math Invariant Ops should not be displayed in the output xlsx sheet.
        op_list = [
            "Reduce", "Transpose", "CropAndResize", "Gather", "GatherElements", "GatherND", "Pad",
            "Pool2d", "Pool3d", "Reshape", "Resize", "StridedSlice", "SpaceToDepth", "DepthToSpace",
            "ChannelShuffle", "Split", "TopK", "Conv2d", "Conv3d", "TransposeConv2d",
            "DepthwiseConv2d", "FullyConnected", "MatMul"
        ]
        #he=int("0x00FF",16)
        for key in dta:
            ename = str(key)
            if (key in dtanodes and dtanodes[key]['type'] in op_list):
                continue
            dty = hex(dta[key]['data_type'])
            #print("hex(data_type)= ",dty , "  data_type= ",dta[key]['data_type'], "  hex(dta[key]['data_type'] & he)= ",hex(dta[key]['data_type'] & he))
            dtyp = ""
            bitw = 0
            #print(dta[key]['bitwidth'],dty,bitw)
            if (dty == "0x008" or dty == "0x008" or dty == "0x016" or dty == "0x032"
                    or dty == "0x064" or dty == "0x108" or dty == "0x116" or dty == "0x132"
                    or dty == "0x164" or dty == "0x308" or dty == "0x316" or dty == "0x332"
                    or dty == "0x408" or dty == "0x416" or dty == "0x432"):
                dtyp = "int"
            if (dty == "0x216" or dty == "0x232"):
                dtyp = "float"
            if (dty == "0x508"):
                dtyp = "bool"
            if (dty[-2:] == "08"):
                bitw = 8
            if (dty[-2:] == "16"):
                bitw = 16
            if (dty[-2:] == "32"):
                bitw = 32
            if (dty[-2:] == "64"):
                bitw = 64
            #print(dta[key],dty,bitw)
            if ("params_count" in dta[key].keys()):
                cou = 0
                #Offset needs to be checked for conversion from negative to 0 in case of per-channel encodings
                if (np.right_shift(dta[key]['data_type'], 8) == 3):
                    if ("scale_offset" in dta[key]['quant_params']):
                        dta[key]['quant_params']['scale_offset']['offset'] = 0
                    elif ("axis_scale_offset" in dta[key]['quant_params']):
                        l = []
                        if ("scale_offsets" in dta[key]['quant_params']["axis_scale_offset"]):
                            while (cou < len(
                                    dta[key]['quant_params']["axis_scale_offset"]["scale_offsets"])
                                   ):
                                dta[key]['quant_params']["axis_scale_offset"]['scale_offsets'][cou][
                                    'offset'] = 0
                                cou = cou + 1
                        elif ("bw_scale_offset" in dta[key]['quant_params']["axis_scale_offset"]):
                            while (cou < len(dta[key]['quant_params']["axis_scale_offset"]
                                             ["bw_scale_offset"])):
                                dta[key]['quant_params']['bw_scale_offset'][cou]['offset'] = 0
                                cou = cou + 1
                    elif ("bw_scale_offset" in dta[key]['quant_params']):
                        dta[key]['quant_params']['bw_scale_offset']['offset'] = 0
                cou = 0
                if ("scale_offset" in dta[key]['quant_params']):
                    out_dict['param_encodings'].update({
                        key: [{
                            'bitwidth': dta[key]['quant_params']['scale_offset']['bitwidth'],
                            'dtype': dtyp,
                            'scale': dta[key]['quant_params']['scale_offset']['scale'],
                            'offset': dta[key]['quant_params']['scale_offset']['offset']
                        }]
                    })
                elif ("axis_scale_offset" in dta[key]['quant_params']):
                    l = []
                    if ("scale_offsets" in dta[key]['quant_params']["axis_scale_offset"]):
                        while (cou < len(
                                dta[key]['quant_params']["axis_scale_offset"]["scale_offsets"])):
                            l.append({
                                'bitwidth':
                                dta[key]['quant_params']["axis_scale_offset"]['scale_offsets'][cou]
                                ['bitwidth'],
                                'dtype':
                                dtyp,
                                'scale':
                                dta[key]['quant_params']["axis_scale_offset"]['scale_offsets'][cou]
                                ['scale'],
                                'offset':
                                dta[key]['quant_params']["axis_scale_offset"]['scale_offsets'][cou]
                                ['offset']
                            })
                            cou = cou + 1
                    elif ("bw_scale_offset" in dta[key]['quant_params']["axis_scale_offset"]):
                        while (cou < len(
                                dta[key]['quant_params']["axis_scale_offset"]["bw_scale_offset"])):
                            l.append({
                                'bitwidth':
                                dta[key]['quant_params']['bw_scale_offset'][cou]['bitwidth'],
                                'dtype':
                                dtyp,
                                'scale':
                                dta[key]['quant_params']['bw_scale_offset'][cou]['scale'],
                                'offset':
                                dta[key]['quant_params']['bw_scale_offset'][cou]['offset']
                            })
                            cou = cou + 1
                    out_dict['param_encodings'].update({key: l})
                elif ("bw_scale_offset" in dta[key]['quant_params']):
                    out_dict['param_encodings'].update({
                        key: [{
                            'bitwidth': dta[key]['quant_params']['bw_scale_offset']['bitwidth'],
                            'dtype': dtyp,
                            'scale': dta[key]['quant_params']['bw_scale_offset']['scale'],
                            'offset': dta[key]['quant_params']['bw_scale_offset']['offset']
                        }]
                    })

            else:
                out_dict['activation_encodings'].update({
                    key: [{
                        'bitwidth': dta[key]['quant_params']['scale_offset']['bitwidth'],
                        'dtype': dtyp,
                        'scale': dta[key]['quant_params']['scale_offset']['scale'],
                        'offset': dta[key]['quant_params']['scale_offset']['offset']
                    }]
                })

            c = c + 1
        '''ctaa=len(data['activation_encodings'].keys())
        ctaq=len(output_dict['activation_encodings'].keys())
        ctpa=len(data['param_encodings'].keys())
        ctpq=len(output_dict['param_encodings'].keys())
        print("activation encodings in aimet=",ctaa)
        print("activation encodings in qnn",ctaq)
        print("param encodings in aimet=",ctpa)
        print("param encodings in qnn",ctpq)'''

        a = 0
        b = 0
        Diff = {}
        pre = self.args.precision
        Diff = difference_dict(out_dict, ac_inp, a, b, pre)

        lst = []
        for k in Diff.keys():
            for r in Diff[k].keys():
                if (k == "activation_encodings"):
                    if ('bitwidth' in Diff[k][r][0].keys()):
                        lst.append("bitwidth")
                    if ('dtype' in Diff[k][r][0].keys()):
                        lst.append("dtype")
                    if ('is_symmetric' in Diff[k][r][0].keys()):
                        lst.append("is_symmetric")
                    if ('max' in Diff[k][r][0].keys()):
                        lst.append("max")
                    if ('min' in Diff[k][r][0].keys()):
                        lst.append("min")
                    if ('offset' in Diff[k][r][0].keys()):
                        lst.append("offset")
                    if ('scale' in Diff[k][r][0].keys()):
                        lst.append("scale")

        if self.args.specific_node is None:
            with xlsxwriter.Workbook(self.encoding_diff_path) as workbook:
                # Add worksheet
                worksheet = workbook.add_worksheet()
                f1 = workbook.add_format({'bold': True, 'font_color': 'red'})
                f2 = workbook.add_format({'bold': True, 'font_color': 'blue'})
                # Write headers
                cou = 2
                worksheet.write(0, 0, 'Encoding_type')
                worksheet.write(0, 1, 'buffer_name')
                if ("bitwidth" in lst):
                    worksheet.write(0, cou, 'bitwidth')
                    cou = cou + 1
                if ("dtype" in lst):
                    worksheet.write(0, cou, 'dtype')
                    cou = cou + 1
                if ("is_symmetric" in lst):
                    worksheet.write(0, cou, 'is_symmetric')
                    cou = cou + 1
                if ("max" in lst):
                    worksheet.write(0, cou, 'max')
                    cou = cou + 1
                if ("min" in lst):
                    worksheet.write(0, cou, 'min')
                    cou = cou + 1
                if ("offset" in lst):
                    worksheet.write(0, cou, 'offset')
                    cou = cou + 1
                if ("scale" in lst):
                    worksheet.write(0, cou, 'scale')
                    cou = cou + 1
                # Write list data
                i = 1
                for k in Diff.keys():
                    worksheet.write(i, 0, k)
                    if (self.args.params_only and k == 'activation_encodings'):
                        continue
                    else:
                        for r in Diff[k].keys():
                            c = 0
                            while (c < len(Diff[k][r])):
                                worksheet.write(i, 0, k)
                                worksheet.write(i, 1, r)
                                #print(Diff[k][r])
                                cout = 2
                                if ("bitwidth" in lst):
                                    if (Diff[k][r][c]['bitwidth'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['bitwidth'], f1)
                                    elif (Diff[k][r][c]['bitwidth'][0] == '|'):
                                        worksheet.write(i, cout, Diff[k][r][c]['bitwidth'][2:], f2)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['bitwidth'])
                                    cout = cout + 1
                                if ("dtype" in lst):
                                    if (Diff[k][r][c]['dtype'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['dtype'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['dtype'])
                                    cout = cout + 1
                                if ("is_symmetric" in lst):
                                    if (Diff[k][r][c]['is_symmetric'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['is_symmetric'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['is_symmetric'])
                                    cout = cout + 1
                                if ("max" in lst):
                                    if (Diff[k][r][c]['max'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['max'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['max'])
                                    cout = cout + 1
                                if ("min" in lst):
                                    if (Diff[k][r][c]['min'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['min'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['min'])
                                    cout = cout + 1
                                if ("offset" in lst):
                                    if (Diff[k][r][c]['offset'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['offset'], f1)
                                    elif (Diff[k][r][c]['offset'][0] == '|'):
                                        worksheet.write(i, cout, Diff[k][r][c]['offset'][2:-2], f2)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['offset'])
                                    cout = cout + 1
                                if ("scale" in lst):
                                    if (Diff[k][r][c]['scale'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['scale'], f1)
                                    elif (Diff[k][r][c]['scale'][0] == '|'):
                                        worksheet.write(i, cout, Diff[k][r][c]['scale'][2:-2], f2)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['scale'])
                                    cout = cout + 1
                                i = i + 1
                                c = c + 1
                        if (self.args.activations_only):
                            break

        if self.args.specific_node:
            with xlsxwriter.Workbook(self.encoding_diff_path) as workbook:
                # Add worksheet
                worksheet = workbook.add_worksheet()
                f1 = workbook.add_format({'bold': True, 'font_color': 'red'})
                f2 = workbook.add_format({'bold': True, 'font_color': 'blue'})
                cou = 2
                worksheet.write(0, 0, 'Encoding_type')
                worksheet.write(0, 1, 'buffer_name')
                if ("bitwidth" in lst):
                    worksheet.write(0, cou, 'bitwidth')
                    cou = cou + 1
                if ("dtype" in lst):
                    worksheet.write(0, cou, 'dtype')
                    cou = cou + 1
                if ("is_symmetric" in lst):
                    worksheet.write(0, cou, 'is_symmetric')
                    cou = cou + 1
                if ("max" in lst):
                    worksheet.write(0, cou, 'max')
                    cou = cou + 1
                if ("min" in lst):
                    worksheet.write(0, cou, 'min')
                    cou = cou + 1
                if ("offset" in lst):
                    worksheet.write(0, cou, 'offset')
                    cou = cou + 1
                if ("scale" in lst):
                    worksheet.write(0, cou, 'scale')
                    cou = cou + 1

                # Write list data
                i = 1
                optz = 0
                for k in Diff.keys():
                    worksheet.write(i, 0, k)
                    for r in Diff[k].keys():
                        c = 0
                        if r == self.args.specific_node:
                            optz = 1
                            while (c < len(Diff[k][r])):
                                worksheet.write(i, 0, k)
                                worksheet.write(i, 1, r)
                                #print(Diff[k][r])

                                cout = 2
                                if ("bitwidth" in lst):
                                    if (Diff[k][r][c]['bitwidth'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['bitwidth'], f1)
                                    elif (Diff[k][r][c]['bitwidth'][0] == '|'):
                                        worksheet.write(i, cout, Diff[k][r][c]['bitwidth'][2:], f2)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['bitwidth'])
                                    cout = cout + 1
                                if ("dtype" in lst):
                                    if (Diff[k][r][c]['dtype'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['dtype'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['dtype'])
                                    cout = cout + 1
                                if ("is_symmetric" in lst):
                                    if (Diff[k][r][c]['is_symmetric'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['is_symmetric'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['is_symmetric'])
                                    cout = cout + 1
                                if ("max" in lst):
                                    if (Diff[k][r][c]['max'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['max'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['max'])
                                    cout = cout + 1
                                if ("min" in lst):
                                    if (Diff[k][r][c]['min'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['min'], f1)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['min'])
                                    cout = cout + 1
                                if ("offset" in lst):
                                    if (Diff[k][r][c]['offset'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['offset'], f1)
                                    elif (Diff[k][r][c]['offset'][0] == '|'):
                                        worksheet.write(i, cout, Diff[k][r][c]['offset'][2:-2], f2)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['offset'])
                                    cout = cout + 1
                                if ("scale" in lst):
                                    if (Diff[k][r][c]['scale'][0] == '*'):
                                        worksheet.write(i, cout, Diff[k][r][c]['scale'], f1)
                                    elif (Diff[k][r][c]['scale'][0] == '|'):
                                        worksheet.write(i, cout, Diff[k][r][c]['scale'][2:-2], f2)
                                    else:
                                        worksheet.write(i, cout, Diff[k][r][c]['scale'])
                                    cout = cout + 1
                                i = i + 1
                                c = c + 1
                if (optz == 0):
                    raise Exception(
                        f"Node with '{self.args.specific_node}' name does not exist. Please enter a valid node name."
                    )

        with open(self.extracted_encodings_path, 'w') as json_write:
            json.dump(out_dict, json_write, indent=4)

        self.check_missing_encodings(extracted_encodings=out_dict, aimet_encodings=ac_inp)

        self._logger.info("Extracted QNN encodings are saved at {}".format(
            os.path.abspath(self.extracted_encodings_path)))
        self._logger.info(
            "Differences in QNN encodings and AIMET encodings are written to {}".format(
                os.path.abspath(self.encoding_diff_path)))

    def compare_encodings_snpe(self):
        try:
            from qti.aisw.dlc_utils import snpe_dlc_utils
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        # Load given SNPE DLC file
        snpe_model = snpe_dlc_utils.ModelInfo()
        snpe_model.load(self.args.input)

        # Fetch model's meta data
        (model_version, converter_command, quantizer_command, converter_version,
         model_copyright) = snpe_model.get_meta_data()

        # Find Major version using value of converter_version
        # Sample value of converter_version variable is 'DLC created with converter version: 2.16.0.231027072756_64280'
        converter_major_version = converter_version.split(':')[-1].strip().split('.')[0]
        self._logger.info(converter_version)

        # Extract both activation and param encodings from the given DLC
        DLC_helper = DLCHelper(self.args.input, converter_major_version)
        extracted_encodings = DLC_helper.extract_dlc_encodings()

        # Dump Extracted SNPE encodings to json file
        with open(self.extracted_encodings_path, 'w') as json_write:
            json.dump(extracted_encodings, json_write, indent=4)

        # load AIMET encodings
        with open(self.args.aimet_encodings_json) as json_file:
            aimet_encodings = json.load(json_file)

        # Generate excel sheet highlighting any mismatches between AIMET and SNPE encodings
        self.generate_excel_sheet(aimet_encodings, extracted_encodings, converter_major_version)

        # Log warnings if any encodings are present in AIMET but not in SNPE and vice-versa
        self.check_missing_encodings(extracted_encodings=extracted_encodings,
                                     aimet_encodings=aimet_encodings)

        self._logger.info("Extracted SNPE encodings are saved at {}".format(
            os.path.abspath(self.extracted_encodings_path)))
        self._logger.info(
            "Differences in SNPE encodings and AIMET encodings are written to {}".format(
                os.path.abspath(self.encoding_diff_path)))

    def generate_excel_sheet(self, aimet_encodings, target_encodings, converter_major_version):
        """
        Helper function to find differences between AIMET and Target encodings.
        """
        with xlsxwriter.Workbook(self.encoding_diff_path) as workbook:
            # Initialize Excel sheet
            worksheet = workbook.add_worksheet()

            # Writer headers to Excel sheet
            if converter_major_version == 1:
                headers = [
                    'Encoding_type', 'buffer_name', 'bitwidth', 'max', 'min', 'offset', 'scale'
                ]
            else:
                headers = [
                    'Encoding_type', 'buffer_name', 'bitwidth', 'dtype', 'is_symmetric', 'max',
                    'min', 'offset', 'scale'
                ]

            headers_idx = {}
            for idx, header in enumerate(headers):
                worksheet.write(0, idx, header)
                headers_idx[header] = idx

            sheet_idx = 1
            warning_format = workbook.add_format({'bold': True, 'font_color': 'red'})
            diff_counts = {}
            dlc_version = 'dlcv3'
            if converter_major_version != 1:
                dlc_version = 'dlcv4'

            # Loop for activations and params
            for encoding_type in aimet_encodings.keys():

                diff_counts[encoding_type] = 0

                if (self.args.params_only and encoding_type == 'activation_encodings') or (
                        self.args.activations_only and encoding_type == 'param_encodings'):
                    continue

                if encoding_type not in target_encodings.keys():
                    continue
                """
                Loop for encodings list present in activations/params.
                if a layer has per-channel quantization then aimet_encoding_list will contain multiple encoding dictionaries corresponding to each channel,
                otherwise only one encoding dictionary will present in aimet_encoding_list
                """
                for encoding_name, aimet_encoding_list in aimet_encodings[encoding_type].items():

                    if self.args.specific_node and encoding_name != self.args.specific_node:
                        continue

                    if encoding_name not in target_encodings[encoding_type].keys():
                        continue

                    for idx, aimet_encoding_dict in enumerate(aimet_encoding_list):

                        target_encoding_dict = target_encodings[encoding_type][encoding_name][idx]
                        worksheet.write(sheet_idx, 0, encoding_type)
                        worksheet.write(sheet_idx, 1, encoding_name)

                        for key in aimet_encoding_dict.keys():
                            if key not in target_encoding_dict.keys():
                                continue

                            # convert below encodings to strings since dtype and is_symmetric are strings in AIMET encodings
                            if converter_major_version != 1 and key in ['dtype', 'is_symmetric']:
                                target_encoding_dict[key] = str(target_encoding_dict[key])

                            # round below encoding values to given precision
                            if key in ['max', 'min', 'scale']:
                                target_encoding_dict[key] = round(target_encoding_dict[key],
                                                                  self.args.precision)
                                aimet_encoding_dict[key] = round(aimet_encoding_dict[key],
                                                                 self.args.precision)

                            # Compare current iteration's encoding
                            if target_encoding_dict[key] != aimet_encoding_dict[key]:
                                # Highlight entry for encoding since AIMET and Target is not matching
                                diff_counts[encoding_type] += 1
                                diff_warning = f"* {dlc_version} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                worksheet.write(sheet_idx, headers_idx[key], diff_warning,
                                                warning_format)
                            else:
                                worksheet.write(sheet_idx, headers_idx[key],
                                                str(target_encoding_dict[key]))
                        sheet_idx = sheet_idx + 1

                    if self.args.specific_node:
                        break

        self._logger.info(
            f"Number of activation encoding differences observed: {diff_counts['activation_encodings']}"
        )
        self._logger.info(
            f"Number of param encoding differences observed: {diff_counts['param_encodings']}")
        self._logger.info(
            f"Total number of encoding differences observed: {diff_counts['activation_encodings']+diff_counts['param_encodings']}"
        )


class DLCHelper():

    def __init__(self, dlc, converter_major_version):
        try:
            from snpe.dlc_utils import modeltools
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        self.converter_major_version = converter_major_version
        if (converter_major_version == '1'):
            self.model = modeltools.Model()
            self.model.load(dlc)
        else:
            self.model = modeltools.IrDlcReader()
            self.cache_reader = modeltools.IrDlcCacheRecordReader()
            self.model.open(dlc)

    def extract_dlc_encodings(self):
        """
        Extracts both activation and param encodings from the given dlc.
        """
        extracted_encodings = {}
        extracted_encodings['activation_encodings'] = self.get_activation_encodings()
        extracted_encodings['param_encodings'] = self.get_param_encodings()

        return extracted_encodings

    def generate_encoding_dict(self, min_value, max_value, delta, offset, bitwidth, data_type=None,
                               is_symmetric=None):
        """
        Helper function to create a dictionary with given encodings data
        """
        # Using OrderedDict to maintain same order as AIMET encodings
        encoding_dict = OrderedDict()
        encoding_dict['bitwidth'] = bitwidth
        if data_type:
            encoding_dict['dtype'] = data_type
        if is_symmetric:
            encoding_dict['is_symmetric'] = is_symmetric
        encoding_dict['max'] = max_value
        encoding_dict['min'] = min_value
        encoding_dict['offset'] = offset
        encoding_dict['scale'] = delta

        return encoding_dict

    def get_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc.
        """
        if self.converter_major_version == '1':
            return self.extract_dlcv3_activation_encodings()
        else:
            return self.extract_dlcv4_activation_encodings()

    def extract_dlcv3_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc with converter version 1.*
        """
        activation_encodings = OrderedDict()
        try:
            for layer in self.model.get_layers():
                if ':0' in layer['name']:
                    continue

                min_value, max_value, delta, offset, bitwidth = self.model.get_tf_output_encoding(
                    layer['name'])[:5]
                encoding_name = layer['output_names'][0]
                encoding_dict = self.generate_encoding_dict(min_value, max_value, delta, offset,
                                                            bitwidth)
                activation_encodings[encoding_name] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )

        return activation_encodings

    def extract_dlcv4_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc with converter version 2.*
        """
        try:
            from qti.aisw.converters.common import ir_graph
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        def extract_encodings(encoding_name, encoding, dtype):
            """
            Helper function to extract bitwidth, min, max, scale and offset params from the given encoding
            """
            encoding_info = None
            if encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
                encoding_info = encoding.encInfo
            elif encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
                encoding_info = encoding.encInfo.axisEncInfo.encInfos[0]

            if encoding_info != None:
                data_type = self.get_aimet_datatype(dtype)
                encoding_dict = self.generate_encoding_dict(
                    encoding_info.min, encoding_info.max, encoding_info.scale, encoding_info.offset,
                    encoding_info.bw, data_type=data_type,
                    is_symmetric=str(bool(encoding.axisEncInfo.axis)))
                activation_encodings[encoding_name] = [encoding_dict]

        graph = self.model.get_ir_graph()
        activation_encodings = OrderedDict()
        try:
            for op in graph.get_ops():
                if ':0' in op.name:
                    continue

                # Extract encodings from inputs of the Op
                for input in op.inputs():
                    if input.is_app_write_tensor():
                        extract_encodings(input.name(), input.get_encoding(),
                                          input.data_type_string())

                # Extract encodings from outputs of the Op
                for output in op.outputs():
                    extract_encodings(output.name(), output.get_encoding(),
                                      output.data_type_string())
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )

        return activation_encodings

    def get_aimet_datatype(self, snpe_dtype):
        """
        Returns AIMET equivalent datatype for given SNPE datatype
        """
        if snpe_dtype in [
                'Int_8', 'Uint_8', 'sFxp_8', 'uFxp_8', 'Int_16', 'Uint_16', 'sFxp_16', 'uFxp_16',
                'Int_32', 'Uint_32', 'sFxp_32', 'uFxp_32', 'Int_64', 'Uint_64'
        ]:
            data_type = 'int'
        elif snpe_dtype in ['Float_16', 'Float_32']:
            data_type = 'float'
        elif snpe_dtype == 'Bool_8':
            data_type = 'bool'
        else:
            data_type = 'undefined'
        return data_type

    def get_param_encodings(self):
        """
        Extracts param encodings from the given dlc.
        """
        if self.converter_major_version == '1':
            return self.extract_dlcv3_param_encodings()
        else:
            return self.extract_dlcv4_param_encodings()

    def extract_dlcv4_param_encodings(self):
        """
        Extracts param encodings from the given dlc with converter version 2.*
        """
        try:
            from qti.aisw.converters.common import ir_graph
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        graph = self.model.get_ir_graph()
        param_encodings = OrderedDict()
        try:
            for op in graph.get_ops():
                if ':0' in op.name:
                    continue

                for input in op.inputs():
                    # consider only static tensors(weights)
                    if input.is_static_tensor():
                        data_type = self.get_aimet_datatype(input.data_type_string())

                        if input.get_encoding(
                        ).type == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
                            # extract per-tensor weight encodings
                            encoding_info = input.get_encoding().encInfo
                            encoding_dict = self.generate_encoding_dict(
                                encoding_info.min, encoding_info.max, encoding_info.scale,
                                encoding_info.offset, encoding_info.bw, data_type=data_type,
                                is_symmetric=str(bool(input.get_encoding().axisEncInfo.axis)))
                            param_encodings[input.name()] = [encoding_dict]
                        elif input.get_encoding(
                        ).type == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET or input.get_encoding(
                        ).type == ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:
                            # extract per-channel weight encodings
                            channel_encodings = []
                            for axis in range(len(input.get_encoding().axisEncInfo.encInfos)):
                                encoding_info = input.get_encoding().axisEncInfo.encInfos[axis]
                                encoding_dict = self.generate_encoding_dict(
                                    encoding_info.min, encoding_info.max, encoding_info.scale,
                                    encoding_info.offset, encoding_info.bw, data_type=data_type,
                                    is_symmetric=str(bool(input.get_encoding().axisEncInfo.axis)))
                                channel_encodings.append(encoding_dict)
                            param_encodings[input.name()] = channel_encodings

        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting param encodings from the given DLC file, error: {e}"
            )

        return param_encodings

    def extract_dlcv3_param_encodings(self):
        """
        Extracts param encodings from the given dlc with converter version 1.*
        """

        param_encodings = OrderedDict()
        for layer in self.model.get_layers():
            if ':0' in layer['name']:
                continue

            try:
                weight_encoding = self.model.get_tf_weight_encoding(layer['name'], 0)
                if weight_encoding is not None:
                    axis = self.model.get_tf_weight_encoding_axis(layer['name'], 0)

                    if axis >= 0:
                        # extract per-channel weight encodings
                        num_elements = self.model.get_tf_weight_encoding_num_elements(
                            layer['name'], 0)

                        channel_encodings = []
                        for channel in range(num_elements):
                            min_value, max_value, delta, offset, bitwidth = self.model.get_tf_weight_encoding_by_element(
                                layer['name'], 0, channel)[:5]
                            encoding_dict = self.generate_encoding_dict(
                                min_value, max_value, delta, offset, bitwidth)
                            channel_encodings.append(encoding_dict)
                        encoding_name = layer['name'] + '.weight'
                        param_encodings[encoding_name] = channel_encodings
                    else:
                        # extract per-tensor weight encodings
                        min_value, max_value, delta, offset, bitwidth = weight_encoding[:5]
                        encoding_name = layer['name'] + '.weight'
                        encoding_dict = self.generate_encoding_dict(min_value, max_value, delta,
                                                                    offset, bitwidth)
                        param_encodings[encoding_name] = [encoding_dict]
            except:
                try:
                    # extract bias encodings
                    bias_encoding = self.model.get_tf_bias_encoding(layer['name'])
                    if bias_encoding is not None:
                        min_value, max_value, delta, offset, bitwidth = bias_encoding[:5]
                        encoding_name = layer['name'] + '.bias'
                        encoding_dict = self.generate_encoding_dict(min_value, max_value, delta,
                                                                    offset, bitwidth)
                        param_encodings[encoding_name] = [encoding_dict]
                except:
                    pass

        return param_encodings
