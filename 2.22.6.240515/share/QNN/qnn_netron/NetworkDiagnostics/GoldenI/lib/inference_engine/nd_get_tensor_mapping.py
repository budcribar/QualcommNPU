# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import json

from lib.utils.nd_constants import Engine, Framework
from lib.framework_diagnosis.nd_framework_runner import FrameworkRunner

class GetTensorMappingRunnerWithFramework(FrameworkRunner):
    def __init__(self, logger, args):
        super(GetTensorMappingRunnerWithFramework, self).__init__(logger, args)
        self._logger = logger
        self.output_file = os.path.join(args.output_dir, "tensor_mapping.json")
        self.engine = args.engine

    def generate_tensor_mapping(self, output_dir):  # type: (Tuple[List[str], List[str]], str) -> None
        try:
            tensor_mapping = {}
            outputs = os.walk(output_dir)
            for _, _, file_list in outputs:
                for file_path in file_list:
                    if not file_path.endswith(".raw"):
                        continue
                    tensor_name = os.path.splitext(file_path)[0]
                    if tensor_name in tensor_mapping:
                        continue
                    if self.engine == Engine.QNN.value:
                        tensor_mapping[tensor_name] = self.framework_instance.get_mapping_for_qnn_node(tensor_name)
                    elif self.engine == Engine.SNPE.value:
                        tensor_mapping[tensor_name] = self.framework_instance.get_mapping_for_snpe_node(tensor_name)
            with open(self.output_file, 'w') as f:
                json.dump(tensor_mapping, f, indent=4)
            self._logger.info("Successfully generated tensor mapping!")
        except Exception as excinfo:
            self._logger.error("Encountered error while generating tensor mapping: {}".format(str(excinfo)))

    def run(self, qnn_output_dir):
        self.load_framework_for_tensor_mapping()
        self.generate_tensor_mapping(qnn_output_dir)
        return self.output_file

class GetTensorMappingRunnerWithGolden():
    def __init__(self, logger, args):
        self._logger = logger
        self.output_file = os.path.join(args.output_dir, "tensor_mapping.json")
        self.engine = args.engine
        self.golden_dir_for_mapping = args.golden_dir_for_mapping
        self.framework = args.framework
        self.version = args.version

        #dictionary key is golden raw file name, value is the original node name
        self.dict_of_golden_dir = {}
        if self.golden_dir_for_mapping:
            for path, _, files in os.walk(self.golden_dir_for_mapping):
                for file in files:
                    rel_path = os.path.relpath(path, self.golden_dir_for_mapping)
                    if rel_path != ".":
                        file = os.path.join(rel_path, file)
                    tensor_name = os.path.splitext(file)[0]
                    tensor_replace = tensor_name
                    # todo: [AISW-36265] Making Tensor_mapping more extensible by having golden_dir and
                    # framework_instance call to the same tensor_mapping
                    if self.framework:
                        #process data if onnx or tensorflow
                        if self.framework == Framework.tensorflow.value:
                            tensor_replace = tensor_name.replace(":", "_")
                            tensor_replace = tensor_replace.replace(".", "_")
                            tensor_replace = tensor_replace.replace("/", "_")
                            if tensor_replace[0].isdigit():
                                tensor_replace = '_' + tensor_replace
                        elif self.framework == Framework.onnx.value:
                            tensor_replace = tensor_name.replace(".", "_")
                            tensor_replace = tensor_replace.replace("/", "_")
                    else:
                        tensor_replace = tensor_name.replace(":", "_")
                        tensor_replace = tensor_replace.replace(".", "_")
                        tensor_replace = tensor_replace.replace("/", "_")
                        if tensor_replace[0].isdigit():
                            tensor_replace = '_' + tensor_replace

                    self.dict_of_golden_dir[tensor_replace] = tensor_name

    def get_mapping_for_qnn_node_with_golden_dir(self, qnn_output):
        #return qnn_output itself if no golden_dir
        if not (self.golden_dir_for_mapping):
            self._logger.warn("NO_GOLDEN_DIR_FOR_TENSOR_MAPPING: Using the qnn output as mapping. {}".format(str(qnn_output)))
            return qnn_output

        #return default tensor mapping if no framework
        if not (self.framework):
            self._logger.warn("NO_FRAMEWORK_FOR_TENSOR_MAPPING: Generate default mapping. {}".format(str(qnn_output)))
            for k in self.dict_of_golden_dir.keys():
                if k == qnn_output:
                    return self.dict_of_golden_dir[k]
                elif k in qnn_output[:-1]:
                    return self.dict_of_golden_dir[k]
                else:
                    return qnn_output

        #currently no support for tflite and caffe
        if self.framework == Framework.tflite.value:
            self._logger.warn("FRAMEWORK_TFLITE_IS_NOT_SUPPORTED: {}".format(str(qnn_output)))
            return " "
        elif self.framework == Framework.caffe2.value:
            self._logger.warn("FRAMEWORK_CAFFE2_IS_NOT_SUPPORTED: {}".format(str(qnn_output)))
            return " "

        if self.framework == "onnx":
            if qnn_output[1:].isdigit():
                qnn_output = qnn_output[1:]
        #support tensorflow, caffe, onnx
        if (qnn_output in self.dict_of_golden_dir.keys()):
            return self.dict_of_golden_dir[qnn_output]
        elif self.framework == "caffe":
            for tensor_name in self.dict_of_golden_dir.keys():
                if (tensor_name in qnn_output):
                    #avoid matching to the bottom of the qnn_output by making the bottom incomplete
                    if (tensor_name in qnn_output[:-1]):
                        return self.dict_of_golden_dir[tensor_name]

        #if no matching, some warning will occur.
        self._logger.warn("GOLDEN_DIR_MAPPING_MISMATCH_TENSOR: {}".format(str(qnn_output)))
        return " "

    def generate_tensor_mapping(self, output_dir):  # type: (Tuple[List[str], List[str]], str) -> None
        try:
            tensor_mapping = {}
            outputs = os.walk(output_dir)
            for _, _, file_list in outputs:
                for file_path in file_list:
                    if not file_path.endswith(".raw"):
                        continue
                    tensor_name = os.path.splitext(file_path)[0]
                    if tensor_name in tensor_mapping:
                        continue
                    if self.engine == Engine.QNN.value:
                        tensor_mapping[tensor_name] = self.get_mapping_for_qnn_node_with_golden_dir(tensor_name)
                    elif self.engine == Engine.SNPE.value:
                        return NotImplementedError
            with open(self.output_file, 'w') as f:
                json.dump(tensor_mapping, f, indent=4)
                self._logger.info("Successfully generated tensor mapping!")
        except Exception as excinfo:
            self._logger.error("Encountered error while generating tensor mapping: {}".format(str(excinfo)))

    def run(self, qnn_output_dir):
        self.generate_tensor_mapping(qnn_output_dir)
        return self.output_file

def TensorMapping(get_mapping_arg, logger):
    output_file = ""
    try:
        if get_mapping_arg.framework and get_mapping_arg.model_path:
            # try mapping tensor with framework environment
            tensor_mapping_runner = GetTensorMappingRunnerWithFramework(logger, get_mapping_arg)
            output_file = tensor_mapping_runner.run(get_mapping_arg.output_dir)
        else:
            raise Exception("Tensor_mapping with framework venv failed because model_path or framework is missing.")
    except Exception as e:
        # mapping tensor with golden_dir, if not, just output default json with no mapping.
        logger.warn("Try to generate tensor_mapping.json with framework env but failed: {}".format(str(e)))
        logger.info("Instead, try generate tensor_mapping.json with golden_dir_for_mapping input")
        tensor_mapping_runner = GetTensorMappingRunnerWithGolden(logger, get_mapping_arg)
        output_file = tensor_mapping_runner.run(get_mapping_arg.output_dir)
    logger.info("Tensor_mapping generated.")
    return output_file
