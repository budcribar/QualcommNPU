#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import itertools
import os
import logging
import platform
from qti.aisw.quantization_checker.utils.Logger import Logger, PrintOptions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from qti.aisw.quantization_checker.utils import utils
from qti.aisw.quantization_checker.utils.FileUtils import FileUtils, ScopedFileUtils
from qti.aisw.quantization_checker.utils.Progress import Progress, ProgressStage
import qti.aisw.quantization_checker.utils.Constants as Constants

class Converter:
    def __init__(self, logger: Logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms):
        self.logger = logger
        self.sdkPath = sdkPath
        self.sdkType = sdkType
        self.inputNetwork = inputNetwork
        self.inputList = inputList
        self.quantizationOptions = quantizationOptions
        self.quantizationAlgorithms = quantizationAlgorithms
        self.fileHelper = FileUtils(self.logger)

    def __buildArgs__(self):
        self.logger.print('Input list: ' + self.inputList, PrintOptions.LOGFILE)
        return buildQuantizationParameterMap(self.quantizationOptions, self.quantizationAlgorithms, self.inputList, self.sdkType)

class TensorflowConverter(Converter):
    def __init__(self, logger: Logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms):
        super().__init__(logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms)
        logging.disable(logging.WARNING)
        self.tf = __import__('tensorflow')
        inputsAndShapes, outputNames = self.__getTfGraphInputsAndOutputs__()
        self.__inputArgs = inputsAndShapes
        self.__outputArgs = outputNames
        self.__tfConverterArgs = self.__buildArgs__()

    def convert(self, env, activationWidth=None, biasWidth=None, weightWidth=None, outputDir=None, quantOverrides=None, inputLayout=None):
        workingDir = os.path.dirname(self.inputNetwork)
        if not outputDir:
            outputDir = workingDir
        with ScopedFileUtils(workingDir, self.fileHelper):
            inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
            outputArgsWithSwitches = ' --out_node '.join(self.__outputArgs)
            baseArgs = ' -d ' + inputArgsWithSwitches + ' --out_node ' + outputArgsWithSwitches + ' -i ' + self.inputNetwork
            if inputLayout != None:
                baseArgs += ' --input_layout ' + ' --input_layout '.join(inputLayout)

            self.logger.print('TENSORFLOW QUANTIZATION COMMANDS AND OUTPUTS\n', PrintOptions.LOGFILE)
            resultsMap = {}
            if self.sdkType.upper() == Constants.SNPE:
                resultsMap, quantizationVariationsWithCommand = self.__convertSnpe(baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, Constants.SNPE_TF_CONVERTER_BIN_NAME, env)
            else:
                resultsMap, quantizationVariationsWithCommand = self.__convertQnn(baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, Constants.QNN_TF_CONVERTER_BIN_NAME, env)

            return resultsMap, quantizationVariationsWithCommand

    def __convertQnn(self, baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, tfConverterBinaryName, env):
        if activationWidth != None:
            baseArgs += ' --act_bw ' + activationWidth
        if biasWidth != None:
            baseArgs += ' --bias_bw ' + biasWidth
        if weightWidth != None:
            baseArgs += ' --weight_bw ' + weightWidth
        if quantOverrides != None:
            baseArgs += ' --quantization_overrides ' + quantOverrides

        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        resultsMap = {}
        quantizationVariationsWithCommand = {}
        if platform.system() == Constants.WINDOWS: sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
        elif platform.system() == Constants.LINUX: sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
        tensorflowConverterBinaryPath = os.path.join(self.sdkPath, sdkBinPath, tfConverterBinaryName)
        for quantizationVariation, params in self.__tfConverterArgs.items():
            self.fileHelper.makeSubdir(os.path.join(outputDir, quantizationVariation))
            outArgs = ' -o ' + os.path.join(outputDir, quantizationVariation, quantizationVariation + '.cpp')
            command = tensorflowConverterBinaryPath + baseArgs + outArgs + params
            quantizationVariationsWithCommand[quantizationVariation] = command
            resultsMap[quantizationVariation] = utils.issueCommandAndWait(command, self.logger, env)
            Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        return resultsMap, quantizationVariationsWithCommand

    def __convertSnpe(self, baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, tfConverterBinaryName, env):
        if platform.system() == Constants.WINDOWS: sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
        elif platform.system() == Constants.LINUX: sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
        unquantizedDlc = os.path.join(outputDir, Constants.UNQUANTIZED, Constants.UNQUANTIZED) + '.dlc'
        outArgs = ' -o ' + unquantizedDlc
        self.logger.print('Converting TF model to SNPE DLC', PrintOptions.LOGFILE)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        tensorflowConverterBinaryPath = os.path.join(self.sdkPath, sdkBinPath, tfConverterBinaryName)
        returnValue = utils.issueCommandAndWait(tensorflowConverterBinaryPath + baseArgs + outArgs, self.logger, env)
        if returnValue != 0:
            return (Constants.UNQUANTIZED, -1)

        quantizerArgs = ''
        if activationWidth != None:
            quantizerArgs += ' --act_bitwidth=' + activationWidth
        if biasWidth != None:
            quantizerArgs += ' --bias_bitwidth=' + biasWidth
        if weightWidth != None:
            quantizerArgs += ' --weights_bitwidth=' + weightWidth
        if quantOverrides != None:
            quantizerArgs += ' --override_params ' + quantOverrides
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        resultsMap = {}
        quantizationVariationsWithCommand = {}

        if platform.system() == Constants.WINDOWS:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
            quantizerBinaryName = Constants.SNPE_QUANTIZER_BIN_NAME_WINDOWS
        elif platform.system() == Constants.LINUX:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
            quantizerBinaryName = Constants.SNPE_QUANTIZER_BIN_NAME_LINUX
        snpeDlcQuantizeBinaryPath = os.path.join(self.sdkPath, sdkBinPath, quantizerBinaryName)

        for quantizationVariation, params in self.__tfConverterArgs.items():
            self.fileHelper.makeSubdir(os.path.join(outputDir, quantizationVariation))
            baseArgs = ' --input_dlc=' + unquantizedDlc + params + quantizerArgs
            outArgs = ' --output_dlc=' + os.path.join(outputDir, quantizationVariation, quantizationVariation + '.dlc')
            command = snpeDlcQuantizeBinaryPath + baseArgs + outArgs
            quantizationVariationsWithCommand[quantizationVariation] = command
            resultsMap[quantizationVariation] = utils.issueCommandAndWait(command, self.logger, env, False)
            Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        return resultsMap, quantizationVariationsWithCommand

    def __getTfGraphInputsNameAndShape__(self, graph_def):
        inputTensors = []
        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name='')
        for op in graph.get_operations():
            if op.type == "Placeholder":
                for output in op.outputs:
                    if output.get_shape().is_fully_defined():
                        inputTensors.append([op.name, output.get_shape().as_list()])
                    else:
                        inputTensors.append([op.name, [None]])

        inputsAndShapes = []
        for inputTensor in inputTensors:
            if None in inputTensor[1]:
                inputTensor = promptUserForInputDims(self.logger, inputTensor)
            listToStr = ','.join(map(str, inputTensor[1]))
            inputsAndShapes.append(inputTensor[0] + ' ' + listToStr)

        return inputsAndShapes

    def __getTfGraphInputsAndOutputs__(self):
        tfGraph = self.__getTfGraph__(self.inputNetwork)
        inputsAndShapes = self.__getTfGraphInputsNameAndShape__(tfGraph)
        outputNames = self.__getTfGraphOutputsName__(tfGraph)
        return (inputsAndShapes, outputNames)

    def __getTfGraphOutputsName__(self, graph_def):
        outputs = []
        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name='')
            ops = self.tf.compat.v1.get_default_graph().get_operations()
            outputs_set = set(ops)
            for op in ops:
                if len(op.inputs) == 0 and op.type != 'Const':#network input nodes detected
                    continue
                else:
                    for input_tensor in op.inputs:
                        if input_tensor.op in outputs_set:
                            outputs_set.remove(input_tensor.op)

        for op in outputs_set:
            outputs.append(op.node_def.name)
        return outputs

    def __getTfGraph__(self, pbFile):
        session = self.tf.compat.v1.Session(graph=self.tf.Graph())
        with session.graph.as_default():
            graph_def = self.tf.compat.v1.GraphDef()
            with open(pbFile, "rb") as f:
                graph_def.ParseFromString(f.read())
            self.tf.import_graph_def(graph_def, name="")
        return graph_def

class TfliteConverter(Converter):
    def __init__(self, logger: Logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms):
        super().__init__(logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms)
        logging.disable(logging.WARNING)
        self.tf = __import__('tensorflow')
        inputsAndShapes, outputNames = self.__getTfliteGraphInputsAndOutputs__()
        self.__inputArgs = inputsAndShapes
        self.__outputArgs = outputNames
        self.__tfliteConverterArgs = self.__buildArgs__()

    def convert(self, env, activationWidth=None, biasWidth=None, weightWidth=None, outputDir=None, quantOverrides=None, inputLayout=None):
        workingDir = os.path.dirname(self.inputNetwork)
        if not outputDir:
            outputDir = workingDir
        with ScopedFileUtils(workingDir, self.fileHelper):
            inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
            outputArgsWithSwitches = ' --out_node '.join(self.__outputArgs)
            baseArgs = ' -d ' + inputArgsWithSwitches + ' --out_node ' + outputArgsWithSwitches + ' -i ' + self.inputNetwork
            if inputLayout != None:
                baseArgs += ' --input_layout ' + ' --input_layout '.join(inputLayout)

            self.logger.print('TFLITE QUANTIZATION COMMANDS AND OUTPUTS\n', PrintOptions.LOGFILE)
            resultsMap = {}
            if self.sdkType.upper() == 'SNPE':
                resultsMap, quantizationVariationsWithCommand = self.__convertSnpe(baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, Constants.SNPE_TFLITE_CONVERTER_BIN_NAME, env)
            else:
                resultsMap, quantizationVariationsWithCommand = self.__convertQnn(baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, Constants.QNN_TFLITE_CONVERTER_BIN_NAME, env)

            return resultsMap, quantizationVariationsWithCommand

    def __convertQnn(self, baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, workingDir, tfliteConverterBinaryName, env):
        if activationWidth != None:
            baseArgs += ' --act_bw ' + activationWidth
        if biasWidth != None:
            baseArgs += ' --bias_bw ' + biasWidth
        if weightWidth != None:
            baseArgs += ' --weight_bw ' + weightWidth
        if quantOverrides != None:
            baseArgs += ' --quantization_overrides ' + quantOverrides

        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        resultsMap = {}
        quantizationVariationsWithCommand = {}
        if platform.system() == Constants.WINDOWS: sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
        elif platform.system() == Constants.LINUX: sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
        tfliteConverterBinaryPath = os.path.join(self.sdkPath, sdkBinPath, tfliteConverterBinaryName)
        for quantizationVariation, params in self.__tfliteConverterArgs.items():
            self.fileHelper.makeSubdir(quantizationVariation)
            outArgs = ' -o ' + os.path.join(workingDir, quantizationVariation, quantizationVariation + '.cpp')
            command = tfliteConverterBinaryPath + baseArgs + outArgs + params
            quantizationVariationsWithCommand[quantizationVariation] = command
            resultsMap[quantizationVariation] = utils.issueCommandAndWait(command, self.logger, env)
            Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        return resultsMap, quantizationVariationsWithCommand

    def __convertSnpe(self, baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, workingDir, tfliteConverterBinaryName, env):
        if platform.system() == Constants.WINDOWS: sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
        elif platform.system() == Constants.LINUX: sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
        unquantizedDlc = os.path.join(workingDir, Constants.UNQUANTIZED, Constants.UNQUANTIZED) + '.dlc'
        outArgs = ' -o ' + unquantizedDlc
        self.logger.print('Converting TFLite model to SNPE DLC', PrintOptions.LOGFILE)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        tfliteConverterBinaryPath = os.path.join(self.sdkPath, sdkBinPath, tfliteConverterBinaryName)
        returnValue = utils.issueCommandAndWait(tfliteConverterBinaryPath + baseArgs + outArgs, self.logger, env)
        if returnValue != 0:
            return (Constants.UNQUANTIZED, -1)

        quantizerArgs = ''
        if activationWidth != None:
            quantizerArgs += ' --act_bitwidth=' + activationWidth
        if biasWidth != None:
            quantizerArgs += ' --bias_bitwidth=' + biasWidth
        if weightWidth != None:
            quantizerArgs += ' --weights_bitwidth=' + weightWidth
        if quantOverrides != None:
            quantizerArgs += ' --override_params ' + quantOverrides
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        resultsMap = {}
        quantizationVariationsWithCommand = {}

        if platform.system() == Constants.WINDOWS:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
            quantizerBinaryName = Constants.SNPE_QUANTIZER_BIN_NAME_WINDOWS
        elif platform.system() == Constants.LINUX:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
            quantizerBinaryName = Constants.SNPE_QUANTIZER_BIN_NAME_LINUX
        snpeDlcQuantizeBinaryPath = os.path.join(self.sdkPath, sdkBinPath, quantizerBinaryName)

        for quantizationVariation, params in self.__tfliteConverterArgs.items():
            self.fileHelper.makeSubdir(quantizationVariation)
            baseArgs = ' --input_dlc=' + unquantizedDlc + params + quantizerArgs
            outArgs = ' --output_dlc=' + os.path.join(workingDir, quantizationVariation, quantizationVariation + '.dlc')
            command = snpeDlcQuantizeBinaryPath + baseArgs + outArgs
            quantizationVariationsWithCommand[quantizationVariation] = command
            resultsMap[quantizationVariation] = utils.issueCommandAndWait(command, self.logger, env, False)
            Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        return resultsMap, quantizationVariationsWithCommand

    def __getTfliteGraphInputsNameAndShape__(self, graph_def):
        inputsAndShapes = []
        interpreter = self.tf.lite.Interpreter(model_content=graph_def)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()
        for iter_info in input_info:
            dims = iter_info['shape']
            inputShape = []
            for dim in dims:
                inputShape.append(dim)
            listToStr = ','.join(map(str, inputShape))
            inputsAndShapes.append(iter_info['name'] + ' '  + listToStr)
        return inputsAndShapes

    def __getTfliteGraphInputsAndOutputs__(self):
        tfliteGraph = self.__getTfliteGraph__(self.inputNetwork)
        inputsAndShapes = self.__getTfliteGraphInputsNameAndShape__(tfliteGraph)
        outputNames = self.__getTfliteGraphOutputsName__(tfliteGraph)
        return (inputsAndShapes, outputNames)

    def __getTfliteGraphOutputsName__(self, graph_def):
        outputs = []
        interpreter = self.tf.lite.Interpreter(model_content=graph_def)
        interpreter.allocate_tensors()
        output_info = interpreter.get_output_details()
        for iter_info in output_info:
            outputs.append(iter_info['name'])
        return outputs

    def __getTfliteGraph__(self, tfliteFile):
        with open(tfliteFile, 'rb') as fid:
            tfliteGraph = fid.read()
        return tfliteGraph

class OnnxConverter(Converter):
    def __init__(self, logger: Logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms):
        super().__init__(logger, sdkPath, sdkType, inputNetwork, inputList, quantizationOptions, quantizationAlgorithms)
        self.__inputArgs = self.__getOnnxGraphInputs__()
        self.__onnxConverterArgs = self.__buildArgs__()

    def convert(self, env,  activationWidth=None, biasWidth=None, weightWidth=None, outputDir=None, quantOverrides=None, inputLayout=None):
        workingDir = os.path.dirname(self.inputNetwork)
        if not outputDir:
            outputDir = workingDir
        with ScopedFileUtils(workingDir, self.fileHelper):
            inputArgsWithSwitches = ' -d '.join(self.__inputArgs)
            baseArgs = ' -d ' + inputArgsWithSwitches + ' -i ' + self.inputNetwork
            if inputLayout != None:
                baseArgs += ' --input_layout ' + ' --input_layout '.join(inputLayout)

            self.logger.print('ONNX QUANTIZATION COMMANDS AND OUTPUTS\n', PrintOptions.LOGFILE)
            resultsMap = {}
            if self.sdkType.upper() == 'SNPE':
                resultsMap, quantizationVariationsWithCommand = self.__convertSnpe(baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, Constants.SNPE_ONNX_CONVERTER_BIN_NAME, env)
            else:
                resultsMap, quantizationVariationsWithCommand = self.__convertQnn(baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, outputDir, Constants.QNN_ONNX_CONVERTER_BIN_NAME, env)

            return resultsMap, quantizationVariationsWithCommand

    def __convertQnn(self, baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, workingDir, onnxConverterBinaryName, env):
        if activationWidth != None:
            baseArgs += ' --act_bw ' + activationWidth
        if biasWidth != None:
            baseArgs += ' --bias_bw ' + biasWidth
        if weightWidth != None:
            baseArgs += ' --weight_bw ' + weightWidth
        if quantOverrides != None:
            baseArgs += ' --quantization_overrides ' + quantOverrides

        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        resultsMap = {}
        quantizationVariationsWithCommand = {}
        if platform.system() == Constants.WINDOWS: sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
        elif platform.system() == Constants.LINUX: sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
        onnxConverterBinaryPath = os.path.join(self.sdkPath, sdkBinPath, onnxConverterBinaryName)
        for quantizationVariation, params in self.__onnxConverterArgs.items():
            self.fileHelper.makeSubdir(quantizationVariation)
            outArgs = ' -o ' + os.path.join(workingDir, quantizationVariation, quantizationVariation + '.cpp')
            command = onnxConverterBinaryPath + baseArgs + outArgs + params
            quantizationVariationsWithCommand[quantizationVariation] = command
            resultsMap[quantizationVariation] = utils.issueCommandAndWait(command, self.logger, env)
            Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        return resultsMap, quantizationVariationsWithCommand

    def __convertSnpe(self, baseArgs, activationWidth, biasWidth, weightWidth, quantOverrides, workingDir, onnxConverterBinaryName, env):
        if platform.system() == Constants.WINDOWS: sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
        elif platform.system() == Constants.LINUX: sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
        unquantizedDlc = os.path.join(workingDir, Constants.UNQUANTIZED, Constants.UNQUANTIZED) + '.dlc'
        outArgs = ' -o ' + unquantizedDlc
        self.logger.print('Converting ONNX model to SNPE DLC', PrintOptions.LOGFILE)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        onnxConverterBinaryPath = os.path.join(self.sdkPath, sdkBinPath, onnxConverterBinaryName)
        returnValue = utils.issueCommandAndWait(onnxConverterBinaryPath + baseArgs + outArgs, self.logger, env)
        if returnValue != 0:
            return (Constants.UNQUANTIZED, -1)

        quantizerArgs = ''
        if activationWidth != None:
            quantizerArgs += ' --act_bitwidth=' + activationWidth
        if biasWidth != None:
            quantizerArgs += ' --bias_bitwidth=' + biasWidth
        if weightWidth != None:
            quantizerArgs += ' --weights_bitwidth=' + weightWidth
        if quantOverrides != None:
            quantizerArgs += ' --override_params ' + quantOverrides
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        resultsMap = {}
        quantizationVariationsWithCommand = {}

        if platform.system() == Constants.WINDOWS:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_WINDOWS
            quantizerBinaryName = Constants.SNPE_QUANTIZER_BIN_NAME_WINDOWS
        elif platform.system() == Constants.LINUX:
            sdkBinPath = Constants.BIN_PATH_IN_SDK_LINUX
            quantizerBinaryName = Constants.SNPE_QUANTIZER_BIN_NAME_LINUX

        snpeDlcQuantizeBinaryPath = os.path.join(self.sdkPath, sdkBinPath, quantizerBinaryName)
        for quantizationVariation, params in self.__onnxConverterArgs.items():
            self.fileHelper.makeSubdir(quantizationVariation)
            baseArgs = ' --input_dlc=' + unquantizedDlc + params + quantizerArgs
            outArgs = ' --output_dlc=' + os.path.join(workingDir, quantizationVariation, quantizationVariation + '.dlc')
            command = snpeDlcQuantizeBinaryPath + baseArgs + outArgs
            quantizationVariationsWithCommand[quantizationVariation] = command
            resultsMap[quantizationVariation] = utils.issueCommandAndWait(command, self.logger, env, False)
            Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))
        return resultsMap, quantizationVariationsWithCommand

    def __getOnnxGraphInputs__(self):
        import onnx
        model = onnx.load(self.inputNetwork)

        parameterNames = set()
        for tensor in model.graph.initializer:
            parameterNames.add(str(tensor.name))

        inputTensors = []
        for input in model.graph.input:
            inputInfo = []
            name = str(input.name)
            if name in parameterNames:
                continue
            dims = []
            tensorType = input.type.tensor_type
            if (tensorType.HasField("shape")):
                for dim in tensorType.shape.dim:
                    if (dim.HasField("dim_value")):
                        dims.append(dim.dim_value)
                    elif (dim.HasField("dim_param")):
                        dims.append(dim.dim_param)
                    else:
                        dims.append('?')
            else:
                self.logger.print("ERROR: Unknown input shape", PrintOptions.LOGFILE)
            inputInfo = [input.name, dims]
            if not self.__checkOnnxForUnknownDims__(dims):
                inputInfo = promptUserForInputDims(self.logger, inputInfo)
            listToStr = ','.join(map(str, inputInfo[1]))
            inputTensors.append(inputInfo[0] + ' ' + listToStr)

        return inputTensors

    def __checkOnnxForUnknownDims__(self, dims):
        return all(isinstance(dim, int) for dim in dims)

def promptUserForInputDims(logger, inputTensor):
    logger.print('Input found with unknown dimensions...', PrintOptions.CONSOLE_LOGFILE)
    logger.print('Please enter the dimensions for the following input: ' + inputTensor[0] + ' ' + ','.join(map(str, inputTensor[1])) + ', in the format B,H,W,C: ', PrintOptions.CONSOLE_LOGFILE)
    dimensions = input()
    strToList = dimensions.split(',')
    if len(strToList) != 4:
        logger.print('Error parsing input dimensions. Exiting generator tool.', PrintOptions.LOGFILE)
        exit()
    inputTensor[1] = strToList
    return inputTensor

def buildQuantizationParameterMap(quantizationOptions, quantizationAlgorithms, inputList, sdkType):
    parameterMap = {}
    if sdkType.upper() == Constants.QNN:
        quantizationOptionsMasterMap = ['tf', 'enhanced', 'adjusted', 'symmetric']
        quantizationAlgorithmsMasterMap = ['cle', 'pcq']
        parameterMap = dict()
        parameterMap[Constants.UNQUANTIZED] = ''
        if not quantizationOptions:
            quantizationOptions = quantizationOptionsMasterMap
        if not quantizationAlgorithms:
            quantizationAlgorithms = quantizationAlgorithmsMasterMap
        for quantizationOption in quantizationOptions:
            if quantizationOption in quantizationOptionsMasterMap:
                parameterMap[quantizationOption] =  ' --input_list ' + inputList + ' --param_quantizer ' + quantizationOption
                for quantizationAlgorithm in quantizationAlgorithms:
                    if quantizationAlgorithm in quantizationAlgorithmsMasterMap:
                        parameterMap[quantizationOption + ('_' + quantizationAlgorithm if quantizationAlgorithm else '')] =  ' --input_list ' + inputList + ' --param_quantizer ' \
                        + quantizationOption + (' --algorithms ' + quantizationAlgorithm if quantizationAlgorithm and quantizationAlgorithm != 'pcq' else '') + \
                        (' --use_per_channel_quantization' if quantizationAlgorithm == 'pcq' else '')
                        if 'pcq' in quantizationAlgorithms and quantizationAlgorithm != 'pcq':
                            parameterMap[quantizationOption + ('_' + quantizationAlgorithm + '_pcq' if quantizationAlgorithm else '')] =  ' --input_list ' + inputList + ' --param_quantizer ' + quantizationOption + ' --algorithms ' + quantizationAlgorithm + ' --use_per_channel_quantization'
        for key in parameterMap.keys():
            if 'adjusted' not in key and 'unquantized' not in key:
                parameterMap[key] += ' --act_quantizer ' + key.split('_')[0]
    else:
        quantizationOptionsMasterMap = {'tf': '', 'enhanced': '--use_enhanced_quantizer', 'adjusted': '--use_adjusted_weights_quantizer', 'symmetric': '--use_symmetric_quantize_weights'}
        quantizationAlgorithmsMasterMap = {'': '', 'cle': '--optimizations cle', 'bc': '--optimizations bc',  'cle_bc': '--optimizations cle --optimizations bc'}
        quantizationOptionsMap = {}
        quantizationAlgorithmsMap = {}
        if quantizationOptions:
            for quantizationOption in quantizationOptions:
                if quantizationOption in quantizationOptionsMasterMap.keys():
                    quantizationOptionsMap[quantizationOption] = quantizationOptionsMasterMap[quantizationOption]
        else:
            quantizationOptionsMap = quantizationOptionsMasterMap
        if quantizationAlgorithms:
            algoTuples = []
            algos = []
            for i in range(len(quantizationAlgorithms)):
                algoTuples.extend(list(itertools.combinations(quantizationAlgorithms, i+1)))
            for algo in algoTuples:
                algos.append('_'.join(algo))
            algos.insert(0, quantizationAlgorithms[0])
            for quantAlgorithm in algos:
                if quantAlgorithm in quantizationAlgorithmsMasterMap.keys():
                    quantizationAlgorithmsMap[quantAlgorithm] = quantizationAlgorithmsMasterMap[quantAlgorithm]
        else:
            quantizationAlgorithmsMap = quantizationAlgorithmsMasterMap

        inputListArg = ' --input_list=' + inputList
        for quantizationOption in quantizationOptionsMap.keys():
            for quantizationAlgorithm in quantizationAlgorithmsMap.keys():
                parameterMap[quantizationOption + ('_' + quantizationAlgorithm if quantizationAlgorithm else '')] = inputListArg + (' ' + quantizationOptionsMap[quantizationOption] if quantizationOptionsMap[quantizationOption] else '') + (' ' + quantizationAlgorithmsMap[quantizationAlgorithm] if quantizationAlgorithm else '')

    return parameterMap
