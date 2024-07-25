#=============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

from typing import Dict
import numpy as np
import pandas as pd
import os

import qti.aisw.quantization_checker.Results.ResultUtils as ResultUtils
from qti.aisw.quantization_checker.Results.ResultSet import ResultSet
from qti.aisw.quantization_checker.utils.FileUtils import FileUtils
from qti.aisw.quantization_checker.utils.Logger import PrintOptions, NEWLINE
from qti.aisw.quantization_checker.utils.Table import Table, Row, Cell
import qti.aisw.quantization_checker.utils.Constants as Constants

class DataFormatter:
    def __init__(self, outputDir, inputModel, inputList, quantizationVariationsWithCommand, logger):
        self.outputDir = outputDir
        self.inputModel = inputModel
        self.inputList = inputList
        self.quantizationVariationsWithCommand = quantizationVariationsWithCommand
        self.quantizationVariations = list(self.quantizationVariationsWithCommand.keys())
        self.logger = logger
        self.fileHelper = FileUtils(self.logger)
        self.inputResults = ResultSet()
        self.activationResults = ResultSet()
        self.weightResults = ResultSet()
        self.biasResults = ResultSet()
        self.inputsWeightsAndBiasesAnalysisCsvHeader = ['Node Name', 'Tensor Name', 'Passes Verification', 'Threshold for Verification']
        self.activationsAnalysisCsvHeader = self.inputsWeightsAndBiasesAnalysisCsvHeader + ['Input Filename']
        self.htmlHeadBegin = '''
            <html>
            <head><title>{title}</title>
        '''
        self.htmlHeadEnd = '''
            <style>
                h1 {
                    font-family: Arial;
                }
                h2 {
                    font-family: Arial;
                }
                h3 {
                    font-family: Arial;
                }
                #nodes table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: auto !important;
                }
                #top table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: auto !important;
                }
                div table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: auto !important;
                }

                #legend td, th {
                    padding: 5px;
                    width: auto !important;
                    white-space:nowrap;
                }
                #nodes td, th {
                    padding: 5px;
                    width: auto !important;
                    white-space:nowrap;
                }

                #legend tr:nth-child(even) {
                    background: #E0E0E0;
                }
                #nodes tr:nth-child(even) {
                    background: #E0E0E0;
                }

                #nodes tr:hover {
                    background: silver;
                }
                #nodes tr.th:hover {
                    background: silver;
                }
                .pass {
                    color: green;
                }
                .fail {
                    color: red;
                    font-weight: bold;
                }
            </style>
            </head>
        '''
        self.htmlBody = '''
            <body>
                <div>
                    <h1>{title}</h1>
                </div>
                <div>{legend}</div>
                <br/>
                <div>
                    <h2>{summaryTitle}</h2>
                    <h3>{instructions}</h3>
                </div>
                <div id=nodes>{summary}</div>
                <br/>
                <h2>{allNodesTitle}</h2>
                <div id=nodes>{table}</div>
            </body>
            </html>
        '''

    def setInputResults(self, inputResults) -> None:
        self.inputResults = inputResults

    def setActivationsResults(self, activationResults) -> None:
        self.activationResults = activationResults

    def setWeightResults(self, weightResults) -> None:
        self.weightResults = weightResults

    def setBiasResults(self, biasResults) -> None:
        self.biasResults = biasResults

    def __formatDataForHtml(self) -> Dict:
        results = {}
        for quantizationVariation in list(self.quantizationVariationsWithCommand.keys()):
            resultsPerInput = {}
            for inputFile in self.inputList:
                quantOptionResults = []
                quantOptionResults.extend(ResultUtils.getResultsForHtml(quantizationVariation, self.weightResults))
                quantOptionResults.extend(ResultUtils.getResultsForHtml(quantizationVariation, self.biasResults))
                quantOptionResults.extend(ResultUtils.getActivationResultsForHtml(quantizationVariation, inputFile, self.activationResults))
                resultsPerInput[inputFile] = quantOptionResults
            results[quantizationVariation] = resultsPerInput
        results["inputData"] = ResultUtils.getInputResultsForHtml(self.inputResults)
        return results

    def __getMultiIndexForColumns(self, algorithms) -> pd.MultiIndex:
        return pd.MultiIndex.from_product([list(['Quantization Checker Pass/Fail']), list(algorithms)])

    def __getHtmlLegendTable(self, algorithms, descriptions):
        htmlString = '<table id=legend><tr align=left><th colspan=2>Legend:</th></tr><tr><th>Quantization Checker Name</th><th>Description</th></tr>'
        # skip the description column since it is self explanatory and is itself an explanation
        for algorithm in algorithms[:-1]:
            htmlString += '<tr><td>' + algorithm + '</td><td>' + descriptions[algorithm] + '</td></tr>'
        htmlString += '</table>'
        return htmlString

    def printHtml(self) -> None:
        results = self.__formatDataForHtml()
        algorithmNames = ResultUtils.getListOfAlgorithmNamesFromResults(self.quantizationVariations, results)
        readableAlgorithmNames = ResultUtils.translateAlgorithmNamesToDescriptiveNames(algorithmNames)
        algoDescriptions = ResultUtils.translateAlgorithmNameToAlgorithmDescription(readableAlgorithmNames)
        # add the description column manually since we need to interpret the results at the end
        readableAlgorithmNames.append('Description')
        columns = self.__getMultiIndexForColumns(readableAlgorithmNames)
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation not in results:
                continue
            for inputFile in self.inputList:
                noAlgorithms = ResultUtils.getListOfResultsWithoutAlgorithms(results[quantizationVariation][inputFile])
                onlyAlgorithms = ResultUtils.getListOfAlgorithmsFromResults(results[quantizationVariation][inputFile], algorithmNames)
                failedNodes = ResultUtils.getFailedNodes(results[quantizationVariation][inputFile])
                noAlgorithmsFailed = ResultUtils.getListOfResultsWithoutAlgorithms(failedNodes)
                onlyAlgorithmsFailed = ResultUtils.getListOfAlgorithmsFromResults(failedNodes, algorithmNames)

                pd.set_option('display.max_colwidth', None)

                dfOnlyAlgos = pd.DataFrame(onlyAlgorithms, columns=columns)
                dfNoAlgos = pd.DataFrame(noAlgorithms, columns=pd.MultiIndex.from_product([list(['']), list(['Op Name', 'Node Name', 'Node Type', 'Scale', 'Offset'])]))
                dfResults = pd.concat([dfNoAlgos, dfOnlyAlgos], axis=1)
                dfResults = dfResults.set_index([('', 'Op Name'), ('', 'Node Name')], drop=True)
                dfResults.index.names = ['Op Name', 'Node Name']
                dfResults.sort_index(inplace=True)

                dfOnlyAlgosFailed = pd.DataFrame(onlyAlgorithmsFailed, columns=columns)
                dfNoAlgosFailed = pd.DataFrame(noAlgorithmsFailed, columns=pd.MultiIndex.from_product([list(['']), list(['Op Name', 'Node Name', 'Node Type', 'Scale', 'Offset'])]))
                dfResultsFailed = pd.concat([dfNoAlgosFailed, dfOnlyAlgosFailed], axis=1)
                dfResultsFailed = dfResultsFailed.set_index([('', 'Op Name'), ('', 'Node Name')], drop=True)
                dfResultsFailed.index.names = ['Op Name', 'Node Name']
                dfResultsFailed.sort_index(inplace=True)

                htmlOutputDir = os.path.join(self.outputDir, 'html')
                self.fileHelper.makeSubdir(htmlOutputDir)
                inputFilename = os.path.basename(inputFile)
                with open(os.path.join(htmlOutputDir, quantizationVariation + '_' + inputFilename + '.html'), 'w') as f:
                    f.write(self.htmlHeadBegin.format(title=quantizationVariation + ' - ' + inputFilename) + self.htmlHeadEnd + self.htmlBody.format(legend=self.__getHtmlLegendTable(readableAlgorithmNames, algoDescriptions), instructions='Please consult the latest logs for further details on the failures.', summaryTitle='Summary of failed nodes that should be inspected: (Total number of nodes analyzed: ' + str(len(noAlgorithms)) + ' Total number of failed nodes: ' + str(len(noAlgorithmsFailed)) + ')', title='Results for quantizer: ' + quantizationVariation + ' using input file: ' + inputFilename + ' on model: ' + os.path.basename(self.inputModel), allNodesTitle='Results for all nodes:', summary=dfResultsFailed.to_html(justify='center', index=True, escape=False), table=dfResults.to_html(justify='center', index=True, escape=False)))
        self.__printInputDataToHtml(results, htmlOutputDir)

    def __printInputDataToHtml(self, results, htmlOutputDir) -> None:
        algorithmNames = ResultUtils.getListOfAlgorithmNamesFromInputResults(self.inputResults)
        readableAlgorithmNames = ResultUtils.translateAlgorithmNamesToDescriptiveNames(algorithmNames)
        readableAlgorithmNames.append('Description')
        columns = self.__getMultiIndexForColumns(readableAlgorithmNames)
        algoDescriptions = ResultUtils.translateAlgorithmNameToAlgorithmDescription(readableAlgorithmNames)
        noAlgorithms = ResultUtils.getListOfResultsWithoutAlgorithms(results["inputData"])
        filenames = [row[0] for row in noAlgorithms]
        onlyAlgorithms = ResultUtils.getListOfAlgorithmsFromResults(results["inputData"], algorithmNames)
        dfOnlyAlgos = pd.DataFrame(onlyAlgorithms, columns=columns)
        dfNoAlgos = pd.DataFrame(filenames, columns=pd.MultiIndex.from_product([list(['']), list(['File Name'])]))
        dfResults = pd.concat([dfNoAlgos, dfOnlyAlgos], axis=1)
        dfResults = dfResults.set_index([('', 'File Name')], drop=True)
        dfResults.index.names = ['File Name']
        with open(os.path.join(htmlOutputDir, 'input_data_analysis.html'), 'w') as f:
            f.write(self.htmlHeadBegin.format(title='Input Data Analysis') + self.htmlHeadEnd + self.htmlBody.format(legend=self.__getHtmlLegendTable(readableAlgorithmNames, algoDescriptions), instructions='Please consult the latest csv or log files for further details on the analysis.', summaryTitle='Total number of input files analyzed: ' + str(len(filenames)), title='Results for input data analysis for the model: ' + os.path.basename(self.inputModel), allNodesTitle='Analysis for all input files:', summary='', table=dfResults.to_html(justify='center', index=True, escape=False)))

    def __printTableToLog(self, header, results):
        for quantizationVariation in list(self.quantizationVariationsWithCommand.keys()):
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in results:
                continue
            self.logger.print("", PrintOptions.LOGFILE)
            self.logger.print("Results for the " + quantizationVariation + " quantization:", PrintOptions.LOGFILE)
            results[quantizationVariation].insert(0, header)
            table = Table(results[quantizationVariation], True)
            logNoResults = True
            for row in table.getRows():
                cells = row.getCells()
                if cells[2].getString() not in (None, ''):
                    self.logger.print(table.decorate(row), PrintOptions.LOGFILE)
                    if not table.isFirstRow():
                        logNoResults = False
            if logNoResults:
                row = [Cell("N/A") for item in header]
                self.logger.print(table.decorate(Row(row)), PrintOptions.LOGFILE)

    def printLog(self) -> None:
        if self.activationResults is not None:
            self.logger.print(NEWLINE + NEWLINE + '<====ACTIVATIONS ANALYSIS====>' + NEWLINE, PrintOptions.LOGFILE)
            activationsForLog = ResultUtils.formatActivationResultsForLogConsole(self.quantizationVariations, self.activationResults)
            self.__printTableToLog(['Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename', 'Scale', 'Offset'], activationsForLog)

        self.logger.print(NEWLINE + NEWLINE + '<====WEIGHTS ANALYSIS====>' + NEWLINE, PrintOptions.LOGFILE)
        weightsForLog = ResultUtils.formatResultsForLogConsole(self.quantizationVariations, self.weightResults)
        self.__printTableToLog(['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Scale', 'Offset'], weightsForLog)
        self.logger.print(NEWLINE + NEWLINE + '<====BIASES ANALYSIS====>' + NEWLINE, PrintOptions.LOGFILE)
        biasesForLog = ResultUtils.formatResultsForLogConsole(self.quantizationVariations, self.biasResults)
        self.__printTableToLog(['Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Scale', 'Offset'], biasesForLog)

    def __printTableToConsole(self, header, results) -> None:
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in results:
                continue
            self.logger.print("", PrintOptions.CONSOLE)
            self.logger.print("Results for the " + quantizationVariation + " quantization:", PrintOptions.CONSOLE)
            results[quantizationVariation].insert(0, header)
            table = Table(results[quantizationVariation], True)
            logNoResults = True
            for row in table.getRows():
                cells = row.getCells()
                if cells[2].getString() not in (None, ''):
                    self.logger.print(table.decorate(row), PrintOptions.CONSOLE)
                    if not table.isFirstRow():
                        logNoResults = False
            if logNoResults:
                row = [Cell("N/A") for item in header]
                self.logger.print(table.decorate(Row(row)), PrintOptions.CONSOLE)

    def printConsole(self) -> None:
        if self.activationResults is not None:
            self.logger.print(NEWLINE + NEWLINE + '<====ACTIVATIONS ANALYSIS FAILURES====>' + NEWLINE, PrintOptions.CONSOLE)
            activationsForConsole = ResultUtils.formatActivationResultsForLogConsole(self.quantizationVariations, self.activationResults, showOnlyFailedResults=True)
            self.__printTableToConsole(['Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename', 'Scale', 'Offset'], activationsForConsole)

        self.logger.print(NEWLINE + NEWLINE + '<====WEIGHTS ANALYSIS FAILURES====>' + NEWLINE, PrintOptions.CONSOLE)
        weightsForConsole = ResultUtils.formatResultsForLogConsole(self.quantizationVariations, self.weightResults, showOnlyFailedResults=True)
        self.__printTableToConsole(['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Scale', 'Offset'], weightsForConsole)
        self.logger.print(NEWLINE + NEWLINE + '<====BIASES ANALYSIS FAILURES====>' + NEWLINE, PrintOptions.CONSOLE)
        biasesForConsole = ResultUtils.formatResultsForLogConsole(self.quantizationVariations, self.biasResults, showOnlyFailedResults=True)
        self.__printTableToConsole(['Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Scale', 'Offset'], biasesForConsole)

    def printCsv(self) -> None:
        csvOutputDir = os.path.join(self.outputDir, 'csv')
        self.fileHelper.makeSubdir(csvOutputDir)
        inputsForCsv = ResultUtils.formatInputsForCsv(self.inputResults)
        DataFormatter.writeResultsToCsv(csvOutputDir, inputsForCsv, self.inputsWeightsAndBiasesAnalysisCsvHeader, 'input_data_analysis.csv', self.fileHelper)
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation != Constants.UNQUANTIZED and quantizationVariation in self.activationResults.keys():
                activationsForCsv = ResultUtils.formatActivationsForCsv(self.activationResults[quantizationVariation])
                DataFormatter.writeResultsToCsv(csvOutputDir, activationsForCsv, self.activationsAnalysisCsvHeader, quantizationVariation + '_activations.csv', self.fileHelper)
            if quantizationVariation in self.weightResults.keys():
                weightsForCsv = ResultUtils.formatWeightsAndBiasesForCsv(self.weightResults[quantizationVariation])
                DataFormatter.writeResultsToCsv(csvOutputDir, weightsForCsv, self.inputsWeightsAndBiasesAnalysisCsvHeader, quantizationVariation + '_weights.csv', self.fileHelper)
            if quantizationVariation in self.biasResults.keys():
                biasesForCsv = ResultUtils.formatWeightsAndBiasesForCsv(self.biasResults[quantizationVariation])
                DataFormatter.writeResultsToCsv(csvOutputDir, biasesForCsv, self.inputsWeightsAndBiasesAnalysisCsvHeader, quantizationVariation + '_biases.csv', self.fileHelper)

    @staticmethod
    def writeResultsToCsv(csvOutputDir, results, headers, filename, fileHelper) -> None:
        resultsForCsvItems = results[0].items()
        resultsForCsvAlgorithmHeaders = results[1]
        resultsCsvPath = os.path.join(csvOutputDir, filename)
        fileHelper.deleteFile(resultsCsvPath)
        with open(resultsCsvPath, 'a') as resultsCsv:
            for algorithmName, resultsData in resultsForCsvItems:
                np.savetxt(resultsCsv, ['Verifier Name: ' + algorithmName], delimiter=',', fmt='%s', comments='')
                np.savetxt(resultsCsv, resultsData, delimiter=',', fmt='%s', comments='', header=(', ').join(headers[:3] + resultsForCsvAlgorithmHeaders[algorithmName] + headers[3:]))

    def printSummary(self, weightsForSummary, biasesForSummary, activationsForSummary=None):
        self.logger.print('')
        self.logger.print('The following tables show the percentage of failed nodes for each verifier by quantization option. You should find the quantization option that results in the lowest number of failures for all verifiers.')
        self.logger.print('')
        self.__printSummaryTableToConsole(weightsForSummary, biasesForSummary, activationsForSummary)

    def __printSummaryTableToConsole(self, weightsForSummary, biasesForSummary, activationsForSummary = None):
        for verifier in weightsForSummary.keys():
            self.logger.print(verifier.upper() + ': Percentage of Failed Nodes')
            results = [['Tensor Type'] + list(weightsForSummary[verifier].keys()), ['Weights'] + list(weightsForSummary[verifier].values()), ['Biases'] + list(biasesForSummary[verifier].values()), ['Activations'] + list(activationsForSummary[verifier].values())]

            table = Table(results, True)
            table.print(self.logger)
            self.logger.print('')
