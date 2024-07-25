# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Dict
import numpy as np
import pandas as pd
import os

import qti.aisw.accuracy_debugger.lib.quant_checker.result.nd_result_utils as ResultUtils
from qti.aisw.accuracy_debugger.lib.quant_checker.result.nd_result_set import ResultSet
from .nd_table import Table, Row, Cell
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_utils import verify_path

NEWLINE = '\n'


class DataFormatter:

    def __init__(self, args, quant_schemes, input_files, logger=None):
        self._args = args
        self._output_dir = os.path.join(os.path.dirname(os.path.dirname(self._args.output_dir)),
                                        'results')
        self._quant_schemes = quant_schemes
        self._input_files = input_files
        self._logger = logger
        self._input_results = ResultSet()
        self._activation_results = ResultSet()
        self._weight_results = ResultSet()
        self._bias_results = ResultSet()
        self.inputsWeightsAndBiasesAnalysisCsvHeader = [
            'Node Name', 'Tensor Name', 'Passes Verification', 'Threshold for Verification'
        ]
        self.activationsAnalysisCsvHeader = self.inputsWeightsAndBiasesAnalysisCsvHeader + [
            'Input Filename'
        ]
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

    def setInputResults(self, input_results) -> None:
        self._input_results = input_results

    def setActivationsResults(self, activation_results) -> None:
        self._activation_results = activation_results

    def setWeightResults(self, weight_results) -> None:
        self._weight_results = weight_results

    def setBiasResults(self, bias_results) -> None:
        self._bias_results = bias_results

    def _formatDataForHtml(self) -> Dict:
        results = {}
        for quant_scheme in self._quant_schemes:
            per_input_result_map = {}
            for input_file in self._input_files:
                quant_scheme_result_per_input = []
                quant_scheme_result_per_input.extend(
                    ResultUtils.getResultsForHtml(quant_scheme, self._weight_results))
                quant_scheme_result_per_input.extend(
                    ResultUtils.getResultsForHtml(quant_scheme, self._bias_results))
                quant_scheme_result_per_input.extend(
                    ResultUtils.getActivationResultsForHtml(quant_scheme, input_file,
                                                            self._activation_results))
                per_input_result_map[input_file] = quant_scheme_result_per_input
            results[quant_scheme] = per_input_result_map
        results["input_data"] = ResultUtils.getInputResultsForHtml(self._input_results)
        return results

    def _getMultiIndexForColumns(self, algorithms) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [list(['Quantization Checker Pass/Fail']),
             list(algorithms)])

    def _getHtmlLegendTable(self, algorithms, descriptions):
        htmlString = '<table id=legend><tr align=left><th colspan=2>Legend:</th></tr><tr><th>Quantization Checker Name</th><th>Description</th></tr>'
        # skip the description column since it is self explanatory and is itself an explanation
        for algorithm in algorithms[:-1]:
            htmlString += '<tr><td>' + algorithm + '</td><td>' + descriptions[
                algorithm] + '</td></tr>'
        htmlString += '</table>'
        return htmlString

    def printHtml(self) -> None:
        results = self._formatDataForHtml()
        algorithm_names = ResultUtils.getListOfAlgorithmNamesFromResults(
            self._quant_schemes, results)
        readable_algorithm_names = ResultUtils.translateAlgorithmNamesToDescriptiveNames(
            algorithm_names)
        algo_descriptions = ResultUtils.translateAlgorithmNameToAlgorithmDescription(
            readable_algorithm_names)
        # add the description column manually since we need to interpret the results at the end
        readable_algorithm_names.append('Description')
        columns = self._getMultiIndexForColumns(readable_algorithm_names)
        html_output_dir = verify_path(self._output_dir, 'html')
        for quant_scheme in self._quant_schemes:
            if quant_scheme not in results:
                continue

            quant_scheme_html_dir = verify_path(html_output_dir, quant_scheme)
            for input_file in self._input_files:
                no_algorithms = ResultUtils.getListOfResultsWithoutAlgorithms(
                    results[quant_scheme][input_file])
                only_algorithms = ResultUtils.getListOfAlgorithmsFromResults(
                    results[quant_scheme][input_file], algorithm_names)
                failed_nodes = ResultUtils.getFailedNodes(results[quant_scheme][input_file])
                no_algorithms_failed = ResultUtils.getListOfResultsWithoutAlgorithms(failed_nodes)
                only_algorithms_failed = ResultUtils.getListOfAlgorithmsFromResults(
                    failed_nodes, algorithm_names)

                pd.set_option('display.max_colwidth', None)

                df_only_algos = pd.DataFrame(only_algorithms, columns=columns)
                df_no_algos = pd.DataFrame(
                    no_algorithms, columns=pd.MultiIndex.from_product([
                        list(['']),
                        list(['Op Name', 'Node Name', 'Node Type', 'Scale', 'Offset'])
                    ]))
                df_results = pd.concat([df_no_algos, df_only_algos], axis=1)
                df_results = df_results.set_index([('', 'Op Name'), ('', 'Node Name')], drop=True)
                df_results.index.names = ['Op Name', 'Node Name']
                df_results.sort_index(inplace=True)

                df_only_algos_failed = pd.DataFrame(only_algorithms_failed, columns=columns)
                df_no_algos_failed = pd.DataFrame(
                    no_algorithms_failed, columns=pd.MultiIndex.from_product([
                        list(['']),
                        list(['Op Name', 'Node Name', 'Node Type', 'Scale', 'Offset'])
                    ]))
                df_results_failed = pd.concat([df_no_algos_failed, df_only_algos_failed], axis=1)
                df_results_failed = df_results_failed.set_index([('', 'Op Name'),
                                                                 ('', 'Node Name')], drop=True)
                df_results_failed.index.names = ['Op Name', 'Node Name']
                df_results_failed.sort_index(inplace=True)

                input_file_name = os.path.basename(input_file).split('.')[0]
                with open(os.path.join(quant_scheme_html_dir, input_file_name + '.html'), 'w') as f:
                    f.write(
                        self.htmlHeadBegin.format(title=quant_scheme + ' - ' + input_file_name) +
                        self.htmlHeadEnd + self.htmlBody.format(
                            legend=self._getHtmlLegendTable(readable_algorithm_names,
                                                            algo_descriptions), instructions=
                            'Please consult the latest logs for further details on the failures.',
                            summaryTitle=
                            'Summary of failed nodes that should be inspected: (Total number of nodes analyzed: '
                            + str(len(no_algorithms)) + ' Total number of failed nodes: ' +
                            str(len(no_algorithms_failed)) + ')', title='Results for quantizer: ' +
                            quant_scheme + ' using input file: ' + input_file + ' on model: ' +
                            os.path.basename(self._args.model_path),
                            allNodesTitle='Results for all nodes:', summary=df_results_failed.
                            to_html(justify='center', index=True, escape=False),
                            table=df_results.to_html(justify='center', index=True, escape=False)))
        self._printInputDataToHtml(results, html_output_dir)

    def _printInputDataToHtml(self, results, html_output_dir) -> None:
        input_results_dir = verify_path(html_output_dir, 'input_results')
        algorithm_names = ResultUtils.getListOfAlgorithmNamesFromInputResults(self._input_results)
        readable_algorithm_names = ResultUtils.translateAlgorithmNamesToDescriptiveNames(
            algorithm_names)
        readable_algorithm_names.append('Description')
        columns = self._getMultiIndexForColumns(readable_algorithm_names)
        algo_descriptions = ResultUtils.translateAlgorithmNameToAlgorithmDescription(
            readable_algorithm_names)
        no_algorithms = ResultUtils.getListOfResultsWithoutAlgorithms(results["input_data"])
        file_names = [row[0] for row in no_algorithms]
        only_algorithms = ResultUtils.getListOfAlgorithmsFromResults(results["input_data"],
                                                                     algorithm_names)
        df_only_algos = pd.DataFrame(only_algorithms, columns=columns)
        df_no_algos = pd.DataFrame(
            file_names, columns=pd.MultiIndex.from_product([list(['']),
                                                            list(['File Name'])]))
        df_results = pd.concat([df_no_algos, df_only_algos], axis=1)
        df_results = df_results.set_index([('', 'File Name')], drop=True)
        df_results.index.names = ['File Name']
        with open(os.path.join(input_results_dir, 'input_data_analysis.html'), 'w') as f:
            f.write(
                self.htmlHeadBegin.format(title='Input Data Analysis') + self.htmlHeadEnd +
                self.htmlBody.format(
                    legend=self._getHtmlLegendTable(readable_algorithm_names,
                                                    algo_descriptions), instructions=
                    'Please consult the latest csv or log files for further details on the analysis.',
                    summaryTitle='Total number of input files analyzed: ' +
                    str(len(file_names)), title='Results for input data analysis for the model: ' +
                    os.path.basename(self._args.model_path),
                    allNodesTitle='Analysis for all input files:', summary='',
                    table=df_results.to_html(justify='center', index=True, escape=False)))

    def _printTableToLog(self, header, results):
        for quant_scheme in list(self._quant_schemes):
            # skip quantized models which are failed to get converted correctly
            if quant_scheme not in results:
                continue
            self._logger.info(
                get_progress_message("Results for the " + quant_scheme + " quantization:"))
            results[quant_scheme].insert(0, header)
            table = Table(results[quant_scheme], True)
            log_no_results = True
            for row in table.getRows():
                cells = row.getCells()
                if cells[2].getString() not in (None, ''):
                    self._logger.info(get_progress_message(table.decorate(row)))
                    if not table.isFirstRow():
                        log_no_results = False
            if log_no_results:
                row = [Cell("N/A") for item in header]
                self._logger.info(get_progress_message(table.decorate(Row(row))))

    def printLog(self) -> None:
        if self._activation_results is not None:
            self._logger.info(
                get_progress_message(NEWLINE + NEWLINE + '<====ACTIVATIONS ANALYSIS====>' +
                                     NEWLINE))
            activations_for_log = ResultUtils.formatActivationResultsForLogConsole(
                self._quant_schemes, self._activation_results)
            self._printTableToLog([
                'Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference',
                'Threshold Used', 'Algorithm Used', 'Input Filename', 'Scale', 'Offset'
            ], activations_for_log)

        self._logger.info(
            get_progress_message(NEWLINE + NEWLINE + '<====WEIGHTS ANALYSIS====>' + NEWLINE))
        weights_for_log = ResultUtils.formatResultsForLogConsole(self._quant_schemes,
                                                                 self._weight_results)
        self._printTableToLog([
            'Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used',
            'Algorithm Used', 'Scale', 'Offset'
        ], weights_for_log)
        self._logger.info(
            get_progress_message(NEWLINE + NEWLINE + '<====BIASES ANALYSIS====>' + NEWLINE))
        biases_for_log = ResultUtils.formatResultsForLogConsole(self._quant_schemes,
                                                                self._bias_results)
        self._printTableToLog([
            'Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used',
            'Algorithm Used', 'Scale', 'Offset'
        ], biases_for_log)

    def _printTableToConsole(self, header, results) -> None:
        for quant_scheme in self._quant_schemes:
            # skip quantized models which are failed to get converted correctly
            if quant_scheme not in results:
                continue
            self._logger.info(
                get_progress_message("Results for the " + quant_scheme + " quantization:"))
            results[quant_scheme].insert(0, header)
            table = Table(results[quant_scheme], True)
            log_no_results = True
            for row in table.getRows():
                cells = row.getCells()
                if cells[2].getString() not in (None, ''):
                    self._logger.info(get_progress_message(table.decorate(row)))
                    if not table.isFirstRow():
                        log_no_results = False
            if log_no_results:
                row = [Cell("N/A") for item in header]
                self._logger.info(get_progress_message(table.decorate(Row(row))))

    def printConsole(self) -> None:
        if self._activation_results is not None:
            self._logger.info(
                get_progress_message(NEWLINE + NEWLINE + '<====ACTIVATIONS ANALYSIS FAILURES====>' +
                                     NEWLINE))
            activations_for_console = ResultUtils.formatActivationResultsForLogConsole(
                self._quant_schemes, self._activation_results, showOnlyFailedResults=True)
            self._printTableToConsole([
                'Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference',
                'Threshold Used', 'Algorithm Used', 'Input Filename', 'Scale', 'Offset'
            ], activations_for_console)

        self._logger.info(
            get_progress_message(NEWLINE + NEWLINE + '<====WEIGHTS ANALYSIS FAILURES====>' +
                                 NEWLINE))
        weights_for_console = ResultUtils.formatResultsForLogConsole(self._quant_schemes,
                                                                     self._weight_results,
                                                                     showOnlyFailedResults=True)
        self._printTableToConsole([
            'Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used',
            'Algorithm Used', 'Scale', 'Offset'
        ], weights_for_console)
        self._logger.info(
            get_progress_message(NEWLINE + NEWLINE + '<====BIASES ANALYSIS FAILURES====>' +
                                 NEWLINE))
        biases_for_console = ResultUtils.formatResultsForLogConsole(self._quant_schemes,
                                                                    self._bias_results,
                                                                    showOnlyFailedResults=True)
        self._printTableToConsole([
            'Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used',
            'Algorithm Used', 'Scale', 'Offset'
        ], biases_for_console)

    def printCsv(self) -> None:
        csv_output_dir = verify_path(self._output_dir, 'csv')
        input_data_csv_dir = verify_path(csv_output_dir, 'input_data')
        inputs_for_csv = ResultUtils.formatInputsForCsv(self._input_results)
        DataFormatter.write_results_to_csv(input_data_csv_dir, inputs_for_csv,
                                           self.inputsWeightsAndBiasesAnalysisCsvHeader,
                                           'input_data_analysis')
        for quantizationVariation in self._quant_schemes:
            quant_scheme_csv_output_dir = verify_path(csv_output_dir, quantizationVariation)
            # ONLY PRINT ACTIVATION RESULT IN UNQUANTIZED FOLDER AS WE ARE NOT QUANTIZING ACTIVATIONS
            # THIS IS DIFFERENT THAN THE STANDALONE QUANTCHECKER which has check as:
            # quantizationVariation != 'unquantized'
            if quantizationVariation == 'unquantized' and quantizationVariation in self._activation_results.keys(
            ):
                activationsForCsv = ResultUtils.formatActivationsForCsv(
                    self._activation_results[quantizationVariation])
                DataFormatter.write_results_to_csv(quant_scheme_csv_output_dir, activationsForCsv,
                                                   self.activationsAnalysisCsvHeader,
                                                   quantizationVariation + '_activations')
            if quantizationVariation in self._weight_results.keys():
                weights_for_csv = ResultUtils.formatWeightsAndBiasesForCsv(
                    self._weight_results[quantizationVariation])
                self._logger.info(str(quantizationVariation) + str(weights_for_csv))
                DataFormatter.write_results_to_csv(quant_scheme_csv_output_dir, weights_for_csv,
                                                   self.inputsWeightsAndBiasesAnalysisCsvHeader,
                                                   'weight')
            if quantizationVariation in self._bias_results.keys():
                biases_for_csv = ResultUtils.formatWeightsAndBiasesForCsv(
                    self._bias_results[quantizationVariation])
                DataFormatter.write_results_to_csv(quant_scheme_csv_output_dir, biases_for_csv,
                                                   self.inputsWeightsAndBiasesAnalysisCsvHeader,
                                                   'bias')

    @staticmethod
    def write_results_to_csv(csv_output_dir, results, headers, file_type) -> None:
        results_for_csv_items = results[0].items()
        results_for_csv_algorithm_headers = results[1]
        for algorithm_name, resultsData in results_for_csv_items:
            file_name = os.path.join(csv_output_dir, algorithm_name + "_" + file_type + ".csv")
            if os.path.exists(file_name):
                os.remove(file_name)
            with open(file_name, 'w') as csv_file:
                np.savetxt(csv_file, ['Verifier Name: ' + algorithm_name], delimiter=',', fmt='%s',
                           comments='')
                np.savetxt(
                    csv_file, resultsData, delimiter=',', fmt='%s', comments='',
                    header=(', ').join(headers[:3] +
                                       results_for_csv_algorithm_headers[algorithm_name] +
                                       headers[3:]))

    def printSummary(self, weightsForSummary, biasesForSummary, activationsForSummary=None):
        self._logger.info(
            get_progress_message(
                'The following tables show the percentage of failed nodes for each verifier by quantization option. You should find the quantization option that results in the lowest number of failures for all verifiers.'
            ))
        self.__printSummaryTableToConsole(weightsForSummary, biasesForSummary,
                                          activationsForSummary)

    def __printSummaryTableToConsole(self, weightsForSummary, biasesForSummary,
                                     activationsForSummary=None):
        for verifier in weightsForSummary.keys():
            self._logger.info(
                get_progress_message(verifier.upper() + ': Percentage of Failed Nodes'))
            results = [['Tensor Type'] + list(weightsForSummary[verifier].keys()),
                       ['Weights'] + list(weightsForSummary[verifier].values()),
                       ['Biases'] + list(biasesForSummary[verifier].values()),
                       ['Activations'] + list(activationsForSummary[verifier].values())]

            table = Table(results, True)
            table.print(self._logger)

    def formatResults(self, processed_results):
        input_data_result, weight_result, bias_result, activation_result, getWeightFailurePercentage, getBiasFailurePercentage, getActivationFailurePercentage = processed_results
        self.setInputResults(input_data_result)
        self.setActivationsResults(activation_result)
        self.setBiasResults(bias_result)
        self.setWeightResults(weight_result)
        self.printLog()
        if self._args.generate_html:
            self.printHtml()
        if self._args.generate_csv:
            self.printCsv()
        self.printConsole()
        self.printSummary(getWeightFailurePercentage, getBiasFailurePercentage,
                          getActivationFailurePercentage)
