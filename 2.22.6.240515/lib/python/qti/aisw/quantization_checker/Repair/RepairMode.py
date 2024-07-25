#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

from enum import Enum
from typing import List
import numpy as np
from qti.aisw.quantization_checker.utils.Logger import PrintOptions
from qti.aisw.quantization_checker.utils.Op import Op
import qti.aisw.quantization_checker.utils.Verifiers as Verifiers
import qti.aisw.quantization_checker.Repair.Failure as Failure
import qti.aisw.quantization_checker.Repair.RepairUtils as RepairUtils

class Status(Enum):
    OK = 'Success'
    NO_FAILURES = 'There are no failures to repair.'
    ERROR = 'An unknown error occured.'
    MAX_BIT_WIDTH = 'Max bit-width reached, cannot increase bit-width any further.'

class RepairModeStatus:
    def __init__(self, status) -> None:
        self.status = status
        self.description = self.status.value

    def getStatus(self) -> Status:
        return self.status

    def getDescription(self) -> str:
        return self.description

    def setStatus(self, status : Status) -> None:
        self.status = status
        self.description = self.status.value

class RepairMode:
    class RepairModeArgs:
        def __init__(self, biasBitWidth, weightBitWidth, activationBitWidth) -> None:
            self.originalBiasBitWidth = biasBitWidth
            self.originalWeightBitWidth = weightBitWidth
            self.originalActivationBitWidth = activationBitWidth
            self.repairBiasBitWidth = self.originalBiasBitWidth
            self.repairWeightBitWidth = self.originalWeightBitWidth
            self.repairActivationBitWidth = self.originalActivationBitWidth

        def getBiasBitWidth(self, isInRepairMode) -> int:
            if isInRepairMode:
                return self.__getRepairBiasBitWidth()
            else:
                return self.__getOriginalBiasBitWidth()

        def setBiasBitWidth(self, bitWidth) -> None:
            self.repairBiasBitWidth = bitWidth
            Op.setBiasWidth(bitWidth)

        def __getRepairBiasBitWidth(self) -> int:
            return self.repairBiasBitWidth

        def __getOriginalBiasBitWidth(self) -> int:
            return self.originalBiasBitWidth

        def getWeightBitWidth(self, isInRepairMode) -> int:
            if isInRepairMode:
                return self.__getRepairWeightBitWidth()
            else:
                return self.__getOriginalWeightBitWidth()

        def setWeightBitWidth(self, bitWidth) -> None:
            self.repairWeightBitWidth = bitWidth
            Op.setWeightWidth(bitWidth)

        def __getRepairWeightBitWidth(self) -> int:
            return self.repairWeightBitWidth

        def __getOriginalWeightBitWidth(self) -> int:
            return self.originalWeightBitWidth

        def getActivationBitWidth(self, isInRepairMode) -> int:
            if isInRepairMode:
                return self.__getRepairActivationBitWidth()
            else:
                return self.__getOriginalActivationBitWidth()

        def setActivationBitWidth(self, bitWidth) -> None:
            self.repairActivationBitWidth = bitWidth
            Op.setActivationWidth(bitWidth)

        def __getRepairActivationBitWidth(self) -> int:
            return self.repairActivationBitWidth

        def __getOriginalActivationBitWidth(self) -> int:
            return self.originalActivationBitWidth

    class Improvement:
        IMPROVEMENT_DESC = {}
        IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.STATS_FAIL] = 'normalizing the data'
        IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.DATA_RANGE_FAIL] = 'increasing the bit-width from {org_bw} to {repair_bw}'
        IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.SQNR_FAIL] = 'increasing the bit-width from {org_bw} to {repair_bw}'
        IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.MIN_MAX_FAIL] = 'increasing the bit-width from {org_bw} to {repair_bw}'
        IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.MAX_DIFF_FAIL] = 'increasing the bit-width from {org_bw} to {repair_bw}'
        IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL] = 'increasing the bit-width from {org_bw} to {repair_bw}'

        def __init__(self, tensorType: str, originalFailurePercentage: float, verifierDesc: str, quantOption: str, failureType: Verifiers.QcFailedVerifier) -> None:
            self.tensorType = tensorType
            self.originalFailurePercentage = originalFailurePercentage
            self.repairedFailurePercentage = 0.0
            self.repairedQuantOption = None
            self.improvementAmount = 0.0
            self.verifierDesc = verifierDesc
            self.quantOption = quantOption
            self.failureType = failureType
            self.repairModeArgs = None

        def setRepairedFailurePercentage(self, repairedFailurePercentage) -> None:
            self.repairedFailurePercentage = repairedFailurePercentage

        def setRepairedQuantOption(self, repairedQuantOption) -> None:
            self.repairedQuantOption = repairedQuantOption

        def setImprovementAmount(self, improvementAmount) -> None:
            self.improvementAmount = improvementAmount

        def toString(self, repairModeArgs) -> str:
            repairDescription = ''
            if self.improvementAmount > 0.0:
                repairDescription = 'The original percentage of failed nodes for the verifier ' + self.verifierDesc + ' and quantization option ' + self.quantOption + ' is ' + str(self.originalFailurePercentage) + \
                '% with a repaired percentage of failed nodes of ' + str(self.repairedFailurePercentage) + '%, resulting in an improvement of ' + str(self.improvementAmount) + \
                '% for the tensor ' + self.tensorType.value + ' and the quantization option ' + self.repairedQuantOption + '. This was accomplished by ' + self.getImprovementDescription(repairModeArgs)
                if self.quantOption != self.repairedQuantOption:
                    repairDescription += ' and updating the quantization option'
                repairDescription += '.'
            else:
                repairDescription = 'Unfortunately, the attempted repair did not result in an improvement, please use original values.'
            return repairDescription

        def getImprovementDescription(self, repairModeArgs) -> str:
            description = str()
            if self.failureType == Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL:
                description = self.IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL]
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.WEIGHT:
                    description = description.format(org_bw=str(repairModeArgs.getWeightBitWidth(False)), repair_bw=str(repairModeArgs.getWeightBitWidth(True)))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.ACTIVATION:
                    description = description.format(org_bw=self.repairModeArgs.getActivationBitWidth(False), repair_bw=self.repairModeArgs.getActivationBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.BIAS:
                    description = description.format(org_bw=self.repairModeArgs.getBiasBitWidth(False), repair_bw=self.repairModeArgs.getBiasBitWidth(True))
            if self.failureType == Verifiers.QcFailedVerifier.DATA_RANGE_FAIL:
                description = self.IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.DATA_RANGE_FAIL]
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.WEIGHT:
                    description = description.format(org_bw=self.repairModeArgs.getWeightBitWidth(False), repair_bw=self.repairModeArgs.getWeightBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.ACTIVATION:
                    description = description.format(org_bw=self.repairModeArgs.getActivationBitWidth(False), repair_bw=self.repairModeArgs.getActivationBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.BIAS:
                    description = description.format(org_bw=self.repairModeArgs.getBiasBitWidth(False), repair_bw=self.repairModeArgs.getBiasBitWidth(True))
            if self.failureType == Verifiers.QcFailedVerifier.MAX_DIFF_FAIL:
                description = self.IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.MAX_DIFF_FAIL]
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.WEIGHT:
                    description = description.format(org_bw=self.repairModeArgs.getWeightBitWidth(False), repair_bw=self.repairModeArgs.getWeightBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.ACTIVATION:
                    description = description.format(org_bw=self.repairModeArgs.getActivationBitWidth(False), repair_bw=self.repairModeArgs.getActivationBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.BIAS:
                    description = description.format(org_bw=self.repairModeArgs.getBiasBitWidth(False), repair_bw=self.repairModeArgs.getBiasBitWidth(True))
            if self.failureType == Verifiers.QcFailedVerifier.MIN_MAX_FAIL:
                description = self.IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.MIN_MAX_FAIL]
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.WEIGHT:
                    description = description.format(org_bw=str(repairModeArgs.getWeightBitWidth(False)), repair_bw=str(repairModeArgs.getWeightBitWidth(True)))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.ACTIVATION:
                    description = description.format(org_bw=str(repairModeArgs.getActivationBitWidth(False)), repair_bw=str(repairModeArgs.getActivationBitWidth(True)))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.BIAS:
                    description = description.format(org_bw=str(repairModeArgs.getBiasBitWidth(False)), repair_bw=str(repairModeArgs.getBiasBitWidth(True)))
            if self.failureType == Verifiers.QcFailedVerifier.SQNR_FAIL:
                description = self.IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.SQNR_FAIL]
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.WEIGHT:
                    description = description.format(org_bw=self.repairModeArgs.getWeightBitWidth(False), repair_bw=self.repairModeArgs.getWeightBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.ACTIVATION:
                    description = description.format(org_bw=self.repairModeArgs.getActivationBitWidth(False), repair_bw=self.repairModeArgs.getActivationBitWidth(True))
                if self.tensorType == Failure.FAILURE_TENSOR_TYPE.BIAS:
                    description = description.format(org_bw=self.repairModeArgs.getBiasBitWidth(False), repair_bw=self.repairModeArgs.getBiasBitWidth(True))
            if self.failureType == Verifiers.QcFailedVerifier.STATS_FAIL:
                description = self.IMPROVEMENT_DESC[Verifiers.QcFailedVerifier.STATS_FAIL]
            return description

    def __init__(self, enterRepairModeCmdLine, biasBitWidth, weightBitWidth, activationBitWidth, logger) -> None:
        self.__PRE_TEXT__ = '[**REPAIR MODE**]: '
        self.logger = logger
        self.isInRepairMode = False
        self.runTool = True
        self.enterRepairModeCmdLine = enterRepairModeCmdLine
        self.originalFailureList = []
        self.highSeverityFailures = []
        self.medSeverityFailures = []
        self.repairModeArgs = RepairMode.RepairModeArgs(biasBitWidth, weightBitWidth, activationBitWidth)
        self.improvement = None

    def getIsInRepairMode(self) -> bool:
        return self.isInRepairMode

    def enterRepairMode(self) -> None:
        self.isInRepairMode = True

    def exitRepairMode(self) -> None:
        self.isInRepairMode = False
        self.stopRunning()

    def isRunning(self) -> bool:
        return self.runTool

    def stopRunning(self) -> None:
        self.runTool = False

    def getActivationBitWidth(self) -> int:
        return self.repairModeArgs.getActivationBitWidth(self.isInRepairMode)

    def getWeightBitWidth(self) -> int:
        return self.repairModeArgs.getWeightBitWidth(self.isInRepairMode)

    def getBiasBitWidth(self) -> int:
        return self.repairModeArgs.getBiasBitWidth(self.isInRepairMode)

    def doRepair(self, minWeightFailurePercentageByVerfier, minBiasFailurePercentageByVerfier, minActivationFailurePercentageByVerfier) -> None:
        failureList = RepairUtils.buildFailureList(minWeightFailurePercentageByVerfier, minBiasFailurePercentageByVerfier, minActivationFailurePercentageByVerfier)
        RepairUtils.sortFailureList(failureList)

        if not self.getIsInRepairMode():
            self.originalFailureList = failureList

            # TODO: consider using a state machine pattern here instead of if...else...
            if not failureList:
                self.logger.print(self.__PRE_TEXT__ + 'No failures found, failure list is None.', PrintOptions.LOGFILE)
                self.stopRunning()
            elif self.enterRepairModeCmdLine:
                self.logger.print(self.__PRE_TEXT__ + 'The user has elected to choose the repair mode, entering repair mode now...')
                status = RepairModeStatus(Status.NO_FAILURES)
                if failureList:
                    status = self.__repair(failureList)
                if status.getStatus() != Status.OK:
                    self.logger.print(self.__PRE_TEXT__ + 'An error has occurred attempting to repair model: ' + status.getDescription())
                    self.stopRunning()
            elif RepairUtils.hasHighSeverityFailures(failureList):
                self.highSeverityFailures = RepairUtils.getListOfHighSeverityFailures(failureList)
                highSeverityTensorTypes = RepairUtils.getHighSeverityTensorTypes(self.highSeverityFailures)
                userPrompt = 'A large number of ' + RepairUtils.getTensorTypeString(highSeverityTensorTypes) + ' failures have been detected. Would you like to enter repair mode to reduce the number of failures? Y/N'
                self.logger.print(self.__PRE_TEXT__ + userPrompt)
                repairYesNo = input()
                if repairYesNo.upper() == 'Y':
                    self.logger.print(self.__PRE_TEXT__ + 'You have selected to enter the repair mode, please be aware that the repair mode can take some time to complete.')
                    status = self.__repair(self.highSeverityFailures)
                    if status.getStatus() != Status.OK:
                        self.logger.print(self.__PRE_TEXT__ + 'An error has occurred attempting to repair model: ' + status.getDescription())
                        self.stopRunning()
                else:
                    self.logger.print(self.__PRE_TEXT__ + 'You have selected to skip the repair mode. The tool will now exit.')
                    self.stopRunning()
            elif RepairUtils.hasMedSeverityFailures(failureList):
                self.medSeverityFailures = RepairUtils.getListOfMedSeverityFailures(failureList)
                medSeverityTensorTypes = RepairUtils.getMedSeverityTensorTypes(self.medSeverityFailures)
                userPrompt = 'A small number of ' + RepairUtils.getTensorTypeString(medSeverityTensorTypes) + ' failures have been detected. If you would like to use the repair mode feature, please do so using the configuration file option.'
                # warn user but do not repair
                self.logger.print(self.__PRE_TEXT__ + userPrompt)
                self.exitRepairMode()
            else:
                # low number of failures, exit normally
                self.stopRunning()
        else:
            self.logger.print(self.__PRE_TEXT__ + 'Repair mode complete, please check results for differences in performance.')
            improvement = RepairUtils.compareFailureLists(self.originalFailureList, failureList)
            self.improvement.setRepairedQuantOption(failureList[0].getQuantOption())
            self.improvement.setImprovementAmount(improvement[0])
            self.improvement.setRepairedFailurePercentage(improvement[1])
            self.logger.print(self.__PRE_TEXT__ + self.improvement.toString(self.repairModeArgs))
            self.exitRepairMode()

    def __repair(self, failures) -> RepairModeStatus:
        self.enterRepairMode()
        return self.__getSolutionForFailures(failures)

    # TODO: repair all failures, atm we only repair the first one and then return...
    def __getSolutionForFailures(self, failures: List[Failure.Failure]) -> RepairModeStatus:
        for failure in failures:
            if failure.getVerifier() == Verifiers.QcFailedVerifier.STATS_FAIL:
                self.logger.print(self.__PRE_TEXT__ + 'Performing repair for Verifier ' + Verifiers.STATS_DESCRIPTIVE_NAME + ': ' + Verifiers.FAILED_VERIFIERS_SOLN_DESC[Verifiers.QcFailedVerifier.STATS_FAIL])
                self.improvement = RepairMode.Improvement(failure.getTensorType(), failure.getFailurePercentage(), Verifiers.STATS_DESCRIPTIVE_NAME, failure.getQuantOption(), Verifiers.QcFailedVerifier.STATS_FAIL)
                return RepairMode.StatsRepair(self.repairModeArgs)(failure)
            if failure.getVerifier() == Verifiers.QcFailedVerifier.DATA_RANGE_FAIL:
                self.logger.print(self.__PRE_TEXT__ + 'Performing repair for Verifier ' + Verifiers.DATA_RANGE_DESCRIPTIVE_NAME + ': ' + Verifiers.FAILED_VERIFIERS_SOLN_DESC[Verifiers.QcFailedVerifier.DATA_RANGE_FAIL])
                self.improvement = RepairMode.Improvement(failure.getTensorType(), failure.getFailurePercentage(), Verifiers.DATA_RANGE_DESCRIPTIVE_NAME, failure.getQuantOption(), Verifiers.QcFailedVerifier.DATA_RANGE_FAIL)
                return RepairMode.DataRangeRepair(self.repairModeArgs)(failure)
            if failure.getVerifier() == Verifiers.QcFailedVerifier.SQNR_FAIL:
                self.logger.print(self.__PRE_TEXT__ + 'Performing repair for Verifier ' + Verifiers.SQNR_DESCRIPTIVE_NAME + ': ' + Verifiers.FAILED_VERIFIERS_SOLN_DESC[Verifiers.QcFailedVerifier.SQNR_FAIL])
                self.improvement = RepairMode.Improvement(failure.getTensorType(), failure.getFailurePercentage(), Verifiers.SQNR_DESCRIPTIVE_NAME, failure.getQuantOption(), Verifiers.QcFailedVerifier.SQNR_FAIL)
                return RepairMode.SqnrRepair(self.repairModeArgs)(failure)
            if failure.getVerifier() == Verifiers.QcFailedVerifier.MIN_MAX_FAIL:
                self.logger.print(self.__PRE_TEXT__ + 'Performing repair for Verifier ' + Verifiers.MIN_MAX_DESCRIPTIVE_NAME + ': ' + Verifiers.FAILED_VERIFIERS_SOLN_DESC[Verifiers.QcFailedVerifier.MIN_MAX_FAIL])
                self.improvement = RepairMode.Improvement(failure.getTensorType(), failure.getFailurePercentage(), Verifiers.MIN_MAX_DESCRIPTIVE_NAME, failure.getQuantOption(), Verifiers.QcFailedVerifier.MIN_MAX_FAIL)
                return RepairMode.MinMaxRepair(self.repairModeArgs)(failure)
            if failure.getVerifier() == Verifiers.QcFailedVerifier.MAX_DIFF_FAIL:
                self.logger.print(self.__PRE_TEXT__ + 'Performing repair for Verifier ' + Verifiers.MAX_DIFF_DESCRIPTIVE_NAME + ': ' + Verifiers.FAILED_VERIFIERS_SOLN_DESC[Verifiers.QcFailedVerifier.MAX_DIFF_FAIL])
                self.improvement = RepairMode.Improvement(failure.getTensorType(), failure.getFailurePercentage(), Verifiers.MAX_DIFF_DESCRIPTIVE_NAME, failure.getQuantOption(), Verifiers.QcFailedVerifier.MAX_DIFF_FAIL)
                return RepairMode.MaxDiffRepair(self.repairModeArgs)(failure)
            if failure.getVerifier() == Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL:
                self.logger.print(self.__PRE_TEXT__ + 'Performing repair for Verifier ' + Verifiers.DATA_DISTRIBUTION_DESCRIPTIVE_NAME + ': ' + Verifiers.FAILED_VERIFIERS_SOLN_DESC[Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL])
                self.improvement = RepairMode.Improvement(failure.getTensorType(), failure.getFailurePercentage(), Verifiers.DATA_DISTRIBUTION_DESCRIPTIVE_NAME, failure.getQuantOption(), Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL)
                return RepairMode.DataDistributionRepair(self.repairModeArgs)(failure)
            self.logger.print('Current failure percentage for tensor type ' + failure.getTensorType().value + ' for the ' + failure.getQuantOption() + ' quantizer is ' + str(failure.getFailurePercentage()), PrintOptions.LOG)

    class Repair():
        def __init__(self, repairModeArgs) -> None:
            self.repairModeArgs = repairModeArgs

        def increaseBitWidth(self, failure: Failure.Failure) -> RepairModeStatus:
            status = RepairModeStatus(Status.OK)
            if failure.getTensorType() == Failure.FAILURE_TENSOR_TYPE.ACTIVATION:
                if self.repairModeArgs.getActivationBitWidth(isInRepairMode=False) == str(np.iinfo(np.uint8).bits):
                    self.repairModeArgs.setActivationBitWidth(str(np.iinfo(np.uint16).bits))
                else:
                    status.setStatus(Status.MAX_BIT_WIDTH)
            elif failure.getTensorType() == Failure.FAILURE_TENSOR_TYPE.WEIGHT:
                if self.repairModeArgs.getWeightBitWidth(isInRepairMode=False) == str(np.iinfo(np.uint8).bits):
                    self.repairModeArgs.setWeightBitWidth(str(np.iinfo(np.uint16).bits))
                else:
                    status.setStatus(Status.MAX_BIT_WIDTH)
            elif failure.getTensorType() == Failure.FAILURE_TENSOR_TYPE.BIAS:
                if self.repairModeArgs.getBiasBitWidth(isInRepairMode=False) == str(np.iinfo(np.uint8).bits):
                    self.repairModeArgs.setBiasBitWidth(str(np.iinfo(np.uint32).bits))
                else:
                    status.setStatus(Status.MAX_BIT_WIDTH)
            else:
                status.setStatus(Status.ERROR)
            return status

    class StatsRepair(Repair):
        def __init__(self, repairModeArgs) -> None:
            super().__init__(repairModeArgs)

        # make the data symmetric...this should be like batch normalization?
        def __call__(self, failure) -> RepairModeStatus:
            return RepairModeStatus(Status.OK)

    class DataRangeRepair(Repair):
        def __init__(self, repairModeArgs) -> None:
            super().__init__(repairModeArgs)

        # increase bit width...
        def __call__(self, failure) -> RepairModeStatus:
            return super().increaseBitWidth(failure)

    class SqnrRepair(Repair):
        def __init__(self, repairModeArgs) -> None:
            super().__init__(repairModeArgs)

        # use QAT...
        def __call__(self, failure) -> RepairModeStatus:
            return RepairModeStatus(Status.OK)

    class MinMaxRepair(Repair):
        def __init__(self, repairModeArgs) -> None:
            super().__init__(repairModeArgs)

        # increase bit width...
        def __call__(self, failure) -> RepairModeStatus:
            return super().increaseBitWidth(failure)

    class MaxDiffRepair(Repair):
        def __init__(self, repairModeArgs) -> None:
            super().__init__(repairModeArgs)

        # use QAT? increase bit width? or adjust encodings?
        def __call__(self, failure) -> RepairModeStatus:
            return super().increaseBitWidth(failure)

    class DataDistributionRepair(Repair):
        def __init__(self, repairModeArgs) -> None:
            super().__init__(repairModeArgs)

        # increase bit width...
        def __call__(self, failure) -> RepairModeStatus:
            return super().increaseBitWidth(failure)
