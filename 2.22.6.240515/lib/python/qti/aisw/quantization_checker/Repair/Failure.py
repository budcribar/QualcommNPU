#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from enum import Enum
import qti.aisw.quantization_checker.utils.Verifiers as Verifiers

class FAILURE_SEVERITY(Enum):
    HIGH = 1
    MED = 2
    LOW = 3

class FAILURE_TENSOR_TYPE(Enum):
    WEIGHT = 'weight'
    BIAS = 'bias'
    ACTIVATION = 'activation'

class Failure:
    def __init__(self, verifier, failurePercentage, quantOption, tensorType, severityLevel) -> None:
        self.verifier = verifier
        self.failurePercentage = failurePercentage
        self.quantOption = quantOption
        self.tensorType = tensorType
        self.severityLevel = severityLevel

    def getFailurePercentage(self) -> float:
        return self.failurePercentage

    def isHighSeverity(self) -> bool:
        return self.severityLevel == FAILURE_SEVERITY.HIGH

    def isMedSeverity(self) -> bool:
        return self.severityLevel == FAILURE_SEVERITY.MED

    def isLowSeverity(self) -> bool:
        return self.severityLevel == FAILURE_SEVERITY.LOW

    def getTensorType(self) -> FAILURE_TENSOR_TYPE:
        return self.tensorType

    def getVerifier(self) -> Verifiers.QcFailedVerifier:
        return self.verifier

    def getQuantOption(self) -> str:
        return self.quantOption
