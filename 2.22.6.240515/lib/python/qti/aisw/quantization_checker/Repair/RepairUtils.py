#=============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from typing import List
import qti.aisw.quantization_checker.utils.Verifiers as Verifiers
import qti.aisw.quantization_checker.Repair.Failure as Failure

def translateVerifierToQcFailedVerifierEnum(verifier):
    if verifier == Verifiers.SQNR:
        return Verifiers.QcFailedVerifier.SQNR_FAIL
    if verifier == Verifiers.DATA_DISTRIBUTION:
        return Verifiers.QcFailedVerifier.DATA_DISTRIBUTION_FAIL
    if verifier == Verifiers.DATA_RANGE:
        return Verifiers.QcFailedVerifier.DATA_RANGE_FAIL
    if verifier == Verifiers.STATS:
        return Verifiers.QcFailedVerifier.STATS_FAIL
    if verifier == Verifiers.MIN_MAX:
        return Verifiers.QcFailedVerifier.MIN_MAX_FAIL
    if verifier == Verifiers.MAX_DIFF:
        return Verifiers.QcFailedVerifier.MAX_DIFF_FAIL

def getSetOfVerifiers(weightVerifiers, biasVerifiers, activationVerifiers):
    verifiers = {}
    verifiers = set(weightVerifiers)
    verifiers = verifiers.union(set(biasVerifiers))
    verifiers = verifiers.union(set(activationVerifiers))
    return verifiers

def checkForNullAndGreaterThan(varToCheck, greaterThanValue) -> bool:
    if varToCheck and varToCheck[0] > greaterThanValue:
        return True
    else:
        return False

def buildFailureList(minWeightFailurePercentageByVerfier, minBiasFailurePercentageByVerfier, minActivationFailurePercentageByVerfier) -> List[Failure.Failure]:
    # get list of verifiers with no duplicates
    verifiers = getSetOfVerifiers(minWeightFailurePercentageByVerfier.keys(), minBiasFailurePercentageByVerfier.keys(), minActivationFailurePercentageByVerfier.keys())

    failureList = []
    for verifier in verifiers:
        minWeightFailurePercentage = None
        minBiasFailurePercentage = None
        minActivationFailurePercentage = None
        if verifier in minWeightFailurePercentageByVerfier:
            minWeightFailurePercentage = minWeightFailurePercentageByVerfier[verifier]
        if verifier in minBiasFailurePercentageByVerfier:
            minBiasFailurePercentage = minBiasFailurePercentageByVerfier[verifier]
        if verifier in minActivationFailurePercentageByVerfier:
            minActivationFailurePercentage = minActivationFailurePercentageByVerfier[verifier]

        # get the failure severity, save whether it is a weight, bias or activation failure and what verifier it failed
        if checkForNullAndGreaterThan(minWeightFailurePercentage, 25.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minWeightFailurePercentage[0], minWeightFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.WEIGHT, Failure.FAILURE_SEVERITY.HIGH))
        elif checkForNullAndGreaterThan(minWeightFailurePercentage, 10.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minWeightFailurePercentage[0], minWeightFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.WEIGHT, Failure.FAILURE_SEVERITY.MED))
        elif checkForNullAndGreaterThan(minWeightFailurePercentage, 0.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minWeightFailurePercentage[0], minWeightFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.WEIGHT, Failure.FAILURE_SEVERITY.LOW))
        if checkForNullAndGreaterThan(minBiasFailurePercentage, 25.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minBiasFailurePercentage[0], minBiasFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.BIAS, Failure.FAILURE_SEVERITY.HIGH))
        elif checkForNullAndGreaterThan(minBiasFailurePercentage, 10.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minBiasFailurePercentage[0], minBiasFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.BIAS, Failure.FAILURE_SEVERITY.MED))
        elif checkForNullAndGreaterThan(minBiasFailurePercentage, 0.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minBiasFailurePercentage[0], minBiasFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.BIAS, Failure.FAILURE_SEVERITY.LOW))
        if checkForNullAndGreaterThan(minActivationFailurePercentage, 25.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minActivationFailurePercentage[0], minActivationFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.ACTIVATION, Failure.FAILURE_SEVERITY.HIGH))
        elif checkForNullAndGreaterThan(minActivationFailurePercentage, 10.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minActivationFailurePercentage[0], minActivationFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.ACTIVATION, Failure.FAILURE_SEVERITY.MED))
        elif checkForNullAndGreaterThan(minActivationFailurePercentage, 0.0):
            failureList.append(Failure.Failure(translateVerifierToQcFailedVerifierEnum(verifier), minActivationFailurePercentage[0], minActivationFailurePercentage[1], Failure.FAILURE_TENSOR_TYPE.ACTIVATION, Failure.FAILURE_SEVERITY.LOW))
    return failureList

def sortFailureList(failureList: List[Failure.Failure], descending=False) -> None:
    failureList.sort(key=lambda x: x.getFailurePercentage(), reverse=descending)

def compareFailureLists(beforeRepair: List[Failure.Failure], afterRepair: List[Failure.Failure]):
    improvement = 0.0
    for failure in afterRepair:
        if failure.getTensorType() == beforeRepair[0].getTensorType() and failure.getVerifier() == beforeRepair[0].getVerifier():
            if beforeRepair[0].getFailurePercentage() != 0.0:
                improvement = (beforeRepair[0].getFailurePercentage() - failure.getFailurePercentage()) / beforeRepair[0].getFailurePercentage()
            break
    return (improvement * 100, failure.getFailurePercentage())

def hasHighSeverityFailures(failureList) -> bool:
    for failure in failureList:
        if failure.isHighSeverity():
            return True
    return False

def hasMedSeverityFailures(failureList) -> bool:
    for failure in failureList:
        if failure.isMedSeverity():
            return True
    return False

def getListOfHighSeverityFailures(failureList) -> List[Failure.Failure]:
    highSeverityList = []
    for failure in failureList:
        if failure.isHighSeverity():
            highSeverityList.append(failure)
    return highSeverityList

def getListOfMedSeverityFailures(failureList) -> List[Failure.Failure]:
    medSeverityList = []
    for failure in failureList:
        if failure.isMedSeverity():
            medSeverityList.append(failure)
    return medSeverityList

def getHighSeverityTensorTypes(failureList) -> List[str]:
    highSeverityTensorTypes = set()
    for failure in failureList:
        if failure.isHighSeverity():
            highSeverityTensorTypes.add(failure.getTensorType())
    return list(highSeverityTensorTypes)

def getMedSeverityTensorTypes(failureList) -> List[str]:
    medSeverityTensorTypes = set()
    for failure in failureList:
        if failure.isMedSeverity():
            medSeverityTensorTypes.add(failure.getTensorType())
    return list(medSeverityTensorTypes)

def getTensorTypeString(tensorTypes) -> str:
    tensorTypesString = ''
    for tensorType in tensorTypes:
        if tensorTypesString == '':
            tensorTypesString = tensorType.value
        elif tensorType == tensorTypes[-1]:
            tensorTypesString += str(' and ' + tensorType.value)
        else:
            tensorTypesString += (', ' + tensorType.value)
    return tensorTypesString
