#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
from enum import Enum
from qti.aisw.quantization_checker.utils import utils
from qti.aisw.quantization_checker.utils.Logger import PrintOptions

class ProgressStage(Enum):
    GENERATOR = 1
    BUILDER = 2
    RUNNER = 3
    PROCESSOR = 4

class Progress:
    totalProgress = 0
    totalNumberOfModels = 1
    progressPercentage = {ProgressStage.GENERATOR : 0, ProgressStage.BUILDER : 0, ProgressStage.RUNNER : 0, ProgressStage.PROCESSOR : 0}
    perModelProgress = 100
    currModelProgressLimit = 0
    limitBeforeFinalCall = 99
    logger = None

    def setProgressInfo(model, logger, skipGenerator, skipBuilder, skipRunner):
        Progress.logger = logger
        if os.path.isdir(model):
            models = utils.buildModelDict(model)
            Progress.totalNumberOfModels = len(models)
            Progress.perModelProgress = Progress.limitBeforeFinalCall/Progress.totalNumberOfModels
        if skipGenerator and skipBuilder and skipRunner:
            Progress.progressPercentage = {ProgressStage.GENERATOR : 0, ProgressStage.BUILDER : 0, ProgressStage.RUNNER : 0, ProgressStage.PROCESSOR : Progress.perModelProgress}
        elif skipGenerator and skipBuilder:
            Progress.progressPercentage = {ProgressStage.GENERATOR : 0, ProgressStage.BUILDER : 0, ProgressStage.RUNNER : (Progress.perModelProgress * 0.5), ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.5)}
        elif skipGenerator and skipRunner:
            Progress.progressPercentage = {ProgressStage.GENERATOR : 0, ProgressStage.BUILDER : (Progress.perModelProgress * 0.5), ProgressStage.RUNNER : 0, ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.5)}
        elif skipGenerator:
            Progress.progressPercentage = {ProgressStage.GENERATOR : 0, ProgressStage.BUILDER : (Progress.perModelProgress * 0.4), ProgressStage.RUNNER : (Progress.perModelProgress * 0.1), ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.5)}
        elif skipBuilder and skipRunner:
            Progress.progressPercentage = {ProgressStage.GENERATOR : (Progress.perModelProgress * 0.5 / 8), ProgressStage.BUILDER : 0, ProgressStage.RUNNER : 0, ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.5)}
        elif skipBuilder:
            Progress.progressPercentage = {ProgressStage.GENERATOR : (Progress.perModelProgress * 0.4 / 8), ProgressStage.BUILDER : 0, ProgressStage.RUNNER : (Progress.perModelProgress * 0.05), ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.5)}
        elif skipRunner:
            Progress.progressPercentage = {ProgressStage.GENERATOR : (Progress.perModelProgress * 0.4 / 8), ProgressStage.BUILDER : (Progress.perModelProgress * 0.1), ProgressStage.RUNNER : 0, ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.5)}
        else:
            Progress.progressPercentage = {ProgressStage.GENERATOR : (Progress.perModelProgress * 0.3 / 8), ProgressStage.BUILDER : (Progress.perModelProgress * 0.2), ProgressStage.RUNNER : (Progress.perModelProgress * 0.1), ProgressStage.PROCESSOR : (Progress.perModelProgress * 0.4)}

    def updateProgressMessage(stepSize):
        message = "Processing |"
        for i in range(int(stepSize / 2)):
            message = message + "#"
        for i in range(50 - int(stepSize / 2)):
            message = message + " "
        Progress.logger.print(message + "| " + str(int(stepSize)) + "/100 \n", PrintOptions.CONSOLE)

    def updateProgress(step):
        if Progress.totalProgress + step >= Progress.currModelProgressLimit:
            Progress.totalProgress = Progress.currModelProgressLimit
        else:
            Progress.totalProgress = Progress.totalProgress + step
        Progress.updateProgressMessage(Progress.totalProgress)

    def updateProgressLimit():
        Progress.currModelProgressLimit = Progress.currModelProgressLimit + Progress.perModelProgress

    def updateModelProgress():
        Progress.totalProgress = Progress.currModelProgressLimit
        Progress.updateProgressMessage(Progress.totalProgress)

    def finishProcessor():
        Progress.logger.print("Processing completed!!!", PrintOptions.CONSOLE)

    def getStepSize(stage):
        if stage == ProgressStage.GENERATOR:
            return Progress.progressPercentage[ProgressStage.GENERATOR]
        elif stage == ProgressStage.RUNNER:
            return Progress.progressPercentage[ProgressStage.RUNNER]
        elif stage == ProgressStage.PROCESSOR:
            return Progress.progressPercentage[ProgressStage.PROCESSOR]
        elif stage == ProgressStage.BUILDER:
            return Progress.progressPercentage[ProgressStage.BUILDER]
        return 0

    def getRemainingProgress() -> int:
        Progress.currModelProgressLimit = 100
        return 100 - Progress.totalProgress
