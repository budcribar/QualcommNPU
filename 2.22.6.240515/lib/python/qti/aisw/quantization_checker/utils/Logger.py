#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import re
import sys
import time
from enum import Enum

NEWLINE = '\n'

class PrintOptions(Enum):
    CONSOLE = 1
    LOGFILE = 2
    CONSOLE_LOGFILE = 3

class Logger(object):
    def __init__(self, outputDir, logFileName = "Output"):
        self.logFile = None
        self.logFileName = logFileName
        self.outputDir = outputDir

    def initLogFile(self, filenameRoot):
        if not os.path.exists(self.outputDir):
            try:
                os.makedirs(os.path.abspath(self.outputDir), mode=0o755)
            except FileExistsError as fe:
                print("Despite the check, the path somehow exists already: " + fe)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.logFile = open(os.path.join(self.outputDir, filenameRoot + "_" + timestr + ".log"), "w")

    def printToConsole(self, str, flush=True):
        if type(str) == bytes:
            sys.stdout.buffer.write(str)
        else:
            if str.endswith('\n') or str.endswith('\r'):
                print(str, end='', flush=flush)
            else:
                print(str, flush=flush)

    def printToLogFile(self, str, flush=True):
        if not self.logFile:
            self.initLogFile(self.logFileName)
        if type(str) == bytes:
            str = str.decode("utf-8")
        if str.endswith('\n') or str.endswith('\r'):
            self.logFile.write(re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", '', str))
        else:
            self.logFile.write(re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", '', str) + NEWLINE)
        if flush == True:
            self.logFile.flush()

    def printToConsoleAndLog(self, str, flush=True):
        self.printToConsole(str, flush)
        self.printToLogFile(str, flush)

    def print(self, str, printOption=PrintOptions.CONSOLE_LOGFILE, flush=True):
        if printOption == PrintOptions.CONSOLE:
            self.printToConsole(str, flush)
        elif printOption == PrintOptions.LOGFILE:
            self.printToLogFile(str, flush)
        elif printOption == PrintOptions.CONSOLE_LOGFILE:
            self.printToConsoleAndLog(str, flush)

    def flush(self, flushOption=PrintOptions.CONSOLE_LOGFILE):
        if flushOption == PrintOptions.CONSOLE:
            print(flush=True)
        elif flushOption == PrintOptions.LOGFILE and self.logFile is not None:
            self.logFile.flush()
        elif flushOption == PrintOptions.CONSOLE_LOGFILE:
            print(flush=True)
            if self.logFile:
                self.logFile.flush()

    def __del__(self):
        if self.logFile is not None:
            self.logFile.close()

def getLogger(outputDir, modelFile, logFilename):
    return Logger(os.path.join(outputDir, logFilename), logFilename + '_' + os.path.splitext(os.path.basename(modelFile))[0])
