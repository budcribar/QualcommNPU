#=============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import shutil

from qti.aisw.quantization_checker.utils.Logger import PrintOptions

class FileUtils:
    class Stack:
        def __init__(self):
            self.stack = []

        def push(self, item):
            self.stack.append(item)

        def pop(self):
            return self.stack.pop()

    def __init__(self, logger):
        self.logger = logger
        self.dirStack = FileUtils.Stack()

    def makeSubdir(self, dirPathToCreate):
        self.logger.print("Creating subdirectory with possible parents " + dirPathToCreate, PrintOptions.LOGFILE)
        os.makedirs(os.path.abspath(dirPathToCreate), mode=0o755, exist_ok=True)
        return 0

    def changeDir(self, dirPath):
        self.logger.print('Changing directory to: ' + str(dirPath), PrintOptions.LOGFILE)
        result = 0
        try:
            os.chdir(dirPath)
        except OSError as e:
            self.logger.print("Unable to change directories, error: " + str(e), PrintOptions.LOGFILE)
            result = -1
        return result

    def pushDir(self, dirPath):
        # push old directory onto stack
        self.dirStack.push(os.getcwd())
        # if we cannot change directories...
        if self.changeDir(dirPath):
            # pop the old directory
            self.dirStack.pop()

    def popDir(self):
        # get the current working directory
        originalDir = os.getcwd()
        # get the previous directory
        newDir = self.dirStack.pop()
        # change to the previous directory
        if self.changeDir(newDir):
            # if we cannot return to the previous directory, restore the stack...
            self.dirStack.push(newDir)
            originalDir = newDir
        return originalDir

    def copyFile(self, fileToCopy, destination):
        result = 0
        self.logger.print('Copying the following file: ' + fileToCopy + ' to: ' + destination, PrintOptions.LOGFILE)
        if os.path.exists(destination):
            shutil.copy(fileToCopy, destination)
        else:
            self.logger.print('Destination path does not exist.', PrintOptions.LOGFILE)
            result = -1
        return result

    def deleteFile(self, pathToFileToDelete):
        result = 0
        if os.path.exists(pathToFileToDelete):
            self.logger.print('Deleting file: ' + pathToFileToDelete, PrintOptions.LOGFILE)
            try:
                os.remove(pathToFileToDelete)
            except OSError as e:
                self.logger.print('Attempt to delete file failed, exception thrown for: ' + pathToFileToDelete + '\n' + e, PrintOptions.LOGFILE)
                result = -1
        else:
            self.logger.print('Attempt to delete file failed, path does not exist: ' + pathToFileToDelete, PrintOptions.LOGFILE)
            result = -1
        return result

    def deleteDirAndContents(self, pathToDirToDelete):
        result = 0
        if os.path.exists(pathToDirToDelete):
            self.logger.print('Deleting directory and contents: ' + pathToDirToDelete, PrintOptions.LOGFILE)
            try:
                shutil.rmtree(pathToDirToDelete)
            except Exception as e:
                self.logger.print('Attempt to delete directory failed, exception thrown for: ' + pathToDirToDelete + '\n' + e, PrintOptions.LOGFILE)
                result = -1
        else:
            self.logger.print('Attempt to delete directory failed, path does not exist: ' + pathToDirToDelete, PrintOptions.LOGFILE)
            result = -1
        return result

class ScopedFileUtils:
    def __init__(self, dirPath, fileUtils: FileUtils):
        self.dirPath = dirPath
        self.fileUtils = fileUtils

    def __enter__(self):
        self.fileUtils.pushDir(self.dirPath)

    def __exit__(self, exception_type=None, exception_value=None, traceback=None):
        self.fileUtils.popDir()
