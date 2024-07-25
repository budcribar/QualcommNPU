# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class Cell:

    def __init__(self, cellValue):
        self.value = cellValue
        self.length = len(self.getString())

    def getString(self):
        return str(self.value)

    def getLength(self):
        return self.length


class Row:

    def __init__(self, listOfCells=[]):
        self.cells = listOfCells
        self.numCols = len(listOfCells)

    def getNumCols(self):
        return self.numCols

    def getCells(self):
        return self.cells

    def getString(self):
        rowStr = ""
        for cell in self.cells:
            rowStr += " " + cell.getString()
        rowStr += " "
        return rowStr


class Table:
    BOLD = '\033[1m'
    CLEAR = '\033[0m'
    NEWLINE = '\n'

    def __init__(self, tblMatrix, showHeader=False, showBorders=True, variableLengthColums=False):
        self.showHeader = showHeader
        self.showBorders = showBorders
        self.variableLengthColums = variableLengthColums
        self.firstRow = True
        self.table = ""
        self.numCols = 0
        self.numRows = 0
        self.maxCellLength = 0
        self.maxColWidth = []
        self.rows = self.__convertRawDataToTable(tblMatrix)

    def isFirstRow(self):
        return self.firstRow

    def getNumRows(self):
        return self.numRows

    def __convertRawDataToTable(self, tblMatrix):
        self.numRows = len(tblMatrix)
        self.numCols = len(tblMatrix[0])
        self.maxCellLength = 0
        rows = []
        self.maxColWidth = [0] * self.numCols
        for row in tblMatrix:
            cells = []
            for idx, col in enumerate(row):
                currentLength = len(str(col))
                if currentLength > self.maxCellLength:
                    self.maxCellLength = currentLength
                cellObj = Cell(col)
                cells.append(cellObj)
                if currentLength > self.maxColWidth[idx]:
                    self.maxColWidth[idx] = currentLength
            rowObject = Row(cells)
            rows.append(rowObject)
        return rows

    def __drawRows(self):
        rowStr = ""
        for row in self.rows:
            rowStr = self.__drawRow(row)
        self.__drawRowDivider(len(rowStr))

    def __drawRow(self, row):
        return self.decorate(row)

    def __drawRowDivider(self, rowLength, showHeader=False):
        return self.getRowDividerString(rowLength, showHeader)

    def getRowDividerString(self, rowLength, showHeader=False):
        rowDivider = '{:-^{width}}'.format('', width=str(rowLength))
        if showHeader:
            return self.BOLD + rowDivider + self.CLEAR
        else:
            return rowDivider

    def decorate(self, row):
        rowStr = ""
        decoratedRowStr = ""
        if self.showBorders:
            decoratedRowStr = self.__decorateBorders(row)
        else:
            for idx, cell in enumerate(row.cells):
                rowStr += '{: ^{width}}'.format(cell.getString(), width=self.maxColWidth[idx] + 2)
            if self.showHeader and self.firstRow:
                decoratedRowStr = self.BOLD + rowStr + self.CLEAR + self.NEWLINE
                self.firstRow = False
            else:
                decoratedRowStr = rowStr

        self.table += decoratedRowStr

        return decoratedRowStr

    def __decorateBorders(self, row):
        rowStr = ""
        decoratedRowStr = ""
        for idx, cell in enumerate(row.cells):
            rowStr += "|" + '{: ^{width}}'.format(cell.getString(), width=self.maxColWidth[idx] + 2)
        rowStr += "|"
        if self.firstRow:
            decoratedRowStr = self.__drawRowDivider(len(rowStr), self.showHeader) + self.NEWLINE
            if self.showHeader:
                decoratedRowStr += self.BOLD + rowStr + self.CLEAR + self.NEWLINE
                decoratedRowStr += self.__drawRowDivider(len(rowStr),
                                                         self.showHeader) + self.NEWLINE
            else:
                decoratedRowStr += rowStr + self.NEWLINE
            decoratedRowStr += self.__drawRowDivider(len(rowStr)) + self.NEWLINE
            self.firstRow = False
        else:
            decoratedRowStr = rowStr + self.NEWLINE
            decoratedRowStr += self.__drawRowDivider(len(rowStr)) + self.NEWLINE
        return decoratedRowStr

    def getRowStr(self, rowNum):
        return self.rows[rowNum].getString(self.maxCellLength)

    def getTblStr(self):
        return self.table

    def getRows(self):
        return self.rows

    def print(self, printer=None):
        if printer:
            self.__drawRows()
            printer.info(self.table)
        else:
            print(self.table)
