
import numpy as np

class BinaryHeap:

    def __init__(self, *args):

        nargin = len(locals())

        if nargin == 0:
            initialMaxSize = 1
            isMaxHeap = True
        elif nargin == 1:
            initialMaxSize = args[0]
            isMaxHeap = True
        elif nargin == 2:
            initialMaxSize = args[0]
            isMaxHeap = args[1]

        self.initialMaxSize = initialMaxSize
        self.heapArray = [0] + [KeyVal() for i in range(initialMaxSize)]
        self.numInHeap = 0
        self.isMaxHeap = isMaxHeap

    def __lt__(self, obj1, obj2):
        
        if isinstance(obj1, KeyVal) and isinstance(obj2, Keyval):
            val = obj1.key < obj2.key

        elif isinstance(obj1, KeyVal):
            val = obj1.key < obj2

        else:
            NotImplemented

        return val

    def __gt__(self, obj1, obj2):
        
        if isinstance(obj1, KeyVal) and isinstance(obj2, Keyval):
            val = obj1.key > obj2.key

        elif isinstance(obj1, KeyVal):
            val = obj1.key > obj2

        else:
            NotImplemented

        return val

    def heapSize(self):
        return self.numInHeap

    def isEmpty(self):
        return self.numInHeap==0

    def buildHeapFromKeysData(self, keyArray, dataArray):

        numKeys = keyArray.shape[0]
        self.numInHeap = numKeys
        self.heapArray[numKeys] = KeyVal

        for curKey in range(numKeys):
            self.heapArray[curKey] = KeyVal(keyArray[curKey], dataArray[curKey])

        idx = int(np.florr(self.numInHeap/2))

        while idx > 0:
            self.percolateDown(idx)
            idx = idx - 1

    def insert(self, key, value):

        self.numInHeap = self.numInHeap + 1
        hole = self.numInHeap

        if self.isMaxHeap:
            while hole>0 and key>self.heapArray[hole//2]:
                self.heapArray[hole] = self.heapArray[hole//2]
                hole = hole//2
        else:
            while hole>0 and key<self.heapArray[hole//2]:
                self.heapArray[hole] = self.heapArray[hole//2]
                hole = hole//2

        self.heapArray[hole] = KeyVal(key, value).copy()

    def getTop(self):

        if self.numInHeap > 0:
            val = self.heapArray[1].copy()
        else:
            val = 0

        return val

    def deleteTop(self):

        if self.numInHeap == 0:
            val = 0
            return val

        val = self.heapArray[1]

        self.heapArray[1] = self.heapArray[self.numInHeap]
        self.numInHeap = self.numInHeap - 1

        self.percolateDown(1)

        return val

    def percolateDown(self, hole):

        temp = self.heapArray[hole]

        if self.isMaxHeap:

            while 2*hole <= self.numInHeap: 

                child = 2*hole

                if child != self.numInHeap and self.heapArray[child+1] > self.heapArray[child]:
                    child = child + 1

                if self.heapArray[child] > temp:
                    self.heapArray[hole] = self.heapArray[child]

                else:
                    break

                hole = child

        else:

            while 2*hole <= self.numInHeap:
                child = 2*hole

                if child != self.numInHeap and self.heapArray[child+1] < self.heapArray[child]:
                    child = child + 1

                if self.heapArray[child] < temp:
                    self.heapArray[hole] = self.heapArray[child].copy()
                else:
                    break

                hole = child

        self.heapArray[hole] = temp

class KeyVal:

    def __init__(self, *args):

        nargin = len(args)

        if nargin == 0:
            key = 0
            val = 0

        elif nargin == 1: 
            key = args[0]
            val = 0

        if nargin == 2:
            key = args[0]
            val = args[1]

        self.key = key
        self.value = val

    def __lt__(self, obj2):

        if isinstance(obj2, KeyVal):
            val = self.key < obj2.key

        else:
            val = self.key < obj2

        return val

    def __gt__(self, obj2):

        if isinstance(obj2, KeyVal):
            val = self.key > obj2.key

        else: 
            val = self.key > obj2

        return val

    def copy(obj):

        objCopy = KeyVal(obj.key, obj.value)

        return objCopy

class MurtyData:

    def __init__(self, *args):
        
        nargin = len(args)
        
        if nargin == 2:

            A, numVarRow = args

            self.numVarRow = numVarRow

            numCol = A.shape[1]

            self.col4rowLCFull, self.row4colLCFull, self.gainFull, self.u, self.v = assign2DByCol(A)

            if self.gainFull != -1:

                self.activeRow = 0
                self.forbiddenActiveCol = np.zeros(numCol, 'bool')
                self.forbiddenActiveCol[self.col4rowLCFull[0]]=1

        else:

            A, numVarRow, activeRow, forbiddenActiveCols, col4rowInit, row4colInit, col2Scan, uInit, vInit = args

            self.numVarRow = numVarRow

            self.col4rowLCFull, self.row4colLCFull, self.gainFull, self.u, self.v = ShortestPathUpdate(A, activeRow, forbiddenActiveCols, col4rowInit, row4colInit, col2Scan, uInit.copy(), vInit.copy())

            if self.gainFull != -1:
                self.activeRow = activeRow
                self.forbiddenActiveCol = forbiddenActiveCols.copy()
                self.forbiddenActiveCol[self.col4rowLCFull[activeRow]] = 1

        self.A = A

    def split(self, splitList):

        numCol = self.A.shape[1]

        col2Scan = self.col4rowLCFull[self.activeRow:].copy()

        for curRow in range(self.activeRow, self.numVarRow):

            if curRow == self.activeRow:

                forbiddenColumns = self.forbiddenActiveCol.copy()

            else:

                forbiddenColumns = np.zeros(numCol, 'bool')
                forbiddenColumns[self.col4rowLCFull[curRow]] = 1

            row4colInit = self.row4colLCFull.copy()
            col4rowInit = self.col4rowLCFull.copy()
            row4colInit[col4rowInit[curRow]] = 10000
            col4rowInit[curRow] = 10000

            splitHyp = MurtyData(self.A, self.numVarRow, curRow, forbiddenColumns, col4rowInit, row4colInit, col2Scan, self.u, self.v)

            if splitHyp.gainFull != -1:
                splitList.insert(splitHyp,1)
            else:
                del splitHyp

            sel = col2Scan==self.col4rowLCFull[curRow]
            col2Scan = np.delete(col2Scan, sel)

    def __lt__(self, data2):

        if isinstance(data2, MurtyData):
            val = self.gainFull < data2.gainFull
        else:
            val = self.gainFull < data2

        return val

    def __gt__(self, data2):

        if isinstance(data2, MurtyData):
            val = self.gainFull > data2.gainFull

        else:
            val = self.gainFull > data2

        return val

    def disp(data):

        print('Data with col4rowLC:', data.col4rowLCFull, 'and gain:', data.gainFull)

def ShortestPathUpdate(C, activeRow, forbiddenActiveCols, col4row, row4col, col2Scan, u, v):

    numRow, numCol = C.shape
    numCol2Scan = len(col2Scan)

    ScannedRows = np.zeros(numRow, 'int')
    ScannedCol = np.zeros(numCol, 'int')

    sink = -1
    pred = np.zeros(numCol, 'int')
    delta = 0
    curRow = activeRow
    shortestPathCost = np.inf*np.ones(numCol)

    while sink == -1:

        ScannedRows[curRow] = 1

        minVal = np.inf

        for curColScan in range(numCol2Scan):
            curCol = col2Scan[curColScan]
            if curRow == activeRow and forbiddenActiveCols[curCol]==1:
                continue

            reducedCost = delta + C[curRow,curCol] - u[curRow] - v[curCol]

            if reducedCost<shortestPathCost[curCol]:
                pred[curCol] = curRow
                shortestPathCost[curCol] = reducedCost

            if shortestPathCost[curCol]<minVal:
                minVal = shortestPathCost[curCol]
                closestColScan = curColScan

        if np.isinf(minVal):
            gain=-1
            return col4row, row4col, gain, u, v

        closestCol = col2Scan[closestColScan]

        ScannedCol[closestCol] = 1
        numCol2Scan = numCol2Scan - 1
        col2Scan = np.delete(col2Scan, closestColScan)

        delta = shortestPathCost[closestCol]

        if row4col[closestCol]==10000:
            sink=closestCol
        else:
            curRow=row4col[closestCol]

    u[activeRow] = u[activeRow] + delta
    sel = ScannedRows != 0
    sel[activeRow] = 0
    u[sel] = u[sel] + delta - shortestPathCost[col4row[sel]]

    sel = ScannedCol != 0
    v[sel] = v[sel] - delta + shortestPathCost[sel]

    j = sink
    while True:
        i = pred[j]
        row4col[j] = i
        h = col4row[i]
        col4row[i] = j
        j = h

        if i==activeRow:
            break

    gain = 0
    for curRow in range(numRow):
        gain = gain + C[curRow, col4row[curRow]]

    return col4row, row4col, gain, u, v

def kBest2DAssign(*args):

    nargin = len(locals())

    if nargin<3:
        C, k = args
        maximize=False

    elif nargin==3:
        C, k, maximize = args

    numRow, numCol = C.shape

    if maximize:
        CDelta = np.max(C)
        C = -C + CDelta
    else:
        CDelta = np.min(C)
        C = C - CDelta

    didFlip = False
    if numRow>numCol:
        C = C.T
        temp = numRow
        numRow = numCol
        numCol = temp
        didFlip = True

    col4rowBest = np.zeros((numRow, k), 'int')
    row4colBest = np.zeros((numCol, k), 'int')
    gainBest = np.zeros(k)

    numPad = numCol - numRow
    C = np.concatenate([C, np.zeros((numPad, numCol))], axis=0)

    LCHyp = MurtyData(C, numRow)

    if LCHyp.gainFull == -1:
        col4rowBest = []
        row4colBest = []
        gainBest = -1

        return col4rowBest, row4colBest, gainBest

    col4rowBest[:,0] = LCHyp.col4rowLCFull[:numRow].copy()
    row4colBest[:,0] = LCHyp.row4colLCFull.copy()
    gainBest[0] = LCHyp.gainFull
    
    HypList = BinaryHeap(50*k, False)
    HypList.insert(LCHyp, 0)

    for curSweep in range(1, k):

        smallestSol = HypList.deleteTop()
        smallestSol.key.split(HypList)
        smallestSol = HypList.getTop()

        if HypList.heapSize() != 0:

            col4rowBest[:,curSweep] = smallestSol.key.col4rowLCFull[:numRow]
            row4colBest[:,curSweep] = smallestSol.key.row4colLCFull
            gainBest[curSweep] = smallestSol.key.gainFull
        else:
            col4rowBest=col4rowBest[:,:curSweep-1]
            row4colBest = row4colBest[:,:curSweep-1]
            gainBest = gainBest[:curSweep-1]

            break

    del HypList

    if numPad>0:
        sel = row4colBest>numRow-1
        row4colBest[sel] = -1

    if maximize:
        gainBest = -gainBest + CDelta*numRow
    else:
        gainBest = gainBest+CDelta*numRow

    if didFlip:
        temp = row4colBest.copy()
        row4colBest = col4rowBest.copy()
        col4rowBest = temp.copy()

    return col4rowBest, row4colBest, gainBest

def assign2DByCol(C, maximize=False):

    nargin = len(locals())

    if nargin < 2:
        maximize = False

    numRow, numCol = C.shape

    if maximize:
        CDelta = np.max(C)
        C = -C + CDelta 
    else:
        CDelta = np.min(C)
        C = C - CDelta

    didFlip = False
    if numRow > numCol:
        C = C.T
        temp = numRow
        numRow = numCol
        numCol = temp
        didFlip = True

    col4row = -1*np.ones(numRow, 'int')
    row4col = -1*np.ones(numCol, 'int')

    u = np.zeros(numRow)
    v = np.zeros(numCol)

    for curUnassRow in range(numRow):

        sink, pred, u, v = ShortestPath(curUnassRow, u, v, C, col4row, row4col)

        if sink == -1:

            col4row = []
            row4col = []
            gain = -1

            return (col4row, row4col, gain, u, v)

        j = sink
        while True:

            i = pred[j]
            row4col[j] = i
            h = col4row[i]
            col4row[i] = j
            j = h

            if i == curUnassRow:
                break
    
    gain = 0

    for curRow in range(numRow):
        gain = gain + C[curRow, col4row[curRow]]

    if maximize:
        gain = -gain + CDelta*numRow
    else:
        gain = gain + CDelta*numRow

    if didFlip:
        temp = row4col
        row4col = col4row
        col4row = temp

        temp = u
        u = v
        v = temp

    return col4row, row4col, gain, u, v

def ShortestPath(curUnassRow, u, v, C, col4row, row4col):

    numRow, numCol = C.shape
    pred = np.zeros(numCol, 'int')
    ScannedRows = np.zeros(numRow, 'int')
    ScannedCol = np.zeros(numCol, 'int')
    Col2Scan = np.arange(numCol)
    numCol2Scan = numCol

    sink = -1
    delta = 0
    curRow = curUnassRow
    shortestPathCost = np.inf*np.ones(numCol)

    while sink == -1:

        ScannedRows[curRow] = 1

        minVal = np.inf

        for curColScan in range(numCol2Scan):

            curCol = Col2Scan[curColScan]

            reducedCost = delta + C[curRow, curCol] - u[curRow] - v[curCol]

            if reducedCost < shortestPathCost[curCol]:
                pred[curCol] = curRow
                shortestPathCost[curCol] = reducedCost

            if shortestPathCost[curCol] < minVal:
                minVal = shortestPathCost[curCol]
                closestColScan = curColScan

        if np.isinf(minVal):

            sink = -1

            return sink, pred, u, v

        closestCol = Col2Scan[closestColScan]

        ScannedCol[closestCol] = 1
        numCol2Scan = numCol2Scan - 1

        Col2Scan = np.delete(Col2Scan, closestColScan)

        delta = shortestPathCost[closestCol]

        if row4col[closestCol] == -1:
            sink = closestCol
        else:
            curRow = row4col[closestCol]

    u[curUnassRow] = u[curUnassRow] + delta
    sel = ScannedRows != 0
    sel[curUnassRow] = 0
    u[sel] = u[sel] + delta - shortestPathCost[col4row[sel]]

    sel = ScannedCol != 0
    v[sel] = v[sel] - delta + shortestPathCost[sel]

    return sink, pred, u, v

def Murty_MSC(C, K):

    col4row, row4col, gain = kBest2DAssign(C, K)

    N = C.shape[0]
    rows = np.zeros((N, K), 'int')
    cols = np.zeros((N, K), 'int')
    for i in range(K):
        for j in range(N):
            rows[j,i] = j
            cols[j,i] = col4row[j,i]

    return gain, rows.T, cols.T
