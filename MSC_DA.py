
import numpy as np
from crouse import *

class MurtyData_DA_MHHT:

    def __init__(self, *args):
        
        nargin = len(args)
        
        if nargin == 3:

            A, z, numVarRow = args

            self.numVarRow = numVarRow
            self.z = z

            numCol = A.shape[1]

            self.col4rowLCFull, self.row4colLCFull, self.gainFull, self.u, self.v = assign2DByCol(A)

            if self.gainFull != -1:

                self.activeRow = 0
                self.forbiddenActiveCol = np.zeros(numCol, 'bool')
                self.forbiddenActiveCol[self.col4rowLCFull[0]]=1

        else:

            A, numVarRow, z, activeRow, forbiddenActiveCols, col4rowInit, row4colInit, col2Scan, uInit, vInit = args

            self.numVarRow = numVarRow
            self.z = z

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

            splitHyp = MurtyData_DA_MHHT(self.A, self.numVarRow, self.z, curRow, forbiddenColumns, col4rowInit, row4colInit, col2Scan, self.u, self.v)

            if splitHyp.gainFull != -1:
                splitList.insert(splitHyp,1)
            else:
                del splitHyp

            sel = col2Scan==self.col4rowLCFull[curRow]
            col2Scan = np.delete(col2Scan, sel)

    def __lt__(self, data2):

        if isinstance(data2, MurtyData_DA_MHHT):
            val = self.gainFull < data2.gainFull
        else:
            val = self.gainFull < data2

        return val

    def __gt__(self, data2):

        if isinstance(data2, MurtyData_DA_MHHT):
            val = self.gainFull > data2.gainFull

        else:
            val = self.gainFull > data2

        return val

    def disp(data):

        print('Data with col4rowLC:', data.col4rowLCFull, 'and gain:', data.gainFull)

def Murty_MSC_DA_MHHT(Cs, K):

    col4row, row4col, gain, pCols = kBest2DAssign_DA_MHHT(Cs, K)

    N = Cs.shape[1]
    rows = np.zeros((N, K), 'int')
    cols = np.zeros((N, K), 'int')
    for i in range(K):
        for j in range(N):
            rows[j,i] = j
            cols[j,i] = col4row[j,i]

    return gain, rows.T, cols.T, pCols.T

def kBest2DAssign_DA_MHHT(*args):

    nargin = len(locals())

    if nargin<3:
        Cs, k = args
        maximize=False

    elif nargin==3:
        Cs, k, maximize = args

    Z, numRow, numCol = Cs.shape

    if maximize:
        CDelta = np.max(Cs, axis=(1,2)) 
        Cs = np.array([-Cs[z] + CDelta[z] for z in range(len(Cs))])
    else:
        # CDelta = np.min(Cs, axis=(1,2))
        CDelta = np.zeros(len(Cs))
        # Cs = Cs - CDelta
        Cs = np.array([Cs[z] - CDelta[z] for z in range(len(Cs))])

    didFlip = False
    if numRow>numCol:
        Cs = np.transpose(Cs, axes=(0,2,1))
        # C = C.T
        temp = numRow
        numRow = numCol
        numCol = temp
        didFlip = True

    col4rowBest = np.zeros((numRow, k), 'int')
    row4colBest = np.zeros((numCol, k), 'int')
    rowPermsBest = np.zeros(k, 'int')

    gainBest = np.zeros(k)

    numPad = numCol - numRow
    Cs = np.concatenate([Cs, np.zeros((len(Cs), numPad, numCol))], axis=1)
    # C = np.concatenate([C, np.zeros((numPad, numCol))], axis=0)

    # LCHyp = MurtyData(C, numRow)

    # if LCHyp.gainFull == -1:
    #   col4rowBest = []
    #   row4colBest = []
    #   gainBest = -1

    #   return col4rowBest, row4colBest, gainBest

    # col4rowBest[:,0] = LCHyp.col4rowLCFull[:numRow].copy()
    # row4colBest[:,0] = LCHyp.row4colLCFull.copy()
    # gainBest[0] = LCHyp.gainFull
    
    # Now we solve each one and insert it
    HypList = BinaryHeap(50*k, False)
    for z in range(Z):

        LCHyp = MurtyData_DA_MHHT(Cs[z], z, numRow)
        HypList.insert(LCHyp, 0)

        # if LCHyp.gainFull == -1:
        #   col4rowBest = []
        #   row4colBest = []
        #   gainBest = -1

        #   return col4rowBest, row4colBest, gainBest

    # col4rowBest[:,0] = LCHyp.col4rowLCFull[:numRow].copy()
    # row4colBest[:,0] = LCHyp.row4colLCFull.copy()
    # gainBest[0] = LCHyp.gainFull
    # HypList = BinaryHeap(50*k, False)
    # HypList.insert(LCHyp, 0)

    for curSweep in range(k):

        # print('curSweep', curSweep)

        # print('Queue:')
        # for i in range(1,50*k):
        #     try:
        #         print(i, HypList.heapArray[i].key.gainFull)
        #     except:
        #         pass

        # print('Deleting Top')
        smallestSol = HypList.deleteTop()
        # print('smallestSol.key.gainFull', smallestSol.key.gainFull)

        # print('Queue:')
        # for i in range(1,50*k):
        #     try:
        #         print(i, HypList.heapArray[i].key.gainFull)
        #     except:
        #         pass

        # print('Splitting.')
        smallestSol.key.split(HypList)

        # print('Queue post split:')
        # for i in range(1,50*k):
        #     try:
        #         print(i, HypList.heapArray[i].key.gainFull)
        #     except:
        #         pass

        # print('Get top:')
        # smallestSol = HypList.getTop()

        # for i in range(1,50*k):
        #     try:
        #         print(i, HypList.heapArray[i].key.gainFull)
        #     except:
        #         pass

        # print(smallestSol.key.gainFull)
        # print('\n')
        # pdb.set_trace()

        # for i in range(1,50*k):
        #     try:
        #         print(i, HypList.heapArray[i].key.gainFull)
        #     except:
        #         pass

        # print([HypList.heapArray[i].key.gainFull for i in range(1,50*k)])
        
        if HypList.heapSize() != 0:
            col4rowBest[:,curSweep] = smallestSol.key.col4rowLCFull[:numRow]
            row4colBest[:,curSweep] = smallestSol.key.row4colLCFull
            gainBest[curSweep] = smallestSol.key.gainFull
            rowPermsBest[curSweep] = smallestSol.key.z

        else:
            col4rowBest=col4rowBest[:,:curSweep]
            row4colBest = row4colBest[:,:curSweep]
            gainBest = gainBest[:curSweep]

            break

    del HypList

    if numPad>0:
        sel = row4colBest>numRow-1
        row4colBest[sel] = -1

    if maximize:
        gainBest = -gainBest + numRow*CDelta[rowPermsBest]
    else:
        # print('gainBest', gainBest)
        # print('CDelta', CDelta)
        # print('CDelta aug', CDelta*numRow*np.ones(len(CDelta))[rowPermsBest])
        # print('CDelta aug', numRow*CDelta[rowPermsBest])

        # gainBest = gainBest + CDelta*numRow*np.ones(len(CDelta))[rowPermsBest]
        gainBest = gainBest + numRow*CDelta[rowPermsBest]

    if didFlip:
        temp = row4colBest.copy()
        row4colBest = col4rowBest.copy()
        col4rowBest = temp.copy()

    return col4rowBest, row4colBest, gainBest, rowPermsBest

def main():

    K = 5
    C1 = np.random.randn(5, 10,10) + 10

    gain12, rows12, cols12, pCols12 = Murty_MSC_DA_MHHT(C1, K)

    print('pCols12')
    print(pCols12)
    print('\n')

    print('cols12')
    print(cols12)
    print('\n')

    print('gain')
    print(gain12)
    print('\n')

    sols = np.empty(25)

    # Do each at a time. 
    for i in range(5):
        c_tmp = C1[i]
        gain02, rows02, cols02 = Murty_MSC(c_tmp, K)

        print(gain02)
        sols[5*i:5*i+5]=gain02

    print('\n')
    print(np.sort(sols))

if __name__ == '__main__':
    main()