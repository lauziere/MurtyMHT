
import numpy as np
from crouse import *
from scipy.spatial import distance

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

        self.A = A.copy()

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

def kBest2DAssign_DA_MHHT(Cs, k):

    Z, numRow, numCol = Cs.shape

    didFlip = False
    if numRow>numCol:
        Cs = np.transpose(Cs, axes=(0,2,1))
        temp = numRow
        numRow = numCol
        numCol = temp
        didFlip = True

    col4rowBest = np.zeros((numRow, k), 'int')
    row4colBest = np.zeros((numCol, k), 'int')
    rowPermsBest = np.zeros(k, 'int')
    gainBest = np.zeros(k,'float32')

    numPad = numCol - numRow
    Cs = np.concatenate([Cs, np.zeros((Z, numPad, numCol))], axis=1)
    
    # Now we solve each one and insert it
    HypList = BinaryHeap(50*k, False)
    for z in range(Z):

        LCHyp = MurtyData_DA_MHHT(Cs[z], z, numRow)
        HypList.insert(LCHyp, 0)

    for curSweep in range(k):

        smallestSol = HypList.getTop()

        if HypList.heapSize() != 0:
            col4rowBest[:,curSweep] = smallestSol.key.col4rowLCFull[:numRow]
            row4colBest[:,curSweep] = smallestSol.key.row4colLCFull
            gainBest[curSweep] = smallestSol.key.gainFull
            rowPermsBest[curSweep] = smallestSol.key.z

        else:
            col4rowBest=col4rowBest[:,:curSweep]
            row4colBest = row4colBest[:,:curSweep]
            gainBest = gainBest[:curSweep]
            rowPermsBest = rowPermsBest[:curSweep]

            break

        smallestSol = HypList.deleteTop()
        smallestSol.key.split(HypList)

    del HypList

    if numPad>0:
        sel = row4colBest>numRow-1
        row4colBest[sel] = -1

    if didFlip:
        temp = row4colBest.copy()
        row4colBest = col4rowBest.copy()
        col4rowBest = temp.copy()

    return col4rowBest, row4colBest, gainBest, rowPermsBest

def main():

    K = 5
    num_row = 10
    num_col = 10

    C1 = np.zeros((K,num_row,num_col))
    for i in range(K):
        a = 3*np.random.randn(num_row,2)
        b = 4*np.random.randn(num_col,2)+2

        C1[i] = distance.cdist(a,b)

    gain12, rows12, cols12, pCols12 = Murty_MSC_DA_MHHT(C1, K)

    print('K of K')
    print('pCols12')
    print(pCols12)
    print('\n')

    print('cols12')
    print(cols12)
    print('\n')

    print('gain')
    print(gain12)
    print('\n')

    sols = np.empty(K*K, 'float32')
    pcols = np.zeros(K*K,'int')
    cols = np.zeros((K*K,num_row),'int')

    # for i in range(5):
    #     print(C1[pCols12[i],np.arange(10),cols12[i]].sum(), gain12[i])

    # Do each at a time. 
    for i in range(K):
        c_tmp = C1[i].copy()
        gain02, rows02, cols02 = Murty_MSC(c_tmp, K)
        
        sols[K*i:K*i+K]=gain02
        cols[K*i:K*i+K]=cols02
        pcols[K*i:K*i+K]=[i for j in range(K)]

    col_sort = np.argsort(sols)
    pcols = pcols[col_sort]
    sols = sols[col_sort]
    cols = cols[col_sort]

    k_sols = sols[:K]
    k_cols = cols[:K]
    k_pcols = pcols[:K]

    print('K from each K')
    print('pCols')
    print(k_pcols)

    print('kcols')
    print(k_cols)

    print('kgain')
    print(k_sols)

    print(np.array_equiv(cols12,cols[:K]))
    print(k_sols-gain12)

if __name__ == '__main__':
    main()