import sys
sys.path.append("SlidingWindowVideoTDA")
from VideoTools import *
from TDA import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg


def getLaplacianEigs(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    w, v = slinalg.eigsh(L, k=NEigs, sigma = 0, which = 'LM')
    return (w, v)

def getLaplacianEigsDense(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    L = L.toarray()
    w, v = linalg.eigh(L)
    return (w, v)    

def getSSM(X):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    return D

def CSMToBinary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If Kappa = 0, take all neighbors
    If Kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise Kappa is the number of mutual neighbors to consider
    """
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*M))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()


def reorderVideo(XOrig, IDims, dim, Tau, dT, derivWin = 10):
    X = getPCAVideo(XOrig)
    print X.shape
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    D = getSSM(XS)
    Kappa = 0.1
    
    A = CSMToBinary(D, Kappa)
    A = scipy.sparse.coo_matrix(A)
    (w, v) = getLaplacianEigsDense(A, 10)
    theta = np.arctan2(v[:, 2], v[:, 1])
    
    plt.subplot(221)
    plt.imshow(D)
    plt.subplot(222)
    #plt.imshow(M, aspect = 'auto')
    plt.imshow(A.toarray(), interpolation = 'none')
    plt.subplot(223)
    #plt.imshow(v, interpolation = 'none', aspect = 'auto')
    plt.plot(v[:, 1], v[:, 2], '.')
    plt.subplot(224)
    plt.plot(theta)
    plt.show()
    
    idx = np.argsort(theta)
    XNew = XOrig[idx, :]
    saveVideo(XNew, IDims, "reordered.avi")
    

if __name__ == '__main__':
    filename = "jumpingjacks2men.ogg"
    (I, IDims) = loadVideo(filename)
    reorderVideo(I, IDims, 30, 1, 1)
