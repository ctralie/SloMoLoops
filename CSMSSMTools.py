import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def getSSM(X):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    np.fill_diagonal(D, 0)
    D = 0.5*(D + D.T)
    return D

def getCSM(X, Y):
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

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

def getW(D, Kappa, Mu = 0.5):
    """
    Return affinity matrix
    :param D: Self-similarity matrix
    :param Kappa: Fraction of nearest neighbors
    """
    K = int(Kappa*D.shape[0])
    #W(i, j) = exp(-Dij^2/(mu*epsij))
    DSym = 0.5*(D + D.T)
    np.fill_diagonal(DSym, 0)

    Neighbs = np.partition(DSym, K+1, 1)[:, 0:K+1]
    MeanDist = np.mean(Neighbs, 1)*float(K+1)/float(K) #Need this scaling
    #to exclude diagonal element in mean
    #Equation 1 in SNF paper [2] for estimating local neighborhood radii
    #by looking at k nearest neighbors, not including point itself
    Eps = MeanDist[:, None] + MeanDist[None, :] + DSym
    Eps = Eps/3
    W = np.exp(-DSym**2/(2*(Mu*Eps)**2))
    return W

def CSMToBinaryMutual(D, Kappa):
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa).T
    return B1*B2


def getNNAdj(X, Kappa):
    D = getCSM(X, X)
    B = CSMToBinaryMutual(D, Kappa)
    B = sparse.csr_matrix(B)
    return B    
