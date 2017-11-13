import sys
import scipy.sparse as sparse
import scipy.stats
import scipy.sparse.linalg as slinalg
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
from CSMSSMTools import *

def getLaplacianEigs(A, NEigs):
    DEG = sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    w, v = slinalg.eigsh(L, k=NEigs, sigma = 0, which = 'LM')
    return (w, v, L)

def getLaplacianEigsDense(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG.toarray() - A
    w, v = linalg.eigh(L)
    return (w[0:NEigs], v[:, 0:NEigs], L)

def getLapThetas(pv, eig1, eig2):
    """
    Use arctangent of mean-centered eigenvectors as estimates of
    circular coordinates
    """
    v = np.array(pv[:, [eig1, eig2]])
    v = v - np.mean(v, 0, keepdims=True)
    theta = np.arctan2(v[:, 1], v[:, 0])
    thetau = np.unwrap(theta)
    #Without loss of generality, switch theta to overall increasing
    if thetau[-1] - thetau[0] < 0:
        thetau = -thetau
    return (theta, thetau - thetau[0])

def getSlopes(thetas, sWin = 10):
    """
    Estimate smoothed versions of slopes in radians per sample
    2*sWin is the size of the rectangular window used to smooth
    """
    N = len(thetas)
    slopes = np.zeros(N)
    deriv = np.zeros(sWin*2)
    deriv[0:sWin] = np.ones(sWin)
    deriv[sWin:] = -np.ones(sWin)
    slopes[sWin-1:-sWin] = np.convolve(thetas, deriv, 'valid')/float(sWin**2)
    slopes[0:sWin-1] = slopes[sWin-1]
    slopes[-(sWin+1):] = slopes[-(sWin+1)]
    return slopes

def getLapCircularCoordinatesSigma(D, sigma):
    """
    Get circular coordinates using a weighted laplacian 
    :param pD: Distance matrix
    :param sigma: Standard deviation for exponential kernel
    :return {'w':eigenvalues, 'v':eigenvectors, 'theta':Circular coordinates,\
            'thetau':Unwrapped circular coordinates, 'A':Adjacency matrix}
    """
    A = np.exp(-D*D/(2*sigma**2))
    (w, v, L) = getLaplacianEigsDense(A, 3)
    (theta, thetau) = getLapThetas(v, 1, 2)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A}

def getLapCircularCoordinatesThresh(pD, thresh, doPlot = False):
    """
    Get circular coordinates using an unweighted laplacian based
    on binarizing a distance matrix below a certain threshold
    :param pD: Distance matrix
    :param thresh: Threshold below which to include edges
    :return {'w':eigenvalues, 'v':eigenvectors, 'theta':Circular coordinates,\
            'thetau':Unwrapped circular coordinates, 'A':Adjacency matrix}
    """
    D = np.array(pD)
    np.fill_diagonal(D, np.inf)
    A = np.zeros(D.shape)
    A[D <= thresh] = 1
    (w, v, L) = getLaplacianEigsDense(A, 3)
    (theta, thetau) = getLapThetas(v, 1, 2)
    if doPlot:
        plt.subplot(131)
        plt.imshow(D, cmap = 'afmhot', interpolation = 'none')
        plt.subplot(132)
        plt.imshow(A, cmap = 'gray', interpolation = 'none')
        plt.subplot(133)
        plt.imshow(v, cmap = 'afmhot', aspect = 'auto', interpolation = 'none')
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A}

def getLineLaplacian(NPoints):
    I = np.arange(NPoints-1).tolist()
    J = np.arange(NPoints-1)
    J = J + 1
    J = J.tolist()
    IF = np.array(I + J)
    JF = np.array(J + I)
    A = scipy.sparse.coo_matrix((np.ones(len(IF)), (IF, JF)), shape=(NPoints, NPoints)).tocsr()
    DEG = sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    return L

def sinusoidalScore(x, medianFilter = True, doPlot = False):
    """
    Return a score between [0, 1] that indicates how sinusoidal 
    a signal is.  0 for not sinusoidal and 1 for sinusoidal
    """
    A = getLineLaplacian(len(x))
    d = A.dot(x)
    x1 = np.array(x)
    if medianFilter:
        x1 = median_filter(x1, 5)
    #Take care of boundary problems with the laplacian
    x1[0] = x1[1]
    x1[-1] = x1[-2]
    d[0] = d[1]
    d[-1] = d[-2]
    x1 = x/np.sqrt(np.sum(x**2))
    x2 = d/np.sqrt(np.sum(d**2))
    #Metric on projective plane
    score = 1 - np.arccos(np.abs(np.sum(x1*x2)))/(np.pi/2)
    if doPlot:
        plt.plot(x1, 'r')
        plt.plot(x2, 'b')
        plt.title("Score = %g"%score)
    return score

if __name__ == '__main__':
    N = 200
    NPeriods = 1
    t = 2*np.pi*NPeriods*np.arange(N)/N
    s = np.sin(t)
    c = np.cos(t)
    c += np.cos(3*t)
    #c += t
    
    sinusoidalScore(c, True, True)
    plt.show()
