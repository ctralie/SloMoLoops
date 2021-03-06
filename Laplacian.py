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

def getSlowestEigenvectorIdx(v, tol = 0.8):
    """
    Find the adjacent pair of eigenvectors with similar eigenvalues that have
    the smallest number of zero crossings
    :param v: NEigs x k array of eigenvectors arrange along columns
    :param tol: Reject pairs with zero crossings ratio less than tol
    """
    #Compute circular zero crossings and put the low frequencies eigenvectors first
    s = 1.0*(v > 0)
    zcs = np.sum(np.abs(np.sign(s[1::, :]-s[0:-1, :])), 0)
    smallest = np.inf
    i1 = 1
    for i in range(1, len(zcs)-1):
        if tol > -1:
            ratio = min(float(zcs[i])/zcs[i+1], float(zcs[i+1])/zcs[i])
            print("%i and %i, %g"%(i, i+1, ratio))
            if ratio < tol:
                continue
        bothSum = zcs[i] + zcs[i+1]
        if bothSum < smallest:
            smallest = bothSum
            i1 = i
    return (i1, zcs)

def getLapCircularCoordinatesSigma(D, sigma, NEigs = 10, doPlot = False):
    """
    Get circular coordinates using a weighted laplacian 
    :param pD: Distance matrix
    :param sigma: Standard deviation for exponential kernel
    :param NEigs: Maximum number of eigenvectors to compute
    :return {'w':eigenvalues, 'v':eigenvectors, 'theta':Circular coordinates,\
            'thetau':Unwrapped circular coordinates, 'A':Adjacency matrix}
    """
    #A = 0.5*(1+np.tanh((sigma-D)/sigma**2))
    A = np.exp(-D*D/(2*sigma**2))
    #A = np.exp(-D*D/sigma)
    np.fill_diagonal(A, 0)
    NEigs = min(NEigs, A.shape[0])
    (w, v, L) = getLaplacianEigsDense(A, NEigs)
    (i1, zcs) = getSlowestEigenvectorIdx(v)
    if doPlot:
        plt.subplot(411)
        plt.imshow(v, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvectors")
        plt.subplot(412)
        plt.stem(w)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvalues")
        plt.subplot(413)
        plt.plot(zcs)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Zero Crossings, Smallest = (%i, %i)"%(i1, i1+1))
        plt.subplot(414)
        plt.plot(v[:, [i1, i1+1]])
        plt.legend(["%i"%i1, "%i"%(i1+1)])
    (theta, thetau) = getLapThetas(v, i1, i1+1)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'idxs':[i1, i1+1]}

def getLapCircularCoordinatesThresh(pD, thresh, NEigs = 10, doPlot = False):
    """
    Get circular coordinates using an unweighted laplacian based
    on binarizing a distance matrix below a certain threshold
    :param pD: Distance matrix
    :param thresh: Threshold below which to include edges
    :param NEigs: Maximum number of eigenvectors to compute
    :return {'w':eigenvalues, 'v':eigenvectors, 'theta':Circular coordinates,\
            'thetau':Unwrapped circular coordinates, 'A':Adjacency matrix}
    """
    D = np.array(pD)
    np.fill_diagonal(D, np.inf)
    A = np.zeros(D.shape)
    A[D <= thresh] = 1
    (w, v, L) = getLaplacianEigsDense(A, NEigs)
    (i1, zcs) = getSlowestEigenvectorIdx(v)
    if doPlot:
        plt.subplot(411)
        plt.imshow(v, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvectors")
        plt.subplot(412)
        plt.stem(w)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvalues")
        plt.subplot(413)
        plt.plot(zcs)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Zero Crossings, Smallest = (%i, %i)"%(i1, i1+1))
        plt.subplot(414)
        plt.plot(v[:, [i1, i1+1]])
        plt.legend(["%i"%i1, "%i"%(i1+1)])
    (theta, thetau) = getLapThetas(v, i1, i1+1)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'idxs':[i1, i1+1]}

def getLapCircularCoordinatesKNN(D, Kappa, NEigs = 10, percentile = True, doPlot = False):
    """
    Get circular coordinates using a weighted laplacian 
    :param pD: Distance matrix
    :param Kappa: Percentage of mutual nearest neighbors
    :param NEigs: Maximum number of eigenvectors to compute
    :param percentile: Whether to use percentile or tuned mutual nearest neighbors
    :return {'w':eigenvalues, 'v':eigenvectors, 'theta':Circular coordinates,\
            'thetau':Unwrapped circular coordinates, 'A':Adjacency matrix}
    """
    if percentile:
        ds = np.sort(D.flatten())
        sigma = ds[int(Kappa*len(ds))]
        A = D <= sigma
    else:
        A = CSMToBinaryMutual(D, Kappa)
    np.fill_diagonal(A, 0)
    NEigs = min(NEigs, A.shape[0])
    (w, v, L) = getLaplacianEigsDense(A, NEigs)
    (i1, zcs) = getSlowestEigenvectorIdx(v)
    if doPlot:
        plt.subplot(411)
        plt.imshow(v, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvectors")
        plt.subplot(412)
        plt.stem(w)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvalues")
        plt.subplot(413)
        plt.plot(zcs)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Zero Crossings, Smallest = (%i, %i)"%(i1, i1+1))
        plt.subplot(414)
        plt.plot(v[:, [i1, i1+1]])
        plt.legend(["%i"%i1, "%i"%(i1+1)])
    (theta, thetau) = getLapThetas(v, i1, i1+1)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'idxs':[i1, i1+1]}

def getLapCircularCoordinatesKNNWeighted(D, Kappa, NEigs = 10, percentile = True, doPlot = False):
    """
    Get circular coordinates using a weighted laplacian 
    :param pD: Distance matrix
    :param Kappa: Percentage of mutual nearest neighbors
    :param NEigs: Maximum number of eigenvectors to compute
    :param percentile: Whether to use percentile or tuned mutual nearest neighbors
    :return {'w':eigenvalues, 'v':eigenvectors, 'theta':Circular coordinates,\
            'thetau':Unwrapped circular coordinates, 'A':Adjacency matrix}
    """
    if percentile:
        ds = np.sort(D.flatten())
        sigma = ds[int(Kappa*len(ds))]        
    else:
        K = int(Kappa*D.shape[0])
        dNN = np.mean(np.sort(D, 1)[:, 0:K], 1)
        sigma = 0.5*(dNN[:, None] + dNN[None, :])
    A = np.exp(-D*D/(sigma))
    np.fill_diagonal(A, 0)
    NEigs = min(NEigs, A.shape[0])
    (w, v, L) = getLaplacianEigsDense(A, NEigs)
    (i1, zcs) = getSlowestEigenvectorIdx(v)
    if doPlot:
        plt.subplot(411)
        plt.imshow(v, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvectors")
        plt.subplot(412)
        plt.stem(w)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Eigenvalues")
        plt.subplot(413)
        plt.plot(zcs)
        plt.xlim([0, v.shape[1]-1])
        plt.title("Zero Crossings, Smallest = (%i, %i)"%(i1, i1+1))
        plt.subplot(414)
        plt.plot(v[:, [i1, i1+1]])
        plt.legend(["%i"%i1, "%i"%(i1+1)])
    (theta, thetau) = getLapThetas(v, i1, i1+1)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'idxs':[i1, i1+1]}

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

def sinusoidalScores(X, doPlot = False):
    """
    Return a score between [0, 1] that indicates how sinusoidal 
    signals in the columns of X are.  0 for not sinusoidal and 1 for sinusoidal
    """
    A = getLineLaplacian(X.shape[0])
    D = A.dot(X)
    X1 = np.array(X)
    #Take care of boundary problems with the laplacian
    X1[:, 0] = X1[:, 1]
    X1[:, -1] = X1[:, -2]
    D[:, 0] = D[:, 1]
    D[:, -1] = D[:, -2]
    X1 = X1/np.sqrt(np.sum(X1**2, 0))[None, :]
    X2 = D/np.sqrt(np.sum(D**2, 0))[None, :]
    #Metric on projective plane
    scores = 1 - np.arccos(np.abs(np.sum(X1*X2, 0)))/(np.pi/2)
    return scores

def frequencyScores(X):
    """
    Return a score between [0, 1] that indicates the frequency
    of the sinusoids contained in the columns of X
    """
    A = getLineLaplacian(X.shape[0])
    D = A.dot(X)
    X1 = np.array(X)
    #Take care of boundary problems with the laplacian
    X1[:, [0, -1]] = 0
    D[:, [0, -1]] = 0
    return np.sum(X1*D, 0)

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
