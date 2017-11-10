from SlidingWindowVideoTDA.VideoTools import *
from ripser import ripser
from CSMSSMTools import *
from Laplacian import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def getReorderedConsensus1D(X, N, theta, doPlot = False):
    M = X.shape[0]
    d = X.shape[1]
    tu = np.unwrap(theta)
    if tu[-1] - tu[0] < 0:
        tu = -tu
    tu = tu - np.min(tu)
    NPeriods = int(np.round(np.max(tu)/(2*np.pi)*N/M))
    T = N/NPeriods #Period
    print("NPeriods = %i"%NPeriods)
    print("N/NPeriods = %g"%(float(N)/NPeriods))
    tu = N*np.mod(tu, 2*np.pi)/(2*np.pi)
    idx = np.argsort(tu)
    X2 = X[idx, :]
    t1 = tu[idx]
    Z = np.ones((M, N))
    for i in range(M):
        ts = t1[i] + NPeriods*np.arange(d)
        imin = int(np.ceil(np.min(ts)))
        imax = int(np.floor(np.max(ts)))
        t2 = np.arange(imin, imax+1)
        x = interp.spline(ts, X2[i, :], t2)
        #Place into array, considering that there may be
        #collisions after modding
        idx = np.mod(t2, N)
        x = sparse.coo_matrix((x, (np.zeros(len(t2)), idx)), shape=(1, N))
        x = (x.toarray()).flatten()
        counts = sparse.coo_matrix((np.ones(len(t2)), (np.zeros(len(t2)), idx)), shape=(1, N))
        counts = (counts.toarray()).flatten()
        counts[counts == 0] = np.nan
        x = x/counts
        Z[i, :] = x
    if doPlot:
        plt.imshow(Z, aspect = 'auto', interpolation = 'none', cmap = 'afmhot')
        plt.xlabel("Time")
        plt.ylabel("Spline Interpolated Windows")
        plt.show()
    z = np.nanmedian(Z, 0)
    return z

def reorderVideo(XOrig, dim, Tau, dT, derivWin = 10, Weighted = False):
    X = getPCAVideo(XOrig)
    print X.shape
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    Y = XS - np.mean(XS, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
    D = getSSM(Y)
    
    Is = ripser.doRipsFiltrationDM(D, 1, coeff=41)
    I = Is[1]
    thresh = np.argmax(I[:, 1] - I[:, 0])
    thresh = np.mean(I[thresh, :])
    if Weighted:
        res = getLapCircularCoordinatesSigma(D, thresh)
    else:
        res = getLapCircularCoordinatesThresh(D, thresh)
    [w, v, theta, A] = [res['w'], res['v'], res['theta'], res['A']]
    
    plt.subplot(231)
    plt.imshow(D, cmap='afmhot', interpolation='none')
    plt.subplot(232)
    plt.imshow(A, interpolation='none', cmap='gray')
    plt.subplot(233)
    ripser.plotDGM(I)
    plt.title("Thresh = %g"%thresh)
    plt.subplot(234)
    plt.scatter(v[:, 1], v[:, 2], 20, np.arange(v.shape[0]), cmap = 'spectral')
    plt.subplot(235)
    plt.plot(theta)
    plt.show()
    
    idx = np.argsort(theta)
    XNew = XOrig[idx, :]
    return XNew
    

if __name__ == '__main__':
    filename = "jumpingjacks2men.ogg"
    (I, IDims) = loadImageIOVideo(filename)
    XNew = reorderVideo(I, 30, 1, 1)
    saveVideo(XNew, IDims, "reordered.avi")