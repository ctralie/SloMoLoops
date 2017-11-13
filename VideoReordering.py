from SlidingWindowVideoTDA.VideoTools import *
from ripser import ripser
from CSMSSMTools import *
from Laplacian import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def getReorderedConsensus1D(X, N, theta, doPlot = False):
    """
    Given an array of sliding windows and circular coordinates,
    reorder the sliding windows to go through one period
    over the length of the signal, and use the different lags in
    the window to vote on the final samples in the reordered signal.
    Use spline interpolation to fill in intermediate values in 
    each window
    :param X: An Mxd array of windows
    :param N: The original signal length
    :param theta: The circular coordinates
    :param doPlot: Whether to make a plot showing all of the windows
    :return z: An N-length 1D consensus signal going through one period
    """
    #Step 1: Figure out the number of periods that the signal
    #goes through
    M = X.shape[0]
    d = X.shape[1]
    tu = np.unwrap(theta)
    if tu[-1] - tu[0] < 0:
        tu = -tu
    tu = tu - np.min(tu)
    NPeriods = (float(N)/M)*np.max(tu)/(2*np.pi)
    print("NPeriods = %g"%NPeriods)
    print("N/NPeriods = %g"%(float(N)/NPeriods))
    tu = N*np.mod(tu, 2*np.pi)/(2*np.pi)
    idx = np.argsort(tu)
    X2 = X[idx, :]
    t1 = tu[idx]
    
    #Step 2: Go through each window and use it to vote on final samples
    Z = np.ones((M, N))
    for i in range(M):
        #Figure out the range of the window in the final signal
        ts = t1[i] + NPeriods*np.arange(d)
        imin = int(np.ceil(np.min(ts)))
        imax = int(np.floor(np.max(ts)))
        t2 = np.arange(imin, imax+1)
        #Interpolate the window to fill in missing samples
        x = interp.spline(ts, X2[i, :], t2)
        #Place into array, considering that there may be
        #collisions after modding.  In the case of collisions, take
        #the mean of the values that overlap
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
    return np.nanmedian(Z, 0)

def getReorderedConsensusVideo(XOrig, d, theta, doPlot = False):
    #TODO: Finish this
    """
    Given an array of sliding window videos and circular coordinates,
    reorder the sliding windows to go through one period
    over the length of the signal, and use the different lags in
    the window to vote on the final pixels in the reordered video.
    Use linear interpolation to fill in intermediate values in 
    each window.
    :param X: An Mxd array of windows
    :param N: The original signal length
    :param theta: The circular coordinates
    :param doPlot: Whether to make a plot showing all of the windows
    """
    #TODO: For speed, do linear interpolation on PCA coordinates, but
    #perform median on the final pixels

    #Step 1: Figure out the number of periods that the signal
    #goes through
    M = len(theta)
    N = XOrig.shape[0]
    tu = np.unwrap(theta)
    if tu[-1] - tu[0] < 0:
        tu = -tu
    tu = tu - np.min(tu)
    NPeriods = (float(N)/M)*np.max(tu)/(2*np.pi)
    tu = N*np.mod(tu, 2*np.pi)/(2*np.pi)
    idx = np.argsort(tu)
    t1 = tu[idx]
    
    #Step 2: Go through each window and use it to vote on final samples
    XRet = 0*XOrig 
    counts = np.zeros(XRet.shape[0])
    for i in range(M):
        #Figure out the range of the window in the final signal
        ts = t1[i] + NPeriods*np.arange(d)
        imin = int(np.ceil(np.min(ts)))
        imax = int(np.floor(np.max(ts)))
        t2 = np.arange(imin, imax+1)
        #Interpolate the window to fill in missing samples
        
        #Place into array, considering that there may be
        #collisions after modding.  In the case of collisions, take
        #the mean of the values that overlap
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
    return np.nanmedian(Z, 0)

def reorderVideo(XOrig, dim, derivWin = 10, Weighted = False, doSimple = False, doPlot = True):
    """
    Reorder the video based on circular coordinates of a sliding
    window embedding
    :param XOrig: An NFrames x (NPixels*NChannels) video array
    :param dim: Dimension to use in the sliding window
    :param derivWin: Window of derivative to use
    :param Weighted: Whether to use weighted Laplacian
    :param doSimple: If true, do a  simple reordering of the first \
        frames in each window.  Otherwise, do a consensus reordering
    :param doPlot: Whether to plot the circular coordinates
    """
    X = getPCAVideo(XOrig)
    print X.shape
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    XS = getSlidingWindowVideo(X, dim, 1, 1)

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
    
    if doPlot:
        plt.subplot(231)
        plt.imshow(D, cmap='afmhot', interpolation='none')
        plt.title("SSM Sliding Window")
        plt.subplot(232)
        plt.imshow(A, interpolation='none', cmap='gray')
        plt.title("Adjacency Matrix")
        plt.subplot(233)
        ripser.plotDGM(I)
        plt.title("H1, Thresh = %g"%thresh)
        plt.subplot(234)
        plt.scatter(v[:, 1], v[:, 2], 20, np.arange(v.shape[0]), cmap = 'spectral')
        plt.title("Eigenvectors")
        plt.subplot(235)
        plt.plot(theta)
        plt.title("Circular Coordinates")
        plt.show()
    
    if doSimple:
        idx = np.argsort(theta)
        return XOrig[idx, :]
    else:
        return getReorderedConsensusVideo()

if __name__ == '__main__':
    filename = "jumpingjacks2men.ogg"
    (I, IDims) = loadImageIOVideo(filename)
    XNew = reorderVideo(I, 30, doSimple = True)
    saveVideo(XNew, IDims, "reordered.avi")
