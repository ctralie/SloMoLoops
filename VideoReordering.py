from VideoTools import *
from ripser import ripser
from CSMSSMTools import *
from Laplacian import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import scipy.sparse as sparse
import scipy.interpolate as interp

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

def getReorderedConsensusVideo(X, IDims, Mu, VT, dim, theta, doPlot = False, Verbose = False, lookAtVotes = False):
    """
    Given an array of sliding window videos and circular coordinates,
    reorder the sliding windows to go through one period
    over the length of the signal, and use the different lags in
    the window to vote on the final pixels in the reordered video.
    Use linear interpolation to fill in intermediate values in 
    each window.  For speed, do linear interpolation on PCA coordinates
    but perform median on the final pixels
    :param X: An NFrames x NFrames array of mean-subtracted PCA frames \
        (singular value U*S)
    :param IDims: Dimensions of video frames
    :param Mu: A 1 x (NPixels*NChannels) array of the average of all frames
    :param VT: The right singular vectors transposed, to get from PCA\
        back to video.  Dimensions NFrames x (NPixels*NChannels)
    :param dim: The sliding window length
    :param theta: The circular coordinates
    :param doPlot: Whether to make a plot showing all of the windows
    :param Verbose: Whether to print out debugging info
    :param lookAtVotes: Whether to output a video for all of the votes\
        for each frame
    """
    N = X.shape[0]
    M = len(theta)
   
    #Step 1: Figure out the number of periods that the signal
    #goes through
    tu = np.unwrap(theta)
    if tu[-1] - tu[0] < 0:
        tu = -tu
    tu = tu - np.min(tu)
    NPeriods = (float(N)/M)*np.max(tu)/(2*np.pi)
    tu = N*np.mod(tu, 2*np.pi)/(2*np.pi)
    idx = np.argsort(tu)
    t1 = tu[idx]
    
    #Step 2: Go through each window and use it to vote on final samples
    #in projected coordinates
    XInterp = np.nan*np.ones((M, X.shape[0], X.shape[1]))
    pix = np.arange(X.shape[1])
    for i in range(M):
        if Verbose:
            print("Interpolating window %i of %i"%(i+1, M))
        #Figure out the range of the window in the final signal
        ts = t1[i] + NPeriods*np.arange(dim)
        imin = int(np.ceil(np.min(ts)))
        imax = int(np.floor(np.max(ts)))
        t2 = np.arange(imin, imax+1)
        #Interpolate the window to fill in missing samples
        f = scipy.interpolate.interp2d(pix, ts, X[idx[i]:idx[i]+dim, :], kind='linear')
        WinNew = f(pix, t2)
        #saveVideo(WinNew.dot(VT) + Mu, IDims, "%i.avi"%i)
        
        #TODO: Deal with collisions
        XInterp[i, np.mod(t2, N), :] = WinNew
    
    #Step 3: Project the consensus of each frame back, and do a median voting
    print('VT shape:',VT.shape)
    XRet = np.zeros(VT.shape)
    for i in range(VT.shape[0]):
        if Verbose:
            print("Interpolating window %i of %i"%(i+1, N))
        F = XInterp[:, i, :]
        F = F[np.sum(np.isnan(F), 1) == 0, :]
        F = F.dot(VT) + Mu
        if lookAtVotes:
            saveVideo(F, IDims, "%i.avi"%i)
        F = np.median(F, 0)
        #F = np.mean(F, 0)
        XRet[i, :] = F.flatten()
        XRet[i,:] = np.minimum(XRet[i,:],1.0)
        XRet[i,:] = np.maximum(XRet[i,:],0.0)
        mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), np.reshape(XRet[i, :], IDims))
    return XRet
    

def reorderVideo(XOrig, X_feat, dim, derivWin = 10, Weighted = False, doSimple = False, doPlot = True, Verbose = False):
    """
    Reorder the video based on circular coordinates of a sliding
    window embedding
    :param XOrig: An NFrames x (NPixels*NChannels) video array
    :param X_feat: An NFeatFrames x (NPixels*NChannels) video array
    :param dim: Dimension to use in the sliding window
    :param derivWin: Window of derivative to use to help sliding window\
        embedding drift (NOTE: NOT used in final consensus)
    :param Weighted: Whether to use weighted Laplacian
    :param doSimple: If true, do a  simple reordering of the first \
        frames in each window.  Otherwise, do a consensus reordering
    :param doPlot: Whether to plot the circular coordinates
    :param Verbose: Whether to print timing information
    """
    tic = time.time()
    if Verbose:
        print("Doing PCA on video...")
    Mu = np.mean(X_feat, 0)[None, :]
    I = X_feat - Mu
    tic = time.time()
    ICov = I.dot(I.T)
    [lam, U] = linalg.eigh(ICov)
    lam = lam[1::] #Smallest eigenvalue is always zero
    U = U[:, 1::]
    VT = U.T.dot(I)/np.sqrt(lam[:, None])
    X = U*np.sqrt(lam[None, :])
    if Verbose:
        print("Elapsed Time: %g"%(time.time() - tic))
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    XS = getSlidingWindowVideoInteger(X, dim)

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
        print('idx:',idx)
        return XOrig[idx, :]
    else:
        # done for the original video
        Mu_orig = np.mean(XOrig, 0)[None, :]
        I_orig = XOrig - Mu_orig
        ICov_orig = I_orig.dot(I_orig.T)
        [lam_orig, U_orig] = linalg.eigh(ICov_orig)
        lam_orig = lam_orig[1::] #Smallest eigenvalue is always zero
        U_orig = U_orig[:, 1::]
        VT_orig = U_orig.T.dot(I_orig)/np.sqrt(lam_orig[:, None])
        X_proj = U_orig*np.sqrt(lam_orig[None, :])
        print('X proj shape:',X_proj.shape,'X shape:',X.shape)
        return getReorderedConsensusVideo(X_proj, IDims, Mu_orig, VT_orig, dim, theta, doPlot, Verbose, lookAtVotes = False)
        #return getReorderedConsensusVideo(X, IDims, Mu, VT, dim, theta, doPlot, Verbose, lookAtVotes = False)

if __name__ == '__main__':
    from SyntheticVideos import getCircleRotatingVideo
    filename = "jumpingjacks2menlowres.ogg"
    pyr_level=2
    doSimple = False
    I, I_feat, IDims = loadImageIOVideo(filename,pyr_level=pyr_level)
    print('I shape:',I.shape,'I feat shape:',I_feat.shape)
    #(I, IDims) = getCircleRotatingVideo()
    #saveVideo(I, IDims, "circle.avi")
    XNew = reorderVideo(I, I_feat, 10, derivWin = 10, doSimple = doSimple, doPlot = True, Verbose = True)
    prefix = 'simple' if doSimple else 'median'
    saveVideo(XNew, IDims, prefix+"-reordered-"+str(pyr_level)+".avi")
    #saveFrames(XNew, IDims)
