"""
Purpose: The main algorithm implementation of slow motion loops by sliding window
video reordering
"""
from VideoTools import *
from CSMSSMTools import *
from Laplacian import *
from FundamentalFreq import *
from PatchTools import *
from TDA import *
from ripser import Rips
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import scipy.sparse as sparse
import scipy.interpolate as interp
from sklearn import manifold
import time
import argparse
import os

def getReorderedConsensus1D(X, N, thetau, doPlot = False):
    """
    Given an array of sliding windows and circular coordinates,
    reorder the sliding windows to go through one period
    over the length of the signal, and use the different lags in
    the window to vote on the final samples in the reordered signal.
    Use spline interpolation to fill in intermediate values in 
    each window
    :param X: An Mxd array of windows
    :param N: The original signal length
    :param theta: The unwrapped, increasing circular coordinates
    :param doPlot: Whether to make a plot showing all of the windows
    :return z: An N-length 1D consensus signal going through one period
    """
    #Step 1: Figure out the number of periods that the signal
    #goes through
    M = X.shape[0]
    d = X.shape[1]
    NPeriods = (float(N)/M)*np.max(thetau)/(2*np.pi)
    print("NPeriods = %g"%NPeriods)
    print("N/NPeriods = %g"%(float(N)/NPeriods))
    tu = N*np.mod(thetau, 2*np.pi)/(2*np.pi)
    tidx = np.argsort(tu)
    X2 = X[tidx, :]
    t1 = tu[tidx]
    
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
    return (Z, np.nanmedian(Z, 0), tidx)

def getReorderedConsensusVideo(X, IDims, Mu, VT, dim, thetau, tdifflim = -1, \
                                doPlot = False, Verbose = False, lookAtVotes = False):
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
    :param thetau: The unwrapped circular coordinates
    :param tdifflim: The maximum distance away from a nearest frame to interpolate\
                    (default -1: interpolate all frames)
    :param doPlot: Whether to make a plot showing all of the windows
    :param Verbose: Whether to print out debugging info
    :param lookAtVotes: Whether to output a video for all of the votes\
        for each frame
    """
    N = X.shape[0]
    M = len(thetau)
    print("tdifflim = ", tdifflim)

    #Step 1: Figure out the number of periods that the signal
    #goes through
    NPeriods = (float(N)/M)*np.max(thetau)/(2*np.pi)
    tu = N*np.mod(thetau, 2*np.pi)/(2*np.pi)
    idx = np.argsort(tu)
    t1 = tu[idx]
    
    #Step 2: Go through each window and use it to vote on final samples
    #in projected coordinates
    XInterp = np.zeros((M, X.shape[0], X.shape[1]))
    pix = np.arange(X.shape[1])
    for i in range(M):
        if Verbose:
            print("Interpolating window %i of %i"%(i+1, M))
        #Figure out the range of the window in the final signal
        ts = t1[i] + NPeriods*np.arange(dim)
        imin = int(np.ceil(np.min(ts)))
        imax = int(np.floor(np.max(ts)))
        t2 = np.arange(imin, imax+1)
        tdiffs = np.min(np.abs(t2[None, :] - ts[:, None]), 0)
        #Interpolate the window to fill in missing samples
        f = scipy.interpolate.interp2d(pix, ts, X[idx[i]:idx[i]+dim, :], kind='linear')
        WinNew = f(pix, t2)
        if tdifflim > 0:
            WinNew[tdiffs > tdifflim, :] = np.nan
        #saveVideo(WinNew.dot(VT) + Mu, IDims, "%i.avi"%i)
        
        #Put frames into the proper place, dealing with collisions
        counts = np.zeros(XInterp.shape[1])
        for k in range(WinNew.shape[0]):
            if np.sum(np.isnan(WinNew[k, :])) == 0:
                t2idx = np.mod(t2[k], N)
                counts[t2idx] += 1
                XInterp[i, t2idx, :] = WinNew[k, :]
        for k in range(len(counts)):
            if counts[k] > 0:
                XInterp[i, k, :] /= counts[k]
            else:
                XInterp[i, k, :] = np.nan
    
    #Step 3: Project the consensus of each frame back, and do a median voting
    print('VT shape:',VT.shape)
    XRet = np.zeros(VT.shape)
    for i in range(VT.shape[0]):
        if Verbose:
            print("Taking consensus for window %i of %i"%(i+1, N))
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
        #mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), np.reshape(XRet[i, :], IDims))
    return XRet
    
def downsampleVideo(I, IDims, fac):
    INew = []
    IDimsNew = [int(IDims[0]*fac), int(IDims[1]*fac), 3]
    for i in range(I.shape[0]):
        FNew = resizeImage(np.reshape(I[i, :], IDims), IDimsNew)
        INew.append(FNew.flatten())
    return (np.array(INew), IDimsNew)

def upsampleVideo(I, IDims, IDimsNew):
    INew = []
    for i in range(I.shape[0]):
        FNew = resizeImage(np.reshape(I[i, :], IDims), IDimsNew)
        INew.append(FNew.flatten())
    return np.array(INew)

def reorderVideo(XOrig, X_feat, IDims, derivWin = 10, Weighted = False, \
                doSimple = False, doPlot = True, Verbose = False, \
                fileprefix = "", Kappa = -1, Alpha = 0.5, percentile = True, \
                p = 41, returnAnswer = True, doSlidingWindow = True, \
                expandWindow = False, tdifflim = -1):
    """
    Reorder the video based on circular coordinates of a sliding
    window embedding
    :param XOrig: An NFrames x (NPixels*NChannels) video array
    :param X_feat: An NFeatFrames x (NPixels*NChannels) video array
    :param derivWin: Window of derivative to use to help sliding window\
        embedding drift (NOTE: NOT used in final consensus)
    :param Weighted: Whether to use weighted Laplacian
    :param doSimple: If true, do a  simple reordering of the first \
        frames in each window.  Otherwise, do a consensus reordering
    :param doPlot: Whether to plot the circular coordinates
    :param Verbose: Whether to print timing information
    :param Kappa: If it's a number in [0, 1], then use mutual nearest neighbors\
        instead of TDA-based thresholds
    :param Alpha: A number in [0, 1] that weights \
                    Alpha*birthTime + (1-Alpha)*deathTime as threshold
    :param p: Field coefficient to use in rips filtration
    """
    tic = time.time()
    if Verbose:
        print("Doing PCA on video...")
    Mu = np.mean(X_feat, 0)[None, :]
    I = X_feat - Mu
    tic = time.time()
    ICov = I.dot(I.T)
    [lam, U] = linalg.eigh(ICov)
    pos_lam_inds = lam > 1e-10
    lam = lam[pos_lam_inds]
    U = U[:, pos_lam_inds]
    VT = U.T.dot(I)/np.sqrt(lam[:, None])
    X = U*np.sqrt(lam[None, :])
    if Verbose:
        print("Elapsed Time: %g"%(time.time() - tic))
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    XIso = manifold.Isomap(10, 2).fit_transform(X)
    if doPlot:
        plt.figure(figsize=(18, 12))
    dim = 0
    XS = X
    if doSlidingWindow:
        dim = int(np.round(estimateFundamentalFreq(XIso[:, 0], doPlot = doPlot)['maxTau']))
        if expandWindow:
            for k in range(2):
                #Make window size larger, within reason
                if dim*2 < int(0.25*XOrig.shape[0]):
                    dim *= 2
        if doPlot:
            plt.subplot(211)
            plt.title("Chosen Dim = %i"%dim)
            plt.savefig("%s_FundamentalFreq.svg"%fileprefix, bbox_inches = 'tight')
        #Do sliding window
        XS = getSlidingWindowVideoInteger(X, dim)

    #Mean-center and normalize sliding window
    Y = XS - np.mean(XS, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
    D = getSSM(Y)

    if Kappa > 0 and Kappa < 1:
        if Weighted:
            res = getLapCircularCoordinatesKNNWeighted(D, Kappa, percentile = percentile)
        else:
            res = getLapCircularCoordinatesKNN(D, Kappa, percentile = percentile)
    else:
        if Verbose:
            print("Computing H1 on point cloud of size %i..."%D.shape[0])
        tic = time.time()
        r = Rips(coeff=p, maxdim=1)
        Is = r.fit_transform(D, distance_matrix=True)
        if Verbose:
            print("Elapsed Time H1: %g"%(time.time() - tic))
        I = Is[1]
        imax = np.argmax(I[:, 1] - I[:, 0])
        thresh = Alpha*I[imax, 0] + (1-Alpha)*I[imax, 1]
        print("Thresh = %g, %.3g percentile"%(thresh, np.sum(D < thresh)/D.size))
        tic = time.time()
        if Weighted:
            if doPlot:
                plt.clf()
            res = getLapCircularCoordinatesSigma(D, thresh, doPlot = doPlot)
            if doPlot:
                plt.savefig("%s_Laplacian.svg"%fileprefix, bbox_inches = 'tight')
        else:
            res = getLapCircularCoordinatesThresh(D, thresh)
    [w, v, theta, thetau, A, idxs] = [res['w'], res['v'], res['theta'], res['thetau'], res['A'], res['idxs']]

    if doPlot:
        plt.clf()
        plt.subplot(231)
        plt.imshow(D, cmap='afmhot', interpolation='none')
        plt.title("SSM Sliding Window")
        plt.subplot(232)
        plt.imshow(A, interpolation='none', cmap='gray')
        plt.title("Adjacency Matrix")
        plt.subplot(233)
        if not (Kappa > 0 and Kappa < 1):
            lims = [np.min(I), np.max(I)]
            plt.plot(lims, lims, 'k')
            plt.scatter(I[:, 0], I[:, 1], 20)
            plt.scatter(I[imax, 0], I[imax, 1], 40)
            plt.plot([lims[0], thresh], [thresh, thresh], 'k', linestyle='--')
            plt.plot([thresh, thresh], [thresh, lims[0]], 'k', linestyle='--')
            plt.plot([I[imax, 0]]*2, I[imax, :], 'r')
            plt.title("H1, Thresh = %g, Percentile = %.3g"%(thresh, np.sum(D < thresh)/D.size))
        plt.subplot(234)
        plt.imshow(v, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.subplot(235)
        plt.scatter(v[:, idxs[0]], v[:, idxs[1]], 20, np.arange(v.shape[0]), cmap = 'spectral')
        plt.title("Eigs %i and %i"%(idxs[0], idxs[1]))
        plt.subplot(236)
        plt.plot(theta)
        plt.title("Circular Coordinates")
        plt.savefig("%s_CircCoords.svg"%fileprefix, bbox_inches = 'tight')
    
    XRet = None
    if doSimple:
        idx = np.argsort(np.mod(thetau, 2*np.pi))
        if returnAnswer:
            XRet = XOrig[idx, :]
    else:
        # done for the original video
        Mu_orig = np.mean(XOrig, 0)
        I_orig = XOrig - Mu_orig
        ICov_orig = I_orig.dot(I_orig.T)
        [lam_orig, U_orig] = linalg.eigh(ICov_orig)
        pos_lam_orig_inds = lam_orig > 1e-10
        lam_orig = lam_orig[pos_lam_orig_inds] #Smallest eigenvalue is always zero
        U_orig = U_orig[:, pos_lam_orig_inds]
        VT_orig = U_orig.T.dot(I_orig)/np.sqrt(lam_orig[:, None])
        X_proj = U_orig*np.sqrt(lam_orig[None, :])
        XRet = getReorderedConsensusVideo(X_proj, IDims, Mu_orig, VT_orig, dim, thetau, \
            doPlot=doPlot, Verbose=Verbose, lookAtVotes = False, tdifflim = tdifflim)
    return {'X':XRet, 'IDims':IDims, 'theta':theta, 'thetau':thetau, 'A':A, 'v':v}

def get_out_fileprefix(base_filename, inputfilename, do_simple, is_weighted, \
                        is_net_feat, pyr_level=0, layer=0, Kappa=0):
    s = inputfilename.split("/")[-1]
    s = s.split(".")[0]
    filename = "Results/"+s+"-"+str(base_filename)+'-'
    filename = filename+"%g"%Kappa
    filename = filename+'-simple' if do_simple else filename+'-median'
    filename = filename+'-weighted' if is_weighted else filename+'-unweighted'
    filename = filename+'-net-'+str(layer) if is_net_feat else filename+'-img-'+str(pyr_level)
    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-plots', dest='doPlot', action='store_true', help='Show plots of circular coordinates')
    parser.add_argument('--is-median-reorder', dest='median_reorder', action='store_true', help='enable median reordering')
    parser.add_argument('--is-simple-reorder', dest='median_reorder', action='store_false', help='enable simple reordering')
    parser.add_argument('--is-weighted-laplacian', dest='weighted_laplacian', action='store_true', help='enable weighted laplacian')
    parser.add_argument('--is-unweighted-laplacian', dest='weighted_laplacian', action='store_false', help='enable unweighted laplacian')
    parser.add_argument('--is-net-feat', dest='net_feat', action='store_true', help='enable resnet features')
    parser.add_argument('--is-pyr-feat', dest='net_feat', action='store_false', help='enable gaussian pyramid features')
    parser.add_argument('--pyr_level', type=int, default=0, help="pyramid level")
    parser.add_argument('--net_depth', type=int, default=0, help="at what layer do we extract features")
    parser.add_argument('--tdifflim', type=int, default=-1, help="The maximum frame number away from data to interpolate")
    parser.add_argument('--Kappa', type=float, default=0, help="Using nearest neighbors or TDA")
    parser.add_argument('--filename', default='jumpingjacks2menlowres.ogg', help="video filename")
    parser.set_defaults(median_reorder=False)
    parser.set_defaults(weighted_laplacian=False)
    parser.set_defaults(net_feat=False)

    opt = parser.parse_args()

    print("weighted_laplacian = ", opt.weighted_laplacian)
    I, I_feat, IDims = loadVideoResNetFeats(opt.filename,opt.net_depth) if opt.net_feat else loadImageIOVideo(opt.filename,pyr_level=opt.pyr_level)
    print('I shape:',I.shape,'I feat shape:',I_feat.shape)

    if not os.path.exists("Results"):
        os.mkdir("Results")
    fileprefix = get_out_fileprefix('reordered', opt.filename, (not opt.median_reorder), opt.weighted_laplacian, opt.net_feat, opt.pyr_level, opt.net_depth, Kappa = opt.Kappa)
    XNew = reorderVideo(I, I_feat, IDims, derivWin = 0, Weighted = opt.weighted_laplacian, \
                        doSimple = (not opt.median_reorder), doPlot = opt.doPlot, Verbose = True, \
                        fileprefix = fileprefix, Kappa=opt.Kappa, tdifflim=opt.tdifflim)['X']
    saveVideo(XNew, IDims, fileprefix+".avi")
