from VideoTools import *
from VideoReordering import *
import numpy as np
import matplotlib.pyplot as plt

def getVideoPyramid(I, IDims, pyr_level):
    """
    Get a coarse level of the image pyramid for ever frame
    in lieu of the original video
    """
    if pyr_level == 0:
        return I
    from skimage.transform import pyramid_gaussian
    IRet = []
    for i in range(0, I.shape[0]):
        frame = np.reshape(I[i, :], IDims)
        feat_frame = tuple(pyramid_gaussian(frame, pyr_level, downscale = 2))[-1]
        IRet.append(feat_frame.flatten())
    return np.array(IRet)

def getKendallTau(order1, order2):
    """
    Given two global rankings, return the Kendall Tau Score
    """
    N = len(order1)
    rank1 = np.zeros(N)
    rank1[order1] = np.arange(N)
    rank2 = np.zeros(N)
    rank2[order2] = np.arange(N)
    A = np.sign(rank1[None, :] - rank1[:, None])
    B = np.sign(rank2[None, :] - rank2[:, None])
    return np.sum(A*B)/float(N*(N-1))

def getCycles(NCycles, NFinal):
    I, I_feat, IDims = loadImageIOVideo('Videos/SlowMotionTemplate.avi')
    N = I.shape[0]
    theta = np.linspace(0, 2*np.pi, N)
    idx = np.mod(NCycles*np.arange(N), N)
    if NFinal == -1:
        NFinal = N
    idx = idx[0:NFinal]
    theta = theta[idx]
    return {'I':I[idx, :], 'IDims':IDims, 'idx':idx, 'theta':theta}

def doTest():
    res = getCycles(NCycles = 20, NFinal = 250)
    [I, IDims, thetagt] = [res['I'], res['IDims'], res['theta']]
    I += 0.1*np.random.randn(I.shape[0], I.shape[1])
    #I = simulateCameraShake(I, IDims, 20)
    I_feat = getVideoPyramid(I, IDims, 3)
    saveVideo(I, IDims, "simulated.avi")
    Kappa = -1
    Weighted = False
    doSimple = True
    doPlot = True
    fileprefix = "synth"
    res = reorderVideo(I, I_feat, IDims, derivWin = 2, Weighted = Weighted, \
                        doSimple = doSimple, doPlot = doPlot, Verbose = True, \
                        doImageAnalogies = False, Kappa = Kappa, fileprefix = fileprefix)
    theta = np.mod(res['thetau'], 2*np.pi)
    thetagt = thetagt[0:len(theta)]
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.scatter(thetagt, theta)
    plt.subplot(212)
    theta = np.unwrap(theta)
    thetagt = np.unwrap(thetagt)
    rank1 = 0*thetagt
    rank1[np.argsort(thetagt)] = np.arange(len(rank1))
    rank2 = 0*rank1
    rank2[np.argsort(theta)] = np.arange(len(rank2))
    plt.scatter(rank1, rank2)
    plt.savefig("%s_CircCoordsCorr.svg"%fileprefix, bbox_inches = 'tight')
    #saveVideo(res['X'], IDims, fileprefix+".avi")


if __name__ == '__main__':
    doTest()
    #saveVideo(I, IDims, 'out.avi')