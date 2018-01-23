from VideoTools import *
from VideoReordering import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse

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

def getCircleDist(x, y):
    return np.minimum(np.abs(x - y), 2*np.pi-np.abs(x - y))

def getKendallTauCircle(rank1, rank2):
    """
    Given two global rankings, return the Kendall Tau Score
    on the circle
    """
    N = len(rank1)
    diff = rank1[None, :] - rank1[:, None]
    A = np.sign(diff)
    A[np.abs(A) > np.pi] *= -1 #Wrap around
    diff = rank2[None, :] - rank2[:, None]
    B = np.sign(diff)
    B[np.abs(B) > np.pi] *= -1
    return np.sum(A*B)/float(N*(N-1))

def getCycles(NCycles, NFinal, filename):
    I, I_feat, IDims = loadImageIOVideo(filename)
    N = I.shape[0]
    theta = np.linspace(0, 2*np.pi, N)
    idx = np.mod(NCycles*np.arange(N), N)
    if NFinal == -1:
        NFinal = N
    starti = np.random.randint(len(idx)-NFinal)
    idx = idx[starti:starti+NFinal]
    theta = theta[idx]
    theta = np.unwrap(theta)
    theta = theta - np.min(theta)
    theta = np.mod(theta, 2*np.pi)
    return {'I':I[idx, :], 'IDims':IDims, 'idx':idx, 'theta':theta}

def doTest(filename, NCycles, noise, shake, fileprefix = "", Verbose = False, saveVideos = False):
    doPlot = False
    if len(fileprefix) > 1:
        doPlot = True
    doSimple = True
    #Range over: Period, Noise Amount, Shake Amount
    res = getCycles(NCycles = NCycles, NFinal = 300, filename = filename)
    [I, IDims, thetagt] = [res['I'], res['IDims'], res['theta']]
    I += noise*np.random.randn(I.shape[0], I.shape[1])
    if shake > 0:
        I = simulateCameraShake(I, IDims, shake)
    if saveVideos:
        saveVideo(I, IDims, "%s_simulated.avi"%fileprefix)
    ret = {}
    for pyr_level in range(4):
        I_feat = getVideoPyramid(I, IDims, pyr_level)
        for Kappa in [0, 0.05, 0.1, 0.15]:
            for Weighted in [True, False]:
                thisprefix = "%i_%g_%i"%(pyr_level, Kappa, Weighted)
                res = reorderVideo(I, I_feat, IDims, derivWin = 2, Weighted = Weighted, \
                                    doSimple = doSimple, doPlot = doPlot, Verbose = Verbose, \
                                    doImageAnalogies = False, Kappa = Kappa, fileprefix = fileprefix+thisprefix)
                theta = np.mod(res['thetau'], 2*np.pi)
                thetagt = thetagt[0:len(theta)]
                theta = theta[0:len(thetagt)]
                circErr = np.mean(getCircleDist(theta, thetagt))
                rank1 = 0*thetagt
                rank1[np.argsort(thetagt)] = np.arange(len(rank1))
                rank2 = 0*rank1
                rank2[np.argsort(theta)] = np.arange(len(rank2))
                kendTau = getKendallTauCircle(rank1, rank2)
                ret[thisprefix] = [circErr, kendTau]
                if doPlot:
                    plt.figure(figsize=(8, 6))
                    plt.subplot(211)
                    plt.scatter(thetagt, theta)
                    plt.title("err = %.3g"%circErr)
                    plt.subplot(212)
                    plt.scatter(rank1, rank2)
                    plt.title("$\\tau = %.3g$"%kendTau)
                    plt.savefig("%s_CircCoordsCorr.svg"%(fileprefix+thisprefix), bbox_inches = 'tight')
                if saveVideos:
                    saveVideo(res['X'], IDims, (fileprefix+thisprefix)+".avi")
    return ret

def writeBatchHeader(fout, filename):
    
    fout.write("""<html>
    <body><h1>{}</h1>
    <table border = "1">
    <tr><td>NCycles</td><td>Noise</td><td>Shake</td><td>Trial Number</td>
        <td>Pyramid Level</td><td>Kappa</td><td>Weighted</td><td>Circ Error</td><td>Kendall Tau</td></tr>   
    """.format(filename))

def doBatchTests(filename, fout, batchidx = -1):
    idx = 0
    for NCycles in [5, 10, 20, 30, 40]:
        for noise in [0, 1, 2, 3]:
            for shake in [0, 20, 40, 80]:
                for trial in range(20):
                    if idx == batchidx:
                        ret = doTest(filename, NCycles, noise, shake)
                        for item in ret:
                            fout.write("<tr>")
                            fout.write("<td>%i</td><td>%g</td><td>%i</td><td>%i</td>"%(NCycles, noise, shake, trial))
                            fout.write(("<td>%s</td>"*3)%tuple(item.split("_")))
                            fout.write(("<td>%g</td>"*2)%tuple(ret[item]))
                            fout.write("</tr>\n")
                            fout.flush()
                    idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0, help="batch")
    parser.add_argument('--out_dir', type=str, default=".", help="out directory")
    opt = parser.parse_args()

    filename = 'Videos/SlowMotionTemplateSimple.avi'
    #"""
    outdir = opt.out_dir
    batchidx = opt.batch
    print("Doing Batch %i..."%batchidx)
    fout = open("%s/%i.html"%(outdir, batchidx), "w")
    doBatchTests(filename, fout, batchidx)
    fout.close()
    
    """
    fout = open("results.html", "w")
    writeBatchHeader(fout, filename)
    doBatchTests(filename, fout)
    fout.write("</table></html>")
    """