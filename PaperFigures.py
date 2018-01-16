import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interp
from VideoReordering import *
from Laplacian import *
from CSMSSMTools import *
from FundamentalFreq import *
from sklearn.decomposition import PCA
from ripser import ripser
from mpl_toolkits.mplot3d import Axes3D

def drawLineColored(idx, x, C, linewidth = 1):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :], linewidth = linewidth)

def getSlidingWindow(x, dim, Tau, dT):
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(len(x))
    xidx = [] #Index the original samples into the sliding window array (assumes dT evenly divides 1)
    for i in range(NWindows):
        if dT*i == int(dT*i):
            xidx.append(i)
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    while len(xidx) < len(x):
        xidx.append(xidx[-1])
    return (X, xidx)

def MSE(x1, x2):
    return np.mean((x1-x2)**2)

def ReorderingExample1D(seed):
    NPeriods = 20
    SamplesPerPeriod = 12
    N = NPeriods*SamplesPerPeriod
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    cs = [1.0, 1.0, 1.0]
    x = cs[0]*np.cos(t) + cs[1]*np.cos(3*t) + cs[2]*np.cos(5*t)
    np.random.seed(seed)
    x = x + 0.6*np.random.randn(len(x))

    doPlot = False
    dim = int(np.round(estimateFundamentalFreq(x, shortBias = 0.0, doPlot = doPlot)[0]))
    if doPlot:
        plt.show()
    print("dim = %i"%dim)
    (X, xidx) = getSlidingWindow(x, dim, 1, 1)

    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Make ground truth cycle
    t2 = np.linspace(0, 2*np.pi, X.shape[0])
    xgt = cs[0]*np.cos(t2) + cs[1]*np.cos(3*t2) + cs[2]*np.cos(5*t2)

    #Use rips filtration to guide Laplacian
    D = getSSM(X)
    Is = ripser.doRipsFiltrationDM(D, 1, coeff=41)
    I = Is[1]
    thresh = np.argmax(I[:, 1] - I[:, 0])
    thresh = np.mean(I[thresh, :])
    
    #Do both weighted and unweighted Laplacian
    resw = getLapCircularCoordinatesSigma(D, thresh, NEigs = 10)
    resu = getLapCircularCoordinatesThresh(D, thresh, NEigs = 10)
    ress = {'Weighted':resw, 'Unweighted':resu}

    fres = 3
    fig = plt.figure(figsize=(fres*5, fres*3))
    ylims = np.array([-3.5, 3.5])
    plt.subplot(3, 5, 1)
    drawLineColored(np.arange(len(x)), x, C[xidx, :])
    plt.plot([0, 0], ylims*0.9, 'w')
    plt.plot([dim-1]*2, ylims*0.9, 'w')
    plt.plot([0, dim-1], [ylims[0]*0.9]*2, 'w')
    plt.plot([0, dim-1], [ylims[1]*0.9]*2, 'w')
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.ylim(ylims)
    plt.title("Original Signal, Sliding Window d = %i"%dim)
    plt.xlabel("t")

    plt.subplot(3, 5, 2)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    plt.xlabel("Window Index")
    plt.ylabel("Window Index")
    plt.colorbar()
    plt.title("Sliding Window SSM")

    plt.subplot(3, 5, 3)
    lims = [np.min(I), np.max(I)]
    plt.plot(lims, lims, 'k')
    plt.scatter(I[:, 0], I[:, 1], 20)
    imax = np.argmax(I[:, 1] - I[:, 0])
    plt.scatter(I[imax, 0], I[imax, 1], 40)
    plt.plot([lims[0], thresh], [thresh, thresh], 'k', linestyle='--')
    plt.plot([thresh, thresh], [thresh, lims[0]], 'k', linestyle='--')
    plt.plot([I[imax, 0]]*2, I[imax, :], 'r')
    plt.xlabel("Birth Time")
    plt.ylabel("Death Time")
    plt.title("H1 Persistence Diagram")

    for count, resType in enumerate(ress):
        res = ress[resType]
        [w, v, theta, A, idxs] = [res['w'], res['v'], res['theta'], res['A'], res['idxs']]

        tu = np.unwrap(theta)
        if tu[-1] - tu[0] < 0:
            tu = -tu
        tu = tu - np.min(tu)
        ridx = np.argsort(np.mod(tu, 2*np.pi))
        xresort = x[ridx]

        #Do denoising
        (YVotes, y, winidx) = getReorderedConsensus1D(X, X.shape[0], theta, doPlot = doPlot)

        plt.subplot(3, 5, 4+count)
        plt.title("%s Adjacency Matrix"%(resType))
        plt.imshow(np.log(A + 1e-10), cmap='gray', interpolation = 'nearest')
        plt.xlabel("Window Index")
        plt.ylabel("Window Index")
        #plt.imshow(A, cmap = 'gray', interpolation = 'nearest')

        plt.subplot(3, 5, 5*(1+count)+1)
        plt.scatter(v[:, 1], v[:, 2], 20, c=C, edgecolor = 'none')
        plt.xlabel("Eigenvector %i"%idxs[0])
        plt.ylabel("Eigenvector %i"%idxs[1])
        plt.title("%s Laplacian Eigenmap"%resType)
        ax = plt.gca()
        ax.set_axis_bgcolor(plotbgcolor)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.subplot(3, 5, 5*(1+count)+2)
        plt.plot(theta)
        plt.title("Circular Coordinates")
        plt.xlabel("Window Index")
        plt.ylabel("$\\theta$")

        plt.subplot(3, 5, 5*(1+count)+3)
        drawLineColored(np.arange(len(xresort)), xresort, C[ridx], linewidth=1.5)
        #plt.plot(xgt, linestyle='--', color=[0.8]*3, linewidth=4)
        ax = plt.gca()
        ax.set_axis_bgcolor(plotbgcolor)
        plt.ylim(ylims)
        plt.title("Reordered Signal\nMSE = %.3g"%MSE(xresort, xgt))
        plt.xlabel("t'")

        plt.subplot(3, 5, 5*(1+count)+4)
        winidx2orig = 0*winidx
        winidx2orig[winidx] = np.arange(len(winidx))
        for i in range(YVotes.shape[0]):
            idx = winidx2orig[i]
            plt.plot(np.arange(YVotes.shape[1]), YVotes[idx, :], color = C[winidx[idx], :])
        mse = np.mean(np.array([MSE(YVotes[i, :], xgt) for i in range(YVotes.shape[0])]))
        plt.title("Interpolated Window Votes\nAverage MSE = %.3g"%mse)
        plt.xlabel("t'")

        plt.subplot(3, 5, 5*(1+count)+5)
        plt.plot(y, linewidth=4)
        plt.plot(xgt, linestyle='--')
        plt.ylim(ylims)
        plt.title("Spline Consensus reordering\nMSE = %.3g"%MSE(y, xgt))
        plt.xlabel("t'")

if __name__ == '__main__':
    seed = 0
    plt.clf()
    ReorderingExample1D(seed)
    plt.savefig("Paper/Figures/1DExample.svg", bbox_inches = 'tight')